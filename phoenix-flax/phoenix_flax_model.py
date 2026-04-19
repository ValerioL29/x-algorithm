"""Phoenix ranking model ported to Flax."""

from typing import Any, NamedTuple, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


class HashConfig:
    """Configuration for hash-based embeddings."""

    def __init__(self, num_user_hashes: int = 2, num_item_hashes: int = 10, num_author_hashes: int = 10):
        self.num_user_hashes = num_user_hashes
        self.num_item_hashes = num_item_hashes
        self.num_author_hashes = num_author_hashes


class RecsysBatch(NamedTuple):
    """Input batch for the recommendation model."""

    user_hashes: jax.Array
    history_post_hashes: jax.Array
    history_author_hashes: jax.Array
    history_actions: jax.Array
    history_product_surface: jax.Array
    candidate_post_hashes: jax.Array
    candidate_author_hashes: jax.Array
    candidate_product_surface: jax.Array


class TransformerBlock(nn.Module):
    """Single transformer encoder block with causal attention."""

    emb_size: int
    num_heads: int
    widening_factor: float = 2.0
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        """Forward pass.

        Args:
            x: (B, S, D)

        Returns:
            (B, S, D)
        """
        # Pre-norm attention
        residual = x
        x = nn.LayerNorm()(x)
        # Causal mask for self-attention
        seq_len = x.shape[1]
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.emb_size,
            out_features=self.emb_size,
            dropout_rate=self.dropout,
            deterministic=not train,
        )(x, x, mask=mask)
        x = residual + attn_out

        # Pre-norm FFN
        residual = x
        x = nn.LayerNorm()(x)
        ff = nn.Dense(int(self.emb_size * self.widening_factor))(x)
        ff = nn.gelu(ff)
        ff = nn.Dense(self.emb_size)(ff)
        x = residual + ff

        return x


class PhoenixTransformer(nn.Module):
    """Simple transformer encoder."""

    emb_size: int
    num_layers: int
    num_heads: int = 2
    widening_factor: float = 2.0
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        for _ in range(self.num_layers):
            x = TransformerBlock(
                emb_size=self.emb_size,
                num_heads=self.num_heads,
                widening_factor=self.widening_factor,
                dropout=self.dropout,
            )(x, train=train)
        x = nn.LayerNorm()(x)
        return x


class PhoenixFlaxModel(nn.Module):
    """Phoenix ranking model in Flax.

    Architecture mirrors the original Haiku implementation but with
    embedding tables learned inside the model.
    """

    emb_size: int = 128
    num_actions: int = 1
    num_user_hashes: int = 2
    num_item_hashes: int = 10
    num_author_hashes: int = 10
    history_seq_len: int = 1
    candidate_seq_len: int = 1
    product_surface_vocab_size: int = 16
    num_layers: int = 4
    num_heads: int = 2
    dropout: float = 0.0

    # Embedding table sizes
    num_user_embeddings: int = 600
    num_post_embeddings: int = 16300
    num_author_embeddings: int = 14800

    @nn.compact
    def __call__(self, batch: RecsysBatch, train: bool = True) -> jax.Array:
        """Forward pass.

        Args:
            batch: RecsysBatch containing hashes.

        Returns:
            logits: (B, candidate_seq_len, 1) for binary classification.
        """
        # Embedding tables
        user_embed = nn.Embed(num_embeddings=self.num_user_embeddings, features=self.emb_size)
        post_embed = nn.Embed(num_embeddings=self.num_post_embeddings, features=self.emb_size)
        author_embed = nn.Embed(num_embeddings=self.num_author_embeddings, features=self.emb_size)
        surface_embed = nn.Embed(
            num_embeddings=self.product_surface_vocab_size, features=self.emb_size
        )
        action_embed = nn.Dense(self.emb_size)  # projects multi-hot actions

        # Look up embeddings (frozen, matching original Phoenix behavior)
        # user: (B, num_user_hashes, emb_size)
        user_embs = jax.lax.stop_gradient(user_embed(batch.user_hashes))
        # history_post: (B, S, num_item_hashes, emb_size)
        history_post_embs = jax.lax.stop_gradient(post_embed(batch.history_post_hashes))
        history_author_embs = jax.lax.stop_gradient(author_embed(batch.history_author_hashes))
        history_surface_embs = jax.lax.stop_gradient(surface_embed(batch.history_product_surface))
        history_action_embs = action_embed(batch.history_actions)
        # candidate_post: (B, C, num_item_hashes, emb_size)
        candidate_post_embs = jax.lax.stop_gradient(post_embed(batch.candidate_post_hashes))
        candidate_author_embs = jax.lax.stop_gradient(author_embed(batch.candidate_author_hashes))
        candidate_surface_embs = jax.lax.stop_gradient(surface_embed(batch.candidate_product_surface))

        # Reduce blocks (same logic as original, ported to Flax)
        # User reduce
        B = user_embs.shape[0]
        user_flat = user_embs.reshape(B, self.num_user_hashes * self.emb_size)
        user_vec = nn.Dense(self.emb_size)(user_flat)  # (B, emb_size)
        user_vec = user_vec[:, None, :]  # (B, 1, emb_size)

        # History reduce (if history_seq_len > 0)
        S = self.history_seq_len
        if S > 0:
            history_post_flat = history_post_embs.reshape(B, S, self.num_item_hashes * self.emb_size)
            history_author_flat = history_author_embs.reshape(
                B, S, self.num_author_hashes * self.emb_size
            )
            history_combined = jnp.concatenate(
                [history_post_flat, history_author_flat, history_action_embs, history_surface_embs],
                axis=-1,
            )
            history_vec = nn.Dense(self.emb_size)(history_combined)  # (B, S, emb_size)
        else:
            history_vec = jnp.zeros((B, 0, self.emb_size), dtype=user_vec.dtype)

        # Candidate reduce
        C = self.candidate_seq_len
        candidate_post_flat = candidate_post_embs.reshape(B, C, self.num_item_hashes * self.emb_size)
        candidate_author_flat = candidate_author_embs.reshape(
            B, C, self.num_author_hashes * self.emb_size
        )
        candidate_combined = jnp.concatenate(
            [candidate_post_flat, candidate_author_flat, candidate_surface_embs],
            axis=-1,
        )
        candidate_vec = nn.Dense(self.emb_size)(candidate_combined)  # (B, C, emb_size)

        # Build sequence: [user, history, candidate]
        embeddings = jnp.concatenate([user_vec, history_vec, candidate_vec], axis=1)
        # (B, 1 + S + C, emb_size)

        # Transformer
        transformed = PhoenixTransformer(
            emb_size=self.emb_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )(embeddings, train=train)

        # Extract candidate positions
        candidate_start = 1 + S
        candidate_out = transformed[:, candidate_start:, :]  # (B, C, emb_size)

        # Output projection
        logits = nn.Dense(1)(candidate_out)  # (B, C, 1)
        return logits
