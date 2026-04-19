"""Phoenix ranking model implemented with TorchRec."""

from typing import List, Optional

import torch
import torch.nn as nn
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# Criteo feature vocab sizes (from stats.json)
CRITEO_VOCAB_SIZES = {
    "C1": 568,
    "C2": 535,
    "C3": 13542,
    "C4": 16222,
    "C5": 157,
    "C6": 13,
    "C7": 9319,
    "C8": 270,
    "C9": 4,
    "C10": 12828,
    "C11": 4287,
    "C12": 13842,
    "C13": 3056,
    "C14": 27,
    "C15": 5986,
    "C16": 14721,
    "C17": 11,
    "C18": 2912,
    "C19": 1457,
    "C20": 4,
    "C21": 14279,
    "C22": 10,
    "C23": 15,
    "C24": 10980,
    "C25": 54,
    "C26": 8768,
}


def build_embedding_bag_collection(
    embedding_dim: int,
    device: torch.device,
) -> EmbeddingBagCollection:
    """Build an EmbeddingBagCollection for all Criteo categorical features."""
    tables = []
    for feat_name, vocab_size in CRITEO_VOCAB_SIZES.items():
        tables.append(
            EmbeddingBagConfig(
                name=f"{feat_name}_table",
                embedding_dim=embedding_dim,
                num_embeddings=vocab_size,
                feature_names=[feat_name],
            )
        )
    return EmbeddingBagCollection(tables=tables, device=device)


class PhoenixTorchModel(nn.Module):
    """Phoenix-style ranking model using TorchRec embeddings + Transformer.

    Architecture:
      1. EmbeddingBagCollection for 26 Criteo categorical features
      2. Group projections: user (C1-C2), post (C3-C12), author (C13-C22), surface (C23)
      3. Candidate projection combining post + author + surface
      4. Transformer encoder over [user, candidate] sequence
      5. Output logit from the candidate position
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        emb_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 2,
        dropout: float = 0.0,
        dense_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.embedding_bag_collection = embedding_bag_collection
        self.emb_size = emb_size
        self.dense_device = dense_device or torch.device("cpu")

        embedding_dim = embedding_bag_collection.embedding_bag_configs()[0].embedding_dim

        # Group projections
        self.user_proj = nn.Linear(2 * embedding_dim, emb_size, device=self.dense_device)
        self.post_proj = nn.Linear(10 * embedding_dim, emb_size, device=self.dense_device)
        self.author_proj = nn.Linear(10 * embedding_dim, emb_size, device=self.dense_device)
        self.surface_proj = nn.Linear(1 * embedding_dim, emb_size, device=self.dense_device)

        # Candidate combines post + author + surface
        self.candidate_proj = nn.Linear(3 * emb_size, emb_size, device=self.dense_device)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=emb_size * 2,
            dropout=dropout,
            batch_first=True,
            device=self.dense_device,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final output projection
        self.output_proj = nn.Linear(emb_size, 1, device=self.dense_device)

    def forward(self, sparse_features: KeyedJaggedTensor) -> torch.Tensor:
        """Forward pass.

        Args:
            sparse_features: KeyedJaggedTensor with keys C1..C23.

        Returns:
            logits: Tensor of shape (B, 1).
        """
        # 1) Sparse lookup -> KeyedTensor
        pooled: torch.Tensor = self.embedding_bag_collection(sparse_features)
        # pooled.values() shape: (B, 26 * embedding_dim)

        B = pooled.values().shape[0]
        embedding_dim = self.embedding_bag_collection.embedding_bag_configs()[0].embedding_dim

        # Reshape to (B, 26, embedding_dim) to extract per-feature embeddings easily
        # The order follows the table definition order (C1, C2, ..., C26)
        all_embs = pooled.values().reshape(B, 26, embedding_dim)

        # Extract groups
        user_embs = all_embs[:, 0:2, :].reshape(B, -1)          # C1, C2
        post_embs = all_embs[:, 2:12, :].reshape(B, -1)         # C3-C12
        author_embs = all_embs[:, 12:22, :].reshape(B, -1)      # C13-C22
        surface_embs = all_embs[:, 22:23, :].reshape(B, -1)     # C23

        # Project groups
        user_vec = self.user_proj(user_embs)                    # (B, emb_size)
        post_vec = self.post_proj(post_embs)                    # (B, emb_size)
        author_vec = self.author_proj(author_embs)              # (B, emb_size)
        surface_vec = self.surface_proj(surface_embs)           # (B, emb_size)

        # Candidate representation
        candidate_vec = self.candidate_proj(
            torch.cat([post_vec, author_vec, surface_vec], dim=-1)
        )  # (B, emb_size)

        # Sequence: [user, candidate]
        seq = torch.stack([user_vec, candidate_vec], dim=1)     # (B, 2, emb_size)

        # Transformer (causal mask not strictly needed for length 2,
        # but we add it to match the Phoenix skeleton)
        seq_len = seq.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=seq.device)
        transformed = self.transformer(seq, mask=causal_mask, is_causal=True)  # (B, 2, emb_size)

        # Take candidate position (index 1)
        candidate_out = transformed[:, 1, :]                    # (B, emb_size)

        logits = self.output_proj(candidate_out)                # (B, 1)
        return logits
