# Copyright 2026 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script for the Phoenix ranking model on Criteo dataset."""

import glob
import logging
import os
from typing import Iterator, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pyarrow.parquet as pq

from grok import TransformerConfig
from recsys_model import (
    HashConfig,
    PhoenixModel,
    PhoenixModelConfig,
    RecsysBatch,
    RecsysEmbeddings,
    RecsysModelOutput,
)

logger = logging.getLogger(__name__)

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


def load_parquet_files(data_dir: str) -> list[str]:
    """Load all parquet files from a directory."""
    pattern = os.path.join(data_dir, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f"No parquet files found in {data_dir}")
    return files


def read_parquet_batch(file_path: str, batch_size: int, offset: int = 0) -> pd.DataFrame:
    """Read a batch of rows from a parquet file."""
    table = pq.read_table(file_path)
    df = table.to_pandas()
    return df.iloc[offset : offset + batch_size]


class CriteoDataLoader:
    """DataLoader for Criteo dataset mapped to RecsysBatch format."""

    def __init__(
        self,
        data_dir: str,
        emb_size: int,
        batch_size: int,
        hash_config: HashConfig,
        history_seq_len: int = 1,
        candidate_seq_len: int = 1,
        num_actions: int = 1,
        rng_seed: int = 42,
    ):
        self.files = load_parquet_files(data_dir)
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.hash_config = hash_config
        self.history_seq_len = history_seq_len
        self.candidate_seq_len = candidate_seq_len
        self.num_actions = num_actions
        self.rng = np.random.default_rng(rng_seed)

        # Pre-initialize embedding tables for categorical features
        self.embeddings = {}
        for col, vocab_size in CRITEO_VOCAB_SIZES.items():
            self.embeddings[col] = self.rng.normal(
                size=(vocab_size, emb_size)
            ).astype(np.float32)

    def _df_to_batch(
        self, df: pd.DataFrame
    ) -> Tuple[RecsysBatch, RecsysEmbeddings, np.ndarray]:
        """Convert a pandas DataFrame to RecsysBatch, RecsysEmbeddings, and labels."""
        B = len(df)
        hc = self.hash_config

        # Fill NaN with 0 for categorical features
        for col in CRITEO_VOCAB_SIZES:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # Map categorical features to hashes
        user_hashes = np.stack([df["C1"].values, df["C2"].values], axis=1).astype(np.int32)

        candidate_post_hashes = np.stack(
            [df[f"C{i}"].values for i in range(3, 13)], axis=1
        ).astype(np.int32)
        candidate_post_hashes = candidate_post_hashes.reshape(B, 1, hc.num_item_hashes)

        candidate_author_hashes = np.stack(
            [df[f"C{i}"].values for i in range(13, 23)], axis=1
        ).astype(np.int32)
        candidate_author_hashes = candidate_author_hashes.reshape(B, 1, hc.num_author_hashes)

        candidate_product_surface = df["C23"].fillna(0).astype(int).values.reshape(B, 1)

        # History is empty (padding)
        history_post_hashes = np.zeros((B, self.history_seq_len, hc.num_item_hashes), np.int32)
        history_author_hashes = np.zeros((B, self.history_seq_len, hc.num_author_hashes), np.int32)
        history_actions = np.zeros((B, self.history_seq_len, self.num_actions), np.float32)
        history_product_surface = np.zeros((B, self.history_seq_len), np.int32)

        batch = RecsysBatch(
            user_hashes=user_hashes,
            history_post_hashes=history_post_hashes,
            history_author_hashes=history_author_hashes,
            history_actions=history_actions,
            history_product_surface=history_product_surface,
            candidate_post_hashes=candidate_post_hashes,
            candidate_author_hashes=candidate_author_hashes,
            candidate_product_surface=candidate_product_surface,
        )

        # Look up embeddings for hashes
        user_embeddings = np.stack(
            [self.embeddings["C1"][user_hashes[:, 0]], self.embeddings["C2"][user_hashes[:, 1]]],
            axis=1,
        )

        candidate_post_embeddings = np.stack(
            [
                self.embeddings[f"C{i}"][candidate_post_hashes[:, 0, j]]
                for i, j in zip(range(3, 13), range(hc.num_item_hashes))
            ],
            axis=2,
        ).reshape(B, self.candidate_seq_len, hc.num_item_hashes, self.emb_size)

        candidate_author_embeddings = np.stack(
            [
                self.embeddings[f"C{i}"][candidate_author_hashes[:, 0, j]]
                for i, j in zip(range(13, 23), range(hc.num_author_hashes))
            ],
            axis=2,
        ).reshape(B, self.candidate_seq_len, hc.num_author_hashes, self.emb_size)

        history_post_embeddings = np.zeros(
            (B, self.history_seq_len, hc.num_item_hashes, self.emb_size), np.float32
        )
        history_author_embeddings = np.zeros(
            (B, self.history_seq_len, hc.num_author_hashes, self.emb_size), np.float32
        )

        recsys_embeddings = RecsysEmbeddings(
            user_embeddings=user_embeddings,
            history_post_embeddings=history_post_embeddings,
            candidate_post_embeddings=candidate_post_embeddings,
            history_author_embeddings=history_author_embeddings,
            candidate_author_embeddings=candidate_author_embeddings,
        )

        labels = df["label"].fillna(0).values.astype(np.float32).reshape(B, 1, 1)

        return batch, recsys_embeddings, labels

    def iterate(
        self, max_batches: int | None = None
    ) -> Iterator[Tuple[RecsysBatch, RecsysEmbeddings, np.ndarray]]:
        """Iterate over batches from all parquet files."""
        batch_count = 0
        for file_path in self.files:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            num_rows = len(df)
            for start in range(0, num_rows, self.batch_size):
                if max_batches is not None and batch_count >= max_batches:
                    return
                end = min(start + self.batch_size, num_rows)
                batch_df = df.iloc[start:end]
                yield self._df_to_batch(batch_df)
                batch_count += 1


def make_forward_fn(model_config: PhoenixModelConfig):
    """Create the forward function for training."""

    def forward(batch: RecsysBatch, recsys_embeddings: RecsysEmbeddings) -> RecsysModelOutput:
        return model_config.make()(batch, recsys_embeddings)

    return hk.transform(forward)


def make_loss_fn(forward_fn):
    """Create the loss function for binary classification."""

    def loss_fn(
        params: hk.Params,
        batch: RecsysBatch,
        recsys_embeddings: RecsysEmbeddings,
        labels: jax.Array,
    ) -> Tuple[jax.Array, dict]:
        output = forward_fn.apply(params, None, batch, recsys_embeddings)
        logits = output.logits  # [B, C, 1]

        # Binary cross-entropy
        log_probs = jax.nn.log_sigmoid(logits)
        log_probs_neg = jax.nn.log_sigmoid(-logits)
        bce = -(labels * log_probs + (1.0 - labels) * log_probs_neg)
        loss = jnp.mean(bce)

        # Metrics
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        accuracy = jnp.mean(preds == labels)
        avg_pred = jnp.mean(jax.nn.sigmoid(logits))
        avg_label = jnp.mean(labels)

        return loss, {
            "loss": loss,
            "accuracy": accuracy,
            "avg_pred": avg_pred,
            "avg_label": avg_label,
        }

    return loss_fn


def train(
    train_data_dir: str,
    valid_data_dir: str | None = None,
    num_epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    emb_size: int = 64,
    history_seq_len: int = 1,
    candidate_seq_len: int = 1,
    max_batches_per_epoch: int | None = None,
    validate_every_n_batches: int = 100,
    key_size: int = 32,
    num_layers: int = 2,
):
    """Train the Phoenix ranking model on Criteo data."""

    hash_config = HashConfig(
        num_user_hashes=2,
        num_item_hashes=10,
        num_author_hashes=10,
    )

    model_config = PhoenixModelConfig(
        emb_size=emb_size,
        num_actions=1,  # Binary click prediction
        history_seq_len=history_seq_len,
        candidate_seq_len=candidate_seq_len,
        hash_config=hash_config,
        product_surface_vocab_size=16,
        model=TransformerConfig(
            emb_size=emb_size,
            widening_factor=2,
            key_size=key_size,
            num_q_heads=2,
            num_kv_heads=2,
            num_layers=num_layers,
            attn_output_multiplier=0.125,
        ),
        fprop_dtype=jnp.float32,  # Use float32 for CPU training
    )

    # Initialize model
    forward_fn = make_forward_fn(model_config)

    # Dummy data for initialization
    dummy_batch = RecsysBatch(
        user_hashes=jnp.zeros((1, hash_config.num_user_hashes), jnp.int32),
        history_post_hashes=jnp.zeros(
            (1, history_seq_len, hash_config.num_item_hashes), jnp.int32
        ),
        history_author_hashes=jnp.zeros(
            (1, history_seq_len, hash_config.num_author_hashes), jnp.int32
        ),
        history_actions=jnp.zeros((1, history_seq_len, 1), jnp.float32),
        history_product_surface=jnp.zeros((1, history_seq_len), jnp.int32),
        candidate_post_hashes=jnp.zeros(
            (1, candidate_seq_len, hash_config.num_item_hashes), jnp.int32
        ),
        candidate_author_hashes=jnp.zeros(
            (1, candidate_seq_len, hash_config.num_author_hashes), jnp.int32
        ),
        candidate_product_surface=jnp.zeros((1, candidate_seq_len), jnp.int32),
    )
    dummy_embeddings = RecsysEmbeddings(
        user_embeddings=jnp.zeros((1, hash_config.num_user_hashes, emb_size), jnp.float32),
        history_post_embeddings=jnp.zeros(
            (1, history_seq_len, hash_config.num_item_hashes, emb_size), jnp.float32
        ),
        candidate_post_embeddings=jnp.zeros(
            (1, candidate_seq_len, hash_config.num_item_hashes, emb_size), jnp.float32
        ),
        history_author_embeddings=jnp.zeros(
            (1, history_seq_len, hash_config.num_author_hashes, emb_size), jnp.float32
        ),
        candidate_author_embeddings=jnp.zeros(
            (1, candidate_seq_len, hash_config.num_author_hashes, emb_size), jnp.float32
        ),
    )

    rng = jax.random.PRNGKey(42)
    params = forward_fn.init(rng, dummy_batch, dummy_embeddings)

    # Count parameters
    param_count = sum(x.size for x in jax.tree.leaves(params))
    logger.info(f"Model initialized with {param_count:,} parameters")

    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    loss_fn = make_loss_fn(forward_fn)

    @jax.jit
    def train_step(params, opt_state, batch, embeddings, labels):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, embeddings, labels
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    @jax.jit
    def eval_step(params, batch, embeddings, labels):
        loss, metrics = loss_fn(params, batch, embeddings, labels)
        return metrics

    # Data loaders
    train_loader = CriteoDataLoader(
        train_data_dir,
        emb_size=emb_size,
        batch_size=batch_size,
        hash_config=hash_config,
        history_seq_len=history_seq_len,
        candidate_seq_len=candidate_seq_len,
        num_actions=1,
    )

    valid_loader = None
    if valid_data_dir:
        valid_loader = CriteoDataLoader(
            valid_data_dir,
            emb_size=emb_size,
            batch_size=batch_size,
            hash_config=hash_config,
            history_seq_len=history_seq_len,
            candidate_seq_len=candidate_seq_len,
            num_actions=1,
        )

    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")

        # Training
        train_metrics = []
        for batch_idx, (batch, embeddings, labels) in enumerate(
            train_loader.iterate(max_batches=max_batches_per_epoch)
        ):
            # Convert to JAX arrays
            batch = RecsysBatch(
                user_hashes=jnp.array(batch.user_hashes),
                history_post_hashes=jnp.array(batch.history_post_hashes),
                history_author_hashes=jnp.array(batch.history_author_hashes),
                history_actions=jnp.array(batch.history_actions),
                history_product_surface=jnp.array(batch.history_product_surface),
                candidate_post_hashes=jnp.array(batch.candidate_post_hashes),
                candidate_author_hashes=jnp.array(batch.candidate_author_hashes),
                candidate_product_surface=jnp.array(batch.candidate_product_surface),
            )
            embeddings = RecsysEmbeddings(
                user_embeddings=jnp.array(embeddings.user_embeddings),
                history_post_embeddings=jnp.array(embeddings.history_post_embeddings),
                candidate_post_embeddings=jnp.array(embeddings.candidate_post_embeddings),
                history_author_embeddings=jnp.array(embeddings.history_author_embeddings),
                candidate_author_embeddings=jnp.array(embeddings.candidate_author_embeddings),
            )
            labels = jnp.array(labels)

            params, opt_state, metrics = train_step(
                params, opt_state, batch, embeddings, labels
            )
            train_metrics.append(metrics)

            if (batch_idx + 1) % validate_every_n_batches == 0:
                avg_metrics = {
                    k: float(np.mean([float(m[k]) for m in train_metrics[-validate_every_n_batches:]]))
                    for k in train_metrics[0].keys()
                }
                logger.info(f"  Batch {batch_idx + 1}: {avg_metrics}")

        # Epoch summary
        avg_train_metrics = {
            k: float(np.mean([float(m[k]) for m in train_metrics]))
            for k in train_metrics[0].keys()
        }
        logger.info(f"Train metrics: {avg_train_metrics}")

        # Validation
        if valid_loader:
            valid_metrics = []
            for batch, embeddings, labels in valid_loader.iterate(
                max_batches=max_batches_per_epoch
            ):
                batch = RecsysBatch(
                    user_hashes=jnp.array(batch.user_hashes),
                    history_post_hashes=jnp.array(batch.history_post_hashes),
                    history_author_hashes=jnp.array(batch.history_author_hashes),
                    history_actions=jnp.array(batch.history_actions),
                    history_product_surface=jnp.array(batch.history_product_surface),
                    candidate_post_hashes=jnp.array(batch.candidate_post_hashes),
                    candidate_author_hashes=jnp.array(batch.candidate_author_hashes),
                    candidate_product_surface=jnp.array(batch.candidate_product_surface),
                )
                embeddings = RecsysEmbeddings(
                    user_embeddings=jnp.array(embeddings.user_embeddings),
                    history_post_embeddings=jnp.array(embeddings.history_post_embeddings),
                    candidate_post_embeddings=jnp.array(embeddings.candidate_post_embeddings),
                    history_author_embeddings=jnp.array(embeddings.history_author_embeddings),
                    candidate_author_embeddings=jnp.array(embeddings.candidate_author_embeddings),
                )
                labels = jnp.array(labels)

                metrics = eval_step(params, batch, embeddings, labels)
                valid_metrics.append(metrics)

            avg_valid_metrics = {
                k: float(np.mean([float(m[k]) for m in valid_metrics]))
                for k in valid_metrics[0].keys()
            }
            logger.info(f"Valid metrics: {avg_valid_metrics}")

    logger.info("Training complete!")
    return params


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Phoenix ranker on Criteo")
    parser.add_argument(
        "--train-dir",
        default="../datasets/criteo-parquets/criteo-parquet-subset-preprocessed/train",
        help="Path to training parquet files",
    )
    parser.add_argument(
        "--valid-dir",
        default="../datasets/criteo-parquets/criteo-parquet-subset-preprocessed/valid",
        help="Path to validation parquet files",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--small", action="store_true", help="Use a small model config for fast testing")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    emb_size = args.emb_size
    key_size = 32
    num_layers = args.num_layers
    if args.small:
        emb_size = 16
        key_size = 8
        num_layers = 1
        logger.info("Using small model config for fast testing")

    train(
        train_data_dir=args.train_dir,
        valid_data_dir=args.valid_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        emb_size=emb_size,
        max_batches_per_epoch=args.max_batches,
        validate_every_n_batches=args.validate_every,
        key_size=key_size,
        num_layers=num_layers,
    )


if __name__ == "__main__":
    main()
