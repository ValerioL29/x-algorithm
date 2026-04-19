"""Training script for Phoenix Flax ranking model on Criteo dataset."""

import argparse
import glob
import logging
import os
from typing import Iterator, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pyarrow.parquet as pq
from flax.training import train_state

from phoenix_flax_model import HashConfig, PhoenixFlaxModel, RecsysBatch

logger = logging.getLogger(__name__)

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
    pattern = os.path.join(data_dir, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f"No parquet files found in {data_dir}")
    return files


class CriteoFlaxDataLoader:
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

    def _df_to_batch(self, df: pd.DataFrame) -> Tuple[RecsysBatch, jax.Array]:
        """Convert DataFrame to RecsysBatch and labels."""
        B = len(df)
        hc = self.hash_config

        for col in CRITEO_VOCAB_SIZES:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

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

        labels = df["label"].fillna(0).values.astype(np.float32).reshape(B, 1, 1)
        return batch, labels

    def iterate(
        self, max_batches: int | None = None
    ) -> Iterator[Tuple[RecsysBatch, jax.Array]]:
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
                batch, labels = self._df_to_batch(batch_df)
                yield batch, labels
                batch_count += 1


def create_train_state(rng, model, batch, learning_rate):
    """Create Flax TrainState."""

    def init_fn(batch):
        return model.init(rng, batch, train=False)

    params = init_fn(batch)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch, labels):
    """Single training step."""

    def loss_fn(params):
        logits = state.apply_fn(params, batch, train=True)
        # Binary cross-entropy
        log_probs = jax.nn.log_sigmoid(logits)
        log_probs_neg = jax.nn.log_sigmoid(-logits)
        bce = -(labels * log_probs + (1.0 - labels) * log_probs_neg)
        loss = jnp.mean(bce)

        probs = jax.nn.sigmoid(logits)
        preds = (probs > 0.5).astype(jnp.float32)
        accuracy = jnp.mean(preds == labels)
        return loss, {
            "loss": loss,
            "accuracy": accuracy,
            "avg_pred": jnp.mean(probs),
            "avg_label": jnp.mean(labels),
        }

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


def eval_step(params, apply_fn, batch, labels):
    """Single eval step."""
    logits = apply_fn(params, batch, train=False)
    log_probs = jax.nn.log_sigmoid(logits)
    log_probs_neg = jax.nn.log_sigmoid(-logits)
    bce = -(labels * log_probs + (1.0 - labels) * log_probs_neg)
    loss = jnp.mean(bce)
    probs = jax.nn.sigmoid(logits)
    preds = (probs > 0.5).astype(jnp.float32)
    accuracy = jnp.mean(preds == labels)
    return {
        "loss": loss,
        "accuracy": accuracy,
        "avg_pred": jnp.mean(probs),
        "avg_label": jnp.mean(labels),
    }


def train(
    train_data_dir: str,
    valid_data_dir: str | None = None,
    num_epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    emb_size: int = 128,
    num_layers: int = 4,
    num_heads: int = 2,
    max_batches_per_epoch: int | None = None,
    validate_every_n_batches: int = 100,
):
    """Train the Phoenix Flax ranking model."""
    hash_config = HashConfig(
        num_user_hashes=2,
        num_item_hashes=10,
        num_author_hashes=10,
    )

    model = PhoenixFlaxModel(
        emb_size=emb_size,
        num_actions=1,
        num_user_hashes=hash_config.num_user_hashes,
        num_item_hashes=hash_config.num_item_hashes,
        num_author_hashes=hash_config.num_author_hashes,
        history_seq_len=1,
        candidate_seq_len=1,
        product_surface_vocab_size=16,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.0,
        num_user_embeddings=600,
        num_post_embeddings=16500,
        num_author_embeddings=15000,
    )

    # Dummy batch for initialization
    dummy_batch = RecsysBatch(
        user_hashes=jnp.zeros((1, hash_config.num_user_hashes), jnp.int32),
        history_post_hashes=jnp.zeros(
            (1, 1, hash_config.num_item_hashes), jnp.int32
        ),
        history_author_hashes=jnp.zeros(
            (1, 1, hash_config.num_author_hashes), jnp.int32
        ),
        history_actions=jnp.zeros((1, 1, 1), jnp.float32),
        history_product_surface=jnp.zeros((1, 1), jnp.int32),
        candidate_post_hashes=jnp.zeros(
            (1, 1, hash_config.num_item_hashes), jnp.int32
        ),
        candidate_author_hashes=jnp.zeros(
            (1, 1, hash_config.num_author_hashes), jnp.int32
        ),
        candidate_product_surface=jnp.zeros((1, 1), jnp.int32),
    )

    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, model, dummy_batch, learning_rate)

    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    logger.info(f"Model initialized with {param_count:,} parameters")

    train_loader = CriteoFlaxDataLoader(
        train_data_dir,
        emb_size=emb_size,
        batch_size=batch_size,
        hash_config=hash_config,
        history_seq_len=1,
        candidate_seq_len=1,
        num_actions=1,
    )

    valid_loader = None
    if valid_data_dir:
        valid_loader = CriteoFlaxDataLoader(
            valid_data_dir,
            emb_size=emb_size,
            batch_size=batch_size,
            hash_config=hash_config,
            history_seq_len=1,
            candidate_seq_len=1,
            num_actions=1,
        )

    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")
        train_metrics = []
        for batch_idx, (batch, labels) in enumerate(
            train_loader.iterate(max_batches=max_batches_per_epoch)
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
            labels = jnp.array(labels)

            state, metrics = train_step(state, batch, labels)
            train_metrics.append(metrics)

            if (batch_idx + 1) % validate_every_n_batches == 0:
                avg_metrics = {
                    k: float(np.mean([float(m[k]) for m in train_metrics[-validate_every_n_batches:]]))
                    for k in train_metrics[0].keys()
                }
                logger.info(f"  Batch {batch_idx + 1}: {avg_metrics}")

        avg_train = {
            k: float(np.mean([float(m[k]) for m in train_metrics]))
            for k in train_metrics[0].keys()
        }
        logger.info(f"Train metrics: {avg_train}")

        if valid_loader:
            valid_metrics = []
            for batch, labels in valid_loader.iterate(max_batches=max_batches_per_epoch):
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
                labels = jnp.array(labels)
                metrics = eval_step(state.params, state.apply_fn, batch, labels)
                valid_metrics.append(metrics)

            avg_valid = {
                k: float(np.mean([float(m[k]) for m in valid_metrics]))
                for k in valid_metrics[0].keys()
            }
            logger.info(f"Valid metrics: {avg_valid}")

    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Phoenix Flax ranker on Criteo")
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    train(
        train_data_dir=args.train_dir,
        valid_data_dir=args.valid_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        emb_size=args.emb_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_batches_per_epoch=args.max_batches,
        validate_every_n_batches=args.validate_every,
    )


if __name__ == "__main__":
    main()
