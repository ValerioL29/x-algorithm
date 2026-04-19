"""Training script for Phoenix TorchRec ranking model on Criteo dataset."""

import argparse
import glob
import logging
import os
from typing import Iterator, Tuple

import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from phoenix_torch_model import CRITEO_VOCAB_SIZES, build_embedding_bag_collection, PhoenixTorchModel

logger = logging.getLogger(__name__)


def load_parquet_files(data_dir: str) -> list[str]:
    """Load all parquet files from a directory."""
    pattern = os.path.join(data_dir, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f"No parquet files found in {data_dir}")
    return files


class CriteoTorchDataLoader:
    """DataLoader for Criteo dataset that produces TorchRec inputs."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        device: torch.device,
        dense_features: bool = True,
    ):
        self.files = load_parquet_files(data_dir)
        self.batch_size = batch_size
        self.device = device
        self.dense_features = dense_features
        self.cat_cols = [f"C{i}" for i in range(1, 27)]
        self.dense_cols = [f"I{i}" for i in range(1, 14)]

    def _df_to_tensors(
        self, df: pd.DataFrame
    ) -> Tuple[KeyedJaggedTensor, torch.Tensor]:
        """Convert DataFrame to KJT and labels."""
        B = len(df)

        # Fill NaN for categorical features and clamp to valid range
        cat_values = []
        for col in self.cat_cols:
            vals = df[col].fillna(0).astype(int).values
            vocab_size = CRITEO_VOCAB_SIZES[col]
            vals = vals.clip(0, vocab_size - 1)
            cat_values.append(vals)

        # Flatten values: feature-major order
        # TorchRec KJT expects values ordered as:
        # [C1_0, C1_1, ..., C1_B, C2_0, C2_1, ..., C2_B, ..., C26_0, C26_1, ..., C26_B]
        values_list = []
        for col_idx in range(26):
            for b in range(B):
                values_list.append(cat_values[col_idx][b])
        values = torch.tensor(values_list, dtype=torch.int64, device=self.device)

        # Single-hot -> lengths all 1
        lengths = torch.ones(26 * B, dtype=torch.int64, device=self.device)

        sparse_features = KeyedJaggedTensor(
            keys=self.cat_cols,
            values=values,
            lengths=lengths,
        )

        labels = torch.tensor(
            df["label"].fillna(0).values.astype("float32"),
            dtype=torch.float32,
            device=self.device,
        )

        return sparse_features, labels

    def iterate(
        self, max_batches: int | None = None
    ) -> Iterator[Tuple[KeyedJaggedTensor, torch.Tensor]]:
        """Iterate over batches."""
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
                yield self._df_to_tensors(batch_df)
                batch_count += 1


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(
    train_data_dir: str,
    valid_data_dir: str | None = None,
    num_epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    emb_size: int = 128,
    embedding_dim: int = 16,
    num_layers: int = 4,
    num_heads: int = 2,
    max_batches_per_epoch: int | None = None,
    validate_every_n_batches: int = 100,
    device_str: str = "cuda",
):
    """Train the Phoenix TorchRec model."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ebc = build_embedding_bag_collection(embedding_dim=embedding_dim, device=device)
    model = PhoenixTorchModel(
        embedding_bag_collection=ebc,
        emb_size=emb_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.0,
        dense_device=device,
    ).to(device)

    param_count = count_parameters(model)
    logger.info(f"Model initialized with {param_count:,} trainable parameters")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = CriteoTorchDataLoader(
        train_data_dir, batch_size=batch_size, device=device
    )
    valid_loader = None
    if valid_data_dir:
        valid_loader = CriteoTorchDataLoader(
            valid_data_dir, batch_size=batch_size, device=device
        )

    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")
        model.train()

        train_metrics = []
        for batch_idx, (sparse_features, labels) in enumerate(
            train_loader.iterate(max_batches=max_batches_per_epoch)
        ):
            optimizer.zero_grad()
            logits = model(sparse_features).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                accuracy = (preds == labels).float().mean()
                avg_pred = probs.mean()
                avg_label = labels.mean()

            metrics = {
                "loss": loss.item(),
                "accuracy": accuracy.item(),
                "avg_pred": avg_pred.item(),
                "avg_label": avg_label.item(),
            }
            train_metrics.append(metrics)

            if (batch_idx + 1) % validate_every_n_batches == 0:
                avg_metrics = {
                    k: sum(m[k] for m in train_metrics[-validate_every_n_batches:])
                    / len(train_metrics[-validate_every_n_batches:])
                    for k in train_metrics[0].keys()
                }
                logger.info(f"  Batch {batch_idx + 1}: {avg_metrics}")

        avg_train = {
            k: sum(m[k] for m in train_metrics) / len(train_metrics)
            for k in train_metrics[0].keys()
        }
        logger.info(f"Train metrics: {avg_train}")

        if valid_loader:
            model.eval()
            valid_metrics = []
            with torch.no_grad():
                for sparse_features, labels in valid_loader.iterate(
                    max_batches=max_batches_per_epoch
                ):
                    logits = model(sparse_features).squeeze(-1)
                    loss = criterion(logits, labels)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    accuracy = (preds == labels).float().mean()
                    valid_metrics.append(
                        {
                            "loss": loss.item(),
                            "accuracy": accuracy.item(),
                            "avg_pred": probs.mean().item(),
                            "avg_label": labels.mean().item(),
                        }
                    )
            avg_valid = {
                k: sum(m[k] for m in valid_metrics) / len(valid_metrics)
                for k in valid_metrics[0].keys()
            }
            logger.info(f"Valid metrics: {avg_valid}")

    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Phoenix TorchRec ranker on Criteo")
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
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    train(
        train_data_dir=args.train_dir,
        valid_data_dir=args.valid_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        emb_size=args.emb_size,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_batches_per_epoch=args.max_batches,
        validate_every_n_batches=args.validate_every,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
