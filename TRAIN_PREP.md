# Training Preparation Guide — Adapting Custom Datasets to Phoenix

This guide explains how to adapt your own dataset to any of the three Phoenix ranking model implementations and tune hyperparameters for optimal performance.

---

## Table of Contents

1. [Overview of Implementations](#1-overview-of-implementations)
2. [Dataset Format Requirements](#2-dataset-format-requirements)
3. [Quick Start — Minimal Changes](#3-quick-start--minimal-changes)
4. [Adapting the DataLoader](#4-adapting-the-dataloader)
5. [Model Hyperparameters](#5-model-hyperparameters)
6. [Training Hyperparameters](#6-training-hyperparameters)
7. [Tuning Guidelines](#7-tuning-guidelines)
8. [Example: Adding a New Dataset](#8-example-adding-a-new-dataset)
9. [Performance Comparison Reference](#9-performance-comparison-reference)

---

## 1. Overview of Implementations

| Implementation | Path | Framework | Key Characteristic |
|---------------|------|-----------|-------------------|
| **Original** | `phoenix/` | JAX + Haiku | Fixed random embeddings, transformer backbone |
| **TorchRec** | `phoenix-torch/` | PyTorch + TorchRec | Learnable `EmbeddingBagCollection`, `nn.TransformerEncoder` |
| **Flax** | `phoenix-flax/` | JAX + Flax | Fixed random embeddings, `nn.MultiHeadDotProductAttention` |

All three share the same **model skeleton**:
- Categorical feature embeddings (user, post, author, product surface)
- Group projection layers (user → `emb_size`, post → `emb_size`, author → `emb_size`, surface → `emb_size`)
- Candidate fusion (post + author + surface → candidate embedding)
- Transformer encoder over `[user, history, candidate]` sequence
- Binary classification logit from the candidate position

---

## 2. Dataset Format Requirements

The training scripts consume **Parquet files** with the following columns:

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `label` | `int32` | Binary target (0 = no click, 1 = click) |
| `C1`–`C26` | `int64` | 26 categorical features (mapped to embedding lookups) |
| `I1`–`I13` | `float` | *(Optional)* 13 dense features — currently unused in all three implementations |

### Directory Structure

```
dataset/
├── train/
│   ├── 0.xxx.parquet
│   ├── 1.xxx.parquet
│   └── ...
└── valid/
    ├── 0.xxx.parquet
    └── ...
```

> **Note:** The current implementations only use `C1`–`C23` for features and `label` for the target. `C24`–`C26` and all `I*` columns are present but not fed into the model. To use them, see [Adapting the DataLoader](#4-adapting-the-dataloader).

---

## 3. Quick Start — Minimal Changes

To train on a new dataset, you typically only need to modify **two things**:

1. **Vocabulary sizes** (`CRITEO_VOCAB_SIZES` or embedding table dimensions)
2. **File paths** (point `--train-dir` and `--valid-dir` to your dataset)

### Step-by-Step

```bash
# 1. Organize your dataset into train/ and valid/ subdirectories with .parquet files

# 2. Update vocabulary sizes in the model file
#    phoenix/train_ranker.py          (hash embedding table sizes)
#    phoenix-torch/phoenix_torch_model.py
#    phoenix-flax/phoenix_flax_model.py

# 3. Run training
# --- Original JAX/Haiku ---
cd phoenix
uv run python train_ranker.py \
  --train-dir /path/to/your/dataset/train \
  --valid-dir /path/to/your/dataset/valid \
  --epochs 5 --batch-size 256 --lr 1e-3

# --- TorchRec ---
cd phoenix-torch
uv run python train_ranker_torch.py \
  --train-dir /path/to/your/dataset/train \
  --valid-dir /path/to/your/dataset/valid \
  --epochs 5 --batch-size 256 --lr 1e-3 --device cuda

# --- Flax ---
cd phoenix-flax
uv run python train_ranker_flax.py \
  --train-dir /path/to/your/dataset/train \
  --valid-dir /path/to/your/dataset/valid \
  --epochs 5 --batch-size 256 --lr 1e-3
```

---

## 4. Adapting the DataLoader

The DataLoader is the primary interface between your dataset and the model. Here is what to change for each implementation.

### 4.1 Original JAX/Haiku (`phoenix/train_ranker.py`)

**Class:** `CriteoDataLoader`

The loader maps categorical columns to `RecsysBatch` + `RecsysEmbeddings`:
- `C1, C2` → `user_hashes`
- `C3–C12` → `candidate_post_hashes` (10 hashes)
- `C13–C22` → `candidate_author_hashes` (10 hashes)
- `C23` → `candidate_product_surface`

**To add dense features (`I1`–`I13`):**

1. In `_df_to_batch`, extract dense features:
```python
dense_features = df[[f"I{i}" for i in range(1, 14)]].fillna(0).values.astype(np.float32)
```

2. Pass them into the model by extending `RecsysBatch` with a `dense_features` field, then concatenate inside `PhoenixModel.build_inputs()`.

**To change the feature-to-hash mapping:**

Modify the `np.stack` calls in `_df_to_batch()`. For example, if your dataset has only 5 post hashes instead of 10, adjust:
```python
candidate_post_hashes = np.stack(
    [df[f"C{i}"].values for i in range(3, 8)], axis=1  # C3–C7 instead of C3–C12
).astype(np.int32)
```

And update `hash_config.num_item_hashes = 5`.

---

### 4.2 TorchRec (`phoenix-torch/train_ranker_torch.py`)

**Class:** `CriteoTorchDataLoader`

The loader produces a `KeyedJaggedTensor` (KJT) with 26 keys (`C1`–`C26`).

**To add dense features:**

1. In `_df_to_tensors`, extract dense values:
```python
dense_vals = torch.tensor(
    df[[f"I{i}" for i in range(1, 14)]].fillna(0).values.astype("float32"),
    dtype=torch.float32, device=self.device,
)
```

2. Modify `PhoenixTorchModel.forward()` to accept an optional `dense_features` tensor and concatenate after the sparse embeddings.

**To use a different number of categorical features:**

Update `self.cat_cols` and rebuild `CRITEO_VOCAB_SIZES`. The KJT construction is generic — as long as `keys`, `values`, and `lengths` are consistent, TorchRec handles any number of features.

---

### 4.3 Flax (`phoenix-flax/train_ranker_flax.py`)

**Class:** `CriteoFlaxDataLoader`

The loader is nearly identical to the original JAX DataLoader but returns JAX arrays directly.

**To add dense features:**

1. Extract dense features in `_df_to_batch`.
2. Add a `dense_features` field to `RecsysBatch`.
3. In `PhoenixFlaxModel.__call__`, concatenate dense features into the sequence before the transformer.

---

## 5. Model Hyperparameters

These control the **capacity** and **architecture** of the model.

### 5.1 Core Model Parameters

| Parameter | Flag | Description | Typical Range | Default |
|-----------|------|-------------|---------------|---------|
| `emb_size` | `--emb-size` | Embedding dimension for all projections and transformer | 64–512 | 128 |
| `num_layers` | `--num-layers` | Number of transformer layers | 1–8 | 4 |
| `num_heads` | `--num-heads` | Attention heads (must divide `emb_size`) | 2–8 | 2 |
| `widening_factor` | *(hardcoded)* | FFN hidden dim multiplier | 2–4 | 2 |
| `dropout` | *(hardcoded)* | Dropout rate for regularization | 0.0–0.3 | 0.0 |

### 5.2 Embedding-Specific Parameters

| Parameter | JAX/Haiku | TorchRec | Flax | Notes |
|-----------|-----------|----------|------|-------|
| Embedding table sizes | `num_*_embeddings` in `create_example_batch` | `num_embeddings` in `EmbeddingBagConfig` | `num_*_embeddings` in `PhoenixFlaxModel` | Should be ≥ max feature ID + 1 |
| Hash counts | `HashConfig` | N/A (direct lookup) | `HashConfig` | How many hashes per group (user=2, post=10, author=10) |
| `embedding_dim` | N/A (uses `emb_size`) | `--embedding-dim` | N/A (uses `emb_size`) | TorchRec only: table dim before projection |

### 5.3 Where to Change

**JAX/Haiku:**
```python
# In train_ranker.py, build the config:
model_config = PhoenixModelConfig(
    emb_size=128,           # <-- change me
    num_actions=1,
    history_seq_len=1,
    candidate_seq_len=1,
    model=TransformerConfig(
        emb_size=128,
        widening_factor=2,
        key_size=32,        # <-- attention head dim
        num_q_heads=2,      # <-- change me
        num_kv_heads=2,
        num_layers=4,       # <-- change me
        attn_output_multiplier=0.125,
    ),
)
```

**TorchRec:**
```python
# In train_ranker_torch.py or phoenix_torch_model.py:
model = PhoenixTorchModel(
    embedding_bag_collection=ebc,
    emb_size=128,           # <-- change me
    num_layers=4,           # <-- change me
    num_heads=2,            # <-- change me
    dropout=0.0,            # <-- change me
)
```

**Flax:**
```python
# In train_ranker_flax.py:
model = PhoenixFlaxModel(
    emb_size=128,           # <-- change me
    num_actions=1,
    num_layers=4,           # <-- change me
    num_heads=2,            # <-- change me
    dropout=0.0,            # <-- change me
)
```

---

## 6. Training Hyperparameters

These control **optimization behavior**.

| Parameter | Flag | Description | Typical Range | Default |
|-----------|------|-------------|---------------|---------|
| Learning rate | `--lr` | Adam step size | 1e-4 – 1e-2 | 1e-3 |
| Batch size | `--batch-size` | Samples per step | 64–1024 | 256 |
| Epochs | `--epochs` | Full passes through data | 1–20 | 5 |
| Max batches | `--max-batches` | Cap batches per epoch (useful for large datasets / debugging) | None (all) | None |
| Validation frequency | `--validate-every` | Log validation metrics every N batches | 20–500 | 100 |

### Learning Rate Scheduling (Not Implemented by Default)

If you want to add LR decay, here is the pattern for each framework:

**JAX/Haiku:**
```python
import optax
schedule = optax.exponential_decay(1e-3, transition_steps=1000, decay_rate=0.9)
optimizer = optax.adam(schedule)
```

**TorchRec:**
```python
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
# Call scheduler.step() after each epoch
```

**Flax:**
```python
schedule = optax.exponential_decay(1e-3, transition_steps=1000, decay_rate=0.9)
tx = optax.adam(schedule)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
```

---

## 7. Tuning Guidelines

### 7.1 Start Small, Then Scale Up

```bash
# Phase 1: Sanity check (fast)
--small --epochs 1 --max-batches 100 --batch-size 64

# Phase 2: Baseline
--emb-size 64 --num-layers 2 --epochs 3 --batch-size 256

# Phase 3: Scale up if underfitting
--emb-size 128 --num-layers 4 --epochs 5 --batch-size 256

# Phase 4: Large model (if dataset is huge)
--emb-size 256 --num-layers 8 --epochs 10 --batch-size 512
```

### 7.2 Diagnosing Underfitting vs. Overfitting

| Symptom | Train Loss | Valid Loss | Action |
|---------|-----------|------------|--------|
| Underfitting | High (>0.55) | High | ↑ `emb_size`, ↑ `num_layers`, ↑ epochs |
| Overfitting | Low (<0.45) | Much higher | Add dropout, ↓ `emb_size`, freeze embeddings |
| Good fit | Moderate | Close to train | You are done |

### 7.3 Embedding Strategy

| Strategy | Best For | Implementation |
|----------|----------|----------------|
| **Fixed random** (original) | Small/medium datasets, strong generalization | JAX, Flax (with `stop_gradient`) |
| **Learnable** | Large datasets with millions of examples | TorchRec, Flax (remove `stop_gradient`) |

On our Criteo subset (~1.6M rows), **fixed random embeddings generalized better** (75.6% vs. 73.2% valid acc). On a full billion-row dataset, learnable embeddings would likely win.

### 7.4 Batch Size Trade-offs

| Batch Size | Speed | Memory | Gradient Noise | Recommendation |
|------------|-------|--------|----------------|----------------|
| 64 | Slow | Low | High | Debugging only |
| 256 | Fast | Medium | Medium | **Default sweet spot** |
| 512+ | Very fast | High | Low | Use with LR warmup |

---

## 8. Example: Adding a New Dataset

Suppose you have a dataset `my-dataset/` with:
- 10 categorical features: `cat_0`–`cat_9`
- 5 dense features: `dense_0`–`dense_4`
- Target: `clicked`

### Step 1: Prepare Parquet Files

```python
import pandas as pd

df = pd.read_csv("my_data.csv")
df.to_parquet("my-dataset/train/part_0.parquet")
```

### Step 2: Create a Custom DataLoader (TorchRec example)

```python
# In phoenix-torch/train_ranker_torch.py

class MyDatasetLoader:
    def __init__(self, data_dir, batch_size, device):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        self.batch_size = batch_size
        self.device = device
        self.vocab_sizes = {"cat_0": 1000, "cat_1": 500, ...}  # <-- your vocabs

    def _df_to_tensors(self, df):
        B = len(df)
        values_list = []
        for col in [f"cat_{i}" for i in range(10)]:
            vals = df[col].fillna(0).astype(int).clip(0, self.vocab_sizes[col] - 1)
            values_list.extend(vals.tolist())

        values = torch.tensor(values_list, dtype=torch.int64, device=self.device)
        lengths = torch.ones(10 * B, dtype=torch.int64, device=self.device)

        sparse = KeyedJaggedTensor(
            keys=[f"cat_{i}" for i in range(10)],
            values=values,
            lengths=lengths,
        )

        # Dense features
        dense = torch.tensor(
            df[[f"dense_{i}" for i in range(5)]].fillna(0).values.astype("float32"),
            dtype=torch.float32, device=self.device,
        )

        labels = torch.tensor(df["clicked"].values.astype("float32"), device=self.device)
        return sparse, dense, labels
```

### Step 3: Update the Model

In `phoenix_torch_model.py`:
1. Change `build_embedding_bag_collection()` to use your 10 features and vocab sizes.
2. In `PhoenixTorchModel.forward()`, add a `dense_features` argument:
```python
def forward(self, sparse_features, dense_features):
    pooled = self.embedding_bag_collection(sparse_features)
    # ... existing group projections ...
    combined = torch.cat([pooled.values(), dense_features], dim=1)
    # ... rest of forward ...
```

### Step 4: Update `train()` to Pass Dense Features

```python
for batch_idx, (sparse, dense, labels) in enumerate(loader.iterate()):
    optimizer.zero_grad()
    logits = model(sparse, dense).squeeze(-1)
    loss = criterion(logits, labels)
    ...
```

### Step 5: Tune

```bash
uv run python train_ranker_torch.py \
  --train-dir ./my-dataset/train \
  --valid-dir ./my-dataset/valid \
  --emb-size 128 --num-layers 4 \
  --epochs 5 --batch-size 256 --lr 1e-3
```

---

## 9. Performance Comparison Reference

Benchmarked on NVIDIA L4, 500 batches, `batch_size=256`, `emb_size=128`, `num_layers=4`:

| Framework | Wall Time | Batches/sec | Valid Acc (Criteo) | GPU Memory |
|-----------|-----------|-------------|-------------------|------------|
| **TorchRec** | **20.3 s** | **~24.6** | 73.2%* | ~22.1 GB |
| JAX/Haiku | 38.2 s | ~13.1 | **75.6%** | ~22.1 GB |
| Flax | 94.3 s | ~5.3 | **75.7%** | ~22.1 GB |

\* TorchRec result uses **learnable embeddings** (overfits). With frozen embeddings, expect ~75% valid acc and slightly faster throughput.

### Recommendation Matrix

| Priority | Recommended Framework | Why |
|----------|----------------------|-----|
| **Speed** | TorchRec | cuDNN fused attention, lowest overhead |
| **Accuracy / Generalization** | JAX/Haiku or Flax | Fixed embeddings force compositional learning |
| **Research / Flexibility** | Flax | Pure functional, easy to modify attention logic |
| **Production / Large Tables** | TorchRec | Native sharding, `DistributedModelParallel` support |

---

## Appendix: File Reference

| File | Role |
|------|------|
| `phoenix/train_ranker.py` | JAX/Haiku trainer + `CriteoDataLoader` |
| `phoenix/recsys_model.py` | `PhoenixModel` (Haiku) + `RecsysBatch` / `RecsysEmbeddings` |
| `phoenix-torch/phoenix_torch_model.py` | `PhoenixTorchModel` (TorchRec) |
| `phoenix-torch/train_ranker_torch.py` | TorchRec trainer + `CriteoTorchDataLoader` |
| `phoenix-flax/phoenix_flax_model.py` | `PhoenixFlaxModel` (Flax) |
| `phoenix-flax/train_ranker_flax.py` | Flax trainer + `CriteoFlaxDataLoader` |

---

*Last updated: 2026-04-17*
