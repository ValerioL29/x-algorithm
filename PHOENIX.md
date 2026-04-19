# PHOENIX.md — Deep Dive into the Phoenix Module

> This document provides a detailed breakdown of the `phoenix/` module, the ML recommendation system powering content ranking and retrieval for the X "For You" feed.

---

## Table of Contents

- [Overview](#overview)
- [Module Structure](#module-structure)
- [Core Architecture](#core-architecture)
- [File-by-File Breakdown](#file-by-file-breakdown)
  - [`grok.py` — Transformer Backbone](#grokpy--transformer-backbone)
  - [`recsys_model.py` — Ranking Model](#recsys_modelpy--ranking-model)
  - [`recsys_retrieval_model.py` — Retrieval Model](#recsys_retrieval_modelpy--retrieval-model)
  - [`runners.py` — Inference Runners & Utilities](#runnerspy--inference-runners--utilities)
  - [`run_ranker.py` & `run_retrieval.py` — Demo Scripts](#run_rankerpy--run_retrievalpy--demo-scripts)
  - [`test_recsys_model.py` & `test_recsys_retrieval_model.py` — Tests](#test_recsys_modelpy--test_recsys_retrieval_modelpy--tests)
- [Data Flow](#data-flow)
- [Key Design Decisions](#key-design-decisions)
- [Running and Testing](#running-and-testing)

---

## Overview

Phoenix is a **two-stage recommendation system** built in Python with JAX and dm-haiku:

1. **Retrieval** (`recsys_retrieval_model.py`): Uses a two-tower architecture to narrow millions of candidates down to hundreds or thousands via approximate nearest neighbor (ANN) search.
2. **Ranking** (`recsys_model.py`): Uses a Grok-based transformer to score and re-rank the retrieved candidates, predicting probabilities for 19 different user engagement actions.

The transformer core in `grok.py` is adapted from the [Grok-1 open source release](https://github.com/xai-org/grok-1) by xAI, with a critical modification: **candidate isolation masking** ensures candidates cannot attend to each other during inference.

---

## Module Structure

```
phoenix/
├── grok.py                          # Transformer blocks (MHA, FFN, RoPE, RMSNorm)
├── recsys_model.py                  # Phoenix ranking model
├── recsys_retrieval_model.py        # Phoenix two-tower retrieval model
├── runners.py                       # Model runners & inference wrappers
├── run_ranker.py                    # Standalone ranker demo
├── run_retrieval.py                 # Standalone retrieval demo
├── test_recsys_model.py             # Tests for attention masking
├── test_recsys_retrieval_model.py   # Tests for retrieval model
├── pyproject.toml                   # uv project config
├── uv.lock                          # Locked dependency tree
└── README.md                        # High-level overview
```

---

## Core Architecture

### Two-Stage Pipeline

```
┌─────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│   User +    │────▶│  STAGE 1: RETRIEVAL     │────▶│  STAGE 2: RANKING       │
│   History   │     │  Two-Tower Model        │     │  Transformer            │
│             │     │  Millions → Top-K       │     │  Top-K → Ranked Feed    │
└─────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

### Candidate Isolation in the Ranker

A defining feature of Phoenix is that **candidate scores must not depend on which other candidates are in the batch**. This is enforced via a custom attention mask (`make_recsys_attn_mask`):

- **User + History positions**: Standard causal attention among themselves.
- **Candidates → User/History**: Full attention (candidates see all context).
- **Candidates → Candidates**: **Self-attention only** (diagonal); candidates are isolated from each other.

This guarantees deterministic, independently computable candidate scores.

---

## File-by-File Breakdown

---

### `grok.py` — Transformer Backbone

This file contains the full transformer stack ported from Grok-1 and adapted for recommendation.

#### `make_recsys_attn_mask(seq_len, candidate_start_offset, dtype)`
Constructs the `[1, 1, seq_len, seq_len]` attention mask used during ranking inference.
- Starts with a full causal mask.
- Zeros out the bottom-right candidate-to-candidate block.
- Restores the diagonal (self-attention) for candidates.

#### `ffn_size(emb_size, widening_factor)`
Computes the hidden size for the feed-forward network, rounding up to the nearest multiple of 8.

#### `TrainingState(NamedTuple)`
Simple container for `hk.Params`.

#### `Linear(hk.Linear)`
Custom linear layer with:
- Weight initialization to zeros (`hk.initializers.Constant(0)`).
- fp32 parameters, cast to the input's forward-pass dtype.
- Optional bias.

#### `RMSNorm(hk.RMSNorm)`
Root Mean Square Layer Normalization with:
- `eps = 1e-5`
- fp32 scale parameter initialized to zeros.
- Computation performed in fp32, output cast back to the forward-pass dtype.

#### `RotaryEmbedding(hk.Module)`
Implements RoPE (Rotary Position Embedding) with base exponent 10,000.
- Supports scalar or per-batch offsets.
- Supports constant position or learned/custom timestep arrays.

#### `MultiHeadAttention(hk.Module)`
Standard multi-head attention with:
- **Grouped Query Attention (GQA)**: `num_q_heads` can be a multiple of `num_kv_heads`.
- RoPE applied to queries and keys.
- Attention logits clipped with `30 * tanh(logits / 30)` for stability.
- Mask broadcasting and softmax in fp32.

#### `MHABlock`, `DenseBlock`, `DecoderLayer`
Modular transformer components:
- `MHABlock`: Self-attention wrapper.
- `DenseBlock`: Gated MLP using `gelu(w1) * v` followed by a projection.
- `DecoderLayer`: Pre-normalization residual block (`RMSNorm → MHA → add → RMSNorm → Dense → add`).

#### `Transformer(hk.Module)`
The full transformer stack.
- **Standard mode** (`candidate_start_offset=None`): causal language modeling mask.
- **Recsys mode** (`candidate_start_offset=int`): applies `make_recsys_attn_mask` for candidate isolation.
- Iterates `DecoderLayer` for `num_layers` times.

#### `TransformerConfig(dataclass)`
Configuration object for `Transformer`:
- `emb_size`, `key_size`, `num_q_heads`, `num_kv_heads`, `num_layers`
- `widening_factor` (default 4.0)
- `attn_output_multiplier` (default 1.0)

---

### `recsys_model.py` — Ranking Model

This file implements `PhoenixModel`, the transformer-based ranker.

#### `HashConfig(dataclass)`
Controls how many hash functions are used for embedding lookups:
- `num_user_hashes` = 2
- `num_item_hashes` = 2
- `num_author_hashes` = 2

Multiple hashes are combined via learned projections to produce a single embedding.

#### `RecsysEmbeddings(NamedTuple)`
Pre-looked-up embeddings passed into the model:
- `user_embeddings`: `[B, num_user_hashes, D]`
- `history_post_embeddings`: `[B, S, num_item_hashes, D]`
- `candidate_post_embeddings`: `[B, C, num_item_hashes, D]`
- `history_author_embeddings`: `[B, S, num_author_hashes, D]`
- `candidate_author_embeddings`: `[B, C, num_author_hashes, D]`

#### `RecsysBatch(NamedTuple)`
Feature data (hashes, actions, product surfaces):
- `user_hashes`, `history_post_hashes`, `history_author_hashes`
- `history_actions`: multi-hot action vectors `[B, S, num_actions]`
- `history_product_surface`: categorical indices `[B, S]`
- `candidate_post_hashes`, `candidate_author_hashes`
- `candidate_product_surface`: categorical indices `[B, C]`

#### `RecsysModelOutput(NamedTuple)`
- `logits`: `[B, num_candidates, num_actions]`

#### `block_user_reduce(...)`
Projects `num_user_hashes * D` down to `D` using a learned matrix.
Returns combined user embedding `[B, 1, D]` and padding mask `[B, 1]`.

#### `block_history_reduce(...)`
Concatenates history post embeddings, author embeddings, action embeddings, and product surface embeddings.
Projects the concatenated vector down to `D`.
Returns `history_embeddings [B, S, D]` and `history_padding_mask [B, S]`.

#### `block_candidate_reduce(...)`
Similar to history reduce but without action embeddings (candidates have no action taken yet).
Returns `candidate_embeddings [B, C, D]` and `candidate_padding_mask [B, C]`.

#### `PhoenixModelConfig(dataclass)`
Configuration for the ranker:
- `model`: `TransformerConfig`
- `emb_size`, `num_actions`
- `history_seq_len` (default 128), `candidate_seq_len` (default 32)
- `hash_config`: `HashConfig`
- `product_surface_vocab_size` (default 16)
- `fprop_dtype` (default `jnp.bfloat16`)

#### `PhoenixModel(hk.Module)`
The ranking model itself.

**`_get_action_embeddings(actions)`**
Maps a multi-hot action vector to an embedding via a learned `[num_actions, D]` projection matrix.
Actions are converted to signed values (`2 * actions - 1`) before projection.

**`_single_hot_to_embeddings(input, vocab_size, emb_size, name)`**
Simple embedding table lookup for categorical features like `product_surface`.

**`_get_unembedding()`**
Returns the `[D, num_actions]` matrix used to project final candidate embeddings to action logits.

**`build_inputs(batch, recsys_embeddings)`**
Orchestrates the input construction:
1. Look up `product_surface` embeddings for history and candidates.
2. Reduce user, history, and candidate embeddings via the block reduce functions.
3. Concatenate along the sequence dimension: `[user, history, candidates]`.
4. Returns `(embeddings, padding_mask, candidate_start_offset)`.

**`__call__(batch, recsys_embeddings)`**
Forward pass:
1. `build_inputs` → embeddings and mask.
2. Run `Transformer` with `candidate_start_offset` for isolation masking.
3. Extract candidate output embeddings.
4. Apply final layer norm and unembedding projection.
5. Return `RecsysModelOutput(logits)`.

---

### `recsys_retrieval_model.py` — Retrieval Model

This file implements the two-tower retrieval system.

#### `RetrievalOutput(NamedTuple)`
- `user_representation`: `[B, D]`
- `top_k_indices`: `[B, K]`
- `top_k_scores`: `[B, K]`

#### `CandidateTower(hk.Module)`
A small MLP that projects post+author embeddings into the shared retrieval space.
- Input: concatenated post and author embeddings (flattened hashes).
- Hidden: `silu(x @ W1)`
- Output: `hidden @ W2`
- **L2 normalized** for dot-product similarity search.

Architecture:
```
post_author_embedding  [B, C, num_hashes * D]
         │
         ▼
    Linear(projection_1)  →  [B, C, 2D]
         │
         ▼
        silu()
         │
         ▼
    Linear(projection_2)  →  [B, C, D]
         │
         ▼
    L2 Normalization
```

#### `PhoenixRetrievalModelConfig(dataclass)`
Same fields as `PhoenixModelConfig` but without `num_actions`.

#### `PhoenixRetrievalModel(hk.Module)`
The two-tower retrieval model.

**`_get_action_embeddings(actions)`** and **`_single_hot_to_embeddings(...)`**
Identical implementations to the ranking model.

**`build_user_representation(batch, recsys_embeddings)`**
1. Constructs user + history embeddings (same as ranker).
2. Runs the `Transformer` **without** `candidate_start_offset` (standard causal attention).
3. Mean-pools over the sequence using the padding mask.
4. L2-normalizes the pooled representation.

**`build_candidate_representation(batch, recsys_embeddings)`**
1. Concatenates candidate post and author embeddings.
2. Passes through `CandidateTower`.
3. Returns normalized representations and a validity mask.

**`_retrieve_top_k(user_representation, corpus_embeddings, top_k, corpus_mask)`**
- Computes dot-product scores: `scores = user_representation @ corpus_embeddings.T`
- Applies optional corpus mask (`-INF` for invalid entries).
- Uses `jax.lax.top_k` to get the best candidates.

**`__call__(batch, recsys_embeddings, corpus_embeddings, top_k, corpus_mask)`**
Public API that encodes the user and retrieves top-k candidates from the corpus.

---

### `runners.py` — Inference Runners & Utilities

This file wraps the raw Haiku models into easy-to-use inference runners and provides data generation utilities.

#### `ACTIONS` (List[str])
The 19 engagement actions predicted by the ranker:
```python
[
    "favorite_score", "reply_score", "repost_score",
    "photo_expand_score", "click_score", "profile_click_score",
    "vqv_score", "share_score", "share_via_dm_score",
    "share_via_copy_link_score", "dwell_score", "quote_score",
    "quoted_click_score", "follow_author_score", "not_interested_score",
    "block_author_score", "mute_author_score", "report_score",
    "dwell_time",
]
```

#### `create_dummy_batch_from_config(...)`
Creates a `RecsysBatch` filled with zeros for model initialization.

#### `create_dummy_embeddings_from_config(...)`
Creates a `RecsysEmbeddings` filled with zeros for model initialization.

#### `BaseModelRunner(ABC)`
Abstract base for model runners.
- `bs_per_device`: batch sizing heuristic.
- `rng_seed`: random seed for initialization.
- Abstract properties: `model`, `_model_name`.
- Abstract method: `make_forward_fn()`.
- `initialize()`: sets `batch_size` based on GPU count and compiles the forward function.

#### `BaseInferenceRunner(ABC)`
Abstract base for inference runners.
- Creates dummy batches/embeddings for initialization.
- Abstract method: `initialize()`.

#### `ModelRunner(BaseModelRunner)`
Runner for the ranking model.
- `make_forward_fn()`: wraps `PhoenixModel.__call__` in `hk.transform`.
- `init(rng, data, embeddings)`: initializes parameters.
- `load_or_init(...)`: convenience wrapper using `rng_seed`.

#### `RecsysInferenceRunner(BaseInferenceRunner)`
High-level inference runner for ranking.

**`initialize()`**
1. Creates dummy data.
2. Calls `runner.initialize()` and `load_or_init()` to get parameters.
3. Builds `hk_rank_candidates`, a transformed function that:
   - Runs the model forward pass.
   - Applies `jax.nn.sigmoid` to logits to get probabilities.
   - Uses `favorite_score` (index 0) as the primary ranking signal.
   - Returns `argsort(-primary_scores)` as `ranked_indices`.
   - Returns per-action probability fields (`p_favorite_score`, `p_reply_score`, etc.).

**`rank(batch, recsys_embeddings)`**
Applies `rank_candidates` with the stored parameters.

#### `RankingOutput(NamedTuple)`
Contains:
- `scores`: full probability tensor `[B, C, num_actions]`
- `ranked_indices`: indices sorted by favorite probability `[B, C]`
- `p_*` fields for each of the 19 actions.

#### `RetrievalModelRunner(BaseModelRunner)`
Runner for the retrieval model.
- `make_forward_fn()`: wraps `PhoenixRetrievalModel.__call__` in `hk.transform`.
- Also calls `build_candidate_representation` during init to ensure all parameters are created.

#### `RecsysRetrievalInferenceRunner(BaseInferenceRunner)`
High-level inference runner for retrieval.

**`initialize()`**
1. Creates dummy data (including a dummy corpus).
2. Initializes parameters.
3. Builds three transformed functions:
   - `hk_encode_user` → `encode_user_fn`
   - `hk_encode_candidates` → `encode_candidates_fn`
   - `hk_retrieve` → `retrieve_fn`

**`encode_user(batch, recsys_embeddings)`**
Returns `[B, D]` user representations.

**`encode_candidates(batch, recsys_embeddings)`**
Returns `[B, C, D]` candidate representations.

**`set_corpus(corpus_embeddings, corpus_post_ids)`**
Stores the pre-computed candidate corpus for retrieval.

**`retrieve(batch, recsys_embeddings, top_k, corpus_embeddings)`**
Returns `RetrievalOutput` with top-k indices and similarity scores.

#### `create_example_batch(...)`
Generates a random `(RecsysBatch, RecsysEmbeddings)` tuple for testing/demos.
- Hash values are in `[1, N)`; `0` is reserved for padding.
- History lengths are randomly truncated to simulate variable-length histories.

#### `create_example_corpus(corpus_size, emb_size, seed)`
Generates random L2-normalized corpus embeddings and sequential post IDs.

---

### `run_ranker.py` & `run_retrieval.py` — Demo Scripts

#### `run_ranker.py`
A standalone script demonstrating the ranking pipeline:
1. Configures a small `PhoenixModelConfig` (`emb_size=128`, 2 layers, 2 heads).
2. Initializes `RecsysInferenceRunner`.
3. Generates a random example batch with 32 history items and 8 candidates.
4. Calls `inference_runner.rank()`.
5. Prints a ranked list with probability bars for all 19 actions.

#### `run_retrieval.py`
A standalone script demonstrating the retrieval pipeline:
1. Configures a `PhoenixRetrievalModelConfig` with the same transformer as the ranker.
2. Initializes `RecsysRetrievalInferenceRunner`.
3. Generates a random example batch for 2 users.
4. Creates a simulated corpus of 1,000 L2-normalized embeddings.
5. Calls `inference_runner.retrieve(top_k=10)`.
6. Prints top-10 post IDs and similarity scores for each user.

---

### `test_recsys_model.py` & `test_recsys_retrieval_model.py` — Tests

#### `test_recsys_model.py`
Tests for `make_recsys_attn_mask`:
- `test_output_shape`: verifies `[1, 1, seq_len, seq_len]` shape.
- `test_user_history_has_causal_attention`: causal mask for prefix positions.
- `test_candidates_attend_to_user_history`: candidates see all user+history context.
- `test_candidates_attend_to_themselves`: diagonal self-attention is preserved.
- `test_candidates_do_not_attend_to_other_candidates`: off-diagonal candidate blocks are zero.
- `test_full_mask_structure`: validates exact mask values for a 6-position example.
- `test_dtype_preserved`: checks `float32` vs `float16` handling.
- `test_single_candidate` and `test_all_candidates`: edge cases.

#### `test_recsys_retrieval_model.py`
Tests using `unittest`:

**`TestCandidateTower`**
- `test_candidate_tower_output_shape`: `[B, C, num_hashes, D] → [B, C, D]`.
- `test_candidate_tower_normalized`: verifies L2 norms are ~1.0.
- `test_candidate_tower_mean_pooling`: shape and normalization check.

**`TestPhoenixRetrievalModel`**
- `test_model_forward`: validates `RetrievalOutput` shapes.
- `test_user_representation_normalized`: L2 norm check on user embeddings.
- `test_candidate_representation_normalized`: L2 norm check on candidate embeddings.
- `test_retrieve_top_k`: validates top-k indices are in range and scores are monotonically non-increasing.

**`TestRetrievalInferenceRunner`**
- `test_runner_initialization`: params are created.
- `test_runner_encode_user`: shape check.
- `test_runner_retrieve`: end-to-end retrieval with a corpus.

---

## Data Flow

### Ranking Pipeline Data Flow

```
RecsysBatch + RecsysEmbeddings
         │
         ▼
┌─────────────────┐
│  build_inputs   │  → concat[user, history, candidates]
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Transformer   │  → with candidate isolation mask
└─────────────────┘
         │
         ▼
   Extract candidates
         │
         ▼
   Layer Norm + Unembedding
         │
         ▼
      Logits [B, C, 19]
         │
         ▼
      Sigmoid → Probabilities
         │
         ▼
   Rank by favorite_score
```

### Retrieval Pipeline Data Flow

```
RecsysBatch + RecsysEmbeddings
         │
    ┌────┴────┐
    ▼         ▼
User Tower  Candidate Tower
    │              │
    ▼              ▼
[B, D]        [B, C, D]
    │              │
    └────┬─────────┘
         ▼
   Dot Product Similarity
         │
         ▼
    jax.lax.top_k
         │
         ▼
   RetrievalOutput
```

---

## Key Design Decisions

### 1. Candidate Isolation
The ranker uses a custom attention mask so candidates cannot attend to each other. This ensures:
- **Score stability**: Adding or removing a candidate does not change the scores of other candidates.
- **Parallel inference**: Each candidate's score is computed independently given the same user+history context.

### 2. Shared Transformer Architecture
The retrieval model's user tower reuses the exact same `Transformer` class as the ranker. This enables knowledge sharing and consistent representation learning across stages.

### 3. Hash-Based Embeddings
Both users and items use multiple hash functions for embedding lookups. This is a common technique for large-scale recommendation systems to handle massive, dynamic vocabularies without maintaining enormous embedding tables.

### 4. Multi-Action Prediction
The ranker predicts 19 distinct engagement probabilities simultaneously from a shared transformer backbone, allowing a single forward pass to estimate likes, replies, reposts, clicks, dwell time, negative signals (not interested, block, mute), etc.

### 5. L2 Normalization for Retrieval
Both user and candidate towers produce L2-normalized embeddings. This means dot-product similarity is equivalent to cosine similarity, which is ideal for ANN indexes like FAISS or ScaNN.

---

## Running and Testing

### Installation
Requires Python >= 3.11 and `uv`.

```bash
cd phoenix
```

### Run Demos

```bash
# Ranking demo
uv run run_ranker.py

# Retrieval demo
uv run run_retrieval.py
```

### Run Tests

```bash
uv run pytest test_recsys_model.py test_recsys_retrieval_model.py
```

### Dependencies
- `jax==0.8.1`
- `dm-haiku>=0.0.13`
- `numpy>=1.26.4`
- `pytest` (dev)

---

*Document generated from source analysis of the `phoenix/` module.*
