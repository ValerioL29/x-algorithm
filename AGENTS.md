# AGENTS.md — X For You Feed Algorithm

> This file is written for AI coding agents. It assumes you know nothing about this repository.

---

## Project Overview

This repository contains the core recommendation system powering the "For You" feed on X. It combines in-network content (from accounts a user follows) with out-of-network content (discovered via ML-based retrieval) and ranks everything using a Grok-based transformer model.

The system is split across four main components:

| Component | Language | Purpose |
|-----------|----------|---------|
| `home-mixer/` | Rust | Orchestration layer — gRPC server that assembles the feed via a staged candidate pipeline. |
| `thunder/` | Rust | In-memory post store and real-time ingestion pipeline for in-network posts. |
| `candidate-pipeline/` | Rust | Reusable framework (traits + executor) for building recommendation pipelines. |
| `phoenix/` | Python (JAX/Haiku) | ML models for retrieval (two-tower) and ranking (transformer with candidate isolation). |

> **Important:** The transformer architecture in `phoenix/` is ported from the [Grok-1 open source release](https://github.com/xai-org/grok-1) by xAI and adapted for recommendation use cases.

---

## Technology Stack

- **Rust**: Services (`home-mixer`, `thunder`, `candidate-pipeline`) using `tokio`, `tonic` (gRPC), `axum`, `clap`, `anyhow`, `log`, `prometheus` metrics, `dashmap`, `lazy_static`.
- **Python**: ML models (`phoenix`) using `jax==0.8.1`, `dm-haiku>=0.0.13`, `numpy>=1.26.4`.
- **Package Management**: `uv` for Python; no standalone Rust package manifests are present in this repo.
- **Protocols**: gRPC with Zstd/Gzip compression; Kafka for event ingestion (in `thunder`).

---

## Build and Test Commands

### Python (Phoenix) — Fully Buildable

Phoenix is the only component that can be built and run out-of-the-box from this repository.

```bash
# Enter the Phoenix directory
cd phoenix

# Install dependencies and run commands via uv
uv run run_ranker.py
uv run run_retrieval.py

# Run tests
uv run pytest test_recsys_model.py test_recsys_retrieval_model.py
```

Configuration for Python is in `phoenix/pyproject.toml`:
- Python >= 3.11 required.
- Ruff is used for linting (`line-length = 100`, `indent-width = 4`).
- `pytest` is the test runner (defined in `dependency-groups.dev`).

### Rust — Source-Only Release

**The Rust crates cannot be compiled standalone.** There are no `Cargo.toml` files at the root or inside the Rust directories because the code depends on numerous internal X/XAI crates that were excluded from the open source release, including:

- `xai_home_mixer_proto`
- `xai_candidate_pipeline`
- `xai_http_server`
- `xai_stats_macro`
- `xai_init_utils`
- `xai_profiling`
- `xai_kafka`
- `xai_post_text`
- `xai_recsys_aggregation`
- `xai_recsys_proto`
- `xai_strato`
- `xai_thunder_proto`
- `xai_twittercontext_proto`
- `xai_uas_thrift`
- `xai_visibility_filtering`
- `xai_wily`

Additionally, some modules inside `home-mixer/` are explicitly excluded:
- `clients/` — gRPC/HTTP client implementations for internal services.
- `params/` — tunable parameters and constants.
- `util/` — shared utilities.

These are referenced in `home-mixer/lib.rs` with comments like `// Excluded from open source release for security reasons`.

**There are no Rust unit tests in this repository.** If you make changes to Rust code, you will not be able to compile or test them without reconstructing the missing internal dependencies.

---

## Code Organization

### `candidate-pipeline/` — Framework

Defines the core abstractions for a recommendation pipeline:

- `candidate_pipeline.rs` — `CandidatePipeline` trait and `execute()` orchestration. Stages run in this order:
  1. `QueryHydrator` (parallel)
  2. `Source` (parallel)
  3. `Hydrator` (parallel)
  4. `Filter` (sequential)
  5. `Scorer` (sequential)
  6. `Selector`
  7. `PostSelectionHydrator` (parallel)
  8. `PostSelectionFilter` (sequential)
  9. `SideEffect` (parallel, fire-and-forget)
- `source.rs`, `hydrator.rs`, `query_hydrator.rs`, `filter.rs`, `scorer.rs`, `selector.rs`, `side_effect.rs` — Trait definitions for each stage.

### `home-mixer/` — Orchestration

Implements the concrete pipeline for the For You feed:

- `main.rs` — Entry point. Starts a gRPC `ScoredPostsService` server via `xai_http_server`.
- `server.rs` — `HomeMixerServer` implements `scored_posts_service_server::ScoredPostsService`. Converts protobuf queries into `ScoredPostsQuery`, runs the pipeline, and returns `ScoredPost` responses.
- `candidate_pipeline/` — Domain models:
  - `phoenix_candidate_pipeline.rs` — Builds the full `CandidatePipeline` with all concrete sources, hydrators, filters, scorers, selectors, and side effects wired together.
  - `candidate.rs`, `candidate_features.rs` — `PostCandidate` and feature structs.
  - `query.rs`, `query_features.rs` — `ScoredPostsQuery` and feature structs.
- `sources/` — `ThunderSource` (in-network) and `PhoenixSource` (out-of-network retrieval).
- `candidate_hydrators/` — Enrich candidates with core data, author info, subscriptions, video duration, visibility filtering results, etc.
- `filters/` — Pre-scoring and post-selection filters (dedup, age, self-posts, muted keywords, socialgraph blocks, etc.).
- `scorers/` — `PhoenixScorer` (ML predictions), `WeightedScorer` (combines action probabilities), `AuthorDiversityScorer`, `OONScorer`.
- `selectors/` — `TopKScoreSelector`.
- `query_hydrators/` — Fetch user action sequences and user features (following list).
- `side_effects/` — `CacheRequestInfoSideEffect`.

### `thunder/` — In-Network Post Store

- `main.rs` — Entry point. Starts gRPC server, initializes `PostStore` and `StratoClient`, consumes Kafka events.
- `thunder_service.rs` — `ThunderServiceImpl` implements `InNetworkPostsService`. Fetches posts from followed accounts with concurrency limiting (semaphore), metrics, and optional Strato fallback for following lists.
- `posts/post_store.rs` — `PostStore` uses `DashMap` for thread-safe in-memory storage. Maintains separate deques per user for original posts, secondary posts (replies/reposts), and video posts. Supports retention-based trimming and delete events.
- `kafka/` — Kafka consumers for tweet create/delete events.
- `deserializer.rs`, `kafka_utils.rs` — Event deserialization and consumer setup.

### `phoenix/` — ML Retrieval & Ranking

- `grok.py` — Core transformer blocks ported from Grok-1 (attention, feed-forward, layer norm, etc.) and `make_recsys_attn_mask()` which implements **candidate isolation** (candidates cannot attend to each other).
- `recsys_model.py` — Ranking model (`PhoenixModelConfig`). Defines `RecsysBatch`, `RecsysEmbeddings`, `RecsysModelOutput`, and hash-based embedding reduction.
- `recsys_retrieval_model.py` — Two-tower retrieval model (`PhoenixRetrievalModelConfig`) with `CandidateTower` and user tower.
- `runners.py` — `RecsysInferenceRunner` and `RecsysRetrievalInferenceRunner` for model initialization, ranking, and retrieval. Also defines the 19 action names (favorite, reply, repost, click, dwell, follow_author, not_interested, block_author, etc.).
- `run_ranker.py` / `run_retrieval.py` — Standalone demo scripts.
- `test_recsys_model.py` — Tests for attention mask correctness (causal history, candidate isolation, self-attention).
- `test_recsys_retrieval_model.py` — Tests for candidate tower output shapes, L2 normalization, and retrieval runner behavior.

---

## Code Style Guidelines

### Python (Phoenix)
- **Formatter/Linter**: Ruff (`line-length = 100`, `indent-width = 4`).
- **Type annotations**: Use `jax.Array`, `jax.typing.ArrayLike`, `NamedTuple`, `dataclass`, and `typing` generics.
- **Docstrings**: Google-style docstrings are common (Args/Returns).
- **Header**: Every `.py` file starts with the Apache 2.0 copyright header.
- **Naming**:
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `SCREAMING_SNAKE_CASE` (e.g., `ACTIONS`)

### Rust
- **Formatting**: Standard `rustfmt` style (no custom config visible).
- **Naming**:
  - Types/traits: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `SCREAMING_SNAKE_CASE`
- **Async**: Heavy use of `tonic::async_trait` for gRPC service traits and pipeline stages.
- **Error handling**: `anyhow::Result` at boundaries; `Result<T, String>` inside pipeline traits.
- **Logging**: Use the `log` crate with structured messages including `request_id` and `stage`.
- **Metrics**: Prometheus-style metrics via `lazy_static!` macros (counters, histograms, gauges with label values).
- **Concurrency**: `tokio::task::spawn_blocking` is used for CPU-intensive or lock-heavy work (e.g., `PostStore` lookups) to avoid blocking the async runtime.

---

## Testing Instructions

- **Python**: Run `uv run pytest test_recsys_model.py test_recsys_retrieval_model.py` inside `phoenix/`.
- **Rust**: There are no Rust tests in this repository. Do not add `#[cfg(test)]` blocks expecting them to compile without the missing internal dependencies.

---

## Deployment and Runtime Architecture

- **Home Mixer** exposes a gRPC `ScoredPostsService` and serves as the entry point for feed requests.
- **Thunder** exposes a gRPC `InNetworkPostsService` and consumes Kafka for real-time post events.
- **Phoenix** is invoked as a gRPC client from `home-mixer` (via `PhoenixPredictionClient` and `PhoenixRetrievalClient`, which are excluded from this release).
- All Rust services use `xai_http_server` to unify HTTP (metrics/health) and gRPC on the same port structure.
- Prometheus metrics are emitted throughout (`POST_STORE_*`, `GET_IN_NETWORK_POSTS_*`, `IN_FLIGHT_REQUESTS`, `REJECTED_REQUESTS`, etc.).

---

## Security Considerations

- **Internal clients excluded**: The actual client implementations that talk to production services (Strato, Gizmoduck, Phoenix, TES, SocialGraph, Visibility Filtering) are not in this repo.
- **mTLS / S2S**: References to `S2S_CRT_PATH`, `S2S_KEY_PATH`, `S2S_CHAIN_PATH` exist in `home-mixer` but the credential loading logic is in the excluded `clients` module.
- **No secrets in repo**: No `.env`, keys, or certificates are present.

---

## License

Apache License 2.0. See `LICENSE` for details.
