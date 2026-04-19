# THUNDER.md — Deep Dive into the Thunder Module

> This document provides a detailed breakdown of the `thunder/` module, the in-memory post store and real-time ingestion pipeline that powers in-network content retrieval for the X "For You" feed.

---

## Table of Contents

- [Overview](#overview)
- [Module Structure](#module-structure)
- [Architecture at a Glance](#architecture-at-a-glance)
- [File-by-File Breakdown](#file-by-file-breakdown)
  - [`main.rs` — Service Entry Point](#mainrs--service-entry-point)
  - [`thunder_service.rs` — gRPC Service Implementation](#thunder_servicers--grpc-service-implementation)
  - [`posts/post_store.rs` — In-Memory Post Store](#postspost_storers--in-memory-post-store)
  - [`deserializer.rs` — Event Deserialization](#deserializerrs--event-deserialization)
  - [`kafka/` — Kafka Consumers](#kafka--kafka-consumers)
  - [`kafka_utils.rs` — Kafka Orchestration](#kafka_utilsrs--kafka-orchestration)
- [Data Flow](#data-flow)
  - [Ingestion Pipeline](#ingestion-pipeline)
  - [Query Pipeline](#query-pipeline)
- [Key Design Decisions](#key-design-decisions)
- [Observability and Metrics](#observability-and-metrics)
- [Missing Dependencies Note](#missing-dependencies-note)

---

## Overview

**Thunder** is a high-performance, in-memory post storage service written in Rust. Its primary responsibilities are:

1. **Real-time ingestion**: Consume tweet create/delete events from Kafka and maintain an up-to-date, in-memory index of posts.
2. **In-network retrieval**: Expose a gRPC `InNetworkPostsService` that returns recent posts from accounts a user follows.
3. **Video support**: Maintain a separate index for video-eligible posts to support video-specific feed requests.

Thunder is designed for low-latency reads and high-throughput writes, using `DashMap` for lock-free concurrent access and `tokio` for async I/O.

---

## Module Structure

```
thunder/
├── main.rs                          # Entry point: init PostStore, StratoClient, gRPC server
├── lib.rs                           # Module declarations
├── thunder_service.rs               # InNetworkPostsService gRPC implementation
├── posts/
│   ├── mod.rs
│   └── post_store.rs                # Core in-memory storage (DashMap + VecDeque)
├── deserializer.rs                  # Thrift / Protobuf event deserialization
├── kafka_utils.rs                   # Kafka setup orchestration
└── kafka/
    ├── mod.rs
    ├── utils.rs                     # Consumer creation + batch deserialization
    ├── tweet_events_listener.rs     # Legacy Thrift-based Kafka consumer
    └── tweet_events_listener_v2.rs  # Protobuf-based Kafka consumer (serving path)
```

> **Note:** Several modules referenced in `lib.rs` (`args`, `config`, `metrics`, `schema`, `strato_client`, `o2`) are **not present** in this open-source release. They depend on internal X/XAI crates. See [Missing Dependencies Note](#missing-dependencies-note).

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              THUNDER SERVICE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐         ┌─────────────────────┐         ┌─────────────┐  │
│   │   Kafka     │────────▶│     PostStore       │────────▶│   gRPC      │  │
│   │  Consumers  │         │  (DashMap in-mem)   │         │   Server    │  │
│   └─────────────┘         └─────────────────────┘         └─────────────┘  │
│          │                         │                              ▲        │
│          │                         │                              │        │
│          ▼                         ▼                              │        │
│   Tweet Create/Delete      Original / Secondary                   │        │
│   Events (Thrift/Proto)    / Video Posts                          │        │
│                                                                  │        │
│                                                   GetInNetworkPosts         │
│                                                        Requests             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File-by-File Breakdown

---

### `main.rs` — Service Entry Point

`main.rs` bootstraps the entire Thunder service.

**Initialization Flow:**
1. **Parse CLI args** via `clap` (`args::Args`).
2. **Initialize `PostStore`** with configurable retention and request timeout.
3. **Initialize `StratoClient`** for fallback following-list fetching.
4. **Create `ThunderServiceImpl`** with a bounded semaphore for concurrency control.
5. **Start HTTP/gRPC server** via `xai_http_server::HttpServer` on configured ports, with Zstd compression.
6. **Start Kafka consumers** via `kafka_utils::start_kafka()`, passing an MPSC channel for catch-up signaling.
7. **If in serving mode**:
   - Wait for all Kafka threads to signal catch-up completion.
   - Call `post_store.finalize_init()` to sort and trim posts.
   - Start the stats logger background task.
   - Start the auto-trim background task (every 2 minutes).
8. **Set readiness probe** and wait for termination.

**Key concurrency primitive:**
- `tokio::sync::mpsc::channel::<i64>(args.kafka_num_threads)`: used to synchronize Kafka catch-up before serving traffic.

---

### `thunder_service.rs` — gRPC Service Implementation

This file implements `InNetworkPostsService` for the `GetInNetworkPosts` RPC.

#### `ThunderServiceImpl`

```rust
pub struct ThunderServiceImpl {
    post_store: Arc<PostStore>,
    strato_client: Arc<StratoClient>,
    request_semaphore: Arc<Semaphore>,
}
```

**Concurrency control:**
- A `tokio::sync::Semaphore` limits in-flight requests to `max_concurrent_requests`.
- If the semaphore is exhausted, requests are **immediately rejected** with `Status::resource_exhausted`.
- `IN_FLIGHT_REQUESTS` and `REJECTED_REQUESTS` Prometheus metrics track load.

**`get_in_network_posts` handler flow:**
1. Acquire semaphore permit (or reject).
2. Start total latency timer (`GET_IN_NETWORK_POSTS_DURATION`).
3. If `following_user_ids` is empty and `debug=true`, fetch following list from `StratoClient`.
4. Clamp `following_user_ids` and `exclude_tweet_ids` to `MAX_INPUT_LIST_SIZE`.
5. Determine `max_results` (video requests use `MAX_VIDEOS_TO_RETURN`).
6. **Spawn blocking task** (`tokio::task::spawn_blocking`) to query `PostStore`:
   - `get_videos_by_users(...)` for video requests.
   - `get_all_posts_by_users(...)` for standard requests.
7. **Score posts by recency** (`score_recent`): sorts by `created_at` descending and truncates to `max_results`.
8. Record metrics and return `GetInNetworkPostsResponse`.

**`analyze_and_report_post_statistics`**
A helper that computes and emits Prometheus metrics for a batch of posts:
- Freshness (seconds since most recent post)
- Time range (oldest → newest)
- Reply ratio
- Unique author count
- Posts per author

These metrics are labeled by stage (`"retrieved"` or `"scored"`).

---

### `posts/post_store.rs` — In-Memory Post Store

The heart of Thunder. `PostStore` maintains a thread-safe, in-memory index of posts using `DashMap` and `VecDeque`.

#### Core Data Structures

```rust
pub struct PostStore {
    posts: Arc<DashMap<i64, LightPost>>,                    // post_id → full post
    original_posts_by_user: Arc<DashMap<i64, VecDeque<TinyPost>>>,   // author_id → originals
    secondary_posts_by_user: Arc<DashMap<i64, VecDeque<TinyPost>>>,  // author_id → replies/RTs
    video_posts_by_user: Arc<DashMap<i64, VecDeque<TinyPost>>>,      // author_id → videos
    deleted_posts: Arc<DashMap<i64, bool>>,
    retention_seconds: u64,
    request_timeout: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TinyPost {
    pub post_id: i64,
    pub created_at: i64,
}
```

- **`TinyPost`**: A minimal reference (ID + timestamp) stored per-user timeline. Keeps per-user memory low.
- **`LightPost`**: The full protobuf-defined post struct (from `xai_thunder_proto`), stored once globally and referenced by ID.

#### Write Operations

**`insert_posts(posts: Vec<LightPost>)`**
1. Filters out future posts and posts older than `retention_seconds`.
2. Sorts by `created_at`.
3. Delegates to `insert_posts_internal`.

**`insert_posts_internal(posts: Vec<LightPost>)`**
For each post:
- Skip if already in `deleted_posts`.
- Insert full post into `posts` map; skip if already present (dedup).
- Create a `TinyPost` reference.
- Append to `original_posts_by_user` (if original) or `secondary_posts_by_user` (if reply/retweet).
- Append to `video_posts_by_user` if video-eligible.

**Video eligibility logic:**
- A post is video-eligible if `has_video` is true.
- If it's a retweet, the eligibility may be derived from the source post (if source has video and is not a reply).
- Replies are never video-eligible.

**`mark_as_deleted(posts: Vec<TweetDeleteEvent>)`**
- Removes post from `posts` map.
- Records post ID in `deleted_posts` set.
- Appends a deletion record to a special `DELETE_EVENT_KEY` timeline.

#### Read Operations

**`get_all_posts_by_users(user_ids, exclude_tweet_ids, start_time, request_user_id)`**
1. Queries `original_posts_by_user` (capped at `MAX_ORIGINAL_POSTS_PER_AUTHOR`).
2. Queries `secondary_posts_by_user` (capped at `MAX_REPLY_POSTS_PER_AUTHOR`).
3. Merges results.

**`get_videos_by_users(user_ids, exclude_tweet_ids, start_time, request_user_id)`**
- Queries `video_posts_by_user` (capped at `MAX_VIDEO_POSTS_PER_AUTHOR`).

**`get_posts_from_map(...)` — unified read path**
Iterates over requested `user_ids`:
1. **Timeout check**: aborts iteration if `request_timeout` exceeded.
2. Retrieves the user's `VecDeque<TinyPost>` from the specified map.
3. Iterates **newest first** (`rev()`), skipping excluded tweet IDs.
4. Looks up full `LightPost` from `posts` map.
5. Filters deleted posts and self-retweets (`source_user_id == request_user_id`).
6. Applies **reply visibility filtering**:
   - If `following_users` set is empty (original posts path), allow all.
   - For secondary posts: a reply is visible only if it replies to a non-reply/non-retweet original post, **or** if it's part of a conversation thread where the in-reply-to chain leads back to the original conversation and the replied-to user is followed.
7. Takes `max_per_user` and accumulates into result vector.

#### Maintenance Operations

**`finalize_init()`**
- Called after Kafka catch-up completes.
- Sorts all user post lists.
- Trims old posts.
- Re-applies deletions (handles out-of-order create/delete in the feeder).

**`trim_old_posts()`**
- Spawns a blocking task.
- Iterates each timeline map and pops from the front while `current_time - created_at > retention_seconds`.
- Removes the full post from `posts` map.
- Shrinks `VecDeque` capacity if over-allocated.
- Removes empty user entries.

**`sort_all_user_posts()`**
- Sorts every `VecDeque` by `created_at` ascending.

**`start_stats_logger(self: Arc<Self>)`**
- Background tokio task that logs stats every 5 seconds:
  - User count
  - Total posts
  - Original / secondary / video / deleted post counts

**`start_auto_trim(self: Arc<Self>, interval_minutes)`**
- Background tokio task that calls `trim_old_posts()` on a schedule.

---

### `deserializer.rs` — Event Deserialization

Thunder supports two event serialization formats:

1. **Thrift (legacy)** — used by `tweet_events_listener.rs`:
   ```rust
   pub fn deserialize_tweet_event(payload: &[u8]) -> Result<TweetEvent>
   pub fn deserialize_event(payload: &[u8]) -> Result<Event>
   ```
   Uses `TBinaryInputProtocol` from the `thrift` crate.

2. **Protobuf (v2 / serving path)** — used by `tweet_events_listener_v2.rs`:
   ```rust
   pub fn deserialize_tweet_event_v2(payload: &[u8]) -> Result<InNetworkEvent>
   ```
   Uses `prost::Message::decode`.

---

### `kafka/` — Kafka Consumers

#### `kafka/utils.rs`

**`create_kafka_consumer(config)`**
- Constructs and starts a `KafkaConsumer` from the `xai_kafka` internal crate.
- Wraps it in `Arc<RwLock<KafkaConsumer>>` for shared access.

**`deserialize_kafka_messages<T, F>(messages, deserializer)`**
- Iterates over `KafkaMessage` payloads.
- Applies the provided deserializer function.
- Tracks `BATCH_PROCESSING_TIME` and `KAFKA_MESSAGES_FAILED_PARSE` metrics.
- Returns a `Vec<T>` of successfully deserialized events.

#### `kafka/tweet_events_listener.rs` — Legacy Consumer

This is the **non-serving / feeder** path that reads raw Thrift `TweetEvent`s and produces protobuf `InNetworkEvent`s to a downstream Kafka topic.

**`start_tweet_event_processing(base_config, producer_config, args)`**
- Spawns `kafka_num_threads` consumer threads.
- Partitions are divided evenly across threads (`partitions_per_thread`).
- Each thread gets its own `KafkaConsumer` with an assigned partition subset.
- Starts a **partition lag monitor** per thread.
- Runs `process_tweet_events(...)` loop.

**`process_tweet_events(consumer, batch_size, producer, post_retention_sec)`**
- Polls Kafka in a loop (`poll(100)`).
- Buffers messages until `batch_size` reached.
- Calls `process_message_batch(...)`.
- Commits offsets after successful batch processing.
- On poll errors, increments `KAFKA_POLL_ERRORS` and sleeps 100ms.

**`process_message_batch(messages, batch_num, producer, post_retention_sec)`**
- Deserializes Thrift `TweetEvent`s.
- Extracts `LightPost`s from `TweetCreateEvent`s (skips nullcast tweets).
- Extracts delete IDs from `TweetDeleteEvent` and `QuotedTweetDeleteEvent`.
- If a `KafkaProducer` is configured, **spawns a tokio task per event** to send `InNetworkEvent` protobufs downstream.
- Logs a milestone every 1000 batches.

**Video eligibility helper:**
```rust
fn is_eligible_video(tweet: &Tweet) -> bool
```
Checks if the first media item is a video with duration ≥ `MIN_VIDEO_DURATION_MS`.

#### `kafka/tweet_events_listener_v2.rs` — Serving Consumer

This is the **serving** path. It reads pre-processed protobuf `InNetworkEvent`s directly and updates the `PostStore`.

**`start_tweet_event_processing_v2(base_config, post_store, args, tx)`**
- Similar threading model to the legacy listener.
- Uses `deserialize_tweet_event_v2` (protobuf).
- Signals catch-up completion via the MPSC sender `tx`.

**`process_tweet_events_v2(consumer, post_store, batch_size, tx, semaphore)`**
- Polls Kafka (`poll(batch_size)`).
- Tracks catch-up state by checking partition lags.
- Once lag is low (`total_lag < partitions * batch_size`), marks `init_data_downloaded = true` and sends a signal on `tx`.
- After catch-up, acquires a `Semaphore` permit (max 3 concurrent updates) before processing batches, reserving CPU for serving requests.
- Deserialization and `PostStore` updates happen inside `tokio::task::spawn_blocking`.

**`deserialize_batch(messages)`**
- Deserializes `InNetworkEvent` protobufs.
- Separates into `Vec<LightPost>` (creates) and `Vec<TweetDeleteEvent>` (deletes).
- Logs throughput every 1000 batches.

---

### `kafka_utils.rs` — Kafka Orchestration

Coordinates which consumer path to start based on `args.is_serving`.

**`start_kafka(args, post_store, user, tx)`**
- If `is_serving`:
  - Configures a `KafkaConsumerConfig` for `IN_NETWORK_EVENTS_TOPIC` (protobuf).
  - Calls `start_tweet_event_processing_v2(...)`.
- If **not** serving:
  - Configures a `KafkaConsumerConfig` for `TWEET_EVENT_TOPIC` (Thrift).
  - Configures a `KafkaProducerConfig` for `IN_NETWORK_EVENTS_TOPIC`.
  - Calls `start_tweet_event_processing(...)`.

Both configs pull SASL/SSL credentials from `args` (or empty env vars in the open-source release).

---

## Data Flow

### Ingestion Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Kafka Topic    │────▶│ tweet_events_    │────▶│    PostStore    │
│ (Thrift/Proto)  │     │ listener_v2      │     │  (DashMap/Deque)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               │ spawn_blocking
                               │ (deserialize + insert)
                               ▼
                        ┌──────────────┐
                        │  LightPost   │
                        │  TinyPost    │
                        │  Deleted IDs │
                        └──────────────┘
```

1. Kafka consumers poll message batches.
2. Batches are deserialized into `LightPost` creates and `TweetDeleteEvent` deletes.
3. `PostStore.insert_posts()` and `PostStore.mark_as_deleted()` update the in-memory indexes inside a blocking task.

### Query Pipeline

```
┌─────────────────┐     ┌─────────────────────────┐     ┌─────────────────┐
│  HomeMixer gRPC │────▶│ ThunderServiceImpl      │────▶│   PostStore     │
│   Request       │     │ get_in_network_posts()  │     │ query by user   │
└─────────────────┘     └─────────────────────────┘     └─────────────────┘
                               │                                │
                               │ Semaphore permit               │ Reverse iterate
                               │ Strato fallback (optional)     │ TinyPost → LightPost
                               ▼                                ▼
                        ┌──────────────┐                ┌──────────────┐
                        │ score_recent │                │ Filter + Cap │
                        │  (by time)   │                │  Return top  │
                        └──────────────┘                └──────────────┘
```

1. gRPC request arrives with `user_id`, `following_user_ids`, and `exclude_tweet_ids`.
2. Thunder acquires a semaphore permit (or rejects).
3. Following list is fetched from Strato if empty (debug mode only in this release).
4. `PostStore` is queried in a `spawn_blocking` task.
5. Posts are scored by recency, truncated to `max_results`, and returned.

---

## Key Design Decisions

### 1. In-Memory Storage with DashMap
Thunder stores all hot data in memory using `DashMap` (a concurrent hash map). This avoids database round-trips and enables sub-millisecond reads at scale. The trade-off is memory footprint, which is managed via:
- Retention-based trimming.
- TinyPost references (only ID + timestamp per user timeline).
- Separate indexes so only relevant post types are scanned.

### 2. Separate Indexes for Post Types
Posts are partitioned into three indexes per user:
- **Original posts** (non-reply, non-retweet)
- **Secondary posts** (replies and retweets)
- **Video posts** (video-eligible originals/retweets)

This allows the query path to scan only the relevant subset and apply different per-author caps.

### 3. Tokio spawn_blocking for Store Operations
All `PostStore` reads and writes run inside `tokio::task::spawn_blocking` to avoid blocking the async runtime while holding locks or iterating over `DashMap`/`VecDeque` structures.

### 4. Concurrency Limits at the Gate
A `Semaphore` at the gRPC service layer provides backpressure. If the system is saturated, it rejects new requests immediately rather than queuing indefinitely.

### 5. Two-Mode Operation: Feeder vs. Serving
- **Feeder mode** (`is_serving = false`): Reads raw Thrift tweet events, enriches them, and produces protobuf `InNetworkEvent`s to a downstream topic.
- **Serving mode** (`is_serving = true`): Reads protobuf `InNetworkEvent`s directly into `PostStore` and answers gRPC queries.

This separation allows independent scaling of ingestion and serving.

### 6. Reply Visibility Filtering
Not all replies from followed users are shown. A reply is filtered out unless:
- It replies to a non-reply, non-retweet original post, **or**
- It is part of a conversation where the in-reply-to chain links back to the original conversation ID, **and** the user being replied to is followed.

This prevents surfacing deep, irrelevant reply threads.

---

## Observability and Metrics

Thunder emits extensive Prometheus metrics. Key families include:

| Metric Prefix | Description |
|---------------|-------------|
| `POST_STORE_*` | Entity counts, requests, timeouts, returned post ratios, deleted post filtering |
| `GET_IN_NETWORK_POSTS_*` | Request latency (with/without Strato), following/excluded sizes, post statistics (freshness, reply ratio, unique authors) |
| `KAFKA_PARTITION_LAG` | Per-partition consumer lag |
| `KAFKA_POLL_ERRORS` | Kafka polling failures |
| `KAFKA_MESSAGES_FAILED_PARSE` | Deserialization errors |
| `BATCH_PROCESSING_TIME` | Time spent processing a Kafka batch |
| `IN_FLIGHT_REQUESTS` | Current active gRPC requests |
| `REJECTED_REQUESTS` | Requests rejected due to semaphore exhaustion |

Additionally, structured `log::info!` / `log::warn!` messages include:
- `user_id`
- Post counts
- Timeout diagnostics
- Kafka milestone logging

---

## Missing Dependencies Note

As noted in the repository's `AGENTS.md`, the Rust code in this repository **cannot be compiled standalone** because it depends on numerous internal X/XAI crates. In the `thunder/` module specifically, the following modules are declared in `lib.rs` but their source files are **excluded** from the open-source release:

- `args` — CLI argument definitions.
- `config` — Constants like `MAX_POSTS_TO_RETURN`, `MAX_INPUT_LIST_SIZE`, retention defaults.
- `metrics` — Prometheus metric definitions and `Timer` helper.
- `schema` — Thrift schemas for `Tweet`, `TweetEvent`, `Event`, etc.
- `strato_client` — gRPC/HTTP client for fetching social graph following lists.
- `o2` — Internal operational tooling.

Additionally, the following external crates are referenced but not included:
- `xai_http_server`
- `xai_thunder_proto`
- `xai_kafka`
- `xai_wily`
- `xai_profiling`

This means **Thunder cannot be built or run from this repository alone**. The source code is provided for architectural reference and educational purposes.

---

*Document generated from source analysis of the `thunder/` module.*
