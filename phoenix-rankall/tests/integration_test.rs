use std::sync::Arc;

use async_trait::async_trait;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

use xai_recsys_rankall::consumer::{ConsumerError, EventConsumer, RawBatch};
use xai_recsys_rankall::metrics::PipelineMetrics;
use xai_recsys_rankall::pipeline::Pipeline;
use xai_recsys_rankall::processor::main_processor::MainProcessor;
use xai_recsys_rankall::processor::metadata_processor::MetadataProcessor;
use xai_recsys_rankall::processor::thrift_types::{
    serialize_binary, EngagementCount, PhoenixRankAllCandidateFlat, PhoenixRankAllObject,
};
use xai_recsys_rankall::store::base::{
    metadata_window_router, prefix_window_router, BaseSnapshotStore,
};
use xai_recsys_rankall::store::snowflake::timestamp_secs_to_snowflake;
use xai_recsys_rankall::store::writer::read_parquet_file;
use xai_recsys_rankall::store::StoreConfig;

struct MockConsumer {
    batches: Vec<RawBatch>,
    index: usize,
}

impl MockConsumer {
    fn new(batches: Vec<RawBatch>) -> Self {
        Self { batches, index: 0 }
    }
}

#[async_trait]
impl EventConsumer for MockConsumer {
    async fn poll_batch(&mut self) -> Result<RawBatch, ConsumerError> {
        if self.index < self.batches.len() {
            let batch = self.batches[self.index].clone();
            self.index += 1;
            Ok(batch)
        } else {
            Err(ConsumerError::Closed)
        }
    }

    async fn commit(&mut self) -> Result<(), ConsumerError> {
        Ok(())
    }

    async fn close(&mut self) -> Result<(), ConsumerError> {
        Ok(())
    }
}

fn recent_snowflake() -> i64 {
    let now = chrono::Utc::now().timestamp() as f64;
    timestamp_secs_to_snowflake(now - 60.0)
}

fn make_core_thrift(post_id: i64, author_id: i64, index_name: &str) -> Vec<u8> {
    serialize_binary(&PhoenixRankAllObject::new(
        Some(post_id),
        Some(author_id),
        Some(index_name.to_string()),
        None::<Vec<i64>>,
    ))
    .unwrap()
}

fn make_metadata_thrift(post_id: i64, author_id: i64, fav_count: i64) -> Vec<u8> {
    use xai_recsys_rankall::processor::thrift_types::MultiModalEmbeddings;
    let ec = EngagementCount::new(
        None::<i64>,
        None::<i64>,
        Some(fav_count),
        None::<i64>,
        None::<i64>,
        None::<i64>,
        None::<i64>,
        None::<i64>,
        None::<i64>,
    );
    serialize_binary(&PhoenixRankAllCandidateFlat::new(
        Some(post_id),
        Some(author_id),
        Some(false),
        Some(true),
        Some(5000_i64),
        Some(ec),
        None::<i64>,
        None::<MultiModalEmbeddings>,
        Some("metadata".to_string()),
    ))
    .unwrap()
}

#[tokio::test]
async fn main_pipeline_end_to_end() {
    let dir = TempDir::new().unwrap();
    let base_id = recent_snowflake();

    let consumer = MockConsumer::new(vec![
        vec![
            make_core_thrift(base_id, 10, "1fav"),
            make_core_thrift(base_id + 1, 20, "1fav"),
            make_core_thrift(base_id + 2, 30, "video"),
        ],
        vec![make_core_thrift(base_id + 3, 40, "1fav")],
    ]);

    let processor = MainProcessor::new();
    let store = BaseSnapshotStore::new(
        StoreConfig {
            output_dir: dir.path().to_path_buf(),
            windows: xai_recsys_rankall::config::PipelineKind::Main.window_configs(),
            compaction_interval_secs: 999,
            versions_to_keep: 3,
            pipeline: "main".to_string(),
            sid_num_levels: 6,
        },
        prefix_window_router(),
        Arc::new(PipelineMetrics::new(prometheus::Registry::new()).unwrap()),
    );

    let metrics = PipelineMetrics::new(prometheus::Registry::new()).unwrap();
    let cancel = CancellationToken::new();

    let pipeline = Pipeline::new(
        Box::new(consumer),
        Box::new(processor),
        Box::new(store),
        metrics,
    );

    pipeline.run(cancel).await.unwrap();

    let symlink = dir.path().join("1fav_1day.parquet");
    assert!(symlink.exists(), "1fav_1day.parquet should exist");

    let batches = read_parquet_file(&symlink).unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3, "3 records with index_name=1fav");

    let video_symlink = dir.path().join("video_2day.parquet");
    assert!(video_symlink.exists(), "video_2day.parquet should exist");

    let video_batches = read_parquet_file(&video_symlink).unwrap();
    let video_rows: usize = video_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(video_rows, 1, "1 record with index_name=video");
}

#[tokio::test]
async fn metadata_pipeline_end_to_end() {
    let dir = TempDir::new().unwrap();
    let base_id = recent_snowflake();

    let consumer = MockConsumer::new(vec![vec![
        make_metadata_thrift(base_id, 10, 50),
        make_metadata_thrift(base_id + 1, 20, 100),
    ]]);

    let processor = MetadataProcessor::new("metadata");
    let store = BaseSnapshotStore::new(
        StoreConfig {
            output_dir: dir.path().to_path_buf(),
            windows: xai_recsys_rankall::config::PipelineKind::Metadata.window_configs(),
            compaction_interval_secs: 999,
            versions_to_keep: 3,
            pipeline: "metadata".to_string(),
            sid_num_levels: 6,
        },
        metadata_window_router(),
        Arc::new(PipelineMetrics::new(prometheus::Registry::new()).unwrap()),
    );

    let metrics = PipelineMetrics::new(prometheus::Registry::new()).unwrap();
    let cancel = CancellationToken::new();

    let pipeline = Pipeline::new(
        Box::new(consumer),
        Box::new(processor),
        Box::new(store),
        metrics,
    );

    pipeline.run(cancel).await.unwrap();

    for window in &["metadata_1day", "metadata_2day", "metadata_3day"] {
        let path = dir.path().join(format!("{window}.parquet"));
        assert!(path.exists(), "{window}.parquet should exist");

        let batches = read_parquet_file(&path).unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2, "{window} should have 2 records");
    }

    let batches = read_parquet_file(&dir.path().join("metadata_1day.parquet")).unwrap();
    let schema = batches[0].schema();
    let col_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(col_names.contains(&"post_id"));
    assert!(col_names.contains(&"author_id"));
    assert!(col_names.contains(&"has_video"));
    assert!(col_names.contains(&"fav_count"));
    assert!(col_names.contains(&"block_count"));
    assert_eq!(
        col_names.len(),
        15,
        "metadata schema should have 15 columns"
    );
}

#[tokio::test]
async fn pipeline_shutdown_flushes_data() {
    let dir = TempDir::new().unwrap();
    let base_id = recent_snowflake();

    struct OneShot {
        batch: Option<RawBatch>,
    }

    #[async_trait]
    impl EventConsumer for OneShot {
        async fn poll_batch(&mut self) -> Result<RawBatch, ConsumerError> {
            if let Some(b) = self.batch.take() {
                Ok(b)
            } else {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                Err(ConsumerError::Closed)
            }
        }
        async fn commit(&mut self) -> Result<(), ConsumerError> {
            Ok(())
        }
        async fn close(&mut self) -> Result<(), ConsumerError> {
            Ok(())
        }
    }

    let consumer = OneShot {
        batch: Some(vec![make_core_thrift(base_id, 10, "1fav")]),
    };

    let processor = MainProcessor::new();
    let windows = vec![xai_recsys_rankall::config::WindowConfig::new("1fav", 24)];
    let store = BaseSnapshotStore::new(
        StoreConfig {
            output_dir: dir.path().to_path_buf(),
            windows,
            compaction_interval_secs: 999,
            versions_to_keep: 3,
            pipeline: "main".to_string(),
            sid_num_levels: 6,
        },
        prefix_window_router(),
        Arc::new(PipelineMetrics::new(prometheus::Registry::new()).unwrap()),
    );

    let metrics = PipelineMetrics::new(prometheus::Registry::new()).unwrap();
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();

    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        cancel_clone.cancel();
    });

    let pipeline = Pipeline::new(
        Box::new(consumer),
        Box::new(processor),
        Box::new(store),
        metrics,
    );

    pipeline.run(cancel).await.unwrap();

    let symlink = dir.path().join("1fav_1day.parquet");
    assert!(symlink.exists(), "data should be flushed on shutdown");

    let batches = read_parquet_file(&symlink).unwrap();
    assert_eq!(batches[0].num_rows(), 1);
}
