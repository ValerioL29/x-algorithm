use std::sync::Arc;

use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::media_info_cache_client::MediaInfoCacheClient;
use xai_candidate_pipeline::component_library::utils::{default_quick_cache, QuickCache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedHydrator};

use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;

type MediaInfoCacheValue = (Option<bool>, Option<i32>);

pub struct MediaInfoHydrator {
    pub media_info_cache_client: Arc<dyn MediaInfoCacheClient + Send + Sync>,
    pub cache: QuickCache<u64, MediaInfoCacheValue>,
}

impl MediaInfoHydrator {
    pub async fn new(media_info_cache_client: Arc<dyn MediaInfoCacheClient + Send + Sync>) -> Self {
        let cache = default_quick_cache();
        Self {
            media_info_cache_client,
            cache,
        }
    }
}

fn min_video_duration_ms(durations: &[i64]) -> Option<i32> {
    durations.iter().copied().min().map(|v| v as i32)
}

#[async_trait]
impl CachedHydrator<ScoredPostsQuery, PostCandidate> for MediaInfoHydrator {
    type CacheKey = u64;

    type CacheValue = MediaInfoCacheValue;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    fn cache_store(&self) -> &dyn CacheStore<Self::CacheKey, Self::CacheValue> {
        &self.cache
    }

    fn cache_key(&self, candidate: &PostCandidate) -> Self::CacheKey {
        candidate.get_original_tweet_id()
    }

    fn cache_value(&self, hydrated: &PostCandidate) -> Self::CacheValue {
        (hydrated.has_media, hydrated.min_video_duration_ms)
    }

    fn hydrate_from_cache(&self, value: Self::CacheValue) -> PostCandidate {
        PostCandidate {
            has_media: value.0,
            min_video_duration_ms: value.1,
            ..Default::default()
        }
    }

    async fn hydrate_from_client(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let tweet_ids: Vec<u64> = candidates
            .iter()
            .map(|c| c.get_original_tweet_id())
            .collect();

        let media_info = self
            .media_info_cache_client
            .multi_get_media_info(&tweet_ids)
            .await;

        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        for tweet_id in tweet_ids {
            let hydrated = match media_info.get(&tweet_id) {
                Some(Ok(Some(info))) => Ok(PostCandidate {
                    has_media: Some(info.has_media),
                    min_video_duration_ms: min_video_duration_ms(&info.video_durations_ms),
                    ..Default::default()
                }),
                Some(Ok(None)) | None => Ok(PostCandidate {
                    has_media: Some(false),
                    min_video_duration_ms: None,
                    ..Default::default()
                }),
                Some(Err(err)) => Err(err.clone()),
            };
            hydrated_candidates.push(hydrated);
        }

        hydrated_candidates
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.has_media = hydrated.has_media;
        candidate.min_video_duration_ms = hydrated.min_video_duration_ms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use xai_candidate_pipeline::component_library::clients::media_info_cache_client::MockMediaInfoCacheClient;
    use xai_candidate_pipeline::hydrator::Hydrator;
    use xai_x_thrift::tweet_media_info::TweetMediaInfo;

    fn mock_client(entries: HashMap<u64, Option<TweetMediaInfo>>) -> Arc<dyn MediaInfoCacheClient> {
        Arc::new(MockMediaInfoCacheClient {
            media_info: entries,
        })
    }

    fn info(has_media: bool, video_durations_ms: Vec<i64>) -> TweetMediaInfo {
        TweetMediaInfo::new(
            has_media,
            has_media,
            !video_durations_ms.is_empty(),
            if has_media { 1 } else { 0 },
            video_durations_ms,
        )
    }

    #[tokio::test]
    async fn test_has_media_and_duration() {
        let mut entries = HashMap::new();
        entries.insert(1001u64, Some(info(true, vec![5000, 3000])));

        let hydrator = MediaInfoHydrator::new(mock_client(entries)).await;

        let candidates = vec![PostCandidate {
            tweet_id: 1001,
            ..Default::default()
        }];

        let result = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert_eq!(result.len(), 1);
        assert!(result[0].is_ok());
        let c = result[0].as_ref().unwrap();
        assert_eq!(c.has_media, Some(true));
        assert_eq!(c.min_video_duration_ms, Some(3000));
    }

    #[tokio::test]
    async fn test_has_media_false_no_duration() {
        let mut entries = HashMap::new();
        entries.insert(1002u64, Some(info(false, vec![])));

        let hydrator = MediaInfoHydrator::new(mock_client(entries)).await;

        let candidates = vec![PostCandidate {
            tweet_id: 1002,
            ..Default::default()
        }];

        let result = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert_eq!(result.len(), 1);
        assert!(result[0].is_ok());
        let c = result[0].as_ref().unwrap();
        assert_eq!(c.has_media, Some(false));
        assert_eq!(c.min_video_duration_ms, None);
    }

    #[tokio::test]
    async fn test_miss_is_no_media() {
        let hydrator = MediaInfoHydrator::new(mock_client(HashMap::new())).await;

        let candidates = vec![PostCandidate {
            tweet_id: 1003,
            ..Default::default()
        }];

        let result = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert_eq!(result.len(), 1);
        assert!(result[0].is_ok());
        let c = result[0].as_ref().unwrap();
        assert_eq!(c.has_media, Some(false));
        assert_eq!(c.min_video_duration_ms, None);
    }

    #[tokio::test]
    async fn test_caching() {
        let mut entries = HashMap::new();
        entries.insert(1004u64, Some(info(true, vec![1200])));

        let hydrator = MediaInfoHydrator::new(mock_client(entries)).await;

        let candidates = vec![PostCandidate {
            tweet_id: 1004,
            ..Default::default()
        }];

        let first = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;
        assert_eq!(first[0].as_ref().unwrap().has_media, Some(true));
        assert_eq!(first[0].as_ref().unwrap().min_video_duration_ms, Some(1200));

        let second = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;
        assert_eq!(second[0].as_ref().unwrap().has_media, Some(true));
        assert_eq!(
            second[0].as_ref().unwrap().min_video_duration_ms,
            Some(1200)
        );
    }

    #[tokio::test]
    async fn test_multiple_candidates() {
        let mut entries = HashMap::new();
        entries.insert(2001u64, Some(info(true, vec![1000, 2000])));
        entries.insert(2002u64, Some(info(false, vec![])));

        let hydrator = MediaInfoHydrator::new(mock_client(entries)).await;

        let candidates = vec![
            PostCandidate {
                tweet_id: 2001,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2002,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2003,
                ..Default::default()
            },
        ];

        let result = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].as_ref().unwrap().has_media, Some(true));
        assert_eq!(
            result[0].as_ref().unwrap().min_video_duration_ms,
            Some(1000)
        );
        assert_eq!(result[1].as_ref().unwrap().has_media, Some(false));
        assert_eq!(result[1].as_ref().unwrap().min_video_duration_ms, None);
        assert_eq!(result[2].as_ref().unwrap().has_media, Some(false));
        assert_eq!(result[2].as_ref().unwrap().min_video_duration_ms, None);
    }
}
