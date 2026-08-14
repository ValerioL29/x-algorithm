use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{AuthorServedMetricsAuthorIds, EnableAuthorServedMetricsExperimentBucket};

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_stats_receiver::global_stats_receiver;

const SERVED_METRIC: &str = "AuthorServedMetrics.Served";
const REQUESTS_METRIC: &str = "AuthorServedMetrics.Requests";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PostType {
    Original,
    Reply,
    Retweet,
}

impl PostType {
    fn classify(candidate: &PostCandidate) -> Self {
        if candidate.retweeted_tweet_id.is_some() {
            PostType::Retweet
        } else if candidate.in_reply_to_tweet_id.is_some() {
            PostType::Reply
        } else {
            PostType::Original
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            PostType::Original => "original",
            PostType::Reply => "reply",
            PostType::Retweet => "retweet",
        }
    }
}

pub struct AuthorServedMetricsSideEffect;

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for AuthorServedMetricsSideEffect {
    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let Some(receiver) = global_stats_receiver() else {
            return Ok(());
        };

        let buckets = input
            .query
            .params
            .experiment_buckets(EnableAuthorServedMetricsExperimentBucket);
        if buckets.is_empty() {
            return Ok(());
        }

        let tracked: HashSet<u64> = input
            .query
            .params
            .get(AuthorServedMetricsAuthorIds)
            .into_iter()
            .collect();
        if tracked.is_empty() {
            return Ok(());
        }

        let counts = aggregate_counts(&input.selected_candidates, &tracked);

        for b in &buckets {
            receiver.incr(
                REQUESTS_METRIC,
                &[("ddg", &b.experiment), ("bucket", &b.bucket)],
                1,
            );
        }

        for ((author_id, post_type), count) in counts {
            let author_str = author_id.to_string();
            for b in &buckets {
                receiver.incr(
                    SERVED_METRIC,
                    &[
                        ("type", post_type.as_str()),
                        ("author_id", &author_str),
                        ("ddg", &b.experiment),
                        ("bucket", &b.bucket),
                    ],
                    count,
                );
            }
        }

        Ok(())
    }
}

fn aggregate_counts(
    candidates: &[PostCandidate],
    tracked: &HashSet<u64>,
) -> HashMap<(u64, PostType), u64> {
    let mut counts: HashMap<(u64, PostType), u64> = HashMap::new();
    for candidate in candidates {
        if tracked.contains(&candidate.author_id) {
            *counts
                .entry((candidate.author_id, PostType::classify(candidate)))
                .or_insert(0) += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn original(author_id: u64) -> PostCandidate {
        PostCandidate {
            author_id,
            ..Default::default()
        }
    }

    fn reply(author_id: u64) -> PostCandidate {
        PostCandidate {
            author_id,
            in_reply_to_tweet_id: Some(999),
            ..Default::default()
        }
    }

    fn retweet(author_id: u64) -> PostCandidate {
        PostCandidate {
            author_id,
            retweeted_tweet_id: Some(999),
            retweeted_user_id: Some(7),
            ..Default::default()
        }
    }

    #[test]
    fn classify_distinguishes_types() {
        assert_eq!(PostType::classify(&original(1)), PostType::Original);
        assert_eq!(PostType::classify(&reply(1)), PostType::Reply);
        assert_eq!(PostType::classify(&retweet(1)), PostType::Retweet);
    }

    #[test]
    fn retweet_takes_precedence_over_reply() {
        let mut candidate = retweet(1);
        candidate.in_reply_to_tweet_id = Some(123);
        assert_eq!(PostType::classify(&candidate), PostType::Retweet);
    }

    #[test]
    fn aggregate_counts_filters_untracked_and_buckets_by_type() {
        let tracked: HashSet<u64> = [10, 20].into_iter().collect();
        let candidates = vec![
            original(10),
            original(10),
            reply(10),
            retweet(20),
            original(30),
            reply(99),
        ];

        let counts = aggregate_counts(&candidates, &tracked);

        assert_eq!(counts.get(&(10, PostType::Original)), Some(&2));
        assert_eq!(counts.get(&(10, PostType::Reply)), Some(&1));
        assert_eq!(counts.get(&(20, PostType::Retweet)), Some(&1));
        assert_eq!(counts.get(&(30, PostType::Original)), None);
        assert_eq!(counts.get(&(99, PostType::Reply)), None);
        assert_eq!(counts.values().sum::<u64>(), 4);
    }

    #[test]
    fn aggregate_counts_empty_when_no_tracked_match() {
        let tracked: HashSet<u64> = [1].into_iter().collect();
        let candidates = vec![original(2), reply(3), retweet(4)];
        assert!(aggregate_counts(&candidates, &tracked).is_empty());
    }
}
