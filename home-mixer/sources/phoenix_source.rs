use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnablePhoenixRetrievalFallback, EnablePhoenixSource, PhoenixMaxResults,
    PhoenixRetrievalInferenceClusterId, PhoenixRetrievalNewUserHistoryThreshold,
    PhoenixRetrievalNewUserInferenceClusterId, PhoenixXdsRetrievalMaxRetries,
};
use crate::util::egress::RetrievalDispatch;
use crate::util::phoenix_request::{build_client_context, build_user_context};
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::PhoenixRetrievalCluster;
use xai_candidate_pipeline::component_library::utils::quality_factor;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;

pub struct PhoenixSource {
    pub dispatch: RetrievalDispatch,
}

impl PhoenixSource {
    fn resolve_cluster(query: &ScoredPostsQuery) -> PhoenixRetrievalCluster {
        let configured_cluster =
            PhoenixRetrievalCluster::parse(&query.params.get(PhoenixRetrievalInferenceClusterId));

        let threshold: u64 = query.params.get(PhoenixRetrievalNewUserHistoryThreshold);
        if threshold > 0 {
            let action_count = query
                .retrieval_sequence
                .as_ref()
                .and_then(|s| s.metadata.as_ref())
                .map(|m| m.length)
                .unwrap_or(0);

            if action_count < threshold {
                return PhoenixRetrievalCluster::parse(
                    &query.params.get(PhoenixRetrievalNewUserInferenceClusterId),
                );
            }
        }

        if let Some(decider) = &query.decider {
            let is_prod = matches!(
                configured_cluster,
                PhoenixRetrievalCluster::Experiment1Fou | PhoenixRetrievalCluster::Experiment2Fou
            );
            if is_prod {
                if decider.enabled("override_retrieval_use_experiment2_fou") {
                    return PhoenixRetrievalCluster::Experiment2Fou;
                }
                if decider.enabled("override_retrieval_use_experiment1_fou") {
                    return PhoenixRetrievalCluster::Experiment1Fou;
                }
            }
        }

        configured_cluster
    }
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for PhoenixSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnablePhoenixSource)
            && (!query.is_topic_request() || query.is_bulk_topic_request())
            && !query.in_network_only
            && !query.has_cached_posts
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let user_id = query.user_id;

        let sequence = query
            .retrieval_sequence
            .as_ref()
            .ok_or_else(|| "PhoenixSource: missing retrieval_sequence".to_string())?;

        let cluster = Self::resolve_cluster(query);
        let client_context = build_client_context(query);
        let user_context = build_user_context(query);

        let response = self
            .dispatch
            .retrieve_with_fallback(
                query,
                cluster,
                user_id,
                sequence.clone(),
                query.columnar_retrieval_sequence.clone(),
                quality_factor::apply(query.params.get(PhoenixMaxResults)),
                vec![],
                None,
                client_context,
                user_context,
                query.params.get(PhoenixXdsRetrievalMaxRetries),
                query.params.get(EnablePhoenixRetrievalFallback),
            )
            .await
            .map_err(|e| format!("PhoenixSource: {e}"))?;

        let candidates: Vec<PostCandidate> = response
            .top_k_candidates
            .into_iter()
            .flat_map(|scored_candidates| scored_candidates.candidates)
            .filter_map(|scored_candidate| scored_candidate.candidate)
            .map(|tweet_info| PostCandidate {
                tweet_id: tweet_info.tweet_id,
                author_id: tweet_info.author_id,
                in_reply_to_tweet_id: (tweet_info.in_reply_to_tweet_id != 0)
                    .then_some(tweet_info.in_reply_to_tweet_id),
                retweeted_tweet_id: (tweet_info.retweeted_tweet_id != 0)
                    .then_some(tweet_info.retweeted_tweet_id),
                served_type: Some(pb::ServedType::ForYouPhoenixRetrieval),
                ..Default::default()
            })
            .collect();

        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::MockRetrievalClient;

    fn source() -> PhoenixSource {
        PhoenixSource {
            dispatch: RetrievalDispatch {
                prod: Arc::new(MockRetrievalClient),
                paths: vec![],
            },
        }
    }

    #[test]
    fn enable_returns_false_when_in_network_only() {
        let query = ScoredPostsQuery {
            in_network_only: true,
            ..Default::default()
        };
        assert!(!source().enable(&query));
    }

    #[test]
    fn enable_returns_false_when_has_cached_posts() {
        let query = ScoredPostsQuery {
            has_cached_posts: true,
            ..Default::default()
        };
        assert!(!source().enable(&query));
    }
}
