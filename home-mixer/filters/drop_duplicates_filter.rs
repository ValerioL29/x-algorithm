use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use rustc_hash::FxHashSet;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct DropDuplicatesFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for DropDuplicatesFilter {
    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let mut seen_ids =
            FxHashSet::with_capacity_and_hasher(candidates.len(), Default::default());
        let mut kept = Vec::with_capacity(candidates.len());
        let mut removed = Vec::new();

        for candidate in candidates {
            if seen_ids.insert(candidate.tweet_id) {
                kept.push(candidate);
            } else {
                removed.push(candidate);
            }
        }

        FilterResult { kept, removed }
    }
}
