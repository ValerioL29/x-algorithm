use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::scorers::ranking_scorer::{RankingScorer, ScoringWeights};
use tonic::async_trait;
use xai_candidate_pipeline::scorer::Scorer;

pub struct PhoenixScoresRankingScorer;

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for PhoenixScoresRankingScorer {
    fn enable(&self, _query: &ScoredPostsQuery) -> bool {
        true
    }

    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let weights = ScoringWeights::from_params(&query.params);
        candidates
            .iter()
            .map(|c| {
                let weighted = RankingScorer::compute_weighted_score(&weights, query, c);
                Ok(PostCandidate {
                    weighted_score: Some(weighted),
                    score: Some(weighted),
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.weighted_score = scored.weighted_score;
        candidate.score = scored.score;
    }
}
