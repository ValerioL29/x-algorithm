use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery};
use crate::util;
use crate::SPAN_LEVEL;
use std::any::{type_name_of_val, Any};
use tracing::{field::Empty, Span};
use xai_stats_receiver::global_stats_receiver;

const KEPT_SCOPE: [(&str, &str); 1] = [("requests", "kept")];
const REMOVED_SCOPE: [(&str, &str); 1] = [("requests", "removed")];

pub struct FilterResult<C> {
    pub kept: Vec<C>,
    pub removed: Vec<C>,
}

pub trait Filter<Q, C>: Any + Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats(latency=Bucket0To50)]
    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "filter", fields(
        name = self.name(),
        input_count = candidates.len(),
        kept_count = Empty,
        removed_count = Empty,
        filter_rate = Empty,
    ))]
    fn run(&self, query: &Q, candidates: Vec<C>) -> FilterResult<C> {
        let result = self.filter(query, candidates);
        let total = result.kept.len() + result.removed.len();
        let rate = if total > 0 {
            result.removed.len() as f64 / total as f64
        } else {
            0.0
        };
        let span = Span::current();
        span.record("kept_count", result.kept.len());
        span.record("removed_count", result.removed.len());
        span.record("filter_rate", format!("{:.3}", rate).as_str());
        #[cfg(feature = "quiet-spans")]
        tracing::info!(
            component = self.name(),
            kept_count = result.kept.len(),
            removed_count = result.removed.len(),
            filter_rate = format!("{rate:.3}"),
            "filter"
        );
        self.stat(&result);
        result
    }

    fn filter(&self, query: &Q, candidates: Vec<C>) -> FilterResult<C>;

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }

    fn stat(&self, result: &FilterResult<C>) {
        if let Some(receiver) = global_stats_receiver() {
            let metric_name = format!("{}.run", self.name());
            receiver.incr(metric_name.as_str(), &KEPT_SCOPE, result.kept.len() as u64);
            receiver.incr(
                metric_name.as_str(),
                &REMOVED_SCOPE,
                result.removed.len() as u64,
            );
        }
    }
}
