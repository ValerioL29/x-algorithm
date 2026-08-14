use std::cell::RefCell;
use std::fmt;
use std::future::Future;
use std::time::Instant;
use tracing::{info, Span};

use crate::candidate_pipeline::PipelineStage;

impl PipelineStage {
    fn summary_name(&self) -> &'static str {
        match self {
            PipelineStage::QueryHydrator => "query_hydrators",
            PipelineStage::DependentQueryHydrator => "dependent_query_hydrators",
            PipelineStage::Source => "sources",
            PipelineStage::Hydrator => "hydrators",
            PipelineStage::PostSelectionHydrator => "post_selection_hydrators",
            PipelineStage::Filter => "filters",
            PipelineStage::PostSelectionFilter => "post_selection_filters",
            PipelineStage::Scorer => "scorers",
            PipelineStage::Selector => "selector",
            PipelineStage::SideEffect => "side_effects",
        }
    }
}

tokio::task_local! {
    static ACTIVE: RefCell<PipelineSummary>;
}

pub async fn scope<F: Future>(fut: F) -> F::Output {
    ACTIVE
        .scope(RefCell::new(PipelineSummary::default()), fut)
        .await
}

pub fn emit(pipeline: &str, start: Instant, result_size: usize) {
    with_active(|summary| {
        info!(
            latency_ms = start.elapsed().as_millis() as u64,
            result_size, "{} Summary:{}", pipeline, summary
        );
    });
}

pub(crate) fn record_source_fetched(name: &str, count: usize) {
    let recorded = with_active(|summary| {
        summary
            .stage_mut(PipelineStage::Source)
            .fetched_per_source
            .push((name.to_string(), count));
    });
    if !recorded {
        info!("Fetched {} candidates", count);
    }
}

pub struct StageStats {
    stage: PipelineStage,
    start: Instant,
}

impl StageStats {
    pub fn begin(stage: PipelineStage) -> Self {
        Self {
            stage,
            start: Instant::now(),
        }
    }

    pub fn record_components(&self, total: usize, enabled: usize) {
        let span = Span::current();
        span.record("total_count", total);
        span.record("enabled_count", enabled);
        with_active(|summary| {
            let stage = summary.stage_mut(self.stage);
            stage.total = total;
            stage.enabled = enabled;
        });
    }

    pub fn finish(self) {
        let latency_ms = self.latency_ms();
        let recorded = with_active(|summary| {
            summary.stage_mut(self.stage).latency_ms = Some(latency_ms);
        });
        if !recorded {
            info!("latency_ms={}", latency_ms);
        }
    }

    pub fn finish_with_size(self, size: usize) {
        let latency_ms = self.latency_ms();
        let recorded = with_active(|summary| {
            let stage = summary.stage_mut(self.stage);
            stage.latency_ms = Some(latency_ms);
            stage.size = Some(size);
        });
        if !recorded {
            info!("latency_ms={} size={}", latency_ms, size);
        }
    }

    pub fn finish_filters(
        self,
        kept: usize,
        removed: usize,
        removed_per_filter: Vec<(String, usize)>,
    ) {
        let total = kept + removed;
        let rate = if total > 0 {
            removed as f64 / total as f64
        } else {
            0.0
        };
        let span = Span::current();
        span.record("kept_count", kept);
        span.record("removed_count", removed);
        span.record("filter_rate", format!("{:.3}", rate).as_str());
        if is_active() {
            with_active(|summary| {
                let stage = summary.stage_mut(self.stage);
                stage.kept = Some(kept);
                stage.removed = Some(removed);
                stage.removed_per_filter = removed_per_filter;
            });
        } else {
            info!(
                "kept {}, removed {} removed_per_filter [{}]",
                kept,
                removed,
                Counts(&removed_per_filter),
            );
        }
    }

    fn latency_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

#[derive(Default)]
struct PipelineSummary {
    stages: Vec<StageSummary>,
}

#[derive(Default)]
struct StageSummary {
    name: &'static str,
    total: usize,
    enabled: usize,
    latency_ms: Option<u64>,
    size: Option<usize>,
    kept: Option<usize>,
    removed: Option<usize>,
    removed_per_filter: Vec<(String, usize)>,
    fetched_per_source: Vec<(String, usize)>,
}

impl PipelineSummary {
    fn stage_mut(&mut self, stage: PipelineStage) -> &mut StageSummary {
        let name = stage.summary_name();
        if let Some(idx) = self.stages.iter().position(|s| s.name == name) {
            &mut self.stages[idx]
        } else {
            self.stages.push(StageSummary {
                name,
                ..Default::default()
            });
            self.stages.last_mut().unwrap()
        }
    }
}

impl fmt::Display for PipelineSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for stage in &self.stages {
            write!(
                f,
                " {}{{total={} enabled={}",
                stage.name, stage.total, stage.enabled
            )?;
            if let Some(latency_ms) = stage.latency_ms {
                write!(f, " latency_ms={}", latency_ms)?;
            }
            if !stage.fetched_per_source.is_empty() {
                write!(f, " fetched=[{}]", Counts(&stage.fetched_per_source))?;
            }
            if let Some(size) = stage.size {
                write!(f, " size={}", size)?;
            }
            if let (Some(kept), Some(removed)) = (stage.kept, stage.removed) {
                write!(f, " kept={} removed={}", kept, removed)?;
                if !stage.removed_per_filter.is_empty() {
                    write!(
                        f,
                        " removed_per_filter=[{}]",
                        Counts(&stage.removed_per_filter)
                    )?;
                }
            }
            write!(f, "}}")?;
        }
        Ok(())
    }
}

struct Counts<'a>(&'a [(String, usize)]);

impl fmt::Display for Counts<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (name, count)) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}={}", name, count)?;
        }
        Ok(())
    }
}

fn is_active() -> bool {
    ACTIVE.try_with(|_| ()).is_ok()
}

fn with_active(f: impl FnOnce(&mut PipelineSummary)) -> bool {
    ACTIVE
        .try_with(|summary| f(&mut summary.borrow_mut()))
        .is_ok()
}
