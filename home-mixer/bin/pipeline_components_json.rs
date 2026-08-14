use std::sync::Arc;
use xai_candidate_pipeline::candidate_pipeline::{
    CandidatePipeline, PipelineComponents, PipelineStage,
};
use xai_home_mixer::candidate_pipeline::following_candidate_pipeline::FollowingCandidatePipeline;
use xai_home_mixer::candidate_pipeline::for_you_candidate_pipeline::ForYouCandidatePipeline;
use xai_home_mixer::candidate_pipeline::phoenix_candidate_pipeline::PhoenixCandidatePipeline;
use xai_home_mixer::candidate_pipeline::phoenix_scores_pipeline::PhoenixScoresPipeline;
use xai_home_mixer::candidate_pipeline::ranked_following_candidate_pipeline::RankedFollowingCandidatePipeline;
use xai_home_mixer::candidate_pipeline::reverse_chron_posts_pipeline::ReverseChronPostsPipeline;
use xai_home_mixer::server::QueryBuilder;
use xai_home_mixer::ScoredPostsServer;

fn escape_json(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\"', "\\\"")
}

fn json_string_list(values: &[String]) -> String {
    let escaped: Vec<String> = values
        .iter()
        .map(|v| format!("\"{}\"", escape_json(v)))
        .collect();
    format!("[{}]", escaped.join(","))
}

fn json_entry(stage: PipelineStage, components: &[String]) -> String {
    let stage_name = format!("{:?}", stage);
    format!(
        "{{\"stage\":\"{}\",\"components\":{}}}",
        escape_json(&stage_name),
        json_string_list(components)
    )
}

fn print_pipeline(name: &str, stages: &[PipelineComponents]) {
    println!("  {{\"pipeline\":\"{name}\",\"stages\":[");
    for (idx, stage) in stages.iter().enumerate() {
        let entry = json_entry(stage.stage, &stage.components);
        let suffix = if idx + 1 == stages.len() { "" } else { "," };
        println!("    {entry}{suffix}");
    }
    println!("  ]}}");
}

#[tokio::main]
async fn main() {
    xai_init_utils::init().rustls();

    let phoenix = PhoenixCandidatePipeline::mock().await;
    let scored_posts = Arc::new(ScoredPostsServer::new(
        QueryBuilder::mock(),
        Arc::new(PhoenixCandidatePipeline::mock().await),
    ));
    let for_you = ForYouCandidatePipeline::mock(Arc::clone(&scored_posts)).await;
    let ranked_following = RankedFollowingCandidatePipeline::mock(scored_posts).await;
    let following = FollowingCandidatePipeline::mock().await;
    let reverse_chron = ReverseChronPostsPipeline::mock().await;
    let phoenix_scores = PhoenixScoresPipeline::mock().await;

    println!("[");
    print_pipeline("PhoenixCandidatePipeline", &phoenix.components());
    println!(",");
    print_pipeline("ForYouCandidatePipeline", &for_you.components());
    println!(",");
    print_pipeline(
        "RankedFollowingCandidatePipeline",
        &ranked_following.components(),
    );
    println!(",");
    print_pipeline("FollowingCandidatePipeline", &following.components());
    println!(",");
    print_pipeline("ReverseChronPostsPipeline", &reverse_chron.components());
    println!(",");
    print_pipeline("PhoenixScoresPipeline", &phoenix_scores.components());
    println!("]");
}
