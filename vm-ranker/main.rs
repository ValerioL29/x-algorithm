use anyhow::{Context, Result};
use axum::Router;
use clap::Parser;
use log::info;
use std::time::Duration;
use tonic::service::Routes;
use xai_http_server::{CancellationToken, GrpcConfig, HttpServer};

use xai_vm_ranker::{
    args::Args, dpp::DppConfig, embedding_store, ranker_service::VMRankerServiceImpl,
    scoring::DppContext,
};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    if args.enable_profiling {
        xai_profiling::spawn_server(3000, CancellationToken::new()).await;
    }

    let (dpp, preload_future) = if args.dpp_enabled {
        let (store, preload_future) = embedding_store::init_store(args.embedding_dim)
            .context("DPP requested but embedding store init failed")?;
        let config = DppConfig {
            top_k: args.dpp_top_k,
            theta: args.dpp_theta,
            max_selected_rank: args.dpp_max_selected_rank,
            debug_viewer_id: args.dpp_debug_viewer_id,
        };
        info!(
            "DPP enabled: top_k={}, theta={}, max_selected_rank={}, embedding_dim={}, debug_viewer_id={}",
            config.top_k,
            config.theta,
            config.max_selected_rank,
            args.embedding_dim,
            config.debug_viewer_id,
        );
        (Some(DppContext { store, config }), Some(preload_future))
    } else {
        info!("DPP rescoring disabled");
        (None, None)
    };

    let ranker_service = VMRankerServiceImpl::new(args.max_concurrent_requests, dpp);
    info!(
        "Initialized VMRankerService with max_concurrent_requests={}",
        args.max_concurrent_requests
    );

    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_service_status("", tonic_health::ServingStatus::NotServing)
        .await;
    let routes = Routes::new(ranker_service.server()).add_service(health_service);

    let grpc_config = GrpcConfig::new(args.grpc_port, routes);

    let mut http_server = HttpServer::builder(
        args.http_port,
        Router::new(),
        CancellationToken::new(),
        Duration::from_secs(10),
    )
    .with_grpc(grpc_config)
    .build()
    .await
    .context("Failed to create HTTP server")?;

    info!("HTTP server on port: {}", args.http_port);
    info!("gRPC server on port: {}", args.grpc_port);
    info!(
        "Metrics server on: http://0.0.0.0:{}/metrics",
        args.http_port
    );

    if let Some(preload) = preload_future {
        preload.await.context("O2 embedding preload failed")?;
    }

    http_server.set_readiness(true);
    health_reporter
        .set_service_status("", tonic_health::ServingStatus::Serving)
        .await;
    info!("HTTP/gRPC server is ready");

    http_server.wait_for_termination().await;
    info!("Server terminated");

    Ok(())
}
