use crate::{
    status::{ERROR, SUCCESS},
    AddTaskResponse, AggregatorAddTaskRequest, AggregatorRole,
};
use anyhow::Context;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use clap::Parser;
use futures::future::try_join_all;
use janus_aggregator::{
    binary_utils::{janus_main, BinaryOptions, CommonBinaryOptions},
    config::{BinaryConfig, CommonConfig},
};
use janus_aggregator_core::{
    datastore::Datastore,
    task::{self, AggregationMode, AggregatorTask, AggregatorTaskParameters},
    SecretBytes,
};
use janus_core::{
    auth_tokens::{AuthenticationToken, AuthenticationTokenHash},
    time::RealClock,
};
use janus_messages::{Duration, HpkeConfig, Time};
use prio::codec::Decode;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{net::SocketAddr, sync::Arc};
use trillium::{Conn, Handler, Status};
use trillium_api::{api, ApiConnExt, Json};
use trillium_proxy::{upstream::IntoUpstreamSelector, Client, Proxy};
use trillium_router::Router;
use trillium_tokio::ClientConfig;
use url::Url;

#[derive(Debug, Serialize)]
struct EndpointResponse {
    status: &'static str,
    endpoint: String,
}

async fn handle_add_task(
    datastore: &Datastore<RealClock>,
    request: AggregatorAddTaskRequest,
) -> anyhow::Result<()> {
    let peer_aggregator_endpoint = match request.role {
        AggregatorRole::Leader => request.helper,
        AggregatorRole::Helper => request.leader,
    };
    let vdaf = request.vdaf.into();
    let leader_authentication_token =
        AuthenticationToken::new_dap_auth_token_from_string(request.leader_authentication_token)
            .context("invalid header value in \"leader_authentication_token\"")?;
    let vdaf_verify_key = SecretBytes::new(
        URL_SAFE_NO_PAD
            .decode(request.vdaf_verify_key)
            .context("invalid base64url content in \"vdaf_verify_key\"")?,
    );
    let time_precision = Duration::from_seconds(request.time_precision);
    let collector_hpke_config_bytes = URL_SAFE_NO_PAD
        .decode(request.collector_hpke_config)
        .context("invalid base64url content in \"collector_hpke_config\"")?;
    let collector_hpke_config = HpkeConfig::get_decoded(&collector_hpke_config_bytes)
        .context("could not parse collector HPKE configuration")?;

    let aggregator_parameters = match (request.role, request.collector_authentication_token) {
        (AggregatorRole::Leader, None) => {
            return Err(anyhow::anyhow!("collector authentication token is missing"))
        }
        (AggregatorRole::Leader, Some(collector_authentication_token)) => {
            AggregatorTaskParameters::Leader {
                aggregator_auth_token: leader_authentication_token,
                collector_auth_token_hash: AuthenticationTokenHash::from(
                    &AuthenticationToken::new_dap_auth_token_from_string(
                        collector_authentication_token,
                    )
                    .context("invalid header value in \"collector_authentication_token\"")?,
                ),
                collector_hpke_config,
            }
        }
        (AggregatorRole::Helper, _) => AggregatorTaskParameters::Helper {
            aggregator_auth_token_hash: AuthenticationTokenHash::from(&leader_authentication_token),
            collector_hpke_config,
            // TODO(#3436): allow callers to specify asynchronous aggregation mode (requires updated interop test design)
            aggregation_mode: AggregationMode::Synchronous,
        },
    };

    let batch_mode = match request.batch_mode {
        1 => task::BatchMode::TimeInterval,
        2 => task::BatchMode::LeaderSelected {
            batch_time_window_size: None,
        },
        _ => {
            return Err(anyhow::anyhow!(
                "invalid batch mode: {}",
                request.batch_mode
            ))
        }
    };

    let task = AggregatorTask::new(
        request.task_id,
        peer_aggregator_endpoint,
        batch_mode,
        vdaf,
        vdaf_verify_key,
        request.task_start.map(Time::from_seconds_since_epoch),
        request.task_end.map(Time::from_seconds_since_epoch),
        None,
        request.min_batch_size,
        time_precision,
        // We can be strict about clock skew since this executable is only intended for use with
        // other aggregators running on the same host.
        Duration::from_seconds(1),
        aggregator_parameters,
    )
    .context("error constructing task")?;

    datastore
        .run_unnamed_tx(move |tx| {
            let task = task.clone();
            Box::pin(async move { tx.put_aggregator_task(&task).await })
        })
        .await
        .context("error adding task to database")
}

async fn make_handler(
    datastore: Arc<Datastore<RealClock>>,
    dap_serving_prefix: String,
    aggregator_address: SocketAddr,
    health_check_peers: Vec<Url>,
) -> anyhow::Result<impl Handler> {
    let upstream = format!("http://{aggregator_address}/").into_upstream();
    let proxy_handler = Proxy::new(
        Client::new(ClientConfig::default()).with_default_pool(),
        upstream,
    );
    let health_check_client = Client::new(ClientConfig::default()).with_default_pool();

    let handler = Router::new()
        .post("internal/test/ready", move |conn: Conn| {
            let health_check_peers = health_check_peers.clone();
            let health_check_client = health_check_client.clone();
            async move {
                let result: Result<_, anyhow::Error> =
                    try_join_all(health_check_peers.iter().map(|peer| {
                        let client = health_check_client.clone();
                        async move {
                            let _ = client.get(peer.as_str()).await?.success()?;
                            Ok(())
                        }
                    }))
                    .await;
                match result {
                    Ok(_) => conn.with_json(&json!({})),
                    Err(_) => conn.with_status(Status::ServiceUnavailable),
                }
            }
        })
        .post(
            "internal/test/endpoint_for_task",
            Json(EndpointResponse {
                status: "success",
                endpoint: dap_serving_prefix.clone(),
            }),
        )
        .post(
            "internal/test/add_task",
            api(
                move |_conn: &mut Conn, Json(request): Json<AggregatorAddTaskRequest>| {
                    let datastore = Arc::clone(&datastore);
                    async move {
                        match handle_add_task(&datastore, request).await {
                            Ok(()) => Json(AddTaskResponse {
                                status: SUCCESS.to_string(),
                                error: None,
                            }),
                            Err(e) => Json(AddTaskResponse {
                                status: ERROR.to_string(),
                                error: Some(format!("{e:?}")),
                            }),
                        }
                    }
                },
            ),
        )
        .all(format!("{dap_serving_prefix}/*"), proxy_handler);
    Ok(handler)
}

#[derive(Debug, Parser)]
#[clap(
    name = "janus_interop_aggregator",
    about = "Janus interoperation test aggregator",
    rename_all = "kebab-case",
    version = env!("CARGO_PKG_VERSION")
)]
pub struct Options {
    #[clap(flatten)]
    common: CommonBinaryOptions,
}

impl BinaryOptions for Options {
    fn common_options(&self) -> &CommonBinaryOptions {
        {
            &self.common
        }
    }
}

fn default_dap_serving_prefix() -> String {
    "/".to_string()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Config {
    #[serde(flatten)]
    common_config: CommonConfig,

    /// Address on which this server should listen for connections and serve its
    /// API endpoints.
    listen_address: SocketAddr,

    /// Path prefix, e.g. `/dap/`, to serve DAP from.
    #[serde(default = "default_dap_serving_prefix")]
    dap_serving_prefix: String,

    /// Address on which the aggregator's HTTP server is listening. DAP requests will be proxied to
    /// this.
    aggregator_address: SocketAddr,

    /// List of URLs to health check endpoints of aggregator subprocesses. The interop aggregator
    /// will check these endpoints for readiness before considering itself ready.
    health_check_peers: Vec<Url>,
}

impl BinaryConfig for Config {
    fn common_config(&self) -> &CommonConfig {
        &self.common_config
    }

    fn common_config_mut(&mut self) -> &mut CommonConfig {
        &mut self.common_config
    }
}

impl Options {
    pub fn run(self) -> anyhow::Result<()> {
        janus_main::<_, _, Config, _, _>(
            "janus_interop_aggregator",
            self,
            RealClock::default(),
            true,
            |ctx| async move {
                ctx.datastore.put_hpke_key().await.unwrap();

                // Run an HTTP server with both the DAP aggregator endpoints and the interoperation test
                // endpoints.
                let handler = make_handler(
                    Arc::new(ctx.datastore),
                    ctx.config.dap_serving_prefix,
                    ctx.config.aggregator_address,
                    ctx.config.health_check_peers,
                )
                .await?;
                trillium_tokio::config()
                    .with_host(&ctx.config.listen_address.ip().to_string())
                    .with_port(ctx.config.listen_address.port())
                    .without_signals()
                    .run_async(handler)
                    .await;

                Ok(())
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Options;
    use clap::CommandFactory;

    #[test]
    fn verify_clap_app() {
        Options::command().debug_assert();
    }
}
