use anyhow::Context;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use clap::Parser;
use janus_aggregator::{
    aggregator::{self, http_handlers::aggregator_handler},
    binary_utils::{janus_main, BinaryOptions, CommonBinaryOptions},
    config::{BinaryConfig, CommonConfig},
};
use janus_aggregator_core::{
    datastore::Datastore,
    task::{self, Task},
    SecretBytes,
};
use janus_core::{
    task::{AuthenticationToken, DapAuthToken},
    time::RealClock,
};
use janus_interop_binaries::{
    status::{ERROR, SUCCESS},
    AddTaskResponse, AggregatorAddTaskRequest, AggregatorRole, HpkeConfigRegistry, Keyring,
};
use janus_messages::{Duration, HpkeConfig, Time};
use opentelemetry::metrics::Meter;
use prio::codec::Decode;
use serde::{Deserialize, Serialize};
use sqlx::{migrate::Migrator, Connection, PgConnection};
use std::{net::SocketAddr, path::Path, sync::Arc};
use tokio::sync::Mutex;
use trillium::{Conn, Handler};
use trillium_api::{api, Json};
use trillium_router::Router;

#[derive(Debug, Serialize)]
struct EndpointResponse {
    status: &'static str,
    endpoint: String,
}

async fn handle_add_task(
    datastore: &Datastore<RealClock>,
    keyring: &Mutex<HpkeConfigRegistry>,
    request: AggregatorAddTaskRequest,
) -> anyhow::Result<()> {
    let vdaf = request.vdaf.into();
    let leader_authentication_token = AuthenticationToken::DapAuth(
        DapAuthToken::try_from(request.leader_authentication_token.into_bytes())
            .context("invalid header value in \"leader_authentication_token\"")?,
    );
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

    let collector_authentication_tokens =
        match (request.role, request.collector_authentication_token) {
            (AggregatorRole::Leader, None) => {
                return Err(anyhow::anyhow!("collector authentication token is missing"))
            }
            (AggregatorRole::Leader, Some(collector_authentication_token)) => {
                Vec::from([AuthenticationToken::DapAuth(
                    DapAuthToken::try_from(collector_authentication_token.into_bytes())
                        .context("invalid header value in \"collector_authentication_token\"")?,
                )])
            }
            (AggregatorRole::Helper, _) => Vec::new(),
        };

    let hpke_keypair = keyring.lock().await.get_random_keypair();

    let query_type = match request.query_type {
        2 => task::QueryType::FixedSize {
            max_batch_size: request
                .max_batch_size
                .ok_or_else(|| anyhow::anyhow!("\"max_batch_size\" is missing"))?,
        },
        _ => {
            return Err(anyhow::anyhow!(
                "invalid query type: {}",
                request.query_type
            ))
        }
    };

    let task = Task::new(
        request.task_id,
        Vec::from([request.leader, request.helper]),
        query_type,
        vdaf,
        request.role.into(),
        Vec::from([vdaf_verify_key]),
        request.max_batch_query_count,
        request.task_expiration.map(Time::from_seconds_since_epoch),
        None,
        request.min_batch_size,
        time_precision,
        // We can be strict about clock skew since this executable is only intended for use with
        // other aggregators running on the same host.
        Duration::from_seconds(1),
        collector_hpke_config,
        Vec::from([leader_authentication_token]),
        collector_authentication_tokens,
        [hpke_keypair],
    )
    .context("error constructing task")?;

    datastore
        .run_tx(move |tx| {
            let task = task.clone();
            Box::pin(async move { tx.put_task(&task).await })
        })
        .await
        .context("error adding task to database")
}

fn make_handler(
    datastore: Arc<Datastore<RealClock>>,
    meter: &Meter,
    dap_serving_prefix: String,
) -> anyhow::Result<impl Handler> {
    let keyring = Keyring::new();
    let dap_handler = aggregator_handler(
        Arc::clone(&datastore),
        RealClock::default(),
        meter,
        aggregator::Config {
            max_upload_batch_size: 100,
            max_upload_batch_write_delay: std::time::Duration::from_millis(100),
            batch_aggregation_shard_count: 32,
        },
    )?;

    let handler = Router::new()
        .all(format!("{dap_serving_prefix}/*"), dap_handler)
        .post("internal/test/ready", Json(serde_json::Map::new()))
        .post(
            "internal/test/endpoint_for_task",
            Json(EndpointResponse {
                status: "success",
                endpoint: dap_serving_prefix,
            }),
        )
        .post(
            "internal/test/add_task",
            api(
                move |_conn: &mut Conn, Json(request): Json<AggregatorAddTaskRequest>| {
                    let datastore = Arc::clone(&datastore);
                    let keyring = keyring.clone();
                    async move {
                        match handle_add_task(&datastore, &keyring.0, request).await {
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
        );
    Ok(handler)
}

#[derive(Debug, Parser)]
#[clap(
    name = "janus_interop_aggregator",
    about = "Janus interoperation test aggregator",
    rename_all = "kebab-case",
    version = env!("CARGO_PKG_VERSION")
)]
struct Options {
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Config {
    #[serde(flatten)]
    common_config: CommonConfig,

    /// Address on which this server should listen for connections and serve its
    /// API endpoints.
    listen_address: SocketAddr,

    /// Path prefix, e.g. `/dap/`, to serve DAP from.
    #[serde(default = "default_dap_serving_prefix")]
    dap_serving_prefix: String,
}

impl BinaryConfig for Config {
    fn common_config(&self) -> &CommonConfig {
        &self.common_config
    }

    fn common_config_mut(&mut self) -> &mut CommonConfig {
        &mut self.common_config
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    janus_main::<_, Options, Config, _, _>(RealClock::default(), |ctx| async move {
        let datastore = Arc::new(ctx.datastore);

        // Apply SQL migrations to database
        let mut connection =
            PgConnection::connect(ctx.config.common_config.database.url.as_str()).await?;
        // Migration scripts are mounted into the container at this path by
        // Dockerfile.interop_aggregator
        let migrator = Migrator::new(Path::new("/etc/janus/migrations")).await?;
        migrator.run(&mut connection).await?;

        // Run an HTTP server with both the DAP aggregator endpoints and the interoperation test
        // endpoints.
        let handler = make_handler(
            Arc::clone(&datastore),
            &ctx.meter,
            ctx.config.dap_serving_prefix,
        )?;
        trillium_tokio::config()
            .with_host(&ctx.config.listen_address.ip().to_string())
            .with_port(ctx.config.listen_address.port())
            .without_signals()
            .run_async(handler)
            .await;

        Ok(())
    })
    .await
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
