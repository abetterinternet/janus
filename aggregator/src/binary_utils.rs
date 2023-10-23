//! Utilities for Janus binaries.

pub mod job_driver;

use crate::{
    config::{BinaryConfig, DbConfig},
    metrics::{install_metrics_exporter, MetricsExporterConfiguration},
    trace::{install_trace_subscriber, OpenTelemetryTraceConfiguration, TraceReloadHandle},
};
use anyhow::{anyhow, Context as _, Result};
use backoff::{future::retry, ExponentialBackoff};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use clap::Parser;
use deadpool::managed::TimeoutType;
use deadpool_postgres::{Manager, Pool, PoolError, Runtime, Timeouts};
use futures::StreamExt;
use janus_aggregator_core::datastore::{Crypter, Datastore};
use janus_core::time::Clock;
use opentelemetry::metrics::Meter;
use ring::aead::{LessSafeKey, UnboundKey, AES_128_GCM};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{self, Debug, Formatter},
    fs,
    future::Future,
    net::SocketAddr,
    panic,
    path::PathBuf,
    str::FromStr,
    sync::Arc,
    time::Duration,
};
use tokio::sync::oneshot;
use tokio_postgres::NoTls;
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;
use trillium::{Handler, Headers, Info, Init, Status};
use trillium_api::{api, Json, State};
use trillium_head::Head;
use trillium_router::Router;
use trillium_tokio::Stopper;

/// Reads, parses, and returns the config referenced by the given options, or None if no config file
/// path was set.
pub fn read_config<Config: BinaryConfig>(options: &CommonBinaryOptions) -> Result<Config> {
    let config_content = fs::read_to_string(&options.config_file)
        .with_context(|| format!("couldn't read config file {:?}", options.config_file))?;
    let mut config: Config = serde_yaml::from_str(&config_content)
        .with_context(|| format!("couldn't parse config file {:?}", options.config_file))?;

    if let Some(OpenTelemetryTraceConfiguration::Otlp(otlp_config)) = &mut config
        .common_config_mut()
        .logging_config
        .open_telemetry_config
    {
        otlp_config
            .metadata
            .extend(options.otlp_tracing_metadata.iter().cloned());
    }
    if let Some(MetricsExporterConfiguration::Otlp(otlp_config)) =
        &mut config.common_config_mut().metrics_config.exporter
    {
        otlp_config
            .metadata
            .extend(options.otlp_metrics_metadata.iter().cloned());
    }

    Ok(config)
}

/// Connects to a database, given a config. `db_password` is mutually exclusive with the database
/// password specified in the connection URL in `db_config`.
pub async fn database_pool(db_config: &DbConfig, db_password: Option<&str>) -> Result<Pool> {
    let mut database_config = tokio_postgres::Config::from_str(db_config.url.as_str())
        .with_context(|| {
            format!(
                "couldn't parse database connect string: {:?}",
                db_config.url
            )
        })?;
    if database_config.get_password().is_some() && db_password.is_some() {
        return Err(anyhow!(
            "database config & password override are both specified"
        ));
    }
    if let Some(pass) = db_password {
        database_config.password(pass);
    }

    let connection_pool_timeout = Duration::from_secs(db_config.connection_pool_timeouts_secs);

    let conn_mgr = Manager::new(database_config, NoTls);
    let pool = Pool::builder(conn_mgr)
        .runtime(Runtime::Tokio1)
        .timeouts(Timeouts {
            wait: Some(connection_pool_timeout),
            create: Some(connection_pool_timeout),
            recycle: Some(connection_pool_timeout),
        })
        .build()
        .context("failed to create database connection pool")?;

    // Attempt to fetch a connection from the connection pool, to check that the database is
    // accessible. This will either create a new database connection or recycle an existing one, and
    // then return the connection back to the pool.
    //
    // Retrying if we encounter timeouts when creating a connection or connection refused errors
    // (which occur if the Cloud SQL Proxy hasn't started yet and manifest as `PoolError::Backend`)
    let _ = retry(
        ExponentialBackoff {
            initial_interval: Duration::from_secs(1),
            max_interval: connection_pool_timeout,
            multiplier: 2.0,
            max_elapsed_time: Some(connection_pool_timeout),
            ..Default::default()
        },
        || async {
            pool.get().await.map_err(|error| match error {
                PoolError::Timeout(TimeoutType::Create) | PoolError::Backend(_) => {
                    debug!(?error, "transient error connecting to database");
                    backoff::Error::transient(error)
                }
                _ => backoff::Error::permanent(error),
            })
        },
    )
    .await
    .context("couldn't make connection to database")?;

    Ok(pool)
}

/// Connects to a datastore, given a connection pool to the underlying database. `datastore_keys`
/// is a list of AES-128-GCM keys, encoded in base64 with no padding, used to protect secret values
/// stored in the datastore; it must not be empty.
pub async fn datastore<C: Clock>(
    pool: Pool,
    clock: C,
    meter: &Meter,
    datastore_keys: &[String],
    check_schema_version: bool,
) -> Result<Datastore<C>> {
    let datastore_keys = datastore_keys
        .iter()
        .filter(|k| !k.is_empty())
        .map(|k| {
            URL_SAFE_NO_PAD
                .decode(k)
                .context("couldn't base64-decode datastore keys")
                .and_then(|k| {
                    Ok(LessSafeKey::new(
                        UnboundKey::new(&AES_128_GCM, k.as_ref()).map_err(|_| {
                            anyhow!(
                                "couldn't parse datastore keys, expected {} bytes, got {}",
                                AES_128_GCM.key_len(),
                                k.len()
                            )
                        })?,
                    ))
                })
        })
        .collect::<Result<Vec<LessSafeKey>>>()?;
    if datastore_keys.is_empty() {
        return Err(anyhow!("datastore_keys is empty"));
    }

    let datastore = if check_schema_version {
        Datastore::new(pool, Crypter::new(datastore_keys), clock, meter).await?
    } else {
        Datastore::new_without_supported_versions(pool, Crypter::new(datastore_keys), clock, meter)
            .await
    };

    Ok(datastore)
}

/// Options for Janus binaries.
pub trait BinaryOptions: Parser + Debug {
    /// Returns the common options.
    fn common_options(&self) -> &CommonBinaryOptions;
}

#[cfg_attr(doc, doc = "Common options that are used by all Janus binaries.")]
#[derive(Default, Clone, Parser)]
pub struct CommonBinaryOptions {
    /// Path to configuration YAML.
    #[clap(
        long,
        env = "CONFIG_FILE",
        num_args = 1,
        required(true),
        help = "path to configuration file"
    )]
    pub config_file: PathBuf,

    /// Password for the PostgreSQL database connection. If specified, must not be specified in the
    /// connection string.
    #[clap(
        long,
        env = "PGPASSWORD",
        hide_env_values = true,
        help = "PostgreSQL password"
    )]
    pub database_password: Option<String>,

    /// Datastore encryption keys.
    #[clap(
        long,
        env = "DATASTORE_KEYS",
        hide_env_values = true,
        num_args = 1,
        use_value_delimiter = true,
        help = "datastore encryption keys, encoded in url-safe unpadded base64 then comma-separated"
    )]
    pub datastore_keys: Vec<String>,

    /// Additional OTLP/gRPC metadata key/value pairs. (concatenated with those in the logging
    /// configuration sections)
    #[clap(
        long,
        env = "OTLP_TRACING_METADATA",
        value_parser(parse_metadata_entry),
        help = "additional OTLP/gRPC metadata key/value pairs for the tracing exporter",
        value_name = "KEY=value",
        use_value_delimiter = true
    )]
    pub otlp_tracing_metadata: Vec<(String, String)>,

    /// Additional OTLP/gRPC metadata key/value pairs. (concatenated with those in the metrics
    /// configuration sections)
    #[clap(
        long,
        env = "OTLP_METRICS_METADATA",
        value_parser(parse_metadata_entry),
        help = "additional OTLP/gRPC metadata key/value pairs for the metrics exporter",
        value_name = "KEY=value",
        use_value_delimiter = true
    )]
    pub otlp_metrics_metadata: Vec<(String, String)>,
}

impl Debug for CommonBinaryOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Options")
            .field("config_file", &self.config_file)
            .finish()
    }
}

fn parse_metadata_entry(input: &str) -> Result<(String, String)> {
    if let Some(equals) = input.find('=') {
        let (key, rest) = input.split_at(equals);
        let value = &rest[1..];
        Ok((key.to_string(), value.to_string()))
    } else {
        Err(anyhow!(
            "`--otlp-tracing-metadata` and `--otlp-metrics-metadata` must be provided a key and \
             value, joined with an `=`"
        ))
    }
}

/// BinaryContext provides contextual objects related to a Janus binary.
pub struct BinaryContext<C: Clock, Options: BinaryOptions, Config: BinaryConfig> {
    pub clock: C,
    pub options: Options,
    pub config: Config,
    pub datastore: Datastore<C>,
    pub meter: Meter,
    pub stopper: Stopper,
}

pub async fn janus_main<C, Options, Config, F, Fut>(clock: C, f: F) -> anyhow::Result<()>
where
    C: Clock,
    Options: BinaryOptions,
    Config: BinaryConfig,
    F: FnOnce(BinaryContext<C, Options, Config>) -> Fut,
    Fut: Future<Output = anyhow::Result<()>>,
{
    // Parse arguments, then read & parse config.
    let options = Options::parse();
    let config: Config = read_config(options.common_options())?;

    // Install tracing/metrics handlers.
    let (_guards, trace_reload_handle) =
        install_trace_subscriber(&config.common_config().logging_config)
            .context("couldn't install tracing subscriber")?;
    let _metrics_exporter = install_metrics_exporter(&config.common_config().metrics_config)
        .await
        .context("failed to install metrics exporter")?;
    let meter = opentelemetry::global::meter("janus_aggregator");

    // Register signal handler.
    let stopper = Stopper::new();
    setup_signal_handler(stopper.clone()).context("failed to register SIGTERM signal handler")?;

    info!(common_options = ?options.common_options(), ?config, "Starting up");

    // Connect to database.
    let pool = database_pool(
        &config.common_config().database,
        options.common_options().database_password.as_deref(),
    )
    .await
    .context("couldn't create database connection pool")?;
    let datastore = datastore(
        pool,
        clock.clone(),
        &meter,
        &options.common_options().datastore_keys,
        config.common_config().database.check_schema_version,
    )
    .await
    .context("couldn't create datastore")?;

    let health_check_listen_address = config.common_config().health_check_listen_address;
    let zpages_task_handle = tokio::task::spawn(async move {
        zpages_server(health_check_listen_address, trace_reload_handle).await
    });

    let result = f(BinaryContext {
        clock,
        options,
        config,
        datastore,
        meter,
        stopper,
    })
    .await;

    zpages_task_handle.abort();

    result
}

/// A trillium server which serves z-pages, which are utility endpoints for health checks and
/// tracing configuration. It listens on the given address and port. It also takes the reload
/// handle necessary for reloading the tracing_subscriber configuration.
///
/// `/healthz` responds with an empty body and status code 200, which serves as a healthcheck to
/// indicate when Janus has started up.
///
/// `/traceconfigz` responds with the tracing_subscriber configuration, or allows configuring it
/// with a PUT request.
async fn zpages_server(address: SocketAddr, trace_reload_handle: TraceReloadHandle) {
    let handler = zpages_handler(trace_reload_handle);
    trillium_tokio::config()
        .with_port(address.port())
        .with_host(&address.ip().to_string())
        .without_signals()
        .run_async(handler)
        .await;
}

fn zpages_handler(trace_reload_handle: TraceReloadHandle) -> impl Handler {
    (
        Head::new(),
        State(Arc::new(trace_reload_handle)),
        Router::new()
            .get(
                "/healthz",
                |conn: trillium::Conn| async move { conn.ok("") },
            )
            .get("/traceconfigz", api(get_traceconfigz))
            .put("/traceconfigz", api(put_traceconfigz)),
    )
}

async fn get_traceconfigz(
    conn: &mut trillium::Conn,
    State(trace_filter_handler): State<Arc<TraceReloadHandle>>,
) -> Result<Json<TraceconfigzBody>, Status> {
    Ok(Json(TraceconfigzBody {
        filter: trace_filter_handler
            .with_current(|trace_filter| trace_filter.to_string())
            .map_err(|err| {
                conn.set_body(format!("failed to get current filter: {}", err));
                Status::InternalServerError
            })?,
    }))
}

async fn put_traceconfigz(
    conn: &mut trillium::Conn,
    (State(trace_filter_handler), Json(req)): (
        State<Arc<TraceReloadHandle>>,
        Json<TraceconfigzBody>,
    ),
) -> Result<Json<TraceconfigzBody>, Status> {
    trace_filter_handler
        .reload(EnvFilter::new(req.filter))
        .map_err(|err| {
            conn.set_body(format!("failed to update filter: {}", err));
            // Note: reload will accept malformed filter directives, and fall back to the `error`
            // filter. An error indicating malformation is never propagated up, so we can't give
            // a proper HTTP status code. The error state will be revealed to the operator in the
            // response body, however, since it will show the fallback log filter rather than their
            // desired one.
            Status::InternalServerError
        })?;
    Ok(Json(TraceconfigzBody {
        filter: trace_filter_handler
            .with_current(|trace_filter| trace_filter.to_string())
            .map_err(|err| {
                conn.set_body(format!("failed to get current filter: {}", err));
                Status::InternalServerError
            })?,
    }))
}

/// The response and request body used by /traceconfigz for reporting and updating its configuration.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct TraceconfigzBody {
    /// The directive that filters spans and events. This field follows the [`EnvFilter`][1] syntax.
    /// [1]: https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html#directives
    filter: String,
}

/// Register a signal handler for SIGTERM, and stop the [`Stopper`] when a SIGTERM signal is
/// received.
pub fn setup_signal_handler(stopper: Stopper) -> Result<(), std::io::Error> {
    let mut signal_stream = signal_hook_tokio::Signals::new([signal_hook::consts::SIGTERM])?;
    let handle = signal_stream.handle();
    tokio::spawn(async move {
        while let Some(signal) = signal_stream.next().await {
            if signal == signal_hook::consts::SIGTERM {
                stopper.stop();
                handle.close();
                break;
            }
        }
    });
    Ok(())
}

/// Construct a server that listens on the provided [`SocketAddr`] and services requests with
/// `handler`. If the `SocketAddr`'s port is 0, an ephemeral port is used. Returns a `SocketAddr`
/// representing the address and port the server are listening on and a future that can be `await`ed
/// to wait until the server shuts down.
pub async fn setup_server(
    listen_address: SocketAddr,
    response_headers: Headers,
    stopper: Stopper,
    handler: impl Handler,
) -> anyhow::Result<(SocketAddr, impl Future<Output = ()> + 'static)> {
    let (sender, receiver) = oneshot::channel();
    let init = Init::new(|info: Info| async move {
        // Ignore error if the receiver is dropped.
        let _ = sender.send(info.tcp_socket_addr().copied());
    });

    let server_config = trillium_tokio::config()
        .with_port(listen_address.port())
        .with_host(&listen_address.ip().to_string())
        .with_stopper(stopper)
        .without_signals();
    let handler = (init, response_headers, handler);

    let task_handle = tokio::spawn(server_config.run_async(handler));

    let address = receiver
        .await
        .map_err(|err| anyhow!("error waiting for socket address: {err}"))?
        .ok_or_else(|| anyhow!("could not get server's socket address"))?;

    let future = async {
        if let Err(err) = task_handle.await {
            if let Ok(reason) = err.try_into_panic() {
                panic::resume_unwind(reason);
            }
        }
    };

    Ok((address, future))
}

#[cfg(test)]
mod tests {
    use super::CommonBinaryOptions;
    use crate::{
        aggregator::http_handlers::test_util::take_response_body,
        binary_utils::{zpages_handler, TraceconfigzBody},
    };
    use clap::CommandFactory;
    use tracing_subscriber::{layer::SubscriberExt, reload, EnvFilter, Registry};
    use trillium::Status;
    use trillium_testing::prelude::*;

    #[test]
    fn verify_app() {
        CommonBinaryOptions::command().debug_assert()
    }

    #[tokio::test]
    async fn healthz() {
        let (_, filter_handle) = reload::Layer::new(EnvFilter::new("info"));
        let handler = zpages_handler(filter_handle);

        let test_conn = get("/healthz").run_async(&handler).await;
        assert_eq!(test_conn.status(), Some(Status::Ok));
    }

    #[tokio::test]
    async fn traceconfigz() {
        let (filter, filter_handle) = reload::Layer::new(EnvFilter::new("info"));
        let _subscriber = Registry::default().with(filter);

        let handler = zpages_handler(filter_handle);

        let mut test_conn = get("/traceconfigz").run_async(&handler).await;
        assert_eq!(test_conn.status(), Some(Status::Ok));
        assert_eq!(
            serde_json::from_slice::<TraceconfigzBody>(&take_response_body(&mut test_conn).await)
                .unwrap(),
            TraceconfigzBody {
                filter: "info".to_string()
            }
        );

        let req = TraceconfigzBody {
            filter: "debug".to_string(),
        };
        let mut test_conn = dbg!(
            put("/traceconfigz")
                .with_request_body(serde_json::to_vec(&req).unwrap())
                .run_async(&handler)
                .await
        );
        assert_eq!(test_conn.status(), Some(Status::Ok));
        assert_eq!(
            serde_json::from_slice::<TraceconfigzBody>(&take_response_body(&mut test_conn).await)
                .unwrap(),
            req,
        );
    }
}
