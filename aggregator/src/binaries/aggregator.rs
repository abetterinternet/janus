use crate::{
    aggregator::{self, garbage_collector::GarbageCollector, http_handlers::aggregator_handler},
    binary_utils::{setup_server, BinaryContext, BinaryOptions, CommonBinaryOptions},
    cache::GlobalHpkeKeypairCache,
    config::{BinaryConfig, CommonConfig, TaskprovConfig},
};
use anyhow::{Context, Result};
use clap::Parser;
use derivative::Derivative;
use janus_aggregator_api::{self, aggregator_api_handler};
use janus_aggregator_core::datastore::Datastore;
use janus_core::{auth_tokens::AuthenticationToken, time::RealClock, TokioRuntime};
use opentelemetry::metrics::Meter;
use serde::{de, Deserialize, Deserializer, Serialize};
use std::{
    future::{ready, Future},
    pin::Pin,
};
use std::{iter::Iterator, net::SocketAddr, sync::Arc, time::Duration};
use tokio::{join, sync::watch, time::interval};
use tracing::{error, info};
use trillium::{Handler, Headers};
use trillium_router::router;
use url::Url;

pub async fn main_callback(ctx: BinaryContext<RealClock, Options, Config>) -> Result<()> {
    let (sender, _) = watch::channel(None);
    run_aggregator(ctx, sender).await
}

/// This produces a future that runs the aggregator and provides a [`tokio::sync::watch::Receiver`]
/// that returns the socket address that the aggregator server listens on. This is useful when
/// specifying ephemeral socket addresses.
pub fn make_callback_ephemeral_address(
    ctx: BinaryContext<RealClock, Options, Config>,
) -> (
    impl Future<Output = Result<()>> + Send,
    watch::Receiver<Option<SocketAddr>>,
) {
    let (sender, receiver) = watch::channel(None);
    (run_aggregator(ctx, sender), receiver)
}

async fn run_aggregator(
    ctx: BinaryContext<RealClock, Options, Config>,
    sender: watch::Sender<Option<SocketAddr>>,
) -> Result<()> {
    let BinaryContext {
        clock,
        options,
        mut config,
        datastore,
        meter,
        stopper,
    } = ctx;

    let datastore = Arc::new(datastore);
    let response_headers = config
        .response_headers()
        .context("failed to parse response headers")?;

    let mut handlers = (
        aggregator_handler(
            Arc::clone(&datastore),
            clock,
            TokioRuntime,
            &meter,
            config.aggregator_config(),
        )
        .await?,
        None,
    );

    let garbage_collector_future = {
        let datastore = Arc::clone(&datastore);
        let gc_config = config.garbage_collection.take();
        let meter = meter.clone();
        async move {
            if let Some(gc_config) = gc_config {
                let gc = GarbageCollector::new(
                    datastore,
                    &meter,
                    gc_config.report_limit,
                    gc_config.aggregation_limit,
                    gc_config.collection_limit,
                    gc_config.tasks_per_tx,
                    gc_config.concurrent_tx_limit,
                );
                let mut interval = interval(Duration::from_secs(gc_config.gc_frequency_s));
                loop {
                    interval.tick().await;
                    if let Err(err) = gc.run().await {
                        error!(?err, "GC error");
                    }
                }
            }
        }
    };

    let aggregator_api_future: Pin<Box<dyn Future<Output = ()> + Send + 'static>> =
        match build_aggregator_api_handler(&options, &config, &datastore, &meter)? {
            Some((handler, config)) => {
                if let Some(listen_address) = config.listen_address {
                    // Bind the requested address and spawn a future that serves the aggregator API
                    // on it, which we'll `tokio::join!` on below
                    let (aggregator_api_bound_address, aggregator_api_server) = setup_server(
                        listen_address,
                        response_headers.clone(),
                        stopper.clone(),
                        handler,
                    )
                    .await
                    .context("failed to create aggregator API server")?;

                    info!(?aggregator_api_bound_address, "Running aggregator API");

                    Box::pin(aggregator_api_server)
                } else if let Some(path_prefix) = &config.path_prefix {
                    // Create a Trillium handler under the requested path prefix, which we'll add to
                    // the DAP API handler in the setup_server call below
                    info!(
                        aggregator_bound_address = ?config.listen_address,
                        path_prefix,
                        "Serving aggregator API relative to DAP API"
                    );
                    // Append wildcard so that this handler will match anything under the prefix
                    let path_prefix = format!("{path_prefix}/*");
                    handlers.1 = Some(router().all(path_prefix, handler));
                    Box::pin(ready(()))
                } else {
                    unreachable!("the configuration should not have deserialized to this state")
                }
            }
            None => Box::pin(ready(())),
        };

    let (aggregator_bound_address, aggregator_server) = setup_server(
        config.listen_address,
        response_headers,
        stopper.clone(),
        handlers,
    )
    .await
    .context("failed to create aggregator server")?;
    sender.send_replace(Some(aggregator_bound_address));

    info!(?aggregator_bound_address, "Running aggregator");

    join!(
        aggregator_server,
        garbage_collector_future,
        aggregator_api_future
    );
    Ok(())
}

fn build_aggregator_api_handler<'a>(
    options: &Options,
    config: &'a Config,
    datastore: &Arc<Datastore<RealClock>>,
    meter: &Meter,
) -> Result<Option<(impl Handler, &'a AggregatorApi)>> {
    let Some(aggregator_api) = &config.aggregator_api else {
        return Ok(None);
    };
    let aggregator_api_auth_tokens = options
        .aggregator_api_auth_tokens
        .iter()
        .filter(|token| !token.is_empty())
        .map(|token| {
            // Aggregator API auth tokens are always bearer tokens
            AuthenticationToken::new_bearer_token_from_string(token)
                .context("invalid aggregator API auth token")
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Some((
        aggregator_api_handler(
            Arc::clone(datastore),
            janus_aggregator_api::Config {
                auth_tokens: aggregator_api_auth_tokens,
                public_dap_url: aggregator_api.public_dap_url.clone(),
            },
            meter,
        ),
        aggregator_api,
    )))
}

#[derive(Debug, Parser)]
#[clap(
    name = "janus-aggregator",
    about = "DAP aggregator server",
    rename_all = "kebab-case",
    version = env!("CARGO_PKG_VERSION"),
)]
pub struct Options {
    #[clap(flatten)]
    pub common: CommonBinaryOptions,

    /// Aggregator API authentication tokens
    ///
    /// API tokens are encoded in unpadded url-safe base64, then comma-separated.
    #[clap(
        long,
        env = "AGGREGATOR_API_AUTH_TOKENS",
        hide_env_values = true,
        num_args = 0..=1,
        use_value_delimiter = true,
    )]
    pub aggregator_api_auth_tokens: Vec<String>,
}

impl BinaryOptions for Options {
    fn common_options(&self) -> &CommonBinaryOptions {
        &self.common
    }
}

/// A name-value HTTP header pair, that appears in configuration objects.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeaderEntry {
    name: String,
    value: String,
}

/// Options for serving the aggregator API.
#[derive(Clone, Derivative, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[derivative(Debug)]
pub struct AggregatorApi {
    /// Address on which this server should listen for connections to the Janus aggregator API
    /// and serve its API endpoints, independently from the address on which the DAP API is
    /// served. This is mutually exclusive with `path_prefix`.
    pub listen_address: Option<SocketAddr>,
    /// The Janus aggregator API will be served on the same address as the DAP API, but relative
    /// to the provided prefix. e.g., if `path_prefix` is `aggregator-api`, then the DAP API's
    /// uploads endpoint would be `{listen-address}/tasks/{task-id}/reports`, while task IDs
    /// could be obtained from the aggregator API at `{listen-address}/aggregator-api/task_ids`.
    /// This is mutually exclusive with `listen_address`.
    pub path_prefix: Option<String>,
    /// Resource location at which the DAP service managed by this aggregator api can be found
    /// on the public internet. Required.
    #[derivative(Debug(format_with = "std::fmt::Display::fmt"))]
    pub public_dap_url: Url,
}

fn deserialize_aggregator_api<'de, D>(deserializer: D) -> Result<Option<AggregatorApi>, D::Error>
where
    D: Deserializer<'de>,
{
    let aggregator_api: Option<AggregatorApi> = Deserialize::deserialize(deserializer)?;
    if let Some(ref aggregator_api) = aggregator_api {
        match (aggregator_api.listen_address, &aggregator_api.path_prefix) {
            (None, None) => {
                return Err(de::Error::custom(
                    "one of listen_address or path_prefix must be provided",
                ))
            }
            (Some(_), Some(_)) => {
                return Err(de::Error::custom(
                    "only one of listen_address and path_prefix must be specified",
                ))
            }
            _ => {}
        }
    }
    Ok(aggregator_api)
}

/// Non-secret configuration options for a Janus aggregator, deserialized from YAML.
///
/// # Examples
///
/// Configuration serving the aggregator API on its own port, distinct from the DAP API:
///
/// ```
/// # use janus_aggregator::binaries::aggregator::Config;
/// let yaml_config = r#"
/// ---
/// listen_address: "0.0.0.0:8080"
/// aggregator_api:
///   listen_address: "0.0.0.0:8081"
///   public_dap_url: "https://dap.example.test"
/// response_headers:
/// - name: "Example"
///   value: "header value"
/// database:
///   url: "postgres://postgres:postgres@localhost:5432/postgres"
/// logging_config: # logging_config is optional
///   force_json_output: true
/// max_upload_batch_size: 100
/// max_upload_batch_write_delay_ms: 250
/// batch_aggregation_shard_count: 32
/// "#;
///
/// let _decoded: Config = serde_yaml::from_str(yaml_config).unwrap();
/// ```
///
/// Configuration serving the aggregator API relative to the DAP API:
///
/// ```
/// # use janus_aggregator::binaries::aggregator::Config;
/// let yaml_config = r#"
/// ---
/// listen_address: "0.0.0.0:8080"
/// aggregator_api:
///   path_prefix: "aggregator-api"
///   public_dap_url: "https://dap.example.test"
/// response_headers:
/// - name: "Example"
///   value: "header value"
/// database:
///   url: "postgres://postgres:postgres@localhost:5432/postgres"
/// logging_config: # logging_config is optional
///   force_json_output: true
/// max_upload_batch_size: 100
/// max_upload_batch_write_delay_ms: 250
/// batch_aggregation_shard_count: 32
/// taskprov_config:
///   enabled: false
/// "#;
///
/// let _decoded: Config = serde_yaml::from_str(yaml_config).unwrap();
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    #[serde(flatten)]
    pub common_config: CommonConfig,

    #[serde(default)]
    pub taskprov_config: TaskprovConfig,

    #[serde(default)]
    pub garbage_collection: Option<GarbageCollectorConfig>,

    /// Address on which this server should listen for connections to the DAP aggregator API and
    /// serve its API endpoints.
    pub listen_address: SocketAddr,

    /// How to serve the Janus aggregator API. If not set, the aggregator API is not served.
    #[serde(default, deserialize_with = "deserialize_aggregator_api")]
    pub aggregator_api: Option<AggregatorApi>,

    /// Additional headers that will be added to all responses.
    #[serde(default)]
    pub response_headers: Vec<HeaderEntry>,

    /// Defines the maximum size of a batch of uploaded reports which will be written in a single
    /// transaction.
    pub max_upload_batch_size: usize,

    /// Defines the maximum delay in milliseconds before writing a batch of uploaded reports, even
    /// if it has not yet reached `max_batch_upload_size`.
    pub max_upload_batch_write_delay_ms: u64,

    /// Defines the number of shards to break each batch aggregation into. Increasing this value
    /// will reduce the amount of database contention during helper aggregation, while increasing
    /// the cost of collection.
    pub batch_aggregation_shard_count: u64,

    /// Defines the number of shards to break report counters into. Increasing this value will
    /// reduce the amount of database contention during report uploads, while increasing the cost
    /// of getting task metrics.
    #[serde(default = "default_task_counter_shard_count")]
    pub task_counter_shard_count: u64,

    /// Defines how often to refresh the global HPKE configs cache in milliseconds. This affects how
    /// often an aggregator becomes aware of key state changes. If unspecified, default is defined
    /// by [`GlobalHpkeKeypairCache::DEFAULT_REFRESH_INTERVAL`]. You shouldn't normally have to
    /// specify this.
    #[serde(default)]
    pub global_hpke_configs_refresh_interval: Option<u64>,
}

fn default_task_counter_shard_count() -> u64 {
    32
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GarbageCollectorConfig {
    /// How frequently garbage collection is run, in seconds.
    pub gc_frequency_s: u64,

    /// The limit to the number of client report artifacts deleted for a single task by a single run
    /// of the garbage collector.
    pub report_limit: u64,

    /// The limit to the number of aggregation jobs, and related aggregation artifacts, deleted for
    /// a single task by a single run of the garbage collector.
    pub aggregation_limit: u64,

    /// The limit to the number of batches, and related collection artifacts, deleted for a single
    /// task by a single run of the garbage collector.
    pub collection_limit: u64,

    /// The maximum number of tasks to process together for GC in a single database transaction.
    /// Defaults to a single task per database transaction.
    #[serde(default = "default_tasks_per_tx")]
    pub tasks_per_tx: usize,

    /// The maximum number of concurrent database transactions to open at once while processing GC.
    /// Leaving this unset means there is no maximum.
    pub concurrent_tx_limit: Option<usize>,
}

fn default_tasks_per_tx() -> usize {
    1
}

impl Config {
    fn response_headers(&self) -> Result<Headers> {
        self.response_headers
            .iter()
            .map(|entry| {
                Ok((
                    entry.name.as_str().to_owned(),
                    entry.value.as_str().to_owned(),
                ))
            })
            .collect()
    }

    fn aggregator_config(&self) -> aggregator::Config {
        aggregator::Config {
            max_upload_batch_size: self.max_upload_batch_size,
            max_upload_batch_write_delay: Duration::from_millis(
                self.max_upload_batch_write_delay_ms,
            ),
            batch_aggregation_shard_count: self.batch_aggregation_shard_count,
            task_counter_shard_count: self.task_counter_shard_count,
            taskprov_config: self.taskprov_config,
            global_hpke_configs_refresh_interval: match self.global_hpke_configs_refresh_interval {
                Some(duration) => Duration::from_millis(duration),
                None => GlobalHpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
            },
        }
    }
}

impl BinaryConfig for Config {
    fn common_config(&self) -> &CommonConfig {
        &self.common_config
    }

    fn common_config_mut(&mut self) -> &mut CommonConfig {
        &mut self.common_config
    }
}

#[cfg(test)]
mod tests {
    use super::{AggregatorApi, Config, GarbageCollectorConfig, HeaderEntry, Options};
    use crate::{
        aggregator,
        config::{
            test_util::{generate_db_config, generate_metrics_config, generate_trace_config},
            BinaryConfig, CommonConfig, TaskprovConfig,
        },
        metrics::{MetricsExporterConfiguration, OtlpExporterConfiguration},
        trace::{
            OpenTelemetryTraceConfiguration, OtlpTraceConfiguration, TokioConsoleConfiguration,
        },
    };
    use assert_matches::assert_matches;
    use clap::CommandFactory;
    use janus_core::test_util::roundtrip_encoding;
    use std::{
        net::{IpAddr, Ipv4Addr, SocketAddr},
        time::Duration,
    };

    #[test]
    fn verify_app() {
        Options::command().debug_assert()
    }

    #[rstest::rstest]
    #[case::listen_address(AggregatorApi {
        listen_address: Some(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 8081))),
        path_prefix: None,
        public_dap_url: "https://dap.url".parse().unwrap()
    })]
    #[case::path_prefix(AggregatorApi {
        listen_address: None,
        path_prefix: Some("prefix".to_string()),
        public_dap_url: "https://dap.url".parse().unwrap()
    })]
    #[test]
    fn roundtrip_config(#[case] aggregator_api: AggregatorApi) {
        roundtrip_encoding(Config {
            listen_address: SocketAddr::from((Ipv4Addr::UNSPECIFIED, 8080)),
            garbage_collection: Some(GarbageCollectorConfig {
                gc_frequency_s: 60,
                report_limit: 25,
                aggregation_limit: 50,
                collection_limit: 75,
                tasks_per_tx: 15,
                concurrent_tx_limit: Some(23),
            }),
            aggregator_api: Some(aggregator_api),
            common_config: CommonConfig {
                database: generate_db_config(),
                logging_config: generate_trace_config(),
                metrics_config: generate_metrics_config(),
                health_check_listen_address: SocketAddr::from((Ipv4Addr::UNSPECIFIED, 8080)),
            },
            response_headers: Vec::from([HeaderEntry {
                name: "name".to_owned(),
                value: "value".to_owned(),
            }]),
            max_upload_batch_size: 100,
            max_upload_batch_write_delay_ms: 250,
            batch_aggregation_shard_count: 32,
            task_counter_shard_count: 64,
            taskprov_config: TaskprovConfig::default(),
            global_hpke_configs_refresh_interval: None,
        })
    }

    #[test]
    fn config_no_aggregator_api() {
        assert_eq!(
            serde_yaml::from_str::<Config>(
                r#"---
    listen_address: "0.0.0.0:8080"
    database:
        url: "postgres://postgres:postgres@localhost:5432/postgres"
        connection_pool_timeouts_secs: 60
    max_upload_batch_size: 100
    max_upload_batch_write_delay_ms: 250
    batch_aggregation_shard_count: 32
    "#
            )
            .unwrap()
            .aggregator_api,
            None
        );
    }

    #[test]
    fn config_garbage_collection() {
        assert_eq!(
            serde_yaml::from_str::<Config>(
                r#"---
    listen_address: "0.0.0.0:8080"
    database:
        url: "postgres://postgres:postgres@localhost:5432/postgres"
        connection_pool_timeouts_secs: 60
    max_upload_batch_size: 100
    max_upload_batch_write_delay_ms: 250
    batch_aggregation_shard_count: 32
    garbage_collection:
        gc_frequency_s: 60
        report_limit: 25
        aggregation_limit: 50
        collection_limit: 75
    "#
            )
            .unwrap()
            .garbage_collection,
            Some(GarbageCollectorConfig {
                gc_frequency_s: 60,
                report_limit: 25,
                aggregation_limit: 50,
                collection_limit: 75,
                tasks_per_tx: 1,
                concurrent_tx_limit: None,
            }),
        );

        assert_eq!(
            serde_yaml::from_str::<Config>(
                r#"---
    listen_address: "0.0.0.0:8080"
    database:
        url: "postgres://postgres:postgres@localhost:5432/postgres"
        connection_pool_timeouts_secs: 60
    max_upload_batch_size: 100
    max_upload_batch_write_delay_ms: 250
    batch_aggregation_shard_count: 32
    garbage_collection:
        gc_frequency_s: 60
        report_limit: 25
        aggregation_limit: 50
        collection_limit: 75
        tasks_per_tx: 15
        concurrent_tx_limit: 23
    "#
            )
            .unwrap()
            .garbage_collection,
            Some(GarbageCollectorConfig {
                gc_frequency_s: 60,
                report_limit: 25,
                aggregation_limit: 50,
                collection_limit: 75,
                tasks_per_tx: 15,
                concurrent_tx_limit: Some(23),
            }),
        );
    }

    #[test]
    fn config_taskprov() {
        assert_eq!(
            serde_yaml::from_str::<Config>(
                r#"---
    listen_address: "0.0.0.0:8080"
    database:
        url: "postgres://postgres:postgres@localhost:5432/postgres"
        connection_pool_timeouts_secs: 60
    max_upload_batch_size: 100
    max_upload_batch_write_delay_ms: 250
    batch_aggregation_shard_count: 32
    taskprov_config:
        enabled: true
    "#
            )
            .unwrap()
            .taskprov_config,
            TaskprovConfig {
                enabled: true,
                ignore_unknown_differential_privacy_mechanism: false
            },
        );
    }

    #[test]
    fn config_aggregator_api_listen_address() {
        assert_eq!(
            serde_yaml::from_str::<Config>(
                r#"---
    listen_address: "0.0.0.0:8080"
    database:
        url: "postgres://postgres:postgres@localhost:5432/postgres"
        connection_pool_timeouts_secs: 60
    max_upload_batch_size: 100
    max_upload_batch_write_delay_ms: 250
    batch_aggregation_shard_count: 32
    aggregator_api:
        listen_address: "0.0.0.0:8081"
        public_dap_url: "https://dap.url"
    "#
            )
            .unwrap()
            .aggregator_api,
            Some(AggregatorApi {
                listen_address: Some(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 8081))),
                path_prefix: None,
                public_dap_url: "https://dap.url".parse().unwrap()
            })
        );
    }

    #[test]
    fn config_aggregator_api_path_prefix() {
        assert_eq!(
            serde_yaml::from_str::<Config>(
                r#"---
    listen_address: "0.0.0.0:8080"
    database:
        url: "postgres://postgres:postgres@localhost:5432/postgres"
        connection_pool_timeouts_secs: 60
    max_upload_batch_size: 100
    max_upload_batch_write_delay_ms: 250
    batch_aggregation_shard_count: 32
    aggregator_api:
        path_prefix: "aggregator-api"
        public_dap_url: "https://dap.url"
    "#
            )
            .unwrap()
            .aggregator_api,
            Some(AggregatorApi {
                listen_address: None,
                path_prefix: Some("aggregator-api".to_string()),
                public_dap_url: "https://dap.url".parse().unwrap()
            })
        );
    }

    #[test]
    fn config_aggregator_mutually_exclusive() {
        assert_matches!(
            serde_yaml::from_str::<Config>(
                r#"---
    listen_address: "0.0.0.0:8080"
    database:
        url: "postgres://postgres:postgres@localhost:5432/postgres"
        connection_pool_timeouts_secs: 60
    max_upload_batch_size: 100
    max_upload_batch_write_delay_ms: 250
    batch_aggregation_shard_count: 32
    aggregator_api:
        path_prefix: "aggregator-api"
        listen_address: "0.0.0.0:8081"
        public_dap_url: "https://dap.url"
    "#
            ),
            Err(_)
        );
    }

    #[test]
    fn config_aggregator_api_missing_parameters() {
        assert_matches!(
            serde_yaml::from_str::<Config>(
                r#"---
    listen_address: "0.0.0.0:8080"
    database:
        url: "postgres://postgres:postgres@localhost:5432/postgres"
        connection_pool_timeouts_secs: 60
    max_upload_batch_size: 100
    max_upload_batch_write_delay_ms: 250
    batch_aggregation_shard_count: 32
    aggregator_api:
        public_dap_url: "https://dap.url"
    "#
            ),
            Err(_)
        );
    }

    /// Check that configuration fragments in the README and other documentation can be parsed
    /// correctly.
    #[test]
    fn documentation_config_examples() {
        assert_eq!(
            serde_yaml::from_str::<Config>(
                r#"---
    listen_address: "0.0.0.0:8080"
    health_check_listen_address: "0.0.0.0:8080"
    database:
        url: "postgres://postgres:postgres@localhost:5432/postgres"
        connection_pool_timeouts_secs: 60
    logging_config:
        tokio_console_config:
            enabled: true
            listen_address: 127.0.0.1:6669
    max_upload_batch_size: 100
    max_upload_batch_write_delay_ms: 250
    batch_aggregation_shard_count: 32
    "#
            )
            .unwrap()
            .common_config()
            .logging_config
            .tokio_console_config,
            TokioConsoleConfiguration {
                enabled: true,
                listen_address: Some(SocketAddr::new(
                    IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
                    6669,
                )),
            },
        );

        assert_eq!(
            serde_yaml::from_str::<Config>(
                r#"---
    listen_address: "0.0.0.0:8080"
    health_check_listen_address: "0.0.0.0:8080"
    database:
        url: "postgres://postgres:postgres@localhost:5432/postgres"
        connection_pool_timeouts_secs: 60
    logging_config:
        tokio_console_config:
            enabled: true
            listen_address: 127.0.0.1:6669
    max_upload_batch_size: 100
    max_upload_batch_write_delay_ms: 250
    batch_aggregation_shard_count: 32
    "#
            )
            .unwrap()
            .aggregator_config(),
            aggregator::Config {
                max_upload_batch_size: 100,
                max_upload_batch_write_delay: Duration::from_millis(250),
                batch_aggregation_shard_count: 32,
                taskprov_config: TaskprovConfig::default(),
                ..Default::default()
            }
        );

        assert_eq!(
            serde_yaml::from_str::<Config>(
                r#"---
    listen_address: "0.0.0.0:8080"
    health_check_listen_address: "0.0.0.0:8080"
    database:
        url: "postgres://postgres:postgres@localhost:5432/postgres"
        connection_pool_timeouts_secs: 60
    logging_config:
        open_telemetry_config:
            otlp:
                endpoint: "http://localhost:4317"
    max_upload_batch_size: 100
    max_upload_batch_write_delay_ms: 250
    batch_aggregation_shard_count: 32
    "#
            )
            .unwrap()
            .common_config()
            .logging_config
            .open_telemetry_config,
            Some(OpenTelemetryTraceConfiguration::Otlp(
                OtlpTraceConfiguration {
                    endpoint: "http://localhost:4317".to_string(),
                }
            )),
        );

        assert_eq!(
            serde_yaml::from_str::<Config>(
                r#"---
    listen_address: "0.0.0.0:8080"
    health_check_listen_address: "0.0.0.0:8080"
    database:
        url: "postgres://postgres:postgres@localhost:5432/postgres"
        connection_pool_timeouts_secs: 60
    logging_config:
        open_telemetry_config:
            otlp:
                endpoint: "https://api.honeycomb.io:443"
    max_upload_batch_size: 100
    max_upload_batch_write_delay_ms: 250
    batch_aggregation_shard_count: 32
    "#
            )
            .unwrap()
            .common_config()
            .logging_config
            .open_telemetry_config,
            Some(OpenTelemetryTraceConfiguration::Otlp(
                OtlpTraceConfiguration {
                    endpoint: "https://api.honeycomb.io:443".to_string(),
                },
            )),
        );

        assert_eq!(
            serde_yaml::from_str::<Config>(
                r#"---
    listen_address: "0.0.0.0:8080"
    health_check_listen_address: "0.0.0.0:8080"
    database:
        url: "postgres://postgres:postgres@localhost:5432/postgres"
        connection_pool_timeouts_secs: 60
    metrics_config:
        exporter:
            prometheus:
                host: 0.0.0.0
                port: 9464
    max_upload_batch_size: 100
    max_upload_batch_write_delay_ms: 250
    batch_aggregation_shard_count: 32
    "#
            )
            .unwrap()
            .common_config()
            .metrics_config
            .exporter,
            Some(MetricsExporterConfiguration::Prometheus {
                host: Some("0.0.0.0".to_string()),
                port: Some(9464),
            }),
        );

        assert_eq!(
            serde_yaml::from_str::<Config>(
                r#"---
    listen_address: "0.0.0.0:8080"
    health_check_listen_address: "0.0.0.0:8080"
    database:
        url: "postgres://postgres:postgres@localhost:5432/postgres"
        connection_pool_timeouts_secs: 60
    metrics_config:
        exporter:
            otlp:
                endpoint: "https://api.honeycomb.io:443"
    max_upload_batch_size: 100
    max_upload_batch_write_delay_ms: 250
    batch_aggregation_shard_count: 32
    "#
            )
            .unwrap()
            .common_config()
            .metrics_config
            .exporter,
            Some(MetricsExporterConfiguration::Otlp(
                OtlpExporterConfiguration {
                    endpoint: "https://api.honeycomb.io:443".to_string(),
                },
            )),
        );

        serde_yaml::from_str::<Config>(include_str!(
            "../../../docs/samples/basic_config/aggregator.yaml"
        ))
        .unwrap();
        serde_yaml::from_str::<Config>(include_str!(
            "../../../docs/samples/advanced_config/aggregator.yaml"
        ))
        .unwrap();
    }
}
