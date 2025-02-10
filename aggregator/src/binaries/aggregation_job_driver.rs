use crate::{
    aggregator::aggregation_job_driver::AggregationJobDriver,
    binary_utils::{job_driver::JobDriver, BinaryContext, BinaryOptions, CommonBinaryOptions},
    cache::HpkeKeypairCache,
    config::{BinaryConfig, CommonConfig, JobDriverConfig, TaskprovConfig},
};
use anyhow::{Context, Result};
use clap::Parser;
use janus_core::{time::RealClock, TokioRuntime};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc, time::Duration};
use tracing::info;

pub async fn main_callback(ctx: BinaryContext<RealClock, Options, Config>) -> Result<()> {
    const CLIENT_USER_AGENT: &str = concat!(
        env!("CARGO_PKG_NAME"),
        "/",
        env!("CARGO_PKG_VERSION"),
        "/aggregation_job_driver",
    );

    let hpke_configs_refresh_interval = match ctx.config.hpke_configs_refresh_interval {
        Some(duration) => Duration::from_millis(duration),
        None => HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
    };

    let datastore = Arc::new(ctx.datastore);
    let aggregation_job_driver = Arc::new(AggregationJobDriver::new(
        reqwest::Client::builder()
            .user_agent(CLIENT_USER_AGENT)
            .timeout(Duration::from_secs(
                ctx.config.job_driver_config.http_request_timeout_s,
            ))
            .connect_timeout(Duration::from_secs(
                ctx.config
                    .job_driver_config
                    .http_request_connection_timeout_s,
            ))
            .build()
            .context("couldn't create HTTP client")?,
        ctx.config.job_driver_config.retry_config(),
        &ctx.meter,
        ctx.config.batch_aggregation_shard_count,
        ctx.config.task_counter_shard_count,
        hpke_configs_refresh_interval,
        Duration::from_millis(ctx.config.default_async_poll_interval),
    ));
    let lease_duration = Duration::from_secs(ctx.config.job_driver_config.worker_lease_duration_s);

    // Start running.
    let job_driver = Arc::new(JobDriver::new(
        ctx.clock,
        TokioRuntime,
        ctx.meter,
        ctx.stopper,
        Duration::from_secs(ctx.config.job_driver_config.job_discovery_interval_s),
        ctx.config.job_driver_config.max_concurrent_job_workers,
        Duration::from_secs(
            ctx.config
                .job_driver_config
                .worker_lease_clock_skew_allowance_s,
        ),
        aggregation_job_driver
            .make_incomplete_job_acquirer_callback(Arc::clone(&datastore), lease_duration),
        aggregation_job_driver.make_job_stepper_callback(
            Arc::clone(&datastore),
            ctx.config.job_driver_config.maximum_attempts_before_failure,
        ),
    )?);

    info!("Running aggregation job driver");
    job_driver.run().await;

    Ok(())
}

#[derive(Debug, Parser)]
#[clap(
    name = "janus-aggregation-job-driver",
    about = "Janus aggregation job driver",
    rename_all = "kebab-case",
    version = env!("CARGO_PKG_VERSION"),
)]
pub struct Options {
    #[clap(flatten)]
    pub common: CommonBinaryOptions,
}

impl BinaryOptions for Options {
    fn common_options(&self) -> &CommonBinaryOptions {
        &self.common
    }
}

/// Non-secret configuration options for Janus Aggregation Job Driver jobs.
///
/// # Examples
///
/// ```
/// # use janus_aggregator::binaries::aggregation_job_driver::Config;
/// let yaml_config = r#"
/// ---
/// database:
///   url: "postgres://postgres:postgres@localhost:5432/postgres"
/// logging_config: # logging_config is optional
///   force_json_output: true
/// max_concurrent_job_workers: 10
/// job_discovery_interval_s: 10
/// worker_lease_duration_s: 600
/// worker_lease_clock_skew_allowance_s: 60
/// maximum_attempts_before_failure: 5
/// retry_initial_interval_ms: 1000
/// retry_max_interval_ms: 30000
/// retry_max_elapsed_time_ms: 300000
/// batch_aggregation_shard_count: 32
/// task_counter_shard_count: 32
/// taskprov_config:
///   enabled: false
/// "#;
///
/// let _decoded: Config = serde_yaml::from_str(yaml_config).unwrap();
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(flatten)]
    pub common_config: CommonConfig,
    #[serde(flatten)]
    pub job_driver_config: JobDriverConfig,
    #[serde(default)]
    pub taskprov_config: TaskprovConfig,

    /// Defines the number of shards to break each batch aggregation into. Increasing this value
    /// will reduce the amount of database contention during leader aggregation, while increasing
    /// the cost of collection.
    pub batch_aggregation_shard_count: u64,

    /// Defines the number of shards to break report & aggregation metric counters into. Increasing
    /// this value will reduce the amount of database contention during report uploads &
    /// aggregations, while increasing the cost of getting task metrics.
    #[serde(default = "default_task_counter_shard_count")]
    pub task_counter_shard_count: u64,

    /// Defines how often to refresh the HPKE configs cache in milliseconds. This affects how often
    /// an aggregator becomes aware of HPKE key state changes. If unspecified, default is defined by
    /// [`HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL`]. You shouldn't normally have to specify this.
    #[serde(default)]
    pub hpke_configs_refresh_interval: Option<u64>,

    /// Defines how frequently outstanding asynchronous aggregation jobs where this aggregator is
    /// the Leader might be polled if the Helper does not send a Retry-After header, in
    /// milliseconds. (If the Helper does send a Retry-After header, it will be respected.) If
    /// unspecified, the default is one minute.
    #[serde(default = "default_default_async_poll_interval")]
    pub default_async_poll_interval: u64,
}

impl BinaryConfig for Config {
    fn common_config(&self) -> &CommonConfig {
        &self.common_config
    }

    fn common_config_mut(&mut self) -> &mut CommonConfig {
        &mut self.common_config
    }
}

fn default_task_counter_shard_count() -> u64 {
    32
}

fn default_default_async_poll_interval() -> u64 {
    60_000
}

#[cfg(test)]
mod tests {
    use super::{Config, Options};
    use crate::config::{
        default_max_transaction_retries,
        test_util::{generate_db_config, generate_metrics_config, generate_trace_config},
        CommonConfig, JobDriverConfig, TaskprovConfig,
    };
    use clap::CommandFactory;
    use janus_core::test_util::roundtrip_encoding;
    use std::net::{Ipv4Addr, SocketAddr};

    #[test]
    fn verify_app() {
        Options::command().debug_assert()
    }

    #[test]
    fn roundtrip_config() {
        roundtrip_encoding(Config {
            common_config: CommonConfig {
                database: generate_db_config(),
                logging_config: generate_trace_config(),
                metrics_config: generate_metrics_config(),
                health_check_listen_address: SocketAddr::from((Ipv4Addr::UNSPECIFIED, 8080)),
                max_transaction_retries: default_max_transaction_retries(),
                thread_pool_stack_size: None,
            },
            job_driver_config: JobDriverConfig {
                job_discovery_interval_s: 10,
                max_concurrent_job_workers: 10,
                worker_lease_duration_s: 600,
                worker_lease_clock_skew_allowance_s: 60,
                maximum_attempts_before_failure: 5,
                http_request_timeout_s: 10,
                http_request_connection_timeout_s: 30,
                retry_initial_interval_ms: 1000,
                retry_max_interval_ms: 30_000,
                retry_max_elapsed_time_ms: 300_000,
            },
            batch_aggregation_shard_count: 32,
            task_counter_shard_count: 64,
            hpke_configs_refresh_interval: Some(180000),
            default_async_poll_interval: 5_000,
            taskprov_config: TaskprovConfig::default(),
        })
    }

    #[test]
    fn documentation_config_examples() {
        serde_yaml::from_str::<Config>(include_str!(
            "../../../docs/samples/basic_config/aggregation_job_driver.yaml"
        ))
        .unwrap();
        serde_yaml::from_str::<Config>(include_str!(
            "../../../docs/samples/advanced_config/aggregation_job_driver.yaml"
        ))
        .unwrap();
    }
}
