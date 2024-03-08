use crate::{
    aggregator::collection_job_driver::CollectionJobDriver,
    binary_utils::{job_driver::JobDriver, BinaryContext, BinaryOptions, CommonBinaryOptions},
    config::{BinaryConfig, CommonConfig, JobDriverConfig},
};
use anyhow::{Context, Result};
use clap::Parser;
use janus_core::{time::RealClock, TokioRuntime};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc, time::Duration};

pub async fn main_callback(ctx: BinaryContext<RealClock, Options, Config>) -> Result<()> {
    const CLIENT_USER_AGENT: &str = concat!(
        env!("CARGO_PKG_NAME"),
        "/",
        env!("CARGO_PKG_VERSION"),
        "/collection_job_driver"
    );

    let datastore = Arc::new(ctx.datastore);
    let collection_job_driver = Arc::new(CollectionJobDriver::new(
        reqwest::Client::builder()
            .user_agent(CLIENT_USER_AGENT)
            .timeout(Duration::from_secs(
                ctx.config.job_driver_config.http_request_timeout_secs,
            ))
            .connect_timeout(Duration::from_secs(
                ctx.config
                    .job_driver_config
                    .http_request_connection_timeout_secs,
            ))
            .build()
            .context("couldn't create HTTP client")?,
        ctx.config.job_driver_config.retry_config(),
        &ctx.meter,
        ctx.config.batch_aggregation_shard_count,
        Duration::from_secs(ctx.config.min_collection_job_retry_delay_secs),
    ));
    let lease_duration =
        Duration::from_secs(ctx.config.job_driver_config.worker_lease_duration_secs);

    // Start running.
    let job_driver = Arc::new(JobDriver::new(
        ctx.clock,
        TokioRuntime,
        ctx.meter,
        ctx.stopper,
        Duration::from_secs(ctx.config.job_driver_config.job_discovery_interval_secs),
        ctx.config.job_driver_config.max_concurrent_job_workers,
        Duration::from_secs(
            ctx.config
                .job_driver_config
                .worker_lease_clock_skew_allowance_secs,
        ),
        collection_job_driver
            .make_incomplete_job_acquirer_callback(Arc::clone(&datastore), lease_duration),
        collection_job_driver.make_job_stepper_callback(
            Arc::clone(&datastore),
            ctx.config.job_driver_config.maximum_attempts_before_failure,
        ),
    )?);
    job_driver.run().await;

    Ok(())
}

#[derive(Debug, Parser)]
#[clap(
    name = "janus-collect-job-driver",
    about = "Janus collection job driver",
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

/// Non-secret configuration options for Janus collection job driver jobs.
///
/// # Examples
///
/// ```
/// # use janus_aggregator::binaries::collection_job_driver::Config;
/// let yaml_config = r#"
/// ---
/// database:
///   url: "postgres://postgres:postgres@localhost:5432/postgres"
/// logging_config: # logging_config is optional
///   force_json_output: true
/// job_discovery_interval_secs: 10
/// max_concurrent_job_workers: 10
/// worker_lease_duration_secs: 600
/// worker_lease_clock_skew_allowance_secs: 60
/// maximum_attempts_before_failure: 5
/// retry_initial_interval_millis: 1000
/// retry_max_interval_millis: 30000
/// retry_max_elapsed_time_millis: 300000
/// batch_aggregation_shard_count: 32
/// min_collection_job_retry_delay_secs: 600
/// "#;
///
/// let _decoded: Config = serde_yaml::from_str(yaml_config).unwrap();
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    #[serde(flatten)]
    pub common_config: CommonConfig,
    #[serde(flatten)]
    pub job_driver_config: JobDriverConfig,

    /// Defines the number of shards to break each batch aggregation into. Increasing this value
    /// will reduce the amount of database contention during leader aggregation, while increasing
    /// the cost of collection.
    pub batch_aggregation_shard_count: u64,

    /// The minimum duration to wait, in seconds, before retrying a collection job that has been
    /// stepped but was not ready yet because not all included reports had finished aggregation.
    pub min_collection_job_retry_delay_secs: u64,
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
    use super::{Config, Options};
    use crate::config::{
        test_util::{generate_db_config, generate_metrics_config, generate_trace_config},
        CommonConfig, JobDriverConfig,
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
            },
            job_driver_config: JobDriverConfig {
                job_discovery_interval_secs: 10,
                max_concurrent_job_workers: 10,
                worker_lease_duration_secs: 600,
                worker_lease_clock_skew_allowance_secs: 60,
                maximum_attempts_before_failure: 5,
                http_request_timeout_secs: 10,
                http_request_connection_timeout_secs: 30,
                retry_initial_interval_millis: 1000,
                retry_max_interval_millis: 30_000,
                retry_max_elapsed_time_millis: 300_000,
            },
            batch_aggregation_shard_count: 32,
            min_collection_job_retry_delay_secs: 600,
        })
    }

    #[test]
    fn documentation_config_examples() {
        serde_yaml::from_str::<Config>(include_str!(
            "../../../docs/samples/basic_config/collection_job_driver.yaml"
        ))
        .unwrap();
        serde_yaml::from_str::<Config>(include_str!(
            "../../../docs/samples/advanced_config/collection_job_driver.yaml"
        ))
        .unwrap();
    }
}
