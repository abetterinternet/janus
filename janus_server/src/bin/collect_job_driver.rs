use anyhow::Context;
use janus::{message::Duration, time::RealClock};
use janus_server::{
    aggregator::aggregate_share::CollectJobDriver,
    binary_utils::{janus_main, job_driver::JobDriver, BinaryOptions, CommonBinaryOptions},
    config::CollectJobDriverConfig,
};
use std::{fmt::Debug, sync::Arc};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "janus-collect-job-driver",
    about = "Janus collect job driver",
    rename_all = "kebab-case",
    version = env!("CARGO_PKG_VERSION"),
)]
struct Options {
    #[structopt(flatten)]
    common: CommonBinaryOptions,
}

impl BinaryOptions for Options {
    fn common_options(&self) -> &CommonBinaryOptions {
        &self.common
    }
}

const CLIENT_USER_AGENT: &str = concat!(
    env!("CARGO_PKG_NAME"),
    "/",
    env!("CARGO_PKG_VERSION"),
    "/collect_job_driver",
);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    janus_main::<Options, _, _, _, _>(
        RealClock::default(),
        |clock, config: CollectJobDriverConfig, datastore| async move {
            let collect_job_driver = Arc::new(CollectJobDriver::new(
                reqwest::Client::builder()
                    .user_agent(CLIENT_USER_AGENT)
                    .build()
                    .context("couldn't create HTTP client")?,
            ));
            // Start running.
            Arc::new(JobDriver::new(
                Arc::new(datastore),
                clock,
                Duration::from_seconds(config.job_driver_config.min_job_discovery_delay_secs),
                Duration::from_seconds(config.job_driver_config.max_job_discovery_delay_secs),
                config.job_driver_config.max_concurrent_job_workers,
                Duration::from_seconds(config.job_driver_config.worker_lease_duration_secs),
                Duration::from_seconds(
                    config
                        .job_driver_config
                        .worker_lease_clock_skew_allowance_secs,
                ),
                |datastore, lease_duration, maximum_acquire_count| async move {
                    datastore
                        .run_tx(|tx| {
                            Box::pin(async move {
                                tx.acquire_incomplete_collect_jobs(
                                    lease_duration,
                                    maximum_acquire_count,
                                )
                                .await
                            })
                        })
                        .await
                },
                move |datastore, collect_job_lease| {
                    let collect_job_driver = Arc::clone(&collect_job_driver);
                    async move {
                        collect_job_driver
                            .step_collect_job(datastore, collect_job_lease)
                            .await
                    }
                },
            ))
            .run()
            .await;

            Ok(())
        },
    )
    .await
}
