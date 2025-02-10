//! Common functionality for DAP aggregators.

pub use crate::aggregator::error::Error;
use crate::{
    aggregator::{
        aggregate_share::compute_aggregate_share,
        aggregation_job_init::compute_helper_aggregate_init,
        aggregation_job_writer::{
            AggregationJobWriter, AggregationJobWriterMetrics, InitialWrite,
            ReportAggregationUpdate as _, UpdateWrite, WritableReportAggregation,
        },
        batch_mode::{CollectableBatchMode, UploadableBatchMode},
        error::{
            handle_ping_pong_error, BatchMismatch, OptOutReason, ReportRejection,
            ReportRejectionReason,
        },
        report_writer::{ReportWriteBatcher, WritableReport},
    },
    cache::{
        HpkeKeypairCache, PeerAggregatorCache, TaskAggregatorCache,
        TASK_AGGREGATOR_CACHE_DEFAULT_CAPACITY, TASK_AGGREGATOR_CACHE_DEFAULT_TTL,
    },
    config::TaskprovConfig,
    diagnostic::AggregationJobInitForbiddenMutationEvent,
    metrics::{
        aggregate_step_failure_counter, aggregated_report_share_dimension_histogram,
        report_aggregation_success_counter,
    },
};
use aggregation_job_continue::compute_helper_aggregate_continue;
use aws_lc_rs::{
    digest::{digest, SHA256},
    rand::SystemRandom,
    signature::{EcdsaKeyPair, Signature},
};
use backoff::{backoff::Backoff, Notify};
use bytes::Bytes;
#[cfg(feature = "fpvec_bounded_l2")]
use fixed::{
    types::extra::{U15, U31},
    FixedI16, FixedI32,
};
use futures::future::try_join_all;
use http::{header::CONTENT_TYPE, Method};
use itertools::iproduct;
use janus_aggregator_core::{
    batch_mode::AccumulableBatchMode,
    datastore::{
        self,
        models::{
            AggregateShareJob, AggregationJob, AggregationJobState, BatchAggregation,
            BatchAggregationState, CollectionJob, CollectionJobState, LeaderStoredReport,
            ReportAggregation, ReportAggregationState, TaskAggregationCounter,
        },
        Datastore, Error as DatastoreError, Transaction,
    },
    task::{self, AggregationMode, AggregatorTask, BatchMode},
    taskprov::PeerAggregator,
};
#[cfg(feature = "fpvec_bounded_l2")]
use janus_core::vdaf::Prio3FixedPointBoundedL2VecSumBitSize;
use janus_core::{
    auth_tokens::AuthenticationToken,
    hpke::{self, HpkeApplicationInfo, Label},
    retries::{retry_http_request_notify, HttpResponse},
    time::{Clock, DurationExt, IntervalExt, TimeExt},
    vdaf::{
        new_prio3_sum_vec_field64_multiproof_hmacsha256_aes128,
        Prio3SumVecField64MultiproofHmacSha256Aes128, VdafInstance,
    },
    Runtime,
};
use janus_messages::{
    batch_mode::{LeaderSelected, TimeInterval},
    taskprov::TaskConfig,
    AggregateShare, AggregateShareAad, AggregateShareReq, AggregationJobContinueReq,
    AggregationJobId, AggregationJobInitializeReq, AggregationJobResp, AggregationJobStep,
    BatchSelector, CollectionJobId, CollectionJobReq, CollectionJobResp, Duration, HpkeConfig,
    HpkeConfigList, InputShareAad, Interval, PartialBatchSelector, PlaintextInputShare,
    PrepareResp, Report, ReportError, ReportIdChecksum, Role, TaskId,
};
use opentelemetry::{
    metrics::{Counter, Histogram, Meter},
    KeyValue,
};
#[cfg(feature = "fpvec_bounded_l2")]
use prio::vdaf::prio3::Prio3FixedPointBoundedL2VecSum;
#[cfg(feature = "test-util")]
use prio::vdaf::{dummy, PrepareTransition, VdafError};
use prio::{
    codec::{Decode, Encode, ParameterizedDecode},
    dp::DifferentialPrivacyStrategy,
    field::Field64,
    flp::gadgets::{Mul, ParallelSum},
    vdaf::{
        self,
        prio3::{Prio3, Prio3Count, Prio3Histogram, Prio3Sum, Prio3SumVec},
    },
};
use rand::{thread_rng, Rng};
use reqwest::Client;
use std::{
    borrow::Cow,
    collections::HashSet,
    fmt::Debug,
    hash::Hash,
    panic,
    path::PathBuf,
    sync::{Arc, Mutex as SyncMutex},
    time::{Duration as StdDuration, Instant},
};
use tokio::try_join;
use tracing::{debug, error, info, warn, Level};
use url::Url;

pub mod aggregate_share;
pub mod aggregation_job_continue;
pub mod aggregation_job_creator;
pub mod aggregation_job_driver;
pub mod aggregation_job_init;
pub mod aggregation_job_writer;
pub mod batch_creator;
pub mod batch_mode;
pub mod collection_job_driver;
#[cfg(test)]
mod collection_job_tests;
mod error;
pub mod garbage_collector;
pub mod http_handlers;
pub mod key_rotator;
pub mod problem_details;
mod queue;
pub mod report_writer;
#[cfg(test)]
mod taskprov_tests;
#[cfg(feature = "test-util")]
#[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
pub mod test_util;
#[cfg(test)]
mod upload_tests;

/// Aggregator implements a DAP aggregator.
pub struct Aggregator<C: Clock> {
    /// Datastore used for durable storage.
    datastore: Arc<Datastore<C>>,
    /// Clock used to sample time.
    clock: C,
    /// Configuration used for this aggregator.
    cfg: Config,
    /// Cache of task aggregators.
    task_aggregators: TaskAggregatorCache<C>,
    /// Metrics.
    metrics: AggregatorMetrics,

    /// Cache of HPKE keypairs and configs.
    hpke_keypairs: Arc<HpkeKeypairCache>,

    /// Cache of taskprov peer aggregators.
    peer_aggregators: PeerAggregatorCache,
}

#[derive(Clone)]
pub struct AggregatorMetrics {
    /// Counter tracking the number of failed decryptions while handling the
    /// `tasks/{task-id}/reports` endpoint.
    upload_decrypt_failure_counter: Counter<u64>,
    /// Counter tracking the number of failed message decodes while handling the
    /// `tasks/{task-id}/reports` endpoint.
    upload_decode_failure_counter: Counter<u64>,
    /// Counter tracking the number of successfully-aggregated reports.
    report_aggregation_success_counter: Counter<u64>,
    /// Counters tracking the number of failures to step client reports through the aggregation
    /// process.
    aggregate_step_failure_counter: Counter<u64>,
    /// Histogram tracking the VDAF type and dimension of successfully-aggregated reports.
    aggregated_report_share_dimension_histogram: Histogram<u64>,
}

impl AggregatorMetrics {
    fn for_aggregation_job_writer(&self) -> AggregationJobWriterMetrics {
        AggregationJobWriterMetrics {
            report_aggregation_success_counter: self.report_aggregation_success_counter.clone(),
            aggregate_step_failure_counter: self.aggregate_step_failure_counter.clone(),
            aggregated_report_share_dimension_histogram: self
                .aggregated_report_share_dimension_histogram
                .clone(),
        }
    }
}

/// Config represents a configuration for an Aggregator.
#[derive(Debug)]
pub struct Config {
    /// Defines the maximum size of a batch of uploaded reports which will be written in a single
    /// transaction.
    pub max_upload_batch_size: usize,

    /// Defines the maximum delay before writing a batch of uploaded reports, even if it has not yet
    /// reached `max_batch_upload_size`. This is the maximum delay added to the
    /// `tasks/{task-id}/reports` endpoint due to write-batching.
    pub max_upload_batch_write_delay: StdDuration,

    /// Defines the number of shards to break each batch aggregation into. Increasing this value
    /// will reduce the amount of database contention during helper aggregation, while increasing
    /// the cost of collection.
    pub batch_aggregation_shard_count: u64,

    /// Defines the number of shards to break report counters into. Increasing this value will
    /// reduce the amount of database contention during report uploads, while increasing the cost
    /// of getting task metrics.
    pub task_counter_shard_count: u64,

    /// Defines how often to refresh the HPKE configs cache. This affects how often an aggregator
    /// becomes aware of key state changes.
    pub hpke_configs_refresh_interval: StdDuration,

    /// Defines how long tasks should be cached for. This affects how often an aggregator becomes aware
    /// of task parameter changes.
    pub task_cache_ttl: StdDuration,

    /// Defines how many tasks can be cached at once. This affects how much memory the aggregator may
    /// consume for caching tasks.
    pub task_cache_capacity: u64,

    /// The key used to sign HPKE configurations.
    pub hpke_config_signing_key: Option<EcdsaKeyPair>,

    /// Configuration for the taskprov extension.
    pub taskprov_config: TaskprovConfig,

    /// If set, forbidden mutations of resources (e.g., re-using the same aggregation job ID but
    /// with different reports in it) will be logged to files under this path when detected.
    ///
    /// This option is not stable, and not subject to Janus' typical API/config stability promises.
    pub log_forbidden_mutations: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_upload_batch_size: 1,
            max_upload_batch_write_delay: StdDuration::ZERO,
            batch_aggregation_shard_count: 1,
            task_counter_shard_count: 32,
            hpke_configs_refresh_interval: HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
            hpke_config_signing_key: None,
            taskprov_config: TaskprovConfig::default(),
            task_cache_ttl: TASK_AGGREGATOR_CACHE_DEFAULT_TTL,
            task_cache_capacity: TASK_AGGREGATOR_CACHE_DEFAULT_CAPACITY,
            log_forbidden_mutations: None,
        }
    }
}

impl<C: Clock> Aggregator<C> {
    /// Creates a new [`Aggregator`].
    ///
    /// # Errors
    ///
    /// Fails on general datastore errors.
    ///
    /// If there is not at least one HPKE keypair in the database in the [`HpkeKeyState::Active`]
    /// state then this function will fail.
    async fn new<R: Runtime + Send + Sync + 'static>(
        datastore: Arc<Datastore<C>>,
        clock: C,
        runtime: R,
        meter: &Meter,
        cfg: Config,
    ) -> Result<Self, Error> {
        let task_aggregators = TaskAggregatorCache::new(
            Arc::clone(&datastore),
            ReportWriteBatcher::new(
                Arc::clone(&datastore),
                runtime,
                cfg.task_counter_shard_count,
                cfg.max_upload_batch_size,
                cfg.max_upload_batch_write_delay,
            ),
            // If we're in taskprov mode, we can never cache None entries for tasks, since
            // aggregators could insert tasks at any time and expect them to be available across all
            // aggregator replicas.
            !cfg.taskprov_config.enabled,
            cfg.task_cache_capacity,
            cfg.task_cache_ttl,
        );

        let upload_decrypt_failure_counter = meter
            .u64_counter("janus_upload_decrypt_failures")
            .with_description(
                "Number of decryption failures in the tasks/{task-id}/reports endpoint.",
            )
            .with_unit("{error}")
            .build();
        upload_decrypt_failure_counter.add(0, &[]);

        let upload_decode_failure_counter = meter
            .u64_counter("janus_upload_decode_failures")
            .with_description(
                "Number of message decode failures in the tasks/{task-id}/reports endpoint.",
            )
            .with_unit("{error}")
            .build();
        upload_decode_failure_counter.add(0, &[]);

        let report_aggregation_success_counter = report_aggregation_success_counter(meter);
        let aggregate_step_failure_counter = aggregate_step_failure_counter(meter);
        let aggregated_report_share_dimension_histogram =
            aggregated_report_share_dimension_histogram(meter);

        let hpke_keypairs = Arc::new(
            HpkeKeypairCache::new(Arc::clone(&datastore), cfg.hpke_configs_refresh_interval)
                .await?,
        );

        let peer_aggregators = PeerAggregatorCache::new(&datastore).await?;

        Ok(Self {
            datastore,
            clock,
            cfg,
            task_aggregators,
            metrics: AggregatorMetrics {
                upload_decrypt_failure_counter,
                upload_decode_failure_counter,
                report_aggregation_success_counter,
                aggregate_step_failure_counter,
                aggregated_report_share_dimension_histogram,
            },
            hpke_keypairs,
            peer_aggregators,
        })
    }

    /// Handles an HPKE config request.
    ///
    /// The returned value is the encoded HPKE config list (i.e. the response body), and an optional
    /// signature over the body if the aggregator is configured to sign HPKE config responses.
    async fn handle_hpke_config(&self) -> Result<(Vec<u8>, Option<Signature>), Error> {
        // Retrieve HPKE keys & encode the HPKE config list.
        let encoded_hpke_config_list = HpkeConfigList::new(self.hpke_keypairs.configs().to_vec())
            .get_encoded()
            .map_err(Error::MessageEncode)?;

        // If configured to do so, sign the encoded HPKE config list.
        let signature = self
            .cfg
            .hpke_config_signing_key
            .as_ref()
            .map(|key| key.sign(&SystemRandom::new(), &encoded_hpke_config_list))
            .transpose()
            .map_err(|_| Error::Internal("HPKE config list signing error".to_string()))?;

        Ok((encoded_hpke_config_list, signature))
    }

    async fn handle_upload(&self, task_id: &TaskId, report_bytes: &[u8]) -> Result<(), Arc<Error>> {
        let report =
            Report::get_decoded(report_bytes).map_err(|err| Arc::new(Error::MessageDecode(err)))?;

        let task_aggregator = self
            .task_aggregators
            .get(task_id)
            .await?
            .ok_or(Error::UnrecognizedTask(*task_id))?;
        if task_aggregator.task.role() != &Role::Leader {
            return Err(Arc::new(Error::UnrecognizedTask(*task_id)));
        }
        task_aggregator
            .handle_upload(&self.clock, &self.hpke_keypairs, &self.metrics, report)
            .await
    }

    async fn handle_aggregate_init(
        &self,
        task_id: &TaskId,
        aggregation_job_id: &AggregationJobId,
        req_bytes: &[u8],
        auth_token: Option<AuthenticationToken>,
        taskprov_task_config: Option<&TaskConfig>,
    ) -> Result<AggregationJobResp, Error> {
        let task_aggregator = match self.task_aggregators.get(task_id).await? {
            Some(task_aggregator) => {
                if task_aggregator.task.role() != &Role::Helper {
                    return Err(Error::UnrecognizedTask(*task_id));
                }
                if self.cfg.taskprov_config.enabled && taskprov_task_config.is_some() {
                    self.taskprov_authorize_request(
                        &Role::Leader,
                        task_id,
                        taskprov_task_config.unwrap(),
                        auth_token.as_ref(),
                    )
                    .await?;
                } else if !task_aggregator
                    .task
                    .check_aggregator_auth_token(auth_token.as_ref())
                {
                    return Err(Error::UnauthorizedRequest(*task_id));
                }
                task_aggregator
            }
            None if self.cfg.taskprov_config.enabled && taskprov_task_config.is_some() => {
                self.taskprov_opt_in(
                    &Role::Leader,
                    task_id,
                    taskprov_task_config.unwrap(),
                    auth_token.as_ref(),
                )
                .await?;

                // Retry fetching the aggregator, since the last function would have just inserted
                // its task.
                debug!(
                    ?task_id,
                    "taskprov: opt-in successful, retrying task acquisition"
                );
                self.task_aggregators.get(task_id).await?.ok_or_else(|| {
                    Error::Internal("unexpectedly failed to create task".to_string())
                })?
            }
            _ => {
                return Err(Error::UnrecognizedTask(*task_id));
            }
        };

        task_aggregator
            .handle_aggregate_init(
                Arc::clone(&self.datastore),
                Arc::clone(&self.hpke_keypairs),
                &self.metrics,
                self.cfg.batch_aggregation_shard_count,
                self.cfg.task_counter_shard_count,
                aggregation_job_id,
                taskprov_task_config.is_some(),
                self.cfg.log_forbidden_mutations.clone(),
                req_bytes,
            )
            .await
    }

    async fn handle_aggregate_continue(
        &self,
        task_id: &TaskId,
        aggregation_job_id: &AggregationJobId,
        req_bytes: &[u8],
        auth_token: Option<AuthenticationToken>,
        taskprov_task_config: Option<&TaskConfig>,
    ) -> Result<AggregationJobResp, Error> {
        let task_aggregator = self
            .task_aggregators
            .get(task_id)
            .await?
            .ok_or(Error::UnrecognizedTask(*task_id))?;
        if task_aggregator.task.role() != &Role::Helper {
            return Err(Error::UnrecognizedTask(*task_id));
        }

        if self.cfg.taskprov_config.enabled && taskprov_task_config.is_some() {
            self.taskprov_authorize_request(
                &Role::Leader,
                task_id,
                taskprov_task_config.unwrap(),
                auth_token.as_ref(),
            )
            .await?;
        } else if !task_aggregator
            .task
            .check_aggregator_auth_token(auth_token.as_ref())
        {
            return Err(Error::UnauthorizedRequest(*task_id));
        }

        let req =
            AggregationJobContinueReq::get_decoded(req_bytes).map_err(Error::MessageDecode)?;
        // unwrap safety: SHA-256 computed by ring should always be 32 bytes
        let request_hash = digest(&SHA256, req_bytes).as_ref().try_into().unwrap();

        task_aggregator
            .handle_aggregate_continue(
                Arc::clone(&self.datastore),
                &self.metrics,
                self.cfg.batch_aggregation_shard_count,
                self.cfg.task_counter_shard_count,
                aggregation_job_id,
                req,
                request_hash,
            )
            .await
    }

    async fn handle_aggregate_get(
        &self,
        task_id: &TaskId,
        aggregation_job_id: &AggregationJobId,
        auth_token: Option<AuthenticationToken>,
        taskprov_task_config: Option<&TaskConfig>,
        step: AggregationJobStep,
    ) -> Result<AggregationJobResp, Error> {
        let task_aggregator = self
            .task_aggregators
            .get(task_id)
            .await?
            .ok_or(Error::UnrecognizedTask(*task_id))?;
        if task_aggregator.task.role() != &Role::Helper
            || task_aggregator.task.aggregation_mode() != Some(&AggregationMode::Asynchronous)
        {
            return Err(Error::UnrecognizedTask(*task_id));
        }

        if self.cfg.taskprov_config.enabled && taskprov_task_config.is_some() {
            self.taskprov_authorize_request(
                &Role::Leader,
                task_id,
                taskprov_task_config.unwrap(),
                auth_token.as_ref(),
            )
            .await?;
        } else if !task_aggregator
            .task
            .check_aggregator_auth_token(auth_token.as_ref())
        {
            return Err(Error::UnauthorizedRequest(*task_id));
        }

        task_aggregator
            .handle_aggregate_get(Arc::clone(&self.datastore), aggregation_job_id, step)
            .await
    }

    async fn handle_aggregate_delete(
        &self,
        task_id: &TaskId,
        aggregation_job_id: &AggregationJobId,
        auth_token: Option<AuthenticationToken>,
        taskprov_task_config: Option<&TaskConfig>,
    ) -> Result<(), Error> {
        let task_aggregator = self
            .task_aggregators
            .get(task_id)
            .await?
            .ok_or(Error::UnrecognizedTask(*task_id))?;
        if task_aggregator.task.role() != &Role::Helper {
            return Err(Error::UnrecognizedTask(*task_id));
        }

        if self.cfg.taskprov_config.enabled && taskprov_task_config.is_some() {
            self.taskprov_authorize_request(
                &Role::Leader,
                task_id,
                taskprov_task_config.unwrap(),
                auth_token.as_ref(),
            )
            .await?;
        } else if !task_aggregator
            .task
            .check_aggregator_auth_token(auth_token.as_ref())
        {
            return Err(Error::UnauthorizedRequest(*task_id));
        }

        task_aggregator
            .handle_aggregate_delete(&self.datastore, aggregation_job_id)
            .await
    }

    /// Handle a collection job creation request. Only supported by the leader. `req_bytes` is an
    /// encoded [`CollectionJobReq`]. Returns an encoded [`CollectionJobResp`] on success.
    async fn handle_create_collection_job(
        &self,
        task_id: &TaskId,
        collection_job_id: &CollectionJobId,
        req_bytes: &[u8],
        auth_token: Option<AuthenticationToken>,
    ) -> Result<Vec<u8>, Error> {
        let task_aggregator = self
            .task_aggregators
            .get(task_id)
            .await?
            .ok_or(Error::UnrecognizedTask(*task_id))?;
        if task_aggregator.task.role() != &Role::Leader {
            return Err(Error::UnrecognizedTask(*task_id));
        }
        if !task_aggregator
            .task
            .check_collector_auth_token(auth_token.as_ref())
        {
            return Err(Error::UnauthorizedRequest(*task_id));
        }

        task_aggregator
            .handle_create_collection_job(&self.datastore, collection_job_id, req_bytes)
            .await
    }

    /// Handle a GET request for a collection job. `collection_job_id` is the unique identifier for the
    /// collection job parsed out of the request URI. Returns an encoded [`Collection`] if the collect
    /// job has been run to completion, `None` if the collection job has not yet run, or an error
    /// otherwise.
    async fn handle_get_collection_job(
        &self,
        task_id: &TaskId,
        collection_job_id: &CollectionJobId,
        auth_token: Option<AuthenticationToken>,
    ) -> Result<Vec<u8>, Error> {
        let task_aggregator = self
            .task_aggregators
            .get(task_id)
            .await?
            .ok_or(Error::UnrecognizedTask(*task_id))?;
        if task_aggregator.task.role() != &Role::Leader {
            return Err(Error::UnrecognizedTask(*task_id));
        }
        if !task_aggregator
            .task
            .check_collector_auth_token(auth_token.as_ref())
        {
            return Err(Error::UnauthorizedRequest(*task_id));
        }

        task_aggregator
            .handle_get_collection_job(&self.datastore, collection_job_id)
            .await
    }

    /// Handle a DELETE request for a collection job.
    async fn handle_delete_collection_job(
        &self,
        task_id: &TaskId,
        collection_job_id: &CollectionJobId,
        auth_token: Option<AuthenticationToken>,
    ) -> Result<(), Error> {
        let task_aggregator = self
            .task_aggregators
            .get(task_id)
            .await?
            .ok_or(Error::UnrecognizedTask(*task_id))?;
        if task_aggregator.task.role() != &Role::Leader {
            return Err(Error::UnrecognizedTask(*task_id));
        }
        if !task_aggregator
            .task
            .check_collector_auth_token(auth_token.as_ref())
        {
            return Err(Error::UnauthorizedRequest(*task_id));
        }

        task_aggregator
            .handle_delete_collection_job(&self.datastore, collection_job_id)
            .await?;

        Ok(())
    }

    /// Handle an aggregate share request. Only supported by the helper. `req_bytes` is an encoded
    /// [`AggregateShareReq`]. Returns an [`AggregateShare`].
    async fn handle_aggregate_share(
        &self,
        task_id: &TaskId,
        req_bytes: &[u8],
        auth_token: Option<AuthenticationToken>,
        taskprov_task_config: Option<&TaskConfig>,
    ) -> Result<AggregateShare, Error> {
        let task_aggregator = self
            .task_aggregators
            .get(task_id)
            .await?
            .ok_or(Error::UnrecognizedTask(*task_id))?;
        if task_aggregator.task.role() != &Role::Helper {
            return Err(Error::UnrecognizedTask(*task_id));
        }

        // Authorize the request and retrieve the collector's HPKE config. If this is a taskprov task, we
        // have to use the peer aggregator's collector config rather than the main task.
        let collector_hpke_config =
            if self.cfg.taskprov_config.enabled && taskprov_task_config.is_some() {
                let (peer_aggregator, _, _) = self
                    .taskprov_authorize_request(
                        &Role::Leader,
                        task_id,
                        taskprov_task_config.unwrap(),
                        auth_token.as_ref(),
                    )
                    .await?;

                peer_aggregator.collector_hpke_config()
            } else {
                if !task_aggregator
                    .task
                    .check_aggregator_auth_token(auth_token.as_ref())
                {
                    return Err(Error::UnauthorizedRequest(*task_id));
                }

                task_aggregator
                    .task
                    .collector_hpke_config()
                    .ok_or_else(|| {
                        Error::Internal("task is missing collector_hpke_config".to_string())
                    })?
            };

        task_aggregator
            .handle_aggregate_share(
                &self.datastore,
                &self.clock,
                self.cfg.batch_aggregation_shard_count,
                req_bytes,
                collector_hpke_config,
            )
            .await
    }

    /// Opts in or out of a taskprov task.
    #[tracing::instrument(skip(self, aggregator_auth_token), err(level = Level::DEBUG))]
    async fn taskprov_opt_in(
        &self,
        peer_role: &Role,
        task_id: &TaskId,
        task_config: &TaskConfig,
        aggregator_auth_token: Option<&AuthenticationToken>,
    ) -> Result<(), Error> {
        let (peer_aggregator, leader_url, _) = self
            .taskprov_authorize_request(peer_role, task_id, task_config, aggregator_auth_token)
            .await?;

        // TODO(#1647): Check whether task config parameters are acceptable for privacy and
        // availability of the system.

        let vdaf_instance = task_config.vdaf_config().try_into().map_err(|err: &str| {
            Error::InvalidTask(*task_id, OptOutReason::InvalidParameter(err.to_string()))
        })?;

        let vdaf_verify_key = peer_aggregator.derive_vdaf_verify_key(task_id, &vdaf_instance);

        let task_end = task_config.task_start().add(task_config.task_duration())?;

        let task = Arc::new(
            AggregatorTask::new(
                *task_id,
                leader_url,
                BatchMode::try_from(*task_config.batch_mode())?,
                vdaf_instance,
                vdaf_verify_key,
                Some(*task_config.task_start()),
                Some(task_end),
                peer_aggregator.report_expiry_age().cloned(),
                u64::from(*task_config.min_batch_size()),
                *task_config.time_precision(),
                *peer_aggregator.tolerable_clock_skew(),
                task::AggregatorTaskParameters::TaskprovHelper {
                    aggregation_mode: peer_aggregator.aggregation_mode().copied().ok_or_else(
                        || {
                            Error::Internal(
                                "peer aggregator has no aggregation mode specified".to_string(),
                            )
                        },
                    )?,
                },
            )
            .map_err(|err| Error::InvalidTask(*task_id, OptOutReason::TaskParameters(err)))?
            .with_taskprov_task_info(task_config.task_info().to_vec()),
        );
        self.datastore
            .run_tx("taskprov_put_task", |tx| {
                let task = Arc::clone(&task);
                Box::pin(async move { tx.put_aggregator_task(&task).await })
            })
            .await
            .or_else(|error| -> Result<(), Error> {
                match error {
                    // If the task is already in the datastore, then some other request or aggregator
                    // replica beat us to inserting it. They _should_ have inserted all the same parameters
                    // as we would have, so we can proceed as normal.
                    DatastoreError::MutationTargetAlreadyExists => {
                        warn!(
                            ?task_id,
                            ?error,
                            "taskprov: went to insert task into db, but it already exists"
                        );
                        Ok(())
                    }
                    error => Err(error.into()),
                }
            })?;

        info!(?task, ?peer_aggregator, "taskprov: opted into new task");
        Ok(())
    }

    /// Validate and authorize a taskprov request. Returns values necessary for determining whether
    /// we can opt into the task. This function might return an opt-out error for conditions that
    /// are relevant for all DAP workflows (e.g. task end).
    #[tracing::instrument(skip(self, aggregator_auth_token), err(level = Level::DEBUG))]
    async fn taskprov_authorize_request(
        &self,
        peer_role: &Role,
        task_id: &TaskId,
        task_config: &TaskConfig,
        aggregator_auth_token: Option<&AuthenticationToken>,
    ) -> Result<(&PeerAggregator, Url, Url), Error> {
        let peer_aggregator_url = match peer_role {
            Role::Leader => task_config.leader_aggregator_endpoint(),
            Role::Helper => task_config.helper_aggregator_endpoint(),
            _ => panic!("Unexpected role {peer_role}"),
        }
        .try_into()?;

        let peer_aggregator = self
            .peer_aggregators
            .get(&peer_aggregator_url, peer_role)
            .ok_or(Error::InvalidTask(
                *task_id,
                OptOutReason::NoSuchPeer(*peer_role),
            ))?;

        if !aggregator_auth_token
            .map(|t| peer_aggregator.check_aggregator_auth_token(t))
            .unwrap_or(false)
        {
            return Err(Error::UnauthorizedRequest(*task_id));
        }

        let task_end = task_config.task_start().add(task_config.task_duration())?;
        if self.clock.now() > task_end {
            return Err(Error::InvalidTask(*task_id, OptOutReason::TaskEnded));
        }

        debug!(
            ?task_id,
            ?task_config,
            ?peer_aggregator,
            "taskprov: authorized request"
        );
        Ok((
            peer_aggregator,
            task_config.leader_aggregator_endpoint().try_into()?,
            task_config.helper_aggregator_endpoint().try_into()?,
        ))
    }

    #[cfg(feature = "test-util")]
    pub async fn refresh_caches(&self) -> Result<(), Error> {
        self.hpke_keypairs.refresh(&self.datastore).await
    }
}

/// TaskAggregator provides aggregation functionality for a single task.
// TODO(#1307): refactor Aggregator to perform indepedent batched operations (e.g. report handling
// in Aggregate requests) using a parallelized library like Rayon.
#[derive(Debug)]
pub struct TaskAggregator<C: Clock> {
    /// The task being aggregated.
    pub(crate) task: Arc<AggregatorTask>,
    /// VDAF-specific operations.
    vdaf_ops: VdafOps,
    /// Report writer, with support for batching.
    report_writer: Arc<ReportWriteBatcher<C>>,
}

impl<C: Clock> TaskAggregator<C> {
    /// Create a new aggregator. `report_recipient` is used to decrypt reports received by this
    /// aggregator.
    pub fn new(
        task: AggregatorTask,
        report_writer: Arc<ReportWriteBatcher<C>>,
    ) -> Result<Self, Error> {
        let vdaf_ops = match task.vdaf() {
            VdafInstance::Prio3Count => {
                let vdaf = Prio3::new_count(2)?;
                VdafOps::Prio3Count(Arc::new(vdaf))
            }

            VdafInstance::Prio3Sum { max_measurement } => {
                let vdaf = Prio3::new_sum(2, *max_measurement)?;
                VdafOps::Prio3Sum(Arc::new(vdaf))
            }

            VdafInstance::Prio3SumVec {
                bits,
                length,
                chunk_length,
                dp_strategy,
            } => {
                let vdaf = Prio3::new_sum_vec(2, *bits, *length, *chunk_length)?;
                VdafOps::Prio3SumVec(
                    Arc::new(vdaf),
                    vdaf_ops_strategies::Prio3SumVec::from_vdaf_dp_strategy(dp_strategy.clone()),
                )
            }

            VdafInstance::Prio3SumVecField64MultiproofHmacSha256Aes128 {
                proofs,
                bits,
                length,
                chunk_length,
                dp_strategy,
            } => {
                let vdaf = new_prio3_sum_vec_field64_multiproof_hmacsha256_aes128::<
                    ParallelSum<Field64, Mul<Field64>>,
                >(*proofs, *bits, *length, *chunk_length)?;
                VdafOps::Prio3SumVecField64MultiproofHmacSha256Aes128(
                    Arc::new(vdaf),
                    vdaf_ops_strategies::Prio3SumVec::from_vdaf_dp_strategy(dp_strategy.clone()),
                )
            }

            VdafInstance::Prio3Histogram {
                length,
                chunk_length,
                dp_strategy,
            } => {
                let vdaf = Prio3::new_histogram(2, *length, *chunk_length)?;
                VdafOps::Prio3Histogram(
                    Arc::new(vdaf),
                    vdaf_ops_strategies::Prio3Histogram::from_vdaf_dp_strategy(dp_strategy.clone()),
                )
            }

            #[cfg(feature = "fpvec_bounded_l2")]
            VdafInstance::Prio3FixedPointBoundedL2VecSum {
                bitsize,
                dp_strategy,
                length,
            } => match bitsize {
                Prio3FixedPointBoundedL2VecSumBitSize::BitSize16 => {
                    let vdaf: Prio3FixedPointBoundedL2VecSum<FixedI16<U15>> =
                        Prio3::new_fixedpoint_boundedl2_vec_sum(2, *length)?;
                    VdafOps::Prio3FixedPoint16BitBoundedL2VecSum(
                        Arc::new(vdaf),
                        vdaf_ops_strategies::Prio3FixedPointBoundedL2VecSum::from_vdaf_dp_strategy(
                            dp_strategy.clone(),
                        ),
                    )
                }
                Prio3FixedPointBoundedL2VecSumBitSize::BitSize32 => {
                    let vdaf: Prio3FixedPointBoundedL2VecSum<FixedI32<U31>> =
                        Prio3::new_fixedpoint_boundedl2_vec_sum(2, *length)?;
                    VdafOps::Prio3FixedPoint32BitBoundedL2VecSum(
                        Arc::new(vdaf),
                        vdaf_ops_strategies::Prio3FixedPointBoundedL2VecSum::from_vdaf_dp_strategy(
                            dp_strategy.clone(),
                        ),
                    )
                }
            },

            #[cfg(feature = "test-util")]
            VdafInstance::Fake { rounds } => VdafOps::Fake(Arc::new(dummy::Vdaf::new(*rounds))),

            #[cfg(feature = "test-util")]
            VdafInstance::FakeFailsPrepInit => VdafOps::Fake(Arc::new(
                dummy::Vdaf::new(1).with_prep_init_fn(|_| -> Result<(), VdafError> {
                    Err(VdafError::Uncategorized(
                        "FakeFailsPrepInit failed at prep_init".to_string(),
                    ))
                }),
            )),

            #[cfg(feature = "test-util")]
            VdafInstance::FakeFailsPrepStep => {
                VdafOps::Fake(Arc::new(dummy::Vdaf::new(1).with_prep_step_fn(
                    |_| -> Result<PrepareTransition<dummy::Vdaf, 0, 16>, VdafError> {
                        Err(VdafError::Uncategorized(
                            "FakeFailsPrepStep failed at prep_step".to_string(),
                        ))
                    },
                )))
            }

            _ => panic!("VDAF {:?} is not yet supported", task.vdaf()),
        };

        Ok(Self {
            task: Arc::new(task),
            vdaf_ops,
            report_writer,
        })
    }

    async fn handle_upload(
        &self,
        clock: &C,
        hpke_keypairs: &HpkeKeypairCache,
        metrics: &AggregatorMetrics,
        report: Report,
    ) -> Result<(), Arc<Error>> {
        self.vdaf_ops
            .handle_upload(
                clock,
                hpke_keypairs,
                metrics,
                &self.task,
                &self.report_writer,
                report,
            )
            .await
    }

    async fn handle_aggregate_init(
        &self,
        datastore: Arc<Datastore<C>>,
        hpke_keypairs: Arc<HpkeKeypairCache>,
        metrics: &AggregatorMetrics,
        batch_aggregation_shard_count: u64,
        task_counter_shard_count: u64,
        aggregation_job_id: &AggregationJobId,
        require_taskbind_extension: bool,
        log_forbidden_mutations: Option<PathBuf>,
        req_bytes: &[u8],
    ) -> Result<AggregationJobResp, Error> {
        self.vdaf_ops
            .handle_aggregate_init(
                datastore,
                hpke_keypairs,
                metrics,
                Arc::clone(&self.task),
                batch_aggregation_shard_count,
                task_counter_shard_count,
                aggregation_job_id,
                require_taskbind_extension,
                log_forbidden_mutations,
                req_bytes,
            )
            .await
    }

    async fn handle_aggregate_continue(
        &self,
        datastore: Arc<Datastore<C>>,
        metrics: &AggregatorMetrics,
        batch_aggregation_shard_count: u64,
        task_counter_shard_count: u64,
        aggregation_job_id: &AggregationJobId,
        req: AggregationJobContinueReq,
        request_hash: [u8; 32],
    ) -> Result<AggregationJobResp, Error> {
        self.vdaf_ops
            .handle_aggregate_continue(
                datastore,
                metrics,
                Arc::clone(&self.task),
                batch_aggregation_shard_count,
                task_counter_shard_count,
                aggregation_job_id,
                req,
                request_hash,
            )
            .await
    }

    async fn handle_aggregate_get(
        &self,
        datastore: Arc<Datastore<C>>,
        aggregation_job_id: &AggregationJobId,
        step: AggregationJobStep,
    ) -> Result<AggregationJobResp, Error> {
        self.vdaf_ops
            .handle_aggregate_get(datastore, Arc::clone(&self.task), aggregation_job_id, step)
            .await
    }

    async fn handle_aggregate_delete(
        &self,
        datastore: &Datastore<C>,
        aggregation_job_id: &AggregationJobId,
    ) -> Result<(), Error> {
        self.vdaf_ops
            .handle_aggregate_delete(datastore, Arc::clone(&self.task), aggregation_job_id)
            .await
    }

    async fn handle_create_collection_job(
        &self,
        datastore: &Datastore<C>,
        collection_job_id: &CollectionJobId,
        req_bytes: &[u8],
    ) -> Result<Vec<u8>, Error> {
        self.vdaf_ops
            .handle_create_collection_job(
                datastore,
                Arc::clone(&self.task),
                collection_job_id,
                req_bytes,
            )
            .await
    }

    async fn handle_get_collection_job(
        &self,
        datastore: &Datastore<C>,
        collection_job_id: &CollectionJobId,
    ) -> Result<Vec<u8>, Error> {
        self.vdaf_ops
            .handle_get_collection_job(datastore, Arc::clone(&self.task), collection_job_id)
            .await
    }

    async fn handle_delete_collection_job(
        &self,
        datastore: &Datastore<C>,
        collection_job_id: &CollectionJobId,
    ) -> Result<(), Error> {
        self.vdaf_ops
            .handle_delete_collection_job(datastore, Arc::clone(&self.task), collection_job_id)
            .await
    }

    async fn handle_aggregate_share(
        &self,
        datastore: &Datastore<C>,
        clock: &C,
        batch_aggregation_shard_count: u64,
        req_bytes: &[u8],
        collector_hpke_config: &HpkeConfig,
    ) -> Result<AggregateShare, Error> {
        self.vdaf_ops
            .handle_aggregate_share(
                datastore,
                clock,
                Arc::clone(&self.task),
                batch_aggregation_shard_count,
                req_bytes,
                collector_hpke_config,
            )
            .await
    }
}

mod vdaf_ops_strategies {
    use std::sync::Arc;

    use janus_core::vdaf::vdaf_dp_strategies;
    use prio::dp::distributions::PureDpDiscreteLaplace;
    #[cfg(feature = "fpvec_bounded_l2")]
    use prio::dp::distributions::ZCdpDiscreteGaussian;

    #[derive(Debug)]
    pub enum Prio3Histogram {
        NoDifferentialPrivacy,
        PureDpDiscreteLaplace(Arc<PureDpDiscreteLaplace>),
    }

    impl Prio3Histogram {
        pub fn from_vdaf_dp_strategy(dp_strategy: vdaf_dp_strategies::Prio3Histogram) -> Self {
            match dp_strategy {
                vdaf_dp_strategies::Prio3Histogram::NoDifferentialPrivacy => {
                    Prio3Histogram::NoDifferentialPrivacy
                }
                vdaf_dp_strategies::Prio3Histogram::PureDpDiscreteLaplace(s) => {
                    Prio3Histogram::PureDpDiscreteLaplace(Arc::new(s))
                }
            }
        }
    }

    #[derive(Debug)]
    pub enum Prio3SumVec {
        NoDifferentialPrivacy,
        PureDpDiscreteLaplace(Arc<PureDpDiscreteLaplace>),
    }

    impl Prio3SumVec {
        pub fn from_vdaf_dp_strategy(dp_strategy: vdaf_dp_strategies::Prio3SumVec) -> Self {
            match dp_strategy {
                vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy => {
                    Prio3SumVec::NoDifferentialPrivacy
                }
                vdaf_dp_strategies::Prio3SumVec::PureDpDiscreteLaplace(s) => {
                    Prio3SumVec::PureDpDiscreteLaplace(Arc::new(s))
                }
            }
        }
    }

    #[cfg(feature = "fpvec_bounded_l2")]
    #[derive(Debug)]
    pub enum Prio3FixedPointBoundedL2VecSum {
        NoDifferentialPrivacy,
        ZCdpDiscreteGaussian(Arc<ZCdpDiscreteGaussian>),
    }

    #[cfg(feature = "fpvec_bounded_l2")]
    impl Prio3FixedPointBoundedL2VecSum {
        pub fn from_vdaf_dp_strategy(
            dp_strategy: vdaf_dp_strategies::Prio3FixedPointBoundedL2VecSum,
        ) -> Self {
            match dp_strategy {
                vdaf_dp_strategies::Prio3FixedPointBoundedL2VecSum::NoDifferentialPrivacy => {
                    Prio3FixedPointBoundedL2VecSum::NoDifferentialPrivacy
                }
                vdaf_dp_strategies::Prio3FixedPointBoundedL2VecSum::ZCdpDiscreteGaussian(s) => {
                    Prio3FixedPointBoundedL2VecSum::ZCdpDiscreteGaussian(Arc::new(s))
                }
            }
        }
    }
}

/// VdafOps stores VDAF-specific operations for a TaskAggregator in a non-generic way.
#[allow(clippy::enum_variant_names)]
#[derive(Debug)]
enum VdafOps {
    Prio3Count(Arc<Prio3Count>),
    Prio3Sum(Arc<Prio3Sum>),
    Prio3SumVec(Arc<Prio3SumVec>, vdaf_ops_strategies::Prio3SumVec),
    Prio3SumVecField64MultiproofHmacSha256Aes128(
        Arc<Prio3SumVecField64MultiproofHmacSha256Aes128<ParallelSum<Field64, Mul<Field64>>>>,
        vdaf_ops_strategies::Prio3SumVec,
    ),
    Prio3Histogram(Arc<Prio3Histogram>, vdaf_ops_strategies::Prio3Histogram),
    #[cfg(feature = "fpvec_bounded_l2")]
    Prio3FixedPoint16BitBoundedL2VecSum(
        Arc<Prio3FixedPointBoundedL2VecSum<FixedI16<U15>>>,
        vdaf_ops_strategies::Prio3FixedPointBoundedL2VecSum,
    ),
    #[cfg(feature = "fpvec_bounded_l2")]
    Prio3FixedPoint32BitBoundedL2VecSum(
        Arc<Prio3FixedPointBoundedL2VecSum<FixedI32<U31>>>,
        vdaf_ops_strategies::Prio3FixedPointBoundedL2VecSum,
    ),
    #[cfg(feature = "test-util")]
    Fake(Arc<dummy::Vdaf>),
}

/// Emits a match block dispatching on a [`VdafOps`] object. Takes a `&VdafOps` as the first
/// argument, followed by a pseudo-pattern and body. The pseudo-pattern takes variable names for the
/// constructed VDAF and the verify key, a type alias name that the block can use to explicitly
/// specify the VDAF's type, and the name of a const that will be set to the VDAF's verify key
/// length, also for explicitly specifying type parameters.
macro_rules! vdaf_ops_dispatch {
    ($vdaf_ops:expr, ($vdaf:pat_param, $Vdaf:ident, $VERIFY_KEY_LENGTH:ident, $dp_strategy:ident, $DpStrategy:ident) => $body:tt) => {
        match $vdaf_ops {
            crate::aggregator::VdafOps::Prio3Count(vdaf) => {
                let $vdaf = vdaf;
                type $Vdaf = ::prio::vdaf::prio3::Prio3Count;
                const $VERIFY_KEY_LENGTH: usize = ::janus_core::vdaf::VERIFY_KEY_LENGTH_PRIO3;
                type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                let $dp_strategy = &Arc::new(janus_core::dp::NoDifferentialPrivacy);
                let body = $body;
                body
            }

            crate::aggregator::VdafOps::Prio3Sum(vdaf) => {
                let $vdaf = vdaf;
                type $Vdaf = ::prio::vdaf::prio3::Prio3Sum;
                const $VERIFY_KEY_LENGTH: usize = ::janus_core::vdaf::VERIFY_KEY_LENGTH_PRIO3;
                type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                let $dp_strategy = &Arc::new(janus_core::dp::NoDifferentialPrivacy);
                let body = $body;
                body
            }

            crate::aggregator::VdafOps::Prio3SumVec(vdaf, _dp_strategy) => {
                let $vdaf = vdaf;
                type $Vdaf = ::prio::vdaf::prio3::Prio3SumVec;
                const $VERIFY_KEY_LENGTH: usize = ::janus_core::vdaf::VERIFY_KEY_LENGTH_PRIO3;
                match _dp_strategy {
                    vdaf_ops_strategies::Prio3SumVec::NoDifferentialPrivacy => {
                        type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                        let $dp_strategy = &Arc::new(janus_core::dp::NoDifferentialPrivacy);
                        let body = $body;
                        body
                    }
                    vdaf_ops_strategies::Prio3SumVec::PureDpDiscreteLaplace(_strategy) => {
                        type $DpStrategy = ::prio::dp::distributions::PureDpDiscreteLaplace;
                        let $dp_strategy = &_strategy;
                        let body = $body;
                        body
                    }
                }
            }

            crate::aggregator::VdafOps::Prio3SumVecField64MultiproofHmacSha256Aes128(vdaf, _dp_strategy) => {
                let $vdaf = vdaf;
                type $Vdaf = ::janus_core::vdaf::Prio3SumVecField64MultiproofHmacSha256Aes128<
                    ::prio::flp::gadgets::ParallelSum<
                        ::prio::field::Field64,
                        ::prio::flp::gadgets::Mul<::prio::field::Field64>
                    >,
                >;
                const $VERIFY_KEY_LENGTH: usize = 32;
                match _dp_strategy {
                    vdaf_ops_strategies::Prio3SumVec::NoDifferentialPrivacy => {
                        type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                        let $dp_strategy = &Arc::new(janus_core::dp::NoDifferentialPrivacy);
                        let body = $body;
                        body
                    }
                    vdaf_ops_strategies::Prio3SumVec::PureDpDiscreteLaplace(_strategy) => {
                        type $DpStrategy = ::prio::dp::distributions::PureDpDiscreteLaplace;
                        let $dp_strategy = &_strategy;
                        let body = $body;
                        body
                    }
                }
            }

            crate::aggregator::VdafOps::Prio3Histogram(vdaf, _dp_strategy) => {
                let $vdaf = vdaf;
                type $Vdaf = ::prio::vdaf::prio3::Prio3Histogram;
                const $VERIFY_KEY_LENGTH: usize = ::janus_core::vdaf::VERIFY_KEY_LENGTH_PRIO3;
                match _dp_strategy {
                    vdaf_ops_strategies::Prio3Histogram::NoDifferentialPrivacy => {
                        type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                        let $dp_strategy = &Arc::new(janus_core::dp::NoDifferentialPrivacy);
                        let body = $body;
                        body
                    }
                    vdaf_ops_strategies::Prio3Histogram::PureDpDiscreteLaplace(_strategy) => {
                        type $DpStrategy = ::prio::dp::distributions::PureDpDiscreteLaplace;
                        let $dp_strategy = &_strategy;
                        let body = $body;
                        body
                    }
                }
            }

            #[cfg(feature = "fpvec_bounded_l2")]
            // Note that the variable `_dp_strategy` is used if `$dp_strategy`
            // and `$DpStrategy` are given. The underscore suppresses warnings
            // which occur when `vdaf_ops!` is called without these parameters.
            crate::aggregator::VdafOps::Prio3FixedPoint16BitBoundedL2VecSum(vdaf, _dp_strategy) => {
                let $vdaf = vdaf;
                type $Vdaf = ::prio::vdaf::prio3::Prio3FixedPointBoundedL2VecSum<FixedI16<U15>>;
                const $VERIFY_KEY_LENGTH: usize = ::janus_core::vdaf::VERIFY_KEY_LENGTH_PRIO3;

                match _dp_strategy {
                    vdaf_ops_strategies::Prio3FixedPointBoundedL2VecSum::ZCdpDiscreteGaussian(_strategy) => {
                        type $DpStrategy = ::prio::dp::distributions::ZCdpDiscreteGaussian;
                        let $dp_strategy = &_strategy;
                        let body = $body;
                        body
                    },
                    vdaf_ops_strategies::Prio3FixedPointBoundedL2VecSum::NoDifferentialPrivacy => {
                        type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                        let $dp_strategy = &Arc::new(janus_core::dp::NoDifferentialPrivacy);
                        let body = $body;
                        body
                    }
                }
            }

            #[cfg(feature = "fpvec_bounded_l2")]
            // Note that the variable `_dp_strategy` is used if `$dp_strategy`
            // and `$DpStrategy` are given. The underscore suppresses warnings
            // which occur when `vdaf_ops!` is called without these parameters.
            crate::aggregator::VdafOps::Prio3FixedPoint32BitBoundedL2VecSum(vdaf, _dp_strategy) => {
                let $vdaf = vdaf;
                type $Vdaf = ::prio::vdaf::prio3::Prio3FixedPointBoundedL2VecSum<FixedI32<U31>>;
                const $VERIFY_KEY_LENGTH: usize = ::janus_core::vdaf::VERIFY_KEY_LENGTH_PRIO3;

                match _dp_strategy {
                    vdaf_ops_strategies::Prio3FixedPointBoundedL2VecSum::ZCdpDiscreteGaussian(_strategy) => {
                        type $DpStrategy = ::prio::dp::distributions::ZCdpDiscreteGaussian;
                        let $dp_strategy = &_strategy;
                        let body = $body;
                        body
                    },
                    vdaf_ops_strategies::Prio3FixedPointBoundedL2VecSum::NoDifferentialPrivacy => {
                        type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                        let $dp_strategy = &Arc::new(janus_core::dp::NoDifferentialPrivacy);
                        let body = $body;
                        body
                    }
                }
            }

            #[cfg(feature = "test-util")]
            crate::aggregator::VdafOps::Fake(vdaf) => {
                let $vdaf = vdaf;
                type $Vdaf = ::prio::vdaf::dummy::Vdaf;
                const $VERIFY_KEY_LENGTH: usize = 0;
                type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                let $dp_strategy = &Arc::new(janus_core::dp::NoDifferentialPrivacy);
                let body = $body;
                body
            }
        }
    };

    ($vdaf_ops:expr, ($vdaf:pat_param, $Vdaf:ident, $VERIFY_KEY_LENGTH:ident) => $body:tt) => {
        vdaf_ops_dispatch!($vdaf_ops, ($vdaf, $Vdaf, $VERIFY_KEY_LENGTH, _unused, _Unused) => $body)};
}

impl VdafOps {
    #[tracing::instrument(skip_all, fields(task_id = ?task.id()), err(level = Level::DEBUG))]
    async fn handle_upload<C: Clock>(
        &self,
        clock: &C,
        hpke_keypairs: &HpkeKeypairCache,
        metrics: &AggregatorMetrics,
        task: &AggregatorTask,
        report_writer: &ReportWriteBatcher<C>,
        report: Report,
    ) -> Result<(), Arc<Error>> {
        match task.batch_mode() {
            task::BatchMode::TimeInterval => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_upload_generic::<VERIFY_KEY_LENGTH, TimeInterval, VdafType, _>(
                        Arc::clone(vdaf),
                        clock,
                        hpke_keypairs,
                        metrics,
                        task,
                        report_writer,
                        report,
                    )
                    .await
                })
            }
            task::BatchMode::LeaderSelected { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_upload_generic::<VERIFY_KEY_LENGTH, LeaderSelected, VdafType, _>(
                        Arc::clone(vdaf),
                        clock,
                        hpke_keypairs,
                        metrics,
                        task,
                        report_writer,
                        report,
                    )
                    .await
                })
            }
        }
    }

    /// Implements [helper aggregate initialization][1].
    ///
    /// [1]: https://www.ietf.org/archive/id/draft-ietf-ppm-dap-07.html#name-helper-initialization
    #[tracing::instrument(
        skip(self, datastore, hpke_keypairs, metrics, task, req_bytes),
        fields(task_id = ?task.id()),
        err(level = Level::DEBUG)
    )]
    async fn handle_aggregate_init<C: Clock>(
        &self,
        datastore: Arc<Datastore<C>>,
        hpke_keypairs: Arc<HpkeKeypairCache>,
        metrics: &AggregatorMetrics,
        task: Arc<AggregatorTask>,
        batch_aggregation_shard_count: u64,
        task_counter_shard_count: u64,
        aggregation_job_id: &AggregationJobId,
        require_taskbind_extension: bool,
        log_forbidden_mutations: Option<PathBuf>,
        req_bytes: &[u8],
    ) -> Result<AggregationJobResp, Error> {
        match task.batch_mode() {
            task::BatchMode::TimeInterval => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_aggregate_init_generic::<VERIFY_KEY_LENGTH, TimeInterval, VdafType, _>(
                        datastore,
                        hpke_keypairs,
                        Arc::clone(vdaf),
                        metrics,
                        task,
                        batch_aggregation_shard_count,
                        task_counter_shard_count,
                        aggregation_job_id,
                        require_taskbind_extension,
                        log_forbidden_mutations,
                        req_bytes,
                    )
                    .await
                })
            }
            task::BatchMode::LeaderSelected { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_aggregate_init_generic::<VERIFY_KEY_LENGTH, LeaderSelected, VdafType, _>(
                        datastore,
                        hpke_keypairs,
                        Arc::clone(vdaf),
                        metrics,
                        task,
                        batch_aggregation_shard_count,
                        task_counter_shard_count,
                        aggregation_job_id,
                        require_taskbind_extension,
                        log_forbidden_mutations,
                        req_bytes,
                    )
                    .await
                })
            }
        }
    }

    #[tracing::instrument(
        skip(self, datastore, metrics, task, req, request_hash),
        fields(task_id = ?task.id()),
        err(level = Level::DEBUG)
    )]
    async fn handle_aggregate_continue<C: Clock>(
        &self,
        datastore: Arc<Datastore<C>>,
        metrics: &AggregatorMetrics,
        task: Arc<AggregatorTask>,
        batch_aggregation_shard_count: u64,
        task_counter_shard_count: u64,
        aggregation_job_id: &AggregationJobId,
        req: AggregationJobContinueReq,
        request_hash: [u8; 32],
    ) -> Result<AggregationJobResp, Error> {
        match task.batch_mode() {
            task::BatchMode::TimeInterval => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_aggregate_continue_generic::<VERIFY_KEY_LENGTH, TimeInterval, VdafType, _>(
                        datastore,
                        Arc::clone(vdaf),
                        metrics,
                        task,
                        batch_aggregation_shard_count,
                        task_counter_shard_count,
                        aggregation_job_id,
                        req,
                        request_hash,
                    )
                    .await
                })
            }
            task::BatchMode::LeaderSelected { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_aggregate_continue_generic::<VERIFY_KEY_LENGTH, LeaderSelected, VdafType, _>(
                        datastore,
                        Arc::clone(vdaf),
                        metrics,
                        task,
                        batch_aggregation_shard_count,
                        task_counter_shard_count,
                        aggregation_job_id,
                        req,
                        request_hash,
                    )
                    .await
                })
            }
        }
    }

    #[tracing::instrument(skip(self, datastore), fields(task_id = ?task.id()), err(level = Level::DEBUG))]
    async fn handle_aggregate_get<C: Clock>(
        &self,
        datastore: Arc<Datastore<C>>,
        task: Arc<AggregatorTask>,
        aggregation_job_id: &AggregationJobId,
        step: AggregationJobStep,
    ) -> Result<AggregationJobResp, Error> {
        match task.batch_mode() {
            task::BatchMode::TimeInterval => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_aggregate_get_generic::<VERIFY_KEY_LENGTH, TimeInterval, VdafType, _>(
                        datastore,
                        Arc::clone(vdaf),
                        task,
                        aggregation_job_id,
                        step,
                    ).await
                })
            }

            task::BatchMode::LeaderSelected { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_aggregate_get_generic::<VERIFY_KEY_LENGTH, LeaderSelected, VdafType, _>(
                        datastore,
                        Arc::clone(vdaf),
                        task,
                        aggregation_job_id,
                        step,
                    ).await
                })
            }
        }
    }

    #[tracing::instrument(skip(self, datastore), fields(task_id = ?task.id()), err(level = Level::DEBUG))]
    async fn handle_aggregate_delete<C: Clock>(
        &self,
        datastore: &Datastore<C>,
        task: Arc<AggregatorTask>,
        aggregation_job_id: &AggregationJobId,
    ) -> Result<(), Error> {
        match task.batch_mode() {
            task::BatchMode::TimeInterval => {
                vdaf_ops_dispatch!(self, (_, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_aggregate_delete_generic::<VERIFY_KEY_LENGTH, TimeInterval, VdafType, _>(
                        datastore,
                        task,
                        aggregation_job_id,
                    ).await
                })
            }
            task::BatchMode::LeaderSelected { .. } => {
                vdaf_ops_dispatch!(self, (_, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_aggregate_delete_generic::<VERIFY_KEY_LENGTH, LeaderSelected, VdafType, _>(
                        datastore,
                        task,
                        aggregation_job_id,
                    ).await
                })
            }
        }
    }

    async fn handle_upload_generic<const SEED_SIZE: usize, B, A, C>(
        vdaf: Arc<A>,
        clock: &C,
        hpke_keypairs: &HpkeKeypairCache,
        metrics: &AggregatorMetrics,
        task: &AggregatorTask,
        report_writer: &ReportWriteBatcher<C>,
        report: Report,
    ) -> Result<(), Arc<Error>>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        A::InputShare: PartialEq + Send + Sync,
        A::PublicShare: PartialEq + Send + Sync,
        A::AggregationParam: Send + Sync,
        A::AggregateShare: Send + Sync,
        C: Clock,
        B: UploadableBatchMode,
    {
        // Shorthand function for generating an Error::ReportRejected with proper parameters and
        // recording it in the report_writer.
        let reject_report = |reason| {
            let report_id = *report.metadata().id();
            let report_time = *report.metadata().time();
            async move {
                let rejection = ReportRejection::new(*task.id(), report_id, report_time, reason);
                report_writer.write_rejection(rejection).await;
                Ok::<_, Arc<Error>>(Arc::new(Error::ReportRejected(rejection)))
            }
        };

        let report_deadline = clock
            .now()
            .add(task.tolerable_clock_skew())
            .map_err(|err| Arc::new(Error::from(err)))?;

        // Reject reports from too far in the future.
        // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-07.html#section-4.4.2-21
        if report.metadata().time().is_after(&report_deadline) {
            return Err(reject_report(ReportRejectionReason::TooEarly).await?);
        }

        // Reject reports before a task has started.
        if let Some(task_start) = task.task_start() {
            if report.metadata().time().is_before(task_start) {
                return Err(reject_report(ReportRejectionReason::TaskNotStarted).await?);
            }
        }

        // Reject reports after a task has ended.
        // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-07.html#section-4.4.2-20
        if let Some(task_end) = task.task_end() {
            if report.metadata().time().is_after(task_end) {
                return Err(reject_report(ReportRejectionReason::TaskEnded).await?);
            }
        }

        // Reject reports that would be eligible for garbage collection, to prevent replay attacks.
        if let Some(report_expiry_age) = task.report_expiry_age() {
            let report_expiry_time = report
                .metadata()
                .time()
                .add(report_expiry_age)
                .map_err(|err| Arc::new(Error::from(err)))?;
            if clock.now().is_after(&report_expiry_time) {
                return Err(reject_report(ReportRejectionReason::Expired).await?);
            }
        }

        // Decode (and in the case of the leader input share, decrypt) the remaining fields of the
        // report before storing them in the datastore. The spec does not require the
        // `tasks/{task-id}/reports` handler to do this, but it exercises HPKE decryption, saves us
        // the trouble of storing reports we can't use, and lets the aggregation job handler assume
        // the values it reads from the datastore are valid.
        let public_share =
            match A::PublicShare::get_decoded_with_param(vdaf.as_ref(), report.public_share()) {
                Ok(public_share) => public_share,
                Err(err) => {
                    debug!(
                        report.task_id = %task.id(),
                        report.id = ?report.metadata().id(),
                        ?err,
                        "public share decoding failed",
                    );
                    metrics.upload_decode_failure_counter.add(1, &[]);
                    return Err(reject_report(ReportRejectionReason::DecodeFailure).await?);
                }
            };

        let input_share_aad = InputShareAad::new(
            *task.id(),
            report.metadata().clone(),
            report.public_share().to_vec(),
        )
        .get_encoded()
        .map_err(|e| Arc::new(Error::MessageEncode(e)))?;

        // Retrieve the HPKE key indicated by the report & verify that it is known.
        let hpke_keypair =
            match hpke_keypairs.keypair(report.leader_encrypted_input_share().config_id()) {
                Some(hpke_keypair) => hpke_keypair,
                None => {
                    return Err(reject_report(ReportRejectionReason::OutdatedHpkeConfig(
                        *report.leader_encrypted_input_share().config_id(),
                    ))
                    .await?);
                }
            };

        // Verify that we can decrypt & decode the Leader input share with the key we retrieved.
        let decryption_result = hpke::open(
            &hpke_keypair,
            &HpkeApplicationInfo::new(&Label::InputShare, &Role::Client, &Role::Leader),
            report.leader_encrypted_input_share(),
            &input_share_aad,
        );

        let encoded_leader_input_share = match decryption_result {
            Ok(plaintext) => plaintext,
            Err(error) => {
                debug!(
                    report.task_id = %task.id(),
                    report.id = ?report.metadata().id(),
                    ?error,
                    "Report decryption failed",
                );
                metrics.upload_decrypt_failure_counter.add(1, &[]);
                return Err(reject_report(ReportRejectionReason::DecryptFailure).await?);
            }
        };

        let decoded_leader_input_share = PlaintextInputShare::get_decoded(
            &encoded_leader_input_share,
        )
        .and_then(|plaintext_input_share| {
            Ok((
                plaintext_input_share.private_extensions().to_vec(),
                A::InputShare::get_decoded_with_param(
                    &(&vdaf, Role::Leader.index().unwrap()),
                    plaintext_input_share.payload(),
                )?,
            ))
        });

        let (leader_private_extensions, leader_input_share) = match decoded_leader_input_share {
            Ok(leader_input_share) => leader_input_share,
            Err(err) => {
                debug!(
                    report.task_id = %task.id(),
                    report.id = ?report.metadata().id(),
                    ?err,
                    "Leader input share decoding failed",
                );
                metrics.upload_decode_failure_counter.add(1, &[]);
                return Err(reject_report(ReportRejectionReason::DecodeFailure).await?);
            }
        };

        let report = LeaderStoredReport::new(
            *task.id(),
            report.metadata().clone(),
            public_share,
            leader_private_extensions,
            leader_input_share,
            report.helper_encrypted_input_share().clone(),
        );

        report_writer
            .write_report(Box::new(WritableReport::<SEED_SIZE, B, A>::new(
                vdaf, report,
            )))
            .await
    }
}

impl VdafOps {
    async fn check_aggregate_init_idempotency<const SEED_SIZE: usize, B, A, C>(
        tx: &Transaction<'_, C>,
        vdaf: &A,
        task_id: &TaskId,
        request_hash: [u8; 32],
        mutating_aggregation_job: &AggregationJob<SEED_SIZE, B, A>,
        mutating_report_aggregations: impl IntoIterator<Item = &ReportAggregation<SEED_SIZE, A>>,
        log_forbidden_mutations: Option<PathBuf>,
    ) -> Result<Option<Vec<PrepareResp>>, datastore::Error>
    where
        B: AccumulableBatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + 'static + Send + Sync,
        C: Clock,
        for<'a> A::PrepareState: ParameterizedDecode<(&'a A, usize)>,
    {
        let existing_aggregation_job = match tx
            .get_aggregation_job::<SEED_SIZE, B, A>(task_id, mutating_aggregation_job.id())
            .await?
        {
            Some(existing_aggregation_job) => existing_aggregation_job,
            None => return Ok(None),
        };

        if existing_aggregation_job.state() == &AggregationJobState::Deleted {
            return Err(datastore::Error::User(
                Error::DeletedAggregationJob(*task_id, *mutating_aggregation_job.id()).into(),
            ));
        }

        if existing_aggregation_job.last_request_hash() != Some(request_hash) {
            if let Some(log_forbidden_mutations) = log_forbidden_mutations {
                let original_report_ids: Vec<_> = tx
                    .get_report_aggregations_for_aggregation_job(
                        vdaf,
                        &Role::Helper,
                        task_id,
                        mutating_aggregation_job.id(),
                        existing_aggregation_job.aggregation_parameter(),
                    )
                    .await?
                    .iter()
                    .map(|ra| *ra.report_id())
                    .collect();
                let mutating_request_report_ids: Vec<_> = mutating_report_aggregations
                    .into_iter()
                    .map(|ra| *ra.report_id())
                    .collect();
                let event = AggregationJobInitForbiddenMutationEvent {
                    task_id: *task_id,
                    aggregation_job_id: *mutating_aggregation_job.id(),
                    original_request_hash: existing_aggregation_job.last_request_hash(),
                    original_report_ids,
                    original_batch_id: format!(
                        "{:?}",
                        existing_aggregation_job.partial_batch_identifier()
                    ),
                    original_aggregation_parameter: existing_aggregation_job
                        .aggregation_parameter()
                        .get_encoded()
                        .map_err(|e| datastore::Error::User(e.into()))?,
                    mutating_request_hash: Some(request_hash),
                    mutating_request_report_ids,
                    mutating_request_batch_id: format!(
                        "{:?}",
                        mutating_aggregation_job.partial_batch_identifier()
                    ),
                    mutating_request_aggregation_parameter: mutating_aggregation_job
                        .aggregation_parameter()
                        .get_encoded()
                        .map_err(|e| datastore::Error::User(e.into()))?,
                };
                let event_id = crate::diagnostic::write_event(
                    log_forbidden_mutations,
                    "agg-job-illegal-mutation",
                    event,
                )
                .await
                .map(|event_id| format!("{event_id:?}"))
                .unwrap_or_else(|error| {
                    tracing::error!(?error, "failed to write hash mismatch event");
                    "no event id".to_string()
                });

                tracing::info!(
                    ?event_id,
                    original_request_hash = existing_aggregation_job
                        .last_request_hash()
                        .map(hex::encode),
                    mutating_request_hash = hex::encode(request_hash),
                    "request hash mismatch on retried aggregation job request",
                );
            }

            return Err(datastore::Error::User(
                Error::ForbiddenMutation {
                    resource_type: "aggregation job",
                    identifier: mutating_aggregation_job.id().to_string(),
                }
                .into(),
            ));
        }

        // This is a repeated request. Send the preparation responses we computed last time.
        return Ok(Some(
            tx.get_report_aggregations_for_aggregation_job(
                vdaf,
                &Role::Helper,
                task_id,
                existing_aggregation_job.id(),
                existing_aggregation_job.aggregation_parameter(),
            )
            .await?
            .iter()
            .filter_map(ReportAggregation::last_prep_resp)
            .cloned()
            .collect(),
        ));
    }

    /// Implements [helper aggregate initialization][1].
    ///
    /// [1]: https://www.ietf.org/archive/id/draft-ietf-ppm-dap-07.html#name-helper-initialization
    async fn handle_aggregate_init_generic<const SEED_SIZE: usize, B, A, C>(
        datastore: Arc<Datastore<C>>,
        hpke_keypairs: Arc<HpkeKeypairCache>,
        vdaf: Arc<A>,
        metrics: &AggregatorMetrics,
        task: Arc<AggregatorTask>,
        batch_aggregation_shard_count: u64,
        task_counter_shard_count: u64,
        aggregation_job_id: &AggregationJobId,
        require_taskbind_extension: bool,
        log_forbidden_mutations: Option<PathBuf>,
        req_bytes: &[u8],
    ) -> Result<AggregationJobResp, Error>
    where
        B: AccumulableBatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + 'static + Send + Sync,
        C: Clock,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        A::PrepareMessage: Send + Sync + PartialEq,
        A::PrepareShare: Send + Sync + PartialEq,
        for<'a> A::PrepareState:
            Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)> + PartialEq,
        A::PublicShare: Send + Sync,
        A::OutputShare: Send + Sync + PartialEq,
    {
        // Unwrap safety: SHA-256 computed by ring should always be 32 bytes.
        let request_hash = digest(&SHA256, req_bytes).as_ref().try_into().unwrap();
        let req = AggregationJobInitializeReq::<B>::get_decoded(req_bytes)
            .map_err(Error::MessageDecode)?;

        // If two ReportShare messages have the same report ID, then the helper MUST abort with
        // error "invalidMessage". (§4.5.1.2)
        let mut seen_report_ids = HashSet::with_capacity(req.prepare_inits().len());
        for prepare_init in req.prepare_inits() {
            if !seen_report_ids.insert(*prepare_init.report_share().metadata().id()) {
                return Err(Error::InvalidMessage(
                    Some(*task.id()),
                    "aggregate request contains duplicate report IDs",
                ));
            }
        }

        // Build initial aggregation job & report aggregations.
        let min_client_timestamp = req
            .prepare_inits()
            .iter()
            .map(|prepare_init| *prepare_init.report_share().metadata().time())
            .min()
            .ok_or_else(|| Error::EmptyAggregation(*task.id()))?;
        let max_client_timestamp = req
            .prepare_inits()
            .iter()
            .map(|prepare_init| *prepare_init.report_share().metadata().time())
            .max()
            .ok_or_else(|| Error::EmptyAggregation(*task.id()))?;
        let client_timestamp_interval = Interval::new(
            min_client_timestamp,
            max_client_timestamp
                .difference(&min_client_timestamp)?
                .add(&Duration::from_seconds(1))?,
        )?;
        let aggregation_job = AggregationJob::<SEED_SIZE, B, A>::new(
            *task.id(),
            *aggregation_job_id,
            A::AggregationParam::get_decoded(req.aggregation_parameter())
                .map_err(Error::MessageDecode)?,
            req.batch_selector().batch_identifier().clone(),
            client_timestamp_interval,
            AggregationJobState::AwaitingRequest,
            AggregationJobStep::from(0),
        )
        .with_last_request_hash(request_hash);

        let report_aggregations = req
            .prepare_inits()
            .iter()
            .enumerate()
            .map(|(ord, prepare_init)| {
                Ok(ReportAggregation::<SEED_SIZE, A>::new(
                    *task.id(),
                    *aggregation_job_id,
                    *prepare_init.report_share().metadata().id(),
                    *prepare_init.report_share().metadata().time(),
                    u64::try_from(ord)?,
                    None,
                    ReportAggregationState::HelperInitProcessing {
                        prepare_init: prepare_init.clone(),
                        require_taskbind_extension,
                    },
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        match task.aggregation_mode() {
            Some(AggregationMode::Synchronous) => {
                Self::handle_aggregate_init_generic_sync(
                    datastore,
                    hpke_keypairs,
                    vdaf,
                    metrics,
                    task,
                    batch_aggregation_shard_count,
                    task_counter_shard_count,
                    log_forbidden_mutations,
                    request_hash,
                    aggregation_job,
                    report_aggregations,
                )
                .await
            }

            Some(AggregationMode::Asynchronous) => {
                Self::handle_aggregate_init_generic_async(
                    datastore,
                    vdaf,
                    metrics,
                    task,
                    batch_aggregation_shard_count,
                    task_counter_shard_count,
                    log_forbidden_mutations,
                    request_hash,
                    aggregation_job,
                    report_aggregations,
                )
                .await
            }

            None => Err(Error::Internal("task has no aggregation mode".to_string())),
        }
    }

    // All report aggregations must be in the HelperInitProcessing state.
    async fn handle_aggregate_init_generic_sync<const SEED_SIZE: usize, B, A, C>(
        datastore: Arc<Datastore<C>>,
        hpke_keypairs: Arc<HpkeKeypairCache>,
        vdaf: Arc<A>,
        metrics: &AggregatorMetrics,
        task: Arc<AggregatorTask>,
        batch_aggregation_shard_count: u64,
        task_counter_shard_count: u64,
        log_forbidden_mutations: Option<PathBuf>,
        request_hash: [u8; 32],
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        report_aggregations: Vec<ReportAggregation<SEED_SIZE, A>>,
    ) -> Result<AggregationJobResp, Error>
    where
        B: AccumulableBatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + 'static + Send + Sync,
        C: Clock,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        A::PrepareMessage: Send + Sync + PartialEq,
        A::PrepareShare: Send + Sync + PartialEq,
        for<'a> A::PrepareState:
            Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)> + PartialEq,
        A::PublicShare: Send + Sync,
        A::OutputShare: Send + Sync + PartialEq,
    {
        // Check if this is a repeated request, and if it is the same as before, send
        // the same response as last time.
        let aggregation_job = Arc::new(aggregation_job);
        let report_aggregations = Arc::new(report_aggregations);
        if let Some(prepare_resps) = datastore
            .run_tx("aggregate_init_idempotecy", |tx| {
                let vdaf = Arc::clone(&vdaf);
                let task = Arc::clone(&task);
                let aggregation_job = Arc::clone(&aggregation_job);
                let report_aggregations = Arc::clone(&report_aggregations);
                let log_forbidden_mutations = log_forbidden_mutations.clone();

                Box::pin(async move {
                    Self::check_aggregate_init_idempotency(
                        tx,
                        vdaf.as_ref(),
                        task.id(),
                        request_hash,
                        &aggregation_job,
                        report_aggregations.iter(),
                        log_forbidden_mutations,
                    )
                    .await
                })
            })
            .await?
        {
            return Ok(AggregationJobResp::Finished { prepare_resps });
        }

        // Compute the next aggregation step.
        let report_aggregations = compute_helper_aggregate_init(
            datastore.clock(),
            hpke_keypairs,
            Arc::clone(&vdaf),
            metrics.clone().into(),
            Arc::clone(&task),
            Arc::clone(&aggregation_job),
            Arc::unwrap_or_clone(report_aggregations),
        )
        .await?;

        // Store data to datastore.
        let prepare_resps = Self::handle_aggregate_init_generic_write(
            datastore,
            vdaf,
            metrics,
            task,
            batch_aggregation_shard_count,
            task_counter_shard_count,
            log_forbidden_mutations,
            request_hash,
            aggregation_job,
            Arc::new(report_aggregations),
        )
        .await?;

        Ok(AggregationJobResp::Finished { prepare_resps })
    }

    // All report aggregations must be in the HelperInitProcessing state.
    async fn handle_aggregate_init_generic_async<const SEED_SIZE: usize, B, A, C>(
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        metrics: &AggregatorMetrics,
        task: Arc<AggregatorTask>,
        batch_aggregation_shard_count: u64,
        task_counter_shard_count: u64,
        log_forbidden_mutations: Option<PathBuf>,
        request_hash: [u8; 32],
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        report_aggregations: Vec<ReportAggregation<SEED_SIZE, A>>,
    ) -> Result<AggregationJobResp, Error>
    where
        B: AccumulableBatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + 'static + Send + Sync,
        C: Clock,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        A::PrepareMessage: Send + Sync + PartialEq,
        A::PrepareShare: Send + Sync + PartialEq,
        for<'a> A::PrepareState:
            Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)> + PartialEq,
        A::PublicShare: Send + Sync,
        A::OutputShare: Send + Sync + PartialEq,
    {
        Self::handle_aggregate_init_generic_write(
            datastore,
            vdaf,
            metrics,
            task,
            batch_aggregation_shard_count,
            task_counter_shard_count,
            log_forbidden_mutations,
            request_hash,
            Arc::new(aggregation_job.with_state(AggregationJobState::Active)),
            Arc::new(
                report_aggregations
                    .into_iter()
                    .map(|ra| WritableReportAggregation::new(ra, None))
                    .collect(),
            ),
        )
        .await?;

        Ok(AggregationJobResp::Processing)
    }

    async fn handle_aggregate_init_generic_write<const SEED_SIZE: usize, B, A, C>(
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        metrics: &AggregatorMetrics,
        task: Arc<AggregatorTask>,
        batch_aggregation_shard_count: u64,
        task_counter_shard_count: u64,
        log_forbidden_mutations: Option<PathBuf>,
        request_hash: [u8; 32],
        aggregation_job: Arc<AggregationJob<SEED_SIZE, B, A>>,
        report_aggregations: Arc<Vec<WritableReportAggregation<SEED_SIZE, A>>>,
    ) -> Result<Vec<PrepareResp>, Error>
    where
        B: AccumulableBatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + 'static + Send + Sync,
        C: Clock,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        A::PrepareMessage: Send + Sync + PartialEq,
        A::PrepareShare: Send + Sync + PartialEq,
        for<'a> A::PrepareState:
            Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)> + PartialEq,
        A::PublicShare: Send + Sync,
        A::OutputShare: Send + Sync + PartialEq,
    {
        let (prepare_resps, counters) = datastore
            .run_tx("aggregate_init_aggregator_write", |tx| {
                let vdaf = Arc::clone(&vdaf);
                let task = Arc::clone(&task);
                let aggregation_job_writer_metrics = metrics.for_aggregation_job_writer();
                let aggregation_job = Arc::clone(&aggregation_job);
                let report_aggregations = Arc::clone(&report_aggregations);
                let log_forbidden_mutations = log_forbidden_mutations.clone();

                Box::pin(async move {
                    // Check if this is a repeated request, and if it is the same as before, send
                    // the same response as last time. We check during the write transaction, even
                    // if we have checked before, to avoid the possibility of races for concurrent
                    // requests.
                    if let Some(prepare_resps) = Self::check_aggregate_init_idempotency(
                        tx,
                        vdaf.as_ref(),
                        task.id(),
                        request_hash,
                        &aggregation_job,
                        report_aggregations.iter().map(|ra| ra.report_aggregation()),
                        log_forbidden_mutations,
                    )
                    .await?
                    {
                        return Ok((prepare_resps, TaskAggregationCounter::default()));
                    }

                    // Write report shares, and ensure this isn't a repeated report aggregation.
                    let report_aggregations = try_join_all(report_aggregations.iter().map(|ra| {
                        let task = Arc::clone(&task);

                        async move {
                            let mut report_aggregation = Cow::Borrowed(ra);
                            match tx
                                .put_scrubbed_report(
                                    task.id(),
                                    ra.report_aggregation().report_id(),
                                    ra.report_aggregation().time(),
                                )
                                .await
                            {
                                Ok(()) => (),
                                Err(datastore::Error::MutationTargetAlreadyExists) => {
                                    report_aggregation = Cow::Owned(
                                        report_aggregation
                                            .into_owned()
                                            .with_failure(ReportError::ReportReplayed),
                                    )
                                }
                                Err(err) => return Err(err),
                            };
                            Ok(report_aggregation)
                        }
                    }))
                    .await?;

                    // Write aggregation job, report aggregations, and batch aggregations.
                    let mut aggregation_job_writer =
                        AggregationJobWriter::<SEED_SIZE, _, _, InitialWrite, _>::new(
                            task,
                            batch_aggregation_shard_count,
                            Some(aggregation_job_writer_metrics),
                        );
                    aggregation_job_writer
                        .put(aggregation_job.as_ref().clone(), report_aggregations)?;
                    let (mut prep_resps_by_agg_job, counters) =
                        aggregation_job_writer.write(tx, vdaf).await?;
                    Ok((
                        prep_resps_by_agg_job
                            .remove(aggregation_job.id())
                            .unwrap_or_default(),
                        counters,
                    ))
                })
            })
            .await?;

        write_task_aggregation_counter(datastore, task_counter_shard_count, *task.id(), counters);

        Ok(prepare_resps)
    }

    async fn handle_aggregate_continue_generic<
        const SEED_SIZE: usize,
        B: AccumulableBatchMode,
        A,
        C: Clock,
    >(
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        metrics: &AggregatorMetrics,
        task: Arc<AggregatorTask>,
        batch_aggregation_shard_count: u64,
        task_counter_shard_count: u64,
        aggregation_job_id: &AggregationJobId,
        req: AggregationJobContinueReq,
        request_hash: [u8; 32],
    ) -> Result<AggregationJobResp, Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        for<'a> A::PrepareState: Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)>,
        A::PrepareShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        A::PublicShare: Send + Sync,
        A::OutputShare: Send + Sync,
    {
        if req.step() == AggregationJobStep::from(0) {
            return Err(Error::InvalidMessage(
                Some(*task.id()),
                "aggregation job cannot be advanced to step 0",
            ));
        }

        let req = Arc::new(req);
        let (response, counters) = datastore
            .run_tx("aggregate_continue", |tx| {
                let vdaf = Arc::clone(&vdaf);
                let metrics = metrics.clone();
                let task = Arc::clone(&task);
                let aggregation_job_id = *aggregation_job_id;
                let req = Arc::clone(&req);

                Box::pin(async move {
                    // Read existing state.
                    let aggregation_job = tx
                        .get_aggregation_job::<SEED_SIZE, B, A>(task.id(), &aggregation_job_id)
                        .await?
                        .ok_or_else(|| {
                            datastore::Error::User(
                                Error::UnrecognizedAggregationJob(*task.id(), aggregation_job_id)
                                    .into(),
                            )
                        })?;
                    let report_aggregations = tx
                        .get_report_aggregations_for_aggregation_job(
                            vdaf.as_ref(),
                            &Role::Helper,
                            task.id(),
                            &aggregation_job_id,
                            aggregation_job.aggregation_parameter(),
                        )
                        .await?;

                    // Deleted aggregation jobs cannot be stepped
                    if *aggregation_job.state() == AggregationJobState::Deleted {
                        return Err(datastore::Error::User(
                            Error::DeletedAggregationJob(*task.id(), *aggregation_job.id()).into(),
                        ));
                    }

                    // Check for a duplicate request, and treat it idempotently.
                    //
                    // If the leader's request is on the same step as our stored aggregation job,
                    // then we probably have already received this message and computed this step,
                    // but the leader never got our response and so retried stepping the job.
                    // TODO(issue #1087): measure how often this happens with a Prometheus metric
                    if aggregation_job.step() == req.step() {
                        match aggregation_job.last_request_hash() {
                            None => {
                                return Err(datastore::Error::User(
                                    Error::Internal(format!(
                                        "aggregation job {aggregation_job_id} is on step {} but \
                                         has no last request hash",
                                        aggregation_job.step(),
                                    ))
                                    .into(),
                                ));
                            }
                            Some(previous_hash) => {
                                if request_hash != previous_hash {
                                    return Err(datastore::Error::User(
                                        Error::ForbiddenMutation {
                                            resource_type: "aggregation job continuation",
                                            identifier: aggregation_job_id.to_string(),
                                        }
                                        .into(),
                                    ));
                                }
                            }
                        }

                        let resp = match task.aggregation_mode() {
                            Some(AggregationMode::Synchronous) => {
                                AggregationJobResp::Finished {
                                    prepare_resps: report_aggregations
                                        .iter()
                                        .filter_map(ReportAggregation::last_prep_resp)
                                        .cloned()
                                        .collect(),
                                }
                            }
                            Some(AggregationMode::Asynchronous) => {
                                AggregationJobResp::Processing
                            }
                            None => {
                                return Err(datastore::Error::User(
                                    Error::Internal("task has no aggregation mode".to_string())
                                        .into(),
                                ))
                            }
                        };

                        return Ok((resp, TaskAggregationCounter::default()));
                    }

                    if aggregation_job.step().increment() != req.step() {
                        // If this is not a replay, the leader should be advancing our state to the
                        // next step and no further.
                        return Err(datastore::Error::User(
                            Error::StepMismatch {
                                task_id: *task.id(),
                                aggregation_job_id,
                                expected_step: aggregation_job.step().increment(),
                                got_step: req.step(),
                            }
                            .into(),
                        ));
                    }

                    // Pair incoming preparation continuation messages with existing report
                    // aggregations.
                    let mut report_aggregations_to_write = Vec::with_capacity(report_aggregations.len());
                    let mut report_aggregations_iter = report_aggregations.into_iter();
                    let mut report_aggregations = Vec::with_capacity(req.prepare_continues().len());
                    for prepare_continue in req.prepare_continues() {
                        let report_aggregation = loop {
                            let report_aggregation = report_aggregations_iter.next().ok_or_else(|| {
                                datastore::Error::User(
                                    Error::InvalidMessage(
                                        Some(*task.id()),
                                        "leader sent unexpected, duplicate, or out-of-order prepare steps",
                                    )
                                    .into(),
                                )
                            })?;
                            if report_aggregation.report_id() != prepare_continue.report_id() {
                                // This report was omitted by the leader because of a prior failure.
                                // Note that the report was dropped (if it's not already in an error
                                // state) and continue.
                                if matches!(
                                    report_aggregation.state(),
                                    ReportAggregationState::HelperContinue { .. }
                                ) {
                                    report_aggregations_to_write.push(WritableReportAggregation::new(
                                        report_aggregation
                                            .with_state(ReportAggregationState::Failed {
                                                report_error: ReportError::ReportDropped,
                                            })
                                            .with_last_prep_resp(None),
                                        None,
                                    ));
                                }
                                continue;
                            }
                            break report_aggregation;
                        };

                        let prepare_state = if let ReportAggregationState::HelperContinue{ prepare_state } = report_aggregation.state() {
                            prepare_state.clone()
                        } else {
                            return Err(datastore::Error::User(
                                Error::InvalidMessage(
                                    Some(*task.id()),
                                    "leader sent prepare step for non-CONTINUE report aggregation",
                                )
                                .into(),
                            ))
                        };

                        report_aggregations.push(report_aggregation
                            .with_state(ReportAggregationState::HelperContinueProcessing {
                                prepare_state,
                                prepare_continue: prepare_continue.clone(),
                            })
                            .with_last_prep_resp(None)
                        );
                    }

                    for report_aggregation in report_aggregations_iter {
                        // This report was omitted by the leader because of a prior failure. Note
                        // that the report was dropped (if it's not already in an error state) and
                        // continue.
                        if matches!(
                            report_aggregation.state(),
                            ReportAggregationState::HelperContinue { .. }
                        ) {
                            report_aggregations_to_write.push(WritableReportAggregation::new(
                                report_aggregation
                                    .with_state(ReportAggregationState::Failed {
                                        report_error: ReportError::ReportDropped,
                                    })
                                    .with_last_prep_resp(None),
                                None,
                            ));
                        }
                    }

                    let aggregation_job = aggregation_job
                        .with_step(req.step()) // Advance the job to the leader's step
                        .with_last_request_hash(request_hash);

                    match task.aggregation_mode() {
                        Some(AggregationMode::Synchronous) => Self::handle_aggregate_continue_generic_sync(
                            tx,
                            task,
                            vdaf,
                            batch_aggregation_shard_count,
                            &metrics,
                            report_aggregations_to_write,
                            aggregation_job,
                            report_aggregations,
                        ).await,

                        Some(AggregationMode::Asynchronous) => Self::handle_aggregate_continue_generic_async(
                            tx,
                            task,
                            vdaf,
                            batch_aggregation_shard_count,
                            &metrics,
                            report_aggregations_to_write,
                            aggregation_job,
                            report_aggregations,
                        ).await,

                        None => Err(Error::Internal("task has no aggregation mode".to_string())),
                    }.map_err(|err| datastore::Error::User(err.into()))
                })
            })
            .await?;

        write_task_aggregation_counter(datastore, task_counter_shard_count, *task.id(), counters);

        Ok(response)
    }

    // All report aggregations must be in the HelperContinueProcessing state.
    async fn handle_aggregate_continue_generic_sync<
        const SEED_SIZE: usize,
        B: AccumulableBatchMode,
        A,
        C: Clock,
    >(
        tx: &Transaction<'_, C>,
        task: Arc<AggregatorTask>,
        vdaf: Arc<A>,
        batch_aggregation_shard_count: u64,
        metrics: &AggregatorMetrics,
        mut report_aggregations_to_write: Vec<WritableReportAggregation<SEED_SIZE, A>>,
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        report_aggregations: Vec<ReportAggregation<SEED_SIZE, A>>,
    ) -> Result<(AggregationJobResp, TaskAggregationCounter), Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        for<'a> A::PrepareState: Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)>,
        A::PrepareShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        A::PublicShare: Send + Sync,
        A::OutputShare: Send + Sync,
    {
        // Compute the next aggregation step.
        // TODO(#224): don't hold DB transaction open while computing VDAF updates?
        let aggregation_job = Arc::new(aggregation_job);
        report_aggregations_to_write.extend(
            compute_helper_aggregate_continue(
                Arc::clone(&vdaf),
                metrics.clone().into(),
                Arc::clone(&task),
                Arc::clone(&aggregation_job),
                report_aggregations,
            )
            .await,
        );

        // Store data to datastore.
        let (prepare_resps, counters) = Self::handle_aggregate_continue_generic_write(
            tx,
            task,
            vdaf,
            batch_aggregation_shard_count,
            metrics,
            Arc::unwrap_or_clone(aggregation_job),
            report_aggregations_to_write,
        )
        .await?;
        Ok((AggregationJobResp::Finished { prepare_resps }, counters))
    }

    // All report aggregations must be in the HelperContinueProcessing state.
    async fn handle_aggregate_continue_generic_async<
        const SEED_SIZE: usize,
        B: AccumulableBatchMode,
        A,
        C: Clock,
    >(
        tx: &Transaction<'_, C>,
        task: Arc<AggregatorTask>,
        vdaf: Arc<A>,
        batch_aggregation_shard_count: u64,
        metrics: &AggregatorMetrics,
        mut report_aggregations_to_write: Vec<WritableReportAggregation<SEED_SIZE, A>>,
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        report_aggregations: Vec<ReportAggregation<SEED_SIZE, A>>,
    ) -> Result<(AggregationJobResp, TaskAggregationCounter), Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        for<'a> A::PrepareState: Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)>,
        A::PrepareShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        A::PublicShare: Send + Sync,
        A::OutputShare: Send + Sync,
    {
        report_aggregations_to_write.extend(
            report_aggregations
                .into_iter()
                .map(|ra| WritableReportAggregation::new(ra, None)),
        );

        let (_, counters) = Self::handle_aggregate_continue_generic_write(
            tx,
            task,
            vdaf,
            batch_aggregation_shard_count,
            metrics,
            aggregation_job.with_state(AggregationJobState::Active),
            report_aggregations_to_write,
        )
        .await?;

        Ok((AggregationJobResp::Processing, counters))
    }

    async fn handle_aggregate_continue_generic_write<
        const SEED_SIZE: usize,
        B: AccumulableBatchMode,
        A,
        C: Clock,
    >(
        tx: &Transaction<'_, C>,
        task: Arc<AggregatorTask>,
        vdaf: Arc<A>,
        batch_aggregation_shard_count: u64,
        metrics: &AggregatorMetrics,
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        report_aggregations: Vec<WritableReportAggregation<SEED_SIZE, A>>,
    ) -> Result<(Vec<PrepareResp>, TaskAggregationCounter), Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        for<'a> A::PrepareState: Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)>,
        A::PrepareShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        A::PublicShare: Send + Sync,
        A::OutputShare: Send + Sync,
    {
        // Sanity-check that we have the correct number of report aggregations.
        assert_eq!(report_aggregations.len(), report_aggregations.capacity());

        // Write accumulated aggregation values back to the datastore; this will mark any reports
        // that can't be aggregated because the batch is collected with error BatchCollected.
        let aggregation_job_id = *aggregation_job.id();
        let mut aggregation_job_writer =
            AggregationJobWriter::<SEED_SIZE, _, _, UpdateWrite, _>::new(
                task,
                batch_aggregation_shard_count,
                Some(metrics.for_aggregation_job_writer()),
            );
        aggregation_job_writer.put(aggregation_job, report_aggregations)?;
        let (mut prep_resps_by_agg_job, counters) = aggregation_job_writer.write(tx, vdaf).await?;
        Ok((
            prep_resps_by_agg_job
                .remove(&aggregation_job_id)
                .unwrap_or_default(),
            counters,
        ))
    }

    /// Handle requests to the helper to get an aggregation job.
    async fn handle_aggregate_get_generic<
        const SEED_SIZE: usize,
        B: AccumulableBatchMode,
        A,
        C: Clock,
    >(
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        task: Arc<AggregatorTask>,
        aggregation_job_id: &AggregationJobId,
        step: AggregationJobStep,
    ) -> Result<AggregationJobResp, Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        for<'a> A::PrepareState: Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)>,
        A::PrepareShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        A::PublicShare: Send + Sync,
        A::OutputShare: Send + Sync,
    {
        Ok(datastore
            .run_tx("get_aggregation_job", |tx| {
                let vdaf = Arc::clone(&vdaf);
                let task = Arc::clone(&task);
                let aggregation_job_id = *aggregation_job_id;

                Box::pin(async move {
                    // Read aggregation job & report aggregations.
                    let aggregation_job = tx
                        .get_aggregation_job::<SEED_SIZE, B, A>(task.id(), &aggregation_job_id)
                        .await?
                        .ok_or_else(|| {
                            datastore::Error::User(
                                Error::UnrecognizedAggregationJob(*task.id(), aggregation_job_id)
                                    .into(),
                            )
                        })?;
                    let report_aggregations = tx
                        .get_report_aggregations_for_aggregation_job(
                            vdaf.as_ref(),
                            &Role::Helper,
                            task.id(),
                            &aggregation_job_id,
                            aggregation_job.aggregation_parameter(),
                        )
                        .await?;

                    // Validate that the request is for the expected step.
                    if aggregation_job.step() != step {
                        return Err(datastore::Error::User(
                            Error::StepMismatch {
                                task_id: *task.id(),
                                aggregation_job_id,
                                expected_step: aggregation_job.step(),
                                got_step: step,
                            }
                            .into(),
                        ));
                    }

                    // Return a value based on the report aggregations.
                    Ok(match aggregation_job.state() {
                        AggregationJobState::Active => AggregationJobResp::Processing,

                        AggregationJobState::AwaitingRequest | AggregationJobState::Finished => {
                            AggregationJobResp::Finished {
                                prepare_resps: report_aggregations
                                    .into_iter()
                                    .filter_map(|ra| ra.last_prep_resp().cloned())
                                    .collect(),
                            }
                        }

                        AggregationJobState::Abandoned => {
                            return Err(datastore::Error::User(
                                Error::AbandonedAggregationJob(*task.id(), *aggregation_job.id())
                                    .into(),
                            ))
                        }

                        AggregationJobState::Deleted => {
                            return Err(datastore::Error::User(
                                Error::DeletedAggregationJob(*task.id(), *aggregation_job.id())
                                    .into(),
                            ))
                        }
                    })
                })
            })
            .await?)
    }

    /// Handle requests to the helper to delete an aggregation job.
    async fn handle_aggregate_delete_generic<
        const SEED_SIZE: usize,
        B: AccumulableBatchMode,
        A,
        C: Clock,
    >(
        datastore: &Datastore<C>,
        task: Arc<AggregatorTask>,
        aggregation_job_id: &AggregationJobId,
    ) -> Result<(), Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        A::AggregationParam: Send + Sync,
        A::AggregateShare: Send + Sync,
        for<'a> A::PrepareState: Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)>,
        A::PrepareShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        A::OutputShare: Send + Sync,
    {
        Ok(datastore
            .run_tx("delete_aggregation_job", |tx| {
                let (task_id, aggregation_job_id) = (*task.id(), *aggregation_job_id);
                Box::pin(async move {
                    let aggregation_job = tx
                        .get_aggregation_job::<SEED_SIZE, B, A>(&task_id, &aggregation_job_id)
                        .await?
                        .ok_or_else(|| {
                            datastore::Error::User(
                                Error::UnrecognizedAggregationJob(task_id, aggregation_job_id)
                                    .into(),
                            )
                        })?
                        .with_state(AggregationJobState::Deleted);

                    tx.update_aggregation_job(&aggregation_job).await?;

                    Ok(())
                })
            })
            .await?)
    }

    /// Handle requests to the leader to create a collection job.
    #[tracing::instrument(
        skip(self, datastore, task, collection_req_bytes),
        fields(task_id = ?task.id()),
        err(level = Level::DEBUG)
    )]
    async fn handle_create_collection_job<C: Clock>(
        &self,
        datastore: &Datastore<C>,
        task: Arc<AggregatorTask>,
        collection_job_id: &CollectionJobId,
        collection_req_bytes: &[u8],
    ) -> Result<Vec<u8>, Error> {
        match task.batch_mode() {
            task::BatchMode::TimeInterval => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_create_collection_job_generic::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        VdafType,
                        _,
                    >(datastore, task, Arc::clone(vdaf), collection_job_id, collection_req_bytes)
                    .await
                })
            }
            task::BatchMode::LeaderSelected { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_create_collection_job_generic::<
                        VERIFY_KEY_LENGTH,
                        LeaderSelected,
                        VdafType,
                        _,
                    >(datastore, task, Arc::clone(vdaf), collection_job_id, collection_req_bytes)
                    .await
                })
            }
        }
    }

    async fn handle_create_collection_job_generic<
        const SEED_SIZE: usize,
        B: CollectableBatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        C: Clock,
    >(
        datastore: &Datastore<C>,
        task: Arc<AggregatorTask>,
        vdaf: Arc<A>,
        collection_job_id: &CollectionJobId,
        req_bytes: &[u8],
    ) -> Result<Vec<u8>, Error>
    where
        A::AggregationParam: 'static + Send + Sync + PartialEq + Eq + Hash,
        A::AggregateShare: Send + Sync,
    {
        let req =
            Arc::new(CollectionJobReq::<B>::get_decoded(req_bytes).map_err(Error::MessageDecode)?);
        let aggregation_param = Arc::new(
            A::AggregationParam::get_decoded(req.aggregation_parameter())
                .map_err(Error::MessageDecode)?,
        );

        datastore
            .run_tx("collect", move |tx| {
                let (task, vdaf, collection_job_id, req, aggregation_param) = (
                    Arc::clone(&task),
                    Arc::clone(&vdaf),
                    *collection_job_id,
                    Arc::clone(&req),
                    Arc::clone(&aggregation_param),
                );
                Box::pin(async move {
                    // Check if this collection job already exists, ensuring that all parameters match.
                    if let Some(collection_job) = tx
                        .get_collection_job::<SEED_SIZE, B, A>(&vdaf, task.id(), &collection_job_id)
                        .await?
                    {
                        if collection_job.query() == req.query()
                            && collection_job.aggregation_parameter() == aggregation_param.as_ref()
                        {
                            debug!(
                                collection_job_id = %collection_job_id,
                                collect_request = ?req,
                                "collection job already exists"
                            );
                            return Ok(());
                        } else {
                            return Err(datastore::Error::User(
                                Error::ForbiddenMutation {
                                    resource_type: "collection job",
                                    identifier: collection_job_id.to_string(),
                                }
                                .into(),
                            ));
                        }
                    }

                    let collection_identifier =
                        B::collection_identifier_for_query(tx, &task, req.query())
                            .await?
                            .ok_or_else(|| {
                                datastore::Error::User(
                                    Error::BatchInvalid(
                                        *task.id(),
                                        "no batch ready for collection".to_string(),
                                    )
                                    .into(),
                                )
                            })?;

                    // Check that the batch interval is valid for the task
                    // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.5.6.1.1
                    if !B::validate_collection_identifier(&task, &collection_identifier) {
                        return Err(datastore::Error::User(
                            Error::BatchInvalid(*task.id(), format!("{collection_identifier}"))
                                .into(),
                        ));
                    }

                    debug!(collect_request = ?req, "Cache miss, creating new collection job");
                    let (_, report_count) = try_join!(
                        B::validate_query_count::<SEED_SIZE, C, A>(
                            tx,
                            &vdaf,
                            &task,
                            &collection_identifier,
                            &aggregation_param,
                        ),
                        B::count_client_reports(tx, &task, &collection_identifier),
                    )?;

                    // Batch size must be validated while handling CollectReq and hence before
                    // creating a collection job.
                    // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.5.6
                    if !task.validate_batch_size(report_count) {
                        return Err(datastore::Error::User(
                            Error::InvalidBatchSize(*task.id(), report_count).into(),
                        ));
                    }

                    tx.put_collection_job(&CollectionJob::<SEED_SIZE, B, A>::new(
                        *task.id(),
                        collection_job_id,
                        req.query().clone(),
                        aggregation_param.as_ref().clone(),
                        collection_identifier,
                        CollectionJobState::Start,
                    ))
                    .await?;

                    Ok(())
                })
            })
            .await?;

        CollectionJobResp::<B>::Processing
            .get_encoded()
            .map_err(Error::MessageEncode)
    }

    /// Handle GET requests to the leader's `tasks/{task-id}/collection_jobs/{collection-job-id}`
    /// endpoint. The return value is an encoded `CollectResp<Q>`.
    /// <https://www.ietf.org/archive/id/draft-ietf-ppm-dap-07.html#name-collecting-results>
    #[tracing::instrument(skip(self, datastore, task), fields(task_id = ?task.id()), err(level = Level::DEBUG))]
    async fn handle_get_collection_job<C: Clock>(
        &self,
        datastore: &Datastore<C>,
        task: Arc<AggregatorTask>,
        collection_job_id: &CollectionJobId,
    ) -> Result<Vec<u8>, Error> {
        match task.batch_mode() {
            task::BatchMode::TimeInterval => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_get_collection_job_generic::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        VdafType,
                        _,
                    >(datastore, task, Arc::clone(vdaf), collection_job_id)
                    .await
                })
            }
            task::BatchMode::LeaderSelected { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_get_collection_job_generic::<
                        VERIFY_KEY_LENGTH,
                        LeaderSelected,
                        VdafType,
                        _,
                    >(datastore, task, Arc::clone(vdaf), collection_job_id)
                    .await
                })
            }
        }
    }

    // return value is an encoded CollectResp<Q>
    async fn handle_get_collection_job_generic<
        const SEED_SIZE: usize,
        B: CollectableBatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        C: Clock,
    >(
        datastore: &Datastore<C>,
        task: Arc<AggregatorTask>,
        vdaf: Arc<A>,
        collection_job_id: &CollectionJobId,
    ) -> Result<Vec<u8>, Error>
    where
        A::AggregationParam: Send + Sync,
        A::AggregateShare: Send + Sync,
    {
        let collection_job = datastore
            .run_tx("get_collection_job", |tx| {
                let (task, vdaf, collection_job_id) =
                    (Arc::clone(&task), Arc::clone(&vdaf), *collection_job_id);
                Box::pin(async move {
                    tx.get_collection_job::<SEED_SIZE, B, A>(&vdaf, task.id(), &collection_job_id)
                        .await?
                        .ok_or_else(|| {
                            datastore::Error::User(
                                Error::UnrecognizedCollectionJob(*task.id(), collection_job_id)
                                    .into(),
                            )
                        })
                })
            })
            .await?;

        match collection_job.state() {
            CollectionJobState::Start => {
                debug!(%collection_job_id, task_id = %task.id(), "collection job has not run yet");
                Ok(CollectionJobResp::<B>::Processing)
            }

            CollectionJobState::Finished {
                report_count,
                client_timestamp_interval,
                encrypted_helper_aggregate_share,
                leader_aggregate_share,
            } => {
                // §4.4.4.3: HPKE encrypt aggregate share to the collector. We store the leader
                // aggregate share *unencrypted* in the datastore so that we can encrypt cached
                // results to the collector HPKE config valid when the current collection job request
                // was made, and not whatever was valid at the time the aggregate share was first
                // computed.
                // However we store the helper's *encrypted* share.

                // TODO(#240): consider fetching freshly encrypted helper aggregate share if it has
                // been long enough since the encrypted helper share was cached -- tricky thing is
                // deciding what "long enough" is.
                debug!(
                    %collection_job_id,
                    task_id = %task.id(),
                    "Serving cached collection job response"
                );
                let encrypted_leader_aggregate_share = hpke::seal(
                    // Unwrap safety: collector_hpke_config is only None for taskprov tasks. Taskprov
                    // is not currently supported for Janus operating as the Leader, so this unwrap
                    // is not reachable.
                    task.collector_hpke_config().unwrap(),
                    &HpkeApplicationInfo::new(
                        &Label::AggregateShare,
                        &Role::Leader,
                        &Role::Collector,
                    ),
                    &leader_aggregate_share
                        .get_encoded()
                        .map_err(Error::MessageEncode)?,
                    &AggregateShareAad::new(
                        *collection_job.task_id(),
                        collection_job
                            .aggregation_parameter()
                            .get_encoded()
                            .map_err(Error::MessageEncode)?,
                        BatchSelector::<B>::new(collection_job.batch_identifier().clone()),
                    )
                    .get_encoded()
                    .map_err(Error::MessageEncode)?,
                )?;

                Ok(CollectionJobResp::<B>::Finished {
                    partial_batch_selector: PartialBatchSelector::new(
                        B::partial_batch_identifier(collection_job.batch_identifier()).clone(),
                    ),
                    report_count: *report_count,
                    interval: *client_timestamp_interval,
                    leader_encrypted_agg_share: encrypted_leader_aggregate_share,
                    helper_encrypted_agg_share: encrypted_helper_aggregate_share.clone(),
                })
            }

            CollectionJobState::Abandoned => Err(Error::AbandonedCollectionJob(
                *task.id(),
                *collection_job_id,
            )),

            CollectionJobState::Deleted => {
                Err(Error::DeletedCollectionJob(*task.id(), *collection_job_id))
            }
        }
        .and_then(|collection_job_resp| {
            collection_job_resp
                .get_encoded()
                .map_err(Error::MessageEncode)
        })
    }

    #[tracing::instrument(skip(self, datastore, task), fields(task_id = ?task.id()), err(level = Level::DEBUG))]
    async fn handle_delete_collection_job<C: Clock>(
        &self,
        datastore: &Datastore<C>,
        task: Arc<AggregatorTask>,
        collection_job_id: &CollectionJobId,
    ) -> Result<(), Error> {
        match task.batch_mode() {
            task::BatchMode::TimeInterval => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_delete_collection_job_generic::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        VdafType,
                        _,
                    >(datastore, task, Arc::clone(vdaf), collection_job_id)
                    .await
                })
            }
            task::BatchMode::LeaderSelected { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_delete_collection_job_generic::<
                        VERIFY_KEY_LENGTH,
                        LeaderSelected,
                        VdafType,
                        _,
                    >(datastore, task, Arc::clone(vdaf), collection_job_id)
                    .await
                })
            }
        }
    }

    async fn handle_delete_collection_job_generic<
        const SEED_SIZE: usize,
        B: CollectableBatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        C: Clock,
    >(
        datastore: &Datastore<C>,
        task: Arc<AggregatorTask>,
        vdaf: Arc<A>,
        collection_job_id: &CollectionJobId,
    ) -> Result<(), Error>
    where
        A::AggregationParam: Send + Sync,
        A::AggregateShare: Send + Sync + PartialEq + Eq,
    {
        datastore
            .run_tx("delete_collection_job", move |tx| {
                let (task, vdaf, collection_job_id) =
                    (Arc::clone(&task), Arc::clone(&vdaf), *collection_job_id);
                Box::pin(async move {
                    let collection_job = tx
                        .get_collection_job::<SEED_SIZE, B, A>(
                            vdaf.as_ref(),
                            task.id(),
                            &collection_job_id,
                        )
                        .await?
                        .ok_or_else(|| {
                            datastore::Error::User(
                                Error::UnrecognizedCollectionJob(*task.id(), collection_job_id)
                                    .into(),
                            )
                        })?;
                    if collection_job.state() != &CollectionJobState::Deleted {
                        tx.update_collection_job::<SEED_SIZE, B, A>(
                            &collection_job.with_state(CollectionJobState::Deleted),
                        )
                        .await?;
                    }
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Implements the `tasks/{task-id}/aggregate_shares` endpoint for the helper.
    #[tracing::instrument(
        skip(self, datastore, clock, task, req_bytes),
        fields(task_id = ?task.id()),
        err(level = Level::DEBUG)
    )]
    async fn handle_aggregate_share<C: Clock>(
        &self,
        datastore: &Datastore<C>,
        clock: &C,
        task: Arc<AggregatorTask>,
        batch_aggregation_shard_count: u64,
        req_bytes: &[u8],
        collector_hpke_config: &HpkeConfig,
    ) -> Result<AggregateShare, Error> {
        match task.batch_mode() {
            task::BatchMode::TimeInterval => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH, dp_strategy, DpStrategyType) => {
                    Self::handle_aggregate_share_generic::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        DpStrategyType,
                        VdafType,
                        _,
                    >(
                        datastore,
                        clock,
                        task,
                        Arc::clone(vdaf),
                        req_bytes,
                        batch_aggregation_shard_count,
                        collector_hpke_config,
                        Arc::clone(dp_strategy)
                    ).await
                })
            }
            task::BatchMode::LeaderSelected { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, VdafType, VERIFY_KEY_LENGTH, dp_strategy, DpStrategyType) => {
                    Self::handle_aggregate_share_generic::<
                        VERIFY_KEY_LENGTH,
                        LeaderSelected,
                        DpStrategyType,
                        VdafType,
                        _,
                    >(
                        datastore,
                        clock,
                        task,
                        Arc::clone(vdaf),
                        req_bytes,
                        batch_aggregation_shard_count,
                        collector_hpke_config,
                        Arc::clone(dp_strategy)
                    ).await
                })
            }
        }
    }

    async fn handle_aggregate_share_generic<
        const SEED_SIZE: usize,
        B: CollectableBatchMode,
        S: DifferentialPrivacyStrategy + Send + Clone + Send + Sync + 'static,
        A: vdaf::AggregatorWithNoise<SEED_SIZE, 16, S> + Send + Sync + 'static,
        C: Clock,
    >(
        datastore: &Datastore<C>,
        clock: &C,
        task: Arc<AggregatorTask>,
        vdaf: Arc<A>,
        req_bytes: &[u8],
        batch_aggregation_shard_count: u64,
        collector_hpke_config: &HpkeConfig,
        dp_strategy: Arc<S>,
    ) -> Result<AggregateShare, Error>
    where
        A::AggregationParam: Send + Sync + Eq + Hash,
        A::AggregateShare: Send + Sync,
    {
        // Decode request, and verify that it is for the current task. We use an assert to check
        // that the task IDs match as this should be guaranteed by the caller.
        let aggregate_share_req =
            Arc::new(AggregateShareReq::<B>::get_decoded(req_bytes).map_err(Error::MessageDecode)?);

        // §4.4.4.3: check that the batch interval meets the requirements from §4.6
        if !B::validate_collection_identifier(
            &task,
            aggregate_share_req.batch_selector().batch_identifier(),
        ) {
            return Err(Error::BatchInvalid(
                *task.id(),
                format!(
                    "{}",
                    aggregate_share_req.batch_selector().batch_identifier()
                ),
            ));
        }

        // Reject requests for aggregation shares that are eligible for GC, to prevent replay
        // attacks.
        if let Some(report_expiry_age) = task.report_expiry_age() {
            if let Some(batch_interval) =
                B::to_batch_interval(aggregate_share_req.batch_selector().batch_identifier())
            {
                let aggregate_share_expiry_time = batch_interval.end().add(report_expiry_age)?;
                if clock.now().is_after(&aggregate_share_expiry_time) {
                    return Err(Error::AggregateShareRequestRejected(
                        *task.id(),
                        "aggregate share request too late".to_string(),
                    ));
                }
            }
        }

        let aggregate_share_job = datastore
            .run_tx("aggregate_share", |tx| {
                let (task, vdaf, aggregate_share_req, dp_strategy) = (
                    Arc::clone(&task),
                    Arc::clone(&vdaf),
                    Arc::clone(&aggregate_share_req),
                    Arc::clone(&dp_strategy),
                );
                Box::pin(async move {
                    // Check if we have already serviced an aggregate share request with these
                    // parameters and serve the cached results if so.
                    let aggregation_param = A::AggregationParam::get_decoded(
                        aggregate_share_req.aggregation_parameter(),
                    )?;
                    if let Some(aggregate_share_job) = tx
                        .get_aggregate_share_job(
                            vdaf.as_ref(),
                            task.id(),
                            aggregate_share_req.batch_selector().batch_identifier(),
                            &aggregation_param,
                        )
                        .await?
                    {
                        debug!(
                            ?aggregate_share_req,
                            "Serving cached aggregate share job result"
                        );
                        return Ok(aggregate_share_job);
                    }

                    // This is a new aggregate share request, compute & validate the response.
                    debug!(
                        ?aggregate_share_req,
                        "Cache miss, computing aggregate share job result"
                    );
                    let aggregation_param = A::AggregationParam::get_decoded(
                        aggregate_share_req.aggregation_parameter(),
                    )?;
                    let (batch_aggregations, _) = try_join!(
                        B::get_batch_aggregations_for_collection_identifier(
                            tx,
                            task.id(),
                            task.time_precision(),
                            vdaf.as_ref(),
                            aggregate_share_req.batch_selector().batch_identifier(),
                            &aggregation_param
                        ),
                        B::validate_query_count::<SEED_SIZE, C, A>(
                            tx,
                            vdaf.as_ref(),
                            &task,
                            aggregate_share_req.batch_selector().batch_identifier(),
                            &aggregation_param,
                        )
                    )?;

                    // To ensure that concurrent aggregations don't write into a
                    // currently-nonexistent batch aggregation, we write (empty) batch
                    // aggregations for any that have not already been written to storage.
                    let empty_batch_aggregations = empty_batch_aggregations(
                        &task,
                        batch_aggregation_shard_count,
                        aggregate_share_req.batch_selector().batch_identifier(),
                        &aggregation_param,
                        &batch_aggregations,
                    );

                    let (mut helper_aggregate_share, report_count, _, checksum) =
                        compute_aggregate_share::<SEED_SIZE, B, A>(&task, &batch_aggregations)
                            .await
                            .map_err(|e| datastore::Error::User(e.into()))?;

                    vdaf.add_noise_to_agg_share(
                        &dp_strategy,
                        &aggregation_param,
                        &mut helper_aggregate_share,
                        report_count.try_into()?,
                    )
                    .map_err(|e| datastore::Error::User(e.into()))?;

                    // Now that we are satisfied that the request is serviceable, we consume
                    // a query by recording the aggregate share request parameters and the
                    // result.
                    let aggregate_share_job = AggregateShareJob::<SEED_SIZE, B, A>::new(
                        *task.id(),
                        aggregate_share_req
                            .batch_selector()
                            .batch_identifier()
                            .clone(),
                        aggregation_param,
                        helper_aggregate_share,
                        report_count,
                        checksum,
                    );
                    try_join!(
                        tx.put_aggregate_share_job(&aggregate_share_job),
                        try_join_all(batch_aggregations.into_iter().map(|ba| async move {
                            tx.update_batch_aggregation(&ba.scrubbed()).await
                        })),
                        try_join_all(empty_batch_aggregations.into_iter().map(|ba| async move {
                            tx.put_batch_aggregation(&ba.scrubbed()).await
                        }))
                    )?;
                    Ok(aggregate_share_job)
                })
            })
            .await?;

        // §4.4.4.3: Verify total report count and the checksum we computed against those reported
        // by the leader.
        if aggregate_share_job.report_count() != aggregate_share_req.report_count()
            || aggregate_share_job.checksum() != aggregate_share_req.checksum()
        {
            return Err(Error::BatchMismatch(Box::new(BatchMismatch {
                task_id: *task.id(),
                own_checksum: *aggregate_share_job.checksum(),
                own_report_count: aggregate_share_job.report_count(),
                peer_checksum: *aggregate_share_req.checksum(),
                peer_report_count: aggregate_share_req.report_count(),
            })));
        }

        // §4.4.4.3: HPKE encrypt aggregate share to the collector. We store *unencrypted* aggregate
        // shares in the datastore so that we can encrypt cached results to the collector HPKE
        // config valid when the current AggregateShareReq was made, and not whatever was valid at
        // the time the aggregate share was first computed.
        let encrypted_aggregate_share = hpke::seal(
            collector_hpke_config,
            &HpkeApplicationInfo::new(&Label::AggregateShare, &Role::Helper, &Role::Collector),
            &aggregate_share_job
                .helper_aggregate_share()
                .get_encoded()
                .map_err(Error::MessageEncode)?,
            &AggregateShareAad::new(
                *task.id(),
                aggregate_share_job
                    .aggregation_parameter()
                    .get_encoded()
                    .map_err(Error::MessageEncode)?,
                aggregate_share_req.batch_selector().clone(),
            )
            .get_encoded()
            .map_err(Error::MessageEncode)?,
        )?;

        Ok(AggregateShare::new(encrypted_aggregate_share))
    }
}

fn write_task_aggregation_counter<C: Clock>(
    datastore: Arc<Datastore<C>>,
    shard_count: u64,
    task_id: TaskId,
    counters: TaskAggregationCounter,
) {
    if counters.is_zero() {
        // Don't spawn a task or interact with the datastore if doing so won't change the state of
        // the datastore.
        return;
    }

    // We write task aggregation counters back in a separate tokio task & datastore transaction,
    // so that any slowness induced by writing the counters (e.g. due to transaction retry) does
    // not slow the main processing. The lack of transactionality between writing the updated
    // aggregation job & updating the counters means that process failure may cause us to leave
    // some counter updates unaccounted for, but that is an acceptable tradeoff.
    let ord = thread_rng().gen_range(0..shard_count);
    tokio::task::spawn(async move {
        let rslt = datastore
            .run_tx("update_task_aggregation_counters", |tx| {
                Box::pin(async move {
                    tx.increment_task_aggregation_counter(&task_id, ord, &counters)
                        .await
                })
            })
            .await;

        if let Err(err) = rslt {
            error!(
                ?task_id,
                ?err,
                "Couldn't increment task aggregation counter"
            );
        }
    });
}

fn empty_batch_aggregations<
    const SEED_SIZE: usize,
    B: CollectableBatchMode,
    A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
>(
    task: &AggregatorTask,
    batch_aggregation_shard_count: u64,
    batch_identifier: &B::BatchIdentifier,
    aggregation_param: &A::AggregationParam,
    batch_aggregations: &[BatchAggregation<SEED_SIZE, B, A>],
) -> Vec<BatchAggregation<SEED_SIZE, B, A>> {
    let existing_batch_aggregations: HashSet<_> = batch_aggregations
        .iter()
        .map(|ba| (ba.batch_identifier(), ba.ord()))
        .collect();
    iproduct!(
        B::batch_identifiers_for_collection_identifier(task.time_precision(), batch_identifier),
        0..batch_aggregation_shard_count
    )
    .filter_map(|(batch_identifier, ord)| {
        if !existing_batch_aggregations.contains(&(&batch_identifier, ord)) {
            Some(BatchAggregation::<SEED_SIZE, B, A>::new(
                *task.id(),
                batch_identifier,
                aggregation_param.clone(),
                ord,
                Interval::EMPTY,
                BatchAggregationState::Collected {
                    aggregate_share: None,
                    report_count: 0,
                    checksum: ReportIdChecksum::default(),
                    aggregation_jobs_created: 0,
                    aggregation_jobs_terminated: 0,
                },
            ))
        } else {
            None
        }
    })
    .collect()
}

#[derive(Clone)]
struct RequestBody {
    content_type: &'static str,
    body: Bytes,
}

struct RequestTimer<'a> {
    // Mutable state.
    start: SyncMutex<Option<Instant>>,

    // Immutable state.
    http_request_duration_histogram: &'a Histogram<f64>,
    domain: Arc<str>,
    endpoint: &'static str,
    method: Arc<str>,
}

impl<'a> RequestTimer<'a> {
    fn new(
        http_request_duration_histogram: &'a Histogram<f64>,
        domain: Arc<str>,
        endpoint: &'static str,
        method: Arc<str>,
    ) -> Self {
        Self {
            start: SyncMutex::new(None),
            http_request_duration_histogram,
            domain,
            endpoint,
            method,
        }
    }

    fn start_attempt(&self) {
        *self.start.lock().unwrap() = Some(Instant::now())
    }

    fn finish_attempt(&self, status: &'static str) {
        let start = self
            .start
            .lock()
            .unwrap()
            .take()
            .expect("RequestTimer: finish_attempt called without calling start_attempt");
        self.http_request_duration_histogram.record(
            start.elapsed().as_secs_f64(),
            &[
                KeyValue::new("status", status),
                KeyValue::new("domain", Arc::clone(&self.domain)),
                KeyValue::new("endpoint", self.endpoint),
                KeyValue::new("method", Arc::clone(&self.method)),
            ],
        )
    }
}

impl<E> Notify<E> for &RequestTimer<'_> {
    fn notify(&mut self, _: E, _: std::time::Duration) {
        self.finish_attempt("error")
    }
}

/// Convenience method to perform an HTTP request to the helper. This includes common
/// metrics and error handling functionality.
#[tracing::instrument(
    skip(
        http_client,
        backoff,
        url,
        request_body,
        auth_token,
        http_request_duration_histogram,
    ),
    fields(url = %url),
    err(level = Level::DEBUG),
)]
async fn send_request_to_helper(
    http_client: &Client,
    backoff: impl Backoff,
    method: Method,
    url: Url,
    route_label: &'static str,
    request_body: Option<RequestBody>,
    auth_token: &AuthenticationToken,
    http_request_duration_histogram: &Histogram<f64>,
) -> Result<HttpResponse, Error> {
    let (auth_header, auth_value) = auth_token.request_authentication();
    let domain = Arc::from(url.domain().unwrap_or_default());
    let method_str = Arc::from(method.as_str());
    let timer = RequestTimer::new(
        http_request_duration_histogram,
        domain,
        route_label,
        method_str,
    );

    let result = retry_http_request_notify(backoff, &timer, || async {
        timer.start_attempt();
        let mut request = http_client
            .request(method.clone(), url.clone())
            .header(auth_header, auth_value.as_str());
        if let Some(request_body) = request_body.clone() {
            request = request
                .header(CONTENT_TYPE, request_body.content_type)
                .body(request_body.body)
        };
        request.send().await
    })
    .await;

    match result {
        // Successful response.
        Ok(response) => {
            timer.finish_attempt("success");
            Ok(response)
        }

        // HTTP-level error.
        Err(Ok(http_error_response)) => {
            timer.finish_attempt("error");
            Err(Error::Http(Box::new(http_error_response)))
        }

        // Network-level error.
        Err(Err(error)) => {
            timer.finish_attempt("error");
            Err(error.into())
        }
    }
}
