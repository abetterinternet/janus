//! Common functionality for DAP aggregators.

pub use crate::aggregator::error::Error;
use crate::{
    aggregator::{
        aggregate_share::compute_aggregate_share,
        error::BatchMismatch,
        query_type::{CollectableQueryType, UploadableQueryType},
        report_writer::{ReportWriteBatcher, WritableReport},
    },
    Operation,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use bytes::Bytes;
#[cfg(feature = "fpvec_bounded_l2")]
use fixed::{
    types::extra::{U15, U31, U63},
    FixedI16, FixedI32, FixedI64,
};
use futures::future::try_join_all;
use http::{header::CONTENT_TYPE, Method};
use itertools::iproduct;
use janus_aggregator_core::{
    datastore::{
        self,
        models::{
            AggregateShareJob, AggregationJob, AggregationJobState, Batch, BatchAggregation,
            BatchAggregationState, BatchState, CollectionJob, CollectionJobState,
            LeaderStoredReport, ReportAggregation, ReportAggregationState,
        },
        Datastore, Transaction,
    },
    query_type::AccumulableQueryType,
    task::{self, Task, VerifyKey},
};
#[cfg(feature = "test-util")]
use janus_core::test_util::dummy_vdaf;
use janus_core::{
    hpke::{self, HpkeApplicationInfo, Label},
    http::response_to_problem_details,
    task::{AuthenticationToken, VdafInstance, PRIO3_VERIFY_KEY_LENGTH},
    time::{Clock, DurationExt, IntervalExt, TimeExt},
};
use janus_messages::{
    problem_type::DapProblemType, query_type::FixedSize, AggregateShare, AggregateShareAad,
    AggregateShareReq, AggregationJobContinueReq, AggregationJobId, AggregationJobInitializeReq,
    AggregationJobResp, AggregationJobRound, BatchSelector, Collection, CollectionJobId,
    CollectionReq, Duration, HpkeCiphertext, HpkeConfigList, InputShareAad, Interval,
    PartialBatchSelector, PlaintextInputShare, PrepareStep, PrepareStepResult, Report,
    ReportIdChecksum, ReportShare, ReportShareError, Role, TaskId,
};
use opentelemetry::{
    metrics::{Counter, Histogram, Meter},
    Context, KeyValue,
};
#[cfg(feature = "fpvec_bounded_l2")]
use prio::vdaf::prio3::Prio3FixedPointBoundedL2VecSumMultithreaded;
#[cfg(feature = "test-util")]
use prio::vdaf::{PrepareTransition, VdafError};
use prio::{
    codec::{Decode, Encode, ParameterizedDecode},
    vdaf::{
        self,
        prio3::{Prio3, Prio3Count, Prio3Histogram, Prio3Sum, Prio3SumVecMultithreaded},
    },
};
use reqwest::Client;
use ring::digest::{digest, SHA256};
use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    fmt::Debug,
    panic,
    sync::Arc,
    time::{Duration as StdDuration, Instant},
};
use tokio::{sync::Mutex, try_join};
use tracing::{debug, info, trace_span, warn};
use url::Url;

pub mod accumulator;
#[cfg(test)]
mod aggregate_init_tests;
pub mod aggregate_share;
pub mod aggregation_job_continue;
pub mod aggregation_job_creator;
pub mod aggregation_job_driver;
pub mod aggregation_job_writer;
pub mod collection_job_driver;
#[cfg(test)]
mod collection_job_tests;
mod error;
pub mod garbage_collector;
pub mod http_handlers;
pub mod problem_details;
pub mod query_type;
pub mod report_writer;

pub(crate) fn aggregate_step_failure_counter(meter: &Meter) -> Counter<u64> {
    let aggregate_step_failure_counter = meter
        .u64_counter("janus_step_failures")
        .with_description(concat!(
            "Failures while stepping aggregation jobs; these failures are ",
            "related to individual client reports rather than entire aggregation jobs."
        ))
        .init();

    // Initialize counters with desired status labels. This causes Prometheus to see the first
    // non-zero value we record.
    for failure_type in [
        "missing_prepare_message",
        "missing_leader_input_share",
        "missing_helper_input_share",
        "prepare_init_failure",
        "prepare_step_failure",
        "prepare_message_failure",
        "unknown_hpke_config_id",
        "decrypt_failure",
        "input_share_decode_failure",
        "public_share_decode_failure",
        "continue_mismatch",
        "accumulate_failure",
        "finish_mismatch",
        "helper_step_failure",
        "plaintext_input_share_decode_failure",
        "duplicate_extension",
        "missing_client_report",
    ] {
        aggregate_step_failure_counter.add(
            &Context::current(),
            0,
            &[KeyValue::new("type", failure_type)],
        );
    }

    aggregate_step_failure_counter
}

/// Aggregator implements a DAP aggregator.
pub struct Aggregator<C: Clock> {
    /// Datastore used for durable storage.
    datastore: Arc<Datastore<C>>,
    /// Clock used to sample time.
    clock: C,
    /// Configuration used for this aggregator.
    cfg: Config,
    /// Report writer, with support for batching.
    report_writer: Arc<ReportWriteBatcher<C>>,
    /// Cache of task aggregators.
    task_aggregators: Mutex<HashMap<TaskId, Arc<TaskAggregator<C>>>>,

    // Metrics.
    /// Counter tracking the number of failed decryptions while handling the /upload endpoint.
    upload_decrypt_failure_counter: Counter<u64>,
    /// Counter tracking the number of failed message decodes while handling the /upload endpoint.
    upload_decode_failure_counter: Counter<u64>,
    /// Counters tracking the number of failures to step client reports through the aggregation
    /// process.
    aggregate_step_failure_counter: Counter<u64>,
}

/// Config represents a configuration for an Aggregator.
#[derive(Debug, PartialEq, Eq)]
pub struct Config {
    /// Defines the maximum size of a batch of uploaded reports which will be written in a single
    /// transaction.
    pub max_upload_batch_size: usize,

    /// Defines the maximum delay before writing a batch of uploaded reports, even if it has not yet
    /// reached `max_batch_upload_size`. This is the maximum delay added to the /upload endpoint due
    /// to write-batching.
    pub max_upload_batch_write_delay: StdDuration,

    /// Defines the number of shards to break each batch aggregation into. Increasing this value
    /// will reduce the amount of database contention during helper aggregation, while increasing
    /// the cost of collection.
    pub batch_aggregation_shard_count: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_upload_batch_size: 1,
            max_upload_batch_write_delay: StdDuration::ZERO,
            batch_aggregation_shard_count: 1,
        }
    }
}

impl<C: Clock> Aggregator<C> {
    fn new(datastore: Arc<Datastore<C>>, clock: C, meter: &Meter, cfg: Config) -> Self {
        let report_writer = Arc::new(ReportWriteBatcher::new(
            Arc::clone(&datastore),
            cfg.max_upload_batch_size,
            cfg.max_upload_batch_write_delay,
        ));

        let upload_decrypt_failure_counter = meter
            .u64_counter("janus_upload_decrypt_failures")
            .with_description("Number of decryption failures in the /upload endpoint.")
            .init();
        upload_decrypt_failure_counter.add(&Context::current(), 0, &[]);

        let upload_decode_failure_counter = meter
            .u64_counter("janus_upload_decode_failures")
            .with_description("Number of message decode failures in the /upload endpoint.")
            .init();
        upload_decode_failure_counter.add(&Context::current(), 0, &[]);

        let aggregate_step_failure_counter = aggregate_step_failure_counter(meter);
        aggregate_step_failure_counter.add(&Context::current(), 0, &[]);

        Self {
            datastore,
            clock,
            cfg,
            report_writer,
            task_aggregators: Mutex::new(HashMap::new()),
            upload_decrypt_failure_counter,
            upload_decode_failure_counter,
            aggregate_step_failure_counter,
        }
    }

    async fn handle_hpke_config(
        &self,
        task_id_base64: Option<&[u8]>,
    ) -> Result<HpkeConfigList, Error> {
        // Task ID is optional in an HPKE config request, but Janus requires it.
        // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.3.1
        let task_id_base64 = task_id_base64.ok_or(Error::MissingTaskId)?;

        let task_id_bytes = URL_SAFE_NO_PAD
            .decode(task_id_base64)
            .map_err(|_| Error::UnrecognizedMessage(None, "task_id"))?;
        let task_id = TaskId::get_decoded(&task_id_bytes)
            .map_err(|_| Error::UnrecognizedMessage(None, "task_id"))?;
        let task_aggregator = self.task_aggregator_for(&task_id).await?;
        Ok(task_aggregator.handle_hpke_config())
    }

    async fn handle_upload(&self, task_id: &TaskId, report_bytes: &[u8]) -> Result<(), Arc<Error>> {
        let report = Report::get_decoded(report_bytes).map_err(|err| Arc::new(Error::from(err)))?;

        let task_aggregator = self.task_aggregator_for(task_id).await?;
        if task_aggregator.task.role() != &Role::Leader {
            return Err(Arc::new(Error::UnrecognizedTask(*task_id)));
        }
        task_aggregator
            .handle_upload(
                &self.clock,
                &self.upload_decrypt_failure_counter,
                &self.upload_decode_failure_counter,
                report,
            )
            .await
    }

    async fn handle_aggregate_init(
        &self,
        task_id: &TaskId,
        aggregation_job_id: &AggregationJobId,
        req_bytes: &[u8],
        auth_token: Option<AuthenticationToken>,
    ) -> Result<AggregationJobResp, Error> {
        let task_aggregator = self.task_aggregator_for(task_id).await?;
        if task_aggregator.task.role() != &Role::Helper {
            return Err(Error::UnrecognizedTask(*task_id));
        }
        if !auth_token
            .map(|t| task_aggregator.task.check_aggregator_auth_token(&t))
            .unwrap_or(false)
        {
            return Err(Error::UnauthorizedRequest(*task_id));
        }

        task_aggregator
            .handle_aggregate_init(
                &self.datastore,
                &self.aggregate_step_failure_counter,
                aggregation_job_id,
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
    ) -> Result<AggregationJobResp, Error> {
        let task_aggregator = self.task_aggregator_for(task_id).await?;
        if task_aggregator.task.role() != &Role::Helper {
            return Err(Error::UnrecognizedTask(*task_id));
        }
        if !auth_token
            .map(|t| task_aggregator.task.check_aggregator_auth_token(&t))
            .unwrap_or(false)
        {
            return Err(Error::UnauthorizedRequest(*task_id));
        }

        let req = AggregationJobContinueReq::get_decoded(req_bytes)?;
        // unwrap safety: SHA-256 computed by ring should always be 32 bytes
        let request_hash = digest(&SHA256, req_bytes).as_ref().try_into().unwrap();

        task_aggregator
            .handle_aggregate_continue(
                &self.datastore,
                &self.aggregate_step_failure_counter,
                self.cfg.batch_aggregation_shard_count,
                aggregation_job_id,
                req,
                request_hash,
            )
            .await
    }

    /// Handle a collection job creation request. Only supported by the leader. `req_bytes` is an
    /// encoded [`CollectionReq`].
    async fn handle_create_collection_job(
        &self,
        task_id: &TaskId,
        collection_job_id: &CollectionJobId,
        req_bytes: &[u8],
        auth_token: Option<AuthenticationToken>,
    ) -> Result<(), Error> {
        let task_aggregator = self.task_aggregator_for(task_id).await?;
        if task_aggregator.task.role() != &Role::Leader {
            return Err(Error::UnrecognizedTask(*task_id));
        }
        if !auth_token
            .map(|t| task_aggregator.task.check_collector_auth_token(&t))
            .unwrap_or(false)
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
    ) -> Result<Option<Vec<u8>>, Error> {
        let task_aggregator = self.task_aggregator_for(task_id).await?;
        if task_aggregator.task.role() != &Role::Leader {
            return Err(Error::UnrecognizedTask(*task_id));
        }
        if !auth_token
            .map(|t| task_aggregator.task.check_collector_auth_token(&t))
            .unwrap_or(false)
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
        let task_aggregator = self.task_aggregator_for(task_id).await?;
        if task_aggregator.task.role() != &Role::Leader {
            return Err(Error::UnrecognizedTask(*task_id));
        }
        if !auth_token
            .map(|t| task_aggregator.task.check_collector_auth_token(&t))
            .unwrap_or(false)
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
    ) -> Result<AggregateShare, Error> {
        let task_aggregator = self.task_aggregator_for(task_id).await?;
        if task_aggregator.task.role() != &Role::Helper {
            return Err(Error::UnrecognizedTask(*task_id));
        }
        if !auth_token
            .map(|t| task_aggregator.task.check_aggregator_auth_token(&t))
            .unwrap_or(false)
        {
            return Err(Error::UnauthorizedRequest(*task_id));
        }

        task_aggregator
            .handle_aggregate_share(
                &self.datastore,
                &self.clock,
                self.cfg.batch_aggregation_shard_count,
                req_bytes,
            )
            .await
    }

    async fn task_aggregator_for(&self, task_id: &TaskId) -> Result<Arc<TaskAggregator<C>>, Error> {
        // TODO(#238): don't cache forever (decide on & implement some cache eviction policy).
        // This is important both to avoid ever-growing resource usage, and to allow aggregators to
        // notice when a task changes (e.g. due to key rotation).

        // Fast path: grab an existing task aggregator if one exists for this task.
        {
            let task_aggs = self.task_aggregators.lock().await;
            if let Some(task_agg) = task_aggs.get(task_id) {
                return Ok(Arc::clone(task_agg));
            }
        }

        // Slow path: retrieve task, create a task aggregator, store it to the cache, then return it.
        let task = self
            .datastore
            .run_tx_with_name("task_aggregator_get_task", |tx| {
                let task_id = *task_id;
                Box::pin(async move { tx.get_task(&task_id).await })
            })
            .await?
            .ok_or(Error::UnrecognizedTask(*task_id))?;
        let task_agg = Arc::new(TaskAggregator::new(task, Arc::clone(&self.report_writer))?);
        {
            let mut task_aggs = self.task_aggregators.lock().await;
            Ok(Arc::clone(task_aggs.entry(*task_id).or_insert(task_agg)))
        }
    }
}

/// TaskAggregator provides aggregation functionality for a single task.
// TODO(#224): refactor Aggregator to perform indepedent batched operations (e.g. report handling in
// Aggregate requests) using a parallelized library like Rayon.
pub struct TaskAggregator<C: Clock> {
    /// The task being aggregated.
    task: Arc<Task>,
    /// VDAF-specific operations.
    vdaf_ops: VdafOps,
    /// Report writer, with support for batching.
    report_writer: Arc<ReportWriteBatcher<C>>,
}

impl<C: Clock> TaskAggregator<C> {
    /// Create a new aggregator. `report_recipient` is used to decrypt reports received by this
    /// aggregator.
    fn new(task: Task, report_writer: Arc<ReportWriteBatcher<C>>) -> Result<Self, Error> {
        let vdaf_ops = match task.vdaf() {
            VdafInstance::Prio3Count => {
                let vdaf = Prio3::new_count(2)?;
                let verify_key = task.primary_vdaf_verify_key()?;
                VdafOps::Prio3Count(Arc::new(vdaf), verify_key)
            }

            VdafInstance::Prio3CountVec { length } => {
                let vdaf = Prio3::new_sum_vec_multithreaded(2, 1, *length)?;
                let verify_key = task.primary_vdaf_verify_key()?;
                VdafOps::Prio3CountVec(Arc::new(vdaf), verify_key)
            }

            VdafInstance::Prio3Sum { bits } => {
                let vdaf = Prio3::new_sum(2, *bits)?;
                let verify_key = task.primary_vdaf_verify_key()?;
                VdafOps::Prio3Sum(Arc::new(vdaf), verify_key)
            }

            VdafInstance::Prio3SumVec { bits, length } => {
                let vdaf = Prio3::new_sum_vec_multithreaded(2, *bits, *length)?;
                let verify_key = task.primary_vdaf_verify_key()?;
                VdafOps::Prio3SumVec(Arc::new(vdaf), verify_key)
            }

            VdafInstance::Prio3Histogram { buckets } => {
                let vdaf = Prio3::new_histogram(2, buckets)?;
                let verify_key = task.primary_vdaf_verify_key()?;
                VdafOps::Prio3Histogram(Arc::new(vdaf), verify_key)
            }

            #[cfg(feature = "fpvec_bounded_l2")]
            VdafInstance::Prio3FixedPoint16BitBoundedL2VecSum { length } => {
                let vdaf: Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI16<U15>> =
                    Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(2, *length)?;
                let verify_key = task.primary_vdaf_verify_key()?;
                VdafOps::Prio3FixedPoint16BitBoundedL2VecSum(Arc::new(vdaf), verify_key)
            }

            #[cfg(feature = "fpvec_bounded_l2")]
            VdafInstance::Prio3FixedPoint32BitBoundedL2VecSum { length } => {
                let vdaf: Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI32<U31>> =
                    Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(2, *length)?;
                let verify_key = task.primary_vdaf_verify_key()?;
                VdafOps::Prio3FixedPoint32BitBoundedL2VecSum(Arc::new(vdaf), verify_key)
            }

            #[cfg(feature = "fpvec_bounded_l2")]
            VdafInstance::Prio3FixedPoint64BitBoundedL2VecSum { length } => {
                let vdaf: Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI64<U63>> =
                    Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(2, *length)?;
                let verify_key = task.primary_vdaf_verify_key()?;
                VdafOps::Prio3FixedPoint64BitBoundedL2VecSum(Arc::new(vdaf), verify_key)
            }

            #[cfg(feature = "test-util")]
            VdafInstance::Fake => VdafOps::Fake(Arc::new(dummy_vdaf::Vdaf::new())),

            #[cfg(feature = "test-util")]
            VdafInstance::FakeFailsPrepInit => VdafOps::Fake(Arc::new(
                dummy_vdaf::Vdaf::new().with_prep_init_fn(|_| -> Result<(), VdafError> {
                    Err(VdafError::Uncategorized(
                        "FakeFailsPrepInit failed at prep_init".to_string(),
                    ))
                }),
            )),

            #[cfg(feature = "test-util")]
            VdafInstance::FakeFailsPrepStep => {
                VdafOps::Fake(Arc::new(dummy_vdaf::Vdaf::new().with_prep_step_fn(
                    || -> Result<PrepareTransition<dummy_vdaf::Vdaf, 0, 16>, VdafError> {
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

    fn handle_hpke_config(&self) -> HpkeConfigList {
        // TODO(#239): consider deciding a better way to determine "primary" (e.g. most-recent) HPKE
        // config/key -- right now it's the one with the maximal config ID, but that will run into
        // trouble if we ever need to wrap-around, which we may since config IDs are effectively a u8.
        HpkeConfigList::new(Vec::from([self
            .task
            .hpke_keys()
            .iter()
            .max_by_key(|(&id, _)| id)
            .unwrap()
            .1
            .config()
            .clone()]))
    }

    async fn handle_upload(
        &self,
        clock: &C,
        upload_decrypt_failure_counter: &Counter<u64>,
        upload_decode_failure_counter: &Counter<u64>,
        report: Report,
    ) -> Result<(), Arc<Error>> {
        self.vdaf_ops
            .handle_upload(
                clock,
                upload_decrypt_failure_counter,
                upload_decode_failure_counter,
                &self.task,
                &self.report_writer,
                report,
            )
            .await
    }

    async fn handle_aggregate_init(
        &self,
        datastore: &Datastore<C>,
        aggregate_step_failure_counter: &Counter<u64>,
        aggregation_job_id: &AggregationJobId,
        req_bytes: &[u8],
    ) -> Result<AggregationJobResp, Error> {
        self.vdaf_ops
            .handle_aggregate_init(
                datastore,
                aggregate_step_failure_counter,
                Arc::clone(&self.task),
                aggregation_job_id,
                req_bytes,
            )
            .await
    }

    async fn handle_aggregate_continue(
        &self,
        datastore: &Datastore<C>,
        aggregate_step_failure_counter: &Counter<u64>,
        batch_aggregation_shard_count: u64,
        aggregation_job_id: &AggregationJobId,
        req: AggregationJobContinueReq,
        request_hash: [u8; 32],
    ) -> Result<AggregationJobResp, Error> {
        self.vdaf_ops
            .handle_aggregate_continue(
                datastore,
                aggregate_step_failure_counter,
                Arc::clone(&self.task),
                batch_aggregation_shard_count,
                aggregation_job_id,
                Arc::new(req),
                request_hash,
            )
            .await
    }

    async fn handle_create_collection_job(
        &self,
        datastore: &Datastore<C>,
        collection_job_id: &CollectionJobId,
        req_bytes: &[u8],
    ) -> Result<(), Error> {
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
    ) -> Result<Option<Vec<u8>>, Error> {
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
    ) -> Result<AggregateShare, Error> {
        self.vdaf_ops
            .handle_aggregate_share(
                datastore,
                clock,
                Arc::clone(&self.task),
                batch_aggregation_shard_count,
                req_bytes,
            )
            .await
    }
}

/// VdafOps stores VDAF-specific operations for a TaskAggregator in a non-generic way.
#[allow(clippy::enum_variant_names)]
enum VdafOps {
    Prio3Count(Arc<Prio3Count>, VerifyKey<PRIO3_VERIFY_KEY_LENGTH>),
    Prio3CountVec(
        Arc<Prio3SumVecMultithreaded>,
        VerifyKey<PRIO3_VERIFY_KEY_LENGTH>,
    ),
    Prio3Sum(Arc<Prio3Sum>, VerifyKey<PRIO3_VERIFY_KEY_LENGTH>),
    Prio3SumVec(
        Arc<Prio3SumVecMultithreaded>,
        VerifyKey<PRIO3_VERIFY_KEY_LENGTH>,
    ),
    Prio3Histogram(Arc<Prio3Histogram>, VerifyKey<PRIO3_VERIFY_KEY_LENGTH>),
    #[cfg(feature = "fpvec_bounded_l2")]
    Prio3FixedPoint16BitBoundedL2VecSum(
        Arc<Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI16<U15>>>,
        VerifyKey<PRIO3_VERIFY_KEY_LENGTH>,
    ),
    #[cfg(feature = "fpvec_bounded_l2")]
    Prio3FixedPoint32BitBoundedL2VecSum(
        Arc<Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI32<U31>>>,
        VerifyKey<PRIO3_VERIFY_KEY_LENGTH>,
    ),
    #[cfg(feature = "fpvec_bounded_l2")]
    Prio3FixedPoint64BitBoundedL2VecSum(
        Arc<Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI64<U63>>>,
        VerifyKey<PRIO3_VERIFY_KEY_LENGTH>,
    ),

    #[cfg(feature = "test-util")]
    Fake(Arc<dummy_vdaf::Vdaf>),
}

/// Emits a match block dispatching on a [`VdafOps`] object. Takes a `&VdafOps` as the first
/// argument, followed by a pseudo-pattern and body. The pseudo-pattern takes variable names for the
/// constructed VDAF and the verify key, a type alias name that the block can use to explicitly
/// specify the VDAF's type, and the name of a const that will be set to the VDAF's verify key
/// length, also for explicitly specifying type parameters.
macro_rules! vdaf_ops_dispatch {
    ($vdaf_ops:expr, ($vdaf:pat_param, $verify_key:pat_param, $Vdaf:ident, $VERIFY_KEY_LENGTH:ident) => $body:tt) => {
        match $vdaf_ops {
            crate::aggregator::VdafOps::Prio3Count(vdaf, verify_key) => {
                let $vdaf = vdaf;
                let $verify_key = verify_key;
                type $Vdaf = ::prio::vdaf::prio3::Prio3Count;
                const $VERIFY_KEY_LENGTH: usize = ::janus_core::task::PRIO3_VERIFY_KEY_LENGTH;
                $body
            }

            crate::aggregator::VdafOps::Prio3CountVec(vdaf, verify_key) => {
                let $vdaf = vdaf;
                let $verify_key = verify_key;
                type $Vdaf = ::prio::vdaf::prio3::Prio3SumVecMultithreaded;
                const $VERIFY_KEY_LENGTH: usize = ::janus_core::task::PRIO3_VERIFY_KEY_LENGTH;
                $body
            }

            crate::aggregator::VdafOps::Prio3Sum(vdaf, verify_key) => {
                let $vdaf = vdaf;
                let $verify_key = verify_key;
                type $Vdaf = ::prio::vdaf::prio3::Prio3Sum;
                const $VERIFY_KEY_LENGTH: usize = ::janus_core::task::PRIO3_VERIFY_KEY_LENGTH;
                $body
            }

            crate::aggregator::VdafOps::Prio3SumVec(vdaf, verify_key) => {
                let $vdaf = vdaf;
                let $verify_key = verify_key;
                type $Vdaf = ::prio::vdaf::prio3::Prio3SumVecMultithreaded;
                const $VERIFY_KEY_LENGTH: usize = ::janus_core::task::PRIO3_VERIFY_KEY_LENGTH;
                $body
            }

            crate::aggregator::VdafOps::Prio3Histogram(vdaf, verify_key) => {
                let $vdaf = vdaf;
                let $verify_key = verify_key;
                type $Vdaf = ::prio::vdaf::prio3::Prio3Histogram;
                const $VERIFY_KEY_LENGTH: usize = ::janus_core::task::PRIO3_VERIFY_KEY_LENGTH;
                $body
            }

            #[cfg(feature = "fpvec_bounded_l2")]
            crate::aggregator::VdafOps::Prio3FixedPoint16BitBoundedL2VecSum(vdaf, verify_key) => {
                let $vdaf = vdaf;
                let $verify_key = verify_key;
                type $Vdaf =
                    ::prio::vdaf::prio3::Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI16<U15>>;
                const $VERIFY_KEY_LENGTH: usize = ::janus_core::task::PRIO3_VERIFY_KEY_LENGTH;
                $body
            }

            #[cfg(feature = "fpvec_bounded_l2")]
            crate::aggregator::VdafOps::Prio3FixedPoint32BitBoundedL2VecSum(vdaf, verify_key) => {
                let $vdaf = vdaf;
                let $verify_key = verify_key;
                type $Vdaf =
                    ::prio::vdaf::prio3::Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI32<U31>>;
                const $VERIFY_KEY_LENGTH: usize = ::janus_core::task::PRIO3_VERIFY_KEY_LENGTH;
                $body
            }

            #[cfg(feature = "fpvec_bounded_l2")]
            crate::aggregator::VdafOps::Prio3FixedPoint64BitBoundedL2VecSum(vdaf, verify_key) => {
                let $vdaf = vdaf;
                let $verify_key = verify_key;
                type $Vdaf =
                    ::prio::vdaf::prio3::Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI64<U63>>;
                const $VERIFY_KEY_LENGTH: usize = ::janus_core::task::PRIO3_VERIFY_KEY_LENGTH;
                $body
            }

            #[cfg(feature = "test-util")]
            crate::aggregator::VdafOps::Fake(vdaf) => {
                let $vdaf = vdaf;
                let $verify_key = &VerifyKey::new([]);
                type $Vdaf = ::janus_core::test_util::dummy_vdaf::Vdaf;
                const $VERIFY_KEY_LENGTH: usize = 0;
                $body
            }
        }
    };
}

impl VdafOps {
    #[tracing::instrument(
        skip(
            self,
            clock,
            upload_decrypt_failure_counter,
            upload_decode_failure_counter,
            task,
            report_writer,
            report
        ),
        fields(task_id = ?task.id()),
        err
    )]
    async fn handle_upload<C: Clock>(
        &self,
        clock: &C,
        upload_decrypt_failure_counter: &Counter<u64>,
        upload_decode_failure_counter: &Counter<u64>,
        task: &Task,
        report_writer: &ReportWriteBatcher<C>,
        report: Report,
    ) -> Result<(), Arc<Error>> {
        match task.query_type() {
            task::QueryType::FixedSize { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, _, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_upload_generic::<VERIFY_KEY_LENGTH, FixedSize, VdafType, _>(
                        Arc::clone(vdaf),
                        clock,
                        upload_decrypt_failure_counter,
                        upload_decode_failure_counter,
                        task,
                        report_writer,
                        report,
                    )
                    .await
                })
            }
        }
    }

    /// Implements the `/aggregate` endpoint for initialization requests for the helper, described
    /// in §4.4.4.1 & §4.4.4.2 of draft-gpew-priv-ppm.
    #[tracing::instrument(
        skip(self, datastore, aggregate_step_failure_counter, task, req_bytes),
        fields(task_id = ?task.id()),
        err
    )]
    async fn handle_aggregate_init<C: Clock>(
        &self,
        datastore: &Datastore<C>,
        aggregate_step_failure_counter: &Counter<u64>,
        task: Arc<Task>,
        aggregation_job_id: &AggregationJobId,
        req_bytes: &[u8],
    ) -> Result<AggregationJobResp, Error> {
        match task.query_type() {
            task::QueryType::FixedSize { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, verify_key, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_aggregate_init_generic::<VERIFY_KEY_LENGTH, FixedSize, VdafType, _>(
                        datastore,
                        vdaf,
                        aggregate_step_failure_counter,
                        task,
                        aggregation_job_id,
                        verify_key,
                        req_bytes,
                    )
                    .await
                })
            }
        }
    }

    #[tracing::instrument(
        skip(self, datastore, aggregate_step_failure_counter, task, req, request_hash),
        fields(task_id = ?task.id()),
        err
    )]
    async fn handle_aggregate_continue<C: Clock>(
        &self,
        datastore: &Datastore<C>,
        aggregate_step_failure_counter: &Counter<u64>,
        task: Arc<Task>,
        batch_aggregation_shard_count: u64,
        aggregation_job_id: &AggregationJobId,
        req: Arc<AggregationJobContinueReq>,
        request_hash: [u8; 32],
    ) -> Result<AggregationJobResp, Error> {
        match task.query_type() {
            task::QueryType::FixedSize { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, _, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_aggregate_continue_generic::<VERIFY_KEY_LENGTH, FixedSize, VdafType, _>(
                        datastore,
                        Arc::clone(vdaf),
                        aggregate_step_failure_counter,
                        task,
                        batch_aggregation_shard_count,
                        aggregation_job_id,
                        req,
                        request_hash,
                    )
                    .await
                })
            }
        }
    }

    async fn handle_upload_generic<const SEED_SIZE: usize, Q, A, C>(
        vdaf: Arc<A>,
        clock: &C,
        upload_decrypt_failure_counter: &Counter<u64>,
        upload_decode_failure_counter: &Counter<u64>,
        task: &Task,
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
        Q: UploadableQueryType,
    {
        // The leader's report is the first one.
        // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.3.2
        if report.encrypted_input_shares().len() != 2 {
            return Err(Arc::new(Error::UnrecognizedMessage(
                Some(*task.id()),
                "unexpected number of encrypted shares in report",
            )));
        }
        let leader_encrypted_input_share =
            &report.encrypted_input_shares()[Role::Leader.index().unwrap()];

        // Verify that the report's HPKE config ID is known.
        // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.3.2
        let hpke_keypair = task
            .hpke_keys()
            .get(leader_encrypted_input_share.config_id())
            .ok_or_else(|| {
                Error::OutdatedHpkeConfig(*task.id(), *leader_encrypted_input_share.config_id())
            })?;

        let report_deadline = clock
            .now()
            .add(task.tolerable_clock_skew())
            .map_err(|err| Arc::new(Error::from(err)))?;

        // Reject reports from too far in the future.
        // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.3.2
        if report.metadata().time().is_after(&report_deadline) {
            return Err(Arc::new(Error::ReportTooEarly(
                *task.id(),
                *report.metadata().id(),
                *report.metadata().time(),
            )));
        }

        // Reject reports after a task has expired.
        // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.3.2
        if let Some(task_expiration) = task.task_expiration() {
            if report.metadata().time().is_after(task_expiration) {
                return Err(Arc::new(Error::ReportRejected(
                    *task.id(),
                    *report.metadata().id(),
                    *report.metadata().time(),
                )));
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
                return Err(Arc::new(Error::ReportRejected(
                    *task.id(),
                    *report.metadata().id(),
                    *report.metadata().time(),
                )));
            }
        }

        // Decode (and in the case of the leader input share, decrypt) the remaining fields of the
        // report before storing them in the datastore. The spec does not require the /upload
        // handler to do this, but it exercises HPKE decryption, saves us the trouble of storing
        // reports we can't use, and lets the aggregation job handler assume the values it reads
        // from the datastore are valid. We don't inform the client if this fails.
        let public_share =
            match A::PublicShare::get_decoded_with_param(vdaf.as_ref(), report.public_share()) {
                Ok(public_share) => public_share,
                Err(err) => {
                    warn!(
                        report.task_id = %task.id(),
                        report.metadata = ?report.metadata(),
                        ?err,
                        "public share decoding failed",
                    );
                    upload_decode_failure_counter.add(&Context::current(), 1, &[]);
                    return Ok(());
                }
            };

        let encoded_leader_plaintext_input_share = match hpke::open(
            hpke_keypair.config(),
            hpke_keypair.private_key(),
            &HpkeApplicationInfo::new(&Label::InputShare, &Role::Client, task.role()),
            leader_encrypted_input_share,
            &InputShareAad::new(
                *task.id(),
                report.metadata().clone(),
                report.public_share().to_vec(),
            )
            .get_encoded(),
        ) {
            Ok(encoded_leader_plaintext_input_share) => encoded_leader_plaintext_input_share,
            Err(error) => {
                info!(
                    report.task_id = %task.id(),
                    report.metadata = ?report.metadata(),
                    ?error,
                    "Report decryption failed",
                );
                upload_decrypt_failure_counter.add(&Context::current(), 1, &[]);
                return Ok(());
            }
        };

        let leader_plaintext_input_share =
            PlaintextInputShare::get_decoded(&encoded_leader_plaintext_input_share)
                .map_err(|err| Arc::new(Error::from(err)))?;

        let leader_input_share = match A::InputShare::get_decoded_with_param(
            &(&vdaf, Role::Leader.index().unwrap()),
            leader_plaintext_input_share.payload(),
        ) {
            Ok(leader_input_share) => leader_input_share,
            Err(err) => {
                warn!(
                    report.task_id = %task.id(),
                    report.metadata = ?report.metadata(),
                    ?err,
                    "Leader input share decoding failed",
                );
                upload_decode_failure_counter.add(&Context::current(), 1, &[]);
                return Ok(());
            }
        };

        let helper_encrypted_input_share =
            &report.encrypted_input_shares()[Role::Helper.index().unwrap()];

        let report = LeaderStoredReport::new(
            *task.id(),
            report.metadata().clone(),
            public_share,
            Vec::from(leader_plaintext_input_share.extensions()),
            leader_input_share,
            helper_encrypted_input_share.clone(),
        );

        report_writer
            .write_report(WritableReport::<SEED_SIZE, Q, A>::new(vdaf, report))
            .await
    }
}

/// Used by the aggregation job initialization handler to represent initialization of a report
/// share.
#[derive(Clone, Debug)]
struct ReportShareData<const SEED_SIZE: usize, A>
where
    A: vdaf::Aggregator<SEED_SIZE, 16>,
{
    report_share: ReportShare,
    report_aggregation: ReportAggregation<SEED_SIZE, A>,
}

impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> ReportShareData<SEED_SIZE, A>
where
    A: vdaf::Aggregator<SEED_SIZE, 16>,
{
    fn new(report_share: ReportShare, report_aggregation: ReportAggregation<SEED_SIZE, A>) -> Self {
        Self {
            report_share,
            report_aggregation,
        }
    }
}

impl VdafOps {
    /// Returns true if the incoming aggregation job matches existing contents of the datastore, in
    /// the sense that no new rows would need to be written to service the job.
    async fn check_aggregation_job_idempotence<'b, const SEED_SIZE: usize, Q, A, C>(
        tx: &Transaction<'b, C>,
        vdaf: &A,
        task: &Task,
        incoming_aggregation_job: &AggregationJob<SEED_SIZE, Q, A>,
        incoming_report_share_data: &[ReportShareData<SEED_SIZE, A>],
    ) -> Result<bool, Error>
    where
        Q: AccumulableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16> + 'static + Send + Sync,
        C: Clock,
        A::AggregationParam: Send + Sync + PartialEq,
        A::AggregateShare: Send + Sync,
        A::PrepareMessage: Send + Sync + PartialEq,
        A::PrepareShare: Send + Sync + PartialEq,
        for<'a> A::PrepareState:
            Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)> + PartialEq,
        A::OutputShare: Send + Sync + PartialEq,
    {
        let existing_aggregation_job = tx
            .get_aggregation_job(task.id(), incoming_aggregation_job.id())
            .await?
            .unwrap_or_else(|| {
                panic!(
                    "found no existing aggregation job for task ID {} and aggregation job ID {}",
                    task.id(),
                    incoming_aggregation_job.id()
                )
            });

        if !existing_aggregation_job.eq(incoming_aggregation_job) {
            return Ok(false);
        }

        // Check the existing report aggregations for this job against the ones in the incoming
        // message.
        let existing_report_aggregations = tx
            .get_report_aggregations_for_aggregation_job(
                vdaf,
                &Role::Helper,
                task.id(),
                incoming_aggregation_job.id(),
            )
            .await?;

        if existing_report_aggregations.len() != incoming_report_share_data.len() {
            return Ok(false);
        }

        // Check each report share in the incoming aggregation job against the already recorded
        // report aggregations. `existing_report_aggregations` preserves the order in which the
        // report shares were seen in the previous `AggregationJobInitReq`, and that order should be
        // preserved in the repeated message, so it's OK to just zip the iterators together.
        if incoming_report_share_data
            .iter()
            .zip(existing_report_aggregations)
            .any(
                |(incoming_report_share_data, existing_report_aggregation)| {
                    !existing_report_aggregation
                        .report_metadata()
                        .eq(incoming_report_share_data.report_share.metadata())
                        || !existing_report_aggregation
                            .eq(&incoming_report_share_data.report_aggregation)
                },
            )
        {
            return Ok(false);
        }

        Ok(true)
    }

    /// Implements the aggregate initialization request portion of the `/aggregate` endpoint for the
    /// helper, described in §4.4.4.1 of draft-gpew-priv-ppm.
    async fn handle_aggregate_init_generic<const SEED_SIZE: usize, Q, A, C>(
        datastore: &Datastore<C>,
        vdaf: &A,
        aggregate_step_failure_counter: &Counter<u64>,
        task: Arc<Task>,
        aggregation_job_id: &AggregationJobId,
        verify_key: &VerifyKey<SEED_SIZE>,
        req_bytes: &[u8],
    ) -> Result<AggregationJobResp, Error>
    where
        Q: AccumulableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16> + 'static + Send + Sync,
        C: Clock,
        A::AggregationParam: Send + Sync + PartialEq,
        A::AggregateShare: Send + Sync,
        A::PrepareMessage: Send + Sync + PartialEq,
        A::PrepareShare: Send + Sync + PartialEq,
        for<'a> A::PrepareState:
            Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)> + PartialEq,
        A::OutputShare: Send + Sync + PartialEq,
    {
        let req = AggregationJobInitializeReq::<Q>::get_decoded(req_bytes)?;

        // If two ReportShare messages have the same report ID, then the helper MUST abort with
        // error "unrecognizedMessage". (§4.4.4.1)
        let mut seen_report_ids = HashSet::with_capacity(req.report_shares().len());
        for share in req.report_shares() {
            if !seen_report_ids.insert(share.metadata().id()) {
                return Err(Error::UnrecognizedMessage(
                    Some(*task.id()),
                    "aggregate request contains duplicate report IDs",
                ));
            }
        }

        // Decrypt shares & prepare initialization states. (§4.4.4.1)
        let mut saw_continue = false;
        let mut report_share_data = Vec::new();
        let agg_param = A::AggregationParam::get_decoded(req.aggregation_parameter())?;
        for (ord, report_share) in req.report_shares().iter().enumerate() {
            let hpke_keypair = task
                .hpke_keys()
                .get(report_share.encrypted_input_share().config_id())
                .ok_or_else(|| {
                    info!(
                        config_id = %report_share.encrypted_input_share().config_id(),
                        "Helper encrypted input share references unknown HPKE config ID"
                    );
                    aggregate_step_failure_counter.add(
                        &Context::current(),
                        1,
                        &[KeyValue::new("type", "unknown_hpke_config_id")],
                    );
                    ReportShareError::HpkeUnknownConfigId
                });

            // If decryption fails, then the aggregator MUST fail with error `hpke-decrypt-error`. (§4.4.2.2)
            let plaintext = hpke_keypair.and_then(|hpke_keypair| {
                hpke::open(
                    hpke_keypair.config(),
                    hpke_keypair.private_key(),
                    &HpkeApplicationInfo::new(&Label::InputShare, &Role::Client, &Role::Helper),
                    report_share.encrypted_input_share(),
                    &InputShareAad::new(
                        *task.id(),
                        report_share.metadata().clone(),
                        report_share.public_share().to_vec(),
                    )
                    .get_encoded(),
                )
                .map_err(|error| {
                    info!(
                        task_id = %task.id(),
                        metadata = ?report_share.metadata(),
                        ?error,
                        "Couldn't decrypt helper's report share"
                    );
                    aggregate_step_failure_counter.add(
                        &Context::current(),
                        1,
                        &[KeyValue::new("type", "decrypt_failure")],
                    );
                    ReportShareError::HpkeDecryptError
                })
            });

            let plaintext_input_share = plaintext.and_then(|plaintext| {
                let plaintext_input_share = PlaintextInputShare::get_decoded(&plaintext).map_err(|error| {
                    info!(task_id = %task.id(), metadata = ?report_share.metadata(), ?error, "Couldn't decode helper's plaintext input share");
                    aggregate_step_failure_counter.add(&Context::current(), 1, &[KeyValue::new("type", "plaintext_input_share_decode_failure")]);
                    ReportShareError::UnrecognizedMessage
                })?;
                // Check for repeated extensions.
                let mut extension_types = HashSet::new();
                if !plaintext_input_share
                    .extensions()
                    .iter()
                    .all(|extension| extension_types.insert(extension.extension_type())) {
                        info!(task_id = %task.id(), metadata = ?report_share.metadata(), "Received report share with duplicate extensions");
                        aggregate_step_failure_counter.add(&Context::current(), 1, &[KeyValue::new("type", "duplicate_extension")]);
                        return Err(ReportShareError::UnrecognizedMessage)
                }
                Ok(plaintext_input_share)
            });

            let input_share = plaintext_input_share.and_then(|plaintext_input_share| {
                A::InputShare::get_decoded_with_param(&(vdaf, Role::Helper.index().unwrap()), plaintext_input_share.payload())
                    .map_err(|error| {
                        info!(task_id = %task.id(), metadata = ?report_share.metadata(), ?error, "Couldn't decode helper's input share");
                        aggregate_step_failure_counter.add(&Context::current(), 1, &[KeyValue::new("type", "input_share_decode_failure")]);
                        ReportShareError::UnrecognizedMessage
                    })
            });

            let public_share = A::PublicShare::get_decoded_with_param(vdaf, report_share.public_share()).map_err(|error|{
                info!(task_id = %task.id(), metadata = ?report_share.metadata(), ?error, "Couldn't decode public share");
                aggregate_step_failure_counter.add(&Context::current(), 1, &[KeyValue::new("type", "public_share_decode_failure")]);
                ReportShareError::UnrecognizedMessage
            });

            let shares = input_share.and_then(|input_share| Ok((public_share?, input_share)));

            // Next, the aggregator runs the preparation-state initialization algorithm for the VDAF
            // associated with the task and computes the first state transition. [...] If either
            // step fails, then the aggregator MUST fail with error `vdaf-prep-error`. (§4.4.2.2)
            let init_rslt = shares.and_then(|(public_share, input_share)| {
                trace_span!("VDAF preparation")
                    .in_scope(|| {
                        vdaf.prepare_init(
                            verify_key.as_bytes(),
                            Role::Helper.index().unwrap(),
                            &agg_param,
                            report_share.metadata().id().as_ref(),
                            &public_share,
                            &input_share,
                        )
                    })
                    .map_err(|error| {
                        info!(
                            task_id = %task.id(),
                            report_id = %report_share.metadata().id(),
                            ?error,
                            "Couldn't prepare_init report share"
                        );
                        aggregate_step_failure_counter.add(
                            &Context::current(),
                            1,
                            &[KeyValue::new("type", "prepare_init_failure")],
                        );
                        ReportShareError::VdafPrepError
                    })
            });

            report_share_data.push(match init_rslt {
                Ok((prep_state, prep_share)) => {
                    saw_continue = true;

                    let encoded_prep_share = prep_share.get_encoded();
                    ReportShareData::new(
                        report_share.clone(),
                        ReportAggregation::<SEED_SIZE, A>::new(
                            *task.id(),
                            *aggregation_job_id,
                            *report_share.metadata().id(),
                            *report_share.metadata().time(),
                            ord.try_into()?,
                            Some(PrepareStep::new(
                                *report_share.metadata().id(),
                                PrepareStepResult::Continued(encoded_prep_share),
                            )),
                            ReportAggregationState::<SEED_SIZE, A>::Waiting(prep_state, None),
                        ),
                    )
                }

                Err(err) => ReportShareData::new(
                    report_share.clone(),
                    ReportAggregation::<SEED_SIZE, A>::new(
                        *task.id(),
                        *aggregation_job_id,
                        *report_share.metadata().id(),
                        *report_share.metadata().time(),
                        ord.try_into()?,
                        Some(PrepareStep::new(
                            *report_share.metadata().id(),
                            PrepareStepResult::Failed(err),
                        )),
                        ReportAggregationState::<SEED_SIZE, A>::Failed(err),
                    ),
                ),
            });
        }

        // Store data to datastore.
        let req = Arc::new(req);
        let min_client_timestamp = req
            .report_shares()
            .iter()
            .map(|report_share| report_share.metadata().time())
            .min()
            .ok_or_else(|| Error::EmptyAggregation(*task.id()))?;
        let max_client_timestamp = req
            .report_shares()
            .iter()
            .map(|report_share| report_share.metadata().time())
            .max()
            .ok_or_else(|| Error::EmptyAggregation(*task.id()))?;
        let client_timestamp_interval = Interval::new(
            *min_client_timestamp,
            max_client_timestamp
                .difference(min_client_timestamp)?
                .add(&Duration::from_seconds(1))?,
        )?;
        let aggregation_job = Arc::new(AggregationJob::<SEED_SIZE, Q, A>::new(
            *task.id(),
            *aggregation_job_id,
            agg_param,
            req.batch_selector().batch_identifier().clone(),
            client_timestamp_interval,
            if saw_continue {
                AggregationJobState::InProgress
            } else {
                AggregationJobState::Finished
            },
            AggregationJobRound::from(0),
        ));

        Ok(datastore
            .run_tx_with_name("aggregate_init", |tx| {
                let (vdaf, task, req, aggregation_job, mut report_share_data) = (
                    vdaf.clone(),
                    Arc::clone(&task),
                    Arc::clone(&req),
                    Arc::clone(&aggregation_job),
                    report_share_data.clone(),
                );

                Box::pin(async move {
                    for report_share_data in &mut report_share_data {
                        // Verify that we haven't seen this report ID and aggregation parameter
                        // before in another aggregation job, and that the report isn't for a batch
                        // interval that has already started collection.
                        let (report_aggregation_exists, conflicting_aggregate_share_jobs) = try_join!(
                            tx.check_other_report_aggregation_exists::<SEED_SIZE, A>(
                                task.id(),
                                report_share_data.report_share.metadata().id(),
                                aggregation_job.aggregation_parameter(),
                                aggregation_job.id(),
                            ),
                            Q::get_conflicting_aggregate_share_jobs::<SEED_SIZE, C, A>(
                                tx,
                                &vdaf,
                                task.id(),
                                req.batch_selector().batch_identifier(),
                                report_share_data.report_share.metadata()
                            ),
                        )?;

                        if report_aggregation_exists {
                            report_share_data.report_aggregation =
                                report_share_data.report_aggregation
                                    .clone()
                                    .with_state(ReportAggregationState::Failed(
                                        ReportShareError::ReportReplayed))
                                    .with_last_prep_step(Some(PrepareStep::new(
                                        *report_share_data.report_share.metadata().id(),
                                        PrepareStepResult::Failed(ReportShareError::ReportReplayed))
                                    ));
                        } else if !conflicting_aggregate_share_jobs.is_empty() {
                            report_share_data.report_aggregation =
                                report_share_data.report_aggregation
                                    .clone()
                                    .with_state(ReportAggregationState::Failed(
                                        ReportShareError::BatchCollected))
                                    .with_last_prep_step(Some(PrepareStep::new(
                                        *report_share_data.report_share.metadata().id(),
                                        PrepareStepResult::Failed(ReportShareError::BatchCollected))
                                    ));
                        }
                    }

                    // Write aggregation job.
                    let replayed_request = match tx.put_aggregation_job(&aggregation_job).await {
                        Ok(_) => false,
                        Err(datastore::Error::MutationTargetAlreadyExists) => {
                            // Slow path: this request is writing an aggregation job that already
                            // exists in the datastore. PUT to an aggregation job is idempotent, so
                            // that's OK, provided the current request is equivalent to what's in
                            // the datastore, which we must now check.
                            if !Self::check_aggregation_job_idempotence(
                                tx,
                                &vdaf,
                                task.borrow(),
                                aggregation_job.borrow(),
                                &report_share_data,
                            )
                            .await
                            .map_err(|e| datastore::Error::User(e.into()))? {
                                return Err(datastore::Error::User(Error::ForbiddenMutation {
                                    resource_type: "aggregation job",
                                    identifier: aggregation_job.id().to_string(),
                                }.into()));
                            }

                            true
                        }
                        Err(e) => return Err(e),
                    };

                    if !replayed_request {
                        // Write report shares & aggregations.
                        report_share_data = try_join_all(report_share_data.into_iter()
                            .map(|mut rsd| { let task = Arc::clone(&task); async move {
                                if let Err(err) = tx.put_report_share(task.id(), &rsd.report_share).await {
                                    match err {
                                        datastore::Error::MutationTargetAlreadyExists => {
                                            rsd.report_aggregation =
                                                rsd.report_aggregation
                                                    .clone()
                                                    .with_state(ReportAggregationState::Failed(
                                                        ReportShareError::ReportReplayed))
                                                    .with_last_prep_step(Some(PrepareStep::new(
                                                        *rsd.report_share.metadata().id(),
                                                        PrepareStepResult::Failed(ReportShareError::ReportReplayed))
                                                    ));
                                        },
                                        err => return Err(err),
                                    }
                                }
                                tx.put_report_aggregation(&rsd.report_aggregation).await?;
                                Ok(rsd)
                            }})).await?;
                    }

                    Ok(Self::aggregation_job_resp_for(report_share_data.into_iter().map(|data| data.report_aggregation)))
                })
            })
            .await?)
    }

    async fn handle_aggregate_continue_generic<
        const SEED_SIZE: usize,
        Q: AccumulableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
        C: Clock,
    >(
        datastore: &Datastore<C>,
        vdaf: Arc<A>,
        aggregate_step_failure_counter: &Counter<u64>,
        task: Arc<Task>,
        batch_aggregation_shard_count: u64,
        aggregation_job_id: &AggregationJobId,
        leader_aggregation_job: Arc<AggregationJobContinueReq>,
        request_hash: [u8; 32],
    ) -> Result<AggregationJobResp, Error>
    where
        A: 'static + Send + Sync,
        A::AggregationParam: Send + Sync,
        A::AggregateShare: Send + Sync,
        for<'a> A::PrepareState: Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)>,
        A::PrepareShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        A::OutputShare: Send + Sync,
    {
        if leader_aggregation_job.round() == AggregationJobRound::from(0) {
            return Err(Error::UnrecognizedMessage(
                Some(*task.id()),
                "aggregation job cannot be advanced to round 0",
            ));
        }

        // TODO(#224): don't hold DB transaction open while computing VDAF updates?
        // TODO(#224): don't do O(n) network round-trips (where n is the number of prepare steps)
        Ok(datastore
            .run_tx_with_name("aggregate_continue", |tx| {
                let (
                    vdaf,
                    aggregate_step_failure_counter,
                    task,
                    aggregation_job_id,
                    leader_aggregation_job,
                ) = (
                    Arc::clone(&vdaf),
                    aggregate_step_failure_counter.clone(),
                    Arc::clone(&task),
                    *aggregation_job_id,
                    Arc::clone(&leader_aggregation_job),
                );

                Box::pin(async move {
                    // Read existing state.
                    let (helper_aggregation_job, report_aggregations) = try_join!(
                        tx.get_aggregation_job::<SEED_SIZE, Q, A>(task.id(), &aggregation_job_id),
                        tx.get_report_aggregations_for_aggregation_job(
                            vdaf.as_ref(),
                            &Role::Helper,
                            task.id(),
                            &aggregation_job_id,
                        ),
                    )?;

                    let helper_aggregation_job = helper_aggregation_job.ok_or_else(|| {
                        datastore::Error::User(
                            Error::UnrecognizedAggregationJob(*task.id(), aggregation_job_id)
                                .into(),
                        )
                    })?;

                    // If the leader's request is on the same round as our stored aggregation job,
                    // then we probably have already received this message and computed this round,
                    // but the leader never got our response and so retried stepping the job.
                    // TODO(issue #1087): measure how often this happens with a Prometheus metric
                    if helper_aggregation_job.round() == leader_aggregation_job.round() {
                        match helper_aggregation_job.last_continue_request_hash() {
                            None => {
                                return Err(datastore::Error::User(
                                    Error::Internal(format!(
                                        "aggregation job {aggregation_job_id} is in round {} but \
                                         has no last request hash",
                                        helper_aggregation_job.round(),
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
                        return Ok(Self::aggregation_job_resp_for(report_aggregations));
                    } else if helper_aggregation_job.round().increment()
                        != leader_aggregation_job.round()
                    {
                        // If this is not a replay, the leader should be advancing our state to the next
                        // round and no further.
                        return Err(datastore::Error::User(
                            Error::RoundMismatch {
                                task_id: *task.id(),
                                aggregation_job_id,
                                expected_round: helper_aggregation_job.round().increment(),
                                got_round: leader_aggregation_job.round(),
                            }
                            .into(),
                        ));
                    }

                    // The leader is advancing us to the next round. Step the aggregation job to
                    // compute the next round of prepare messages and state.
                    Self::step_aggregation_job(
                        tx,
                        task,
                        vdaf,
                        batch_aggregation_shard_count,
                        helper_aggregation_job,
                        report_aggregations,
                        leader_aggregation_job,
                        request_hash,
                        aggregate_step_failure_counter,
                    )
                    .await
                })
            })
            .await?)
    }

    /// Handle requests to the leader to create a collection job.
    #[tracing::instrument(
        skip(self, datastore, task, collection_req_bytes),
        fields(task_id = ?task.id()),
        err
    )]
    async fn handle_create_collection_job<C: Clock>(
        &self,
        datastore: &Datastore<C>,
        task: Arc<Task>,
        collection_job_id: &CollectionJobId,
        collection_req_bytes: &[u8],
    ) -> Result<(), Error> {
        match task.query_type() {
            task::QueryType::FixedSize { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, _, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_create_collection_job_generic::<
                        VERIFY_KEY_LENGTH,
                        FixedSize,
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
        Q: CollectableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        C: Clock,
    >(
        datastore: &Datastore<C>,
        task: Arc<Task>,
        vdaf: Arc<A>,
        collection_job_id: &CollectionJobId,
        req_bytes: &[u8],
    ) -> Result<(), Error>
    where
        A::AggregationParam: 'static + Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
    {
        let req = Arc::new(CollectionReq::<Q>::get_decoded(req_bytes)?);
        let aggregation_param = Arc::new(A::AggregationParam::get_decoded(
            req.aggregation_parameter(),
        )?);

        Ok(datastore
            .run_tx_with_name("collect", move |tx| {
                let (task, vdaf, collection_job_id, req, aggregation_param) = (
                    Arc::clone(&task),
                    Arc::clone(&vdaf),
                    *collection_job_id,
                    Arc::clone(&req),
                    Arc::clone(&aggregation_param),
                );
                Box::pin(async move {
                    let collection_identifier =
                        Q::collection_identifier_for_query(tx, &task, req.query())
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
                    if !Q::validate_collection_identifier(&task, &collection_identifier) {
                        return Err(datastore::Error::User(
                            Error::BatchInvalid(*task.id(), format!("{collection_identifier}"))
                                .into(),
                        ));
                    }

                    // Check if this collection job already exists, ensuring that all parameters match.
                    if let Some(collection_job) = tx
                        .get_collection_job::<SEED_SIZE, Q, A>(&vdaf, &collection_job_id)
                        .await?
                    {
                        if collection_job.batch_identifier() == &collection_identifier
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

                    debug!(collect_request = ?req, "Cache miss, creating new collection job");
                    let (_, report_count, batches, batches_with_reports) = try_join!(
                        Q::validate_query_count::<SEED_SIZE, C, A>(
                            tx,
                            &vdaf,
                            &task,
                            &collection_identifier
                        ),
                        Q::count_client_reports(tx, &task, &collection_identifier),
                        try_join_all(
                            Q::batch_identifiers_for_collection_identifier(
                                &task,
                                &collection_identifier
                            )
                            .map(|batch_identifier| {
                                let task_id = *task.id();
                                let aggregation_param = Arc::clone(&aggregation_param);
                                async move {
                                    let batch = tx
                                        .get_batch::<SEED_SIZE, Q, A>(
                                            &task_id,
                                            &batch_identifier,
                                            &aggregation_param,
                                        )
                                        .await?;
                                    Ok::<_, datastore::Error>((batch_identifier, batch))
                                }
                            }),
                        ),
                        try_join_all(
                            Q::batch_identifiers_for_collection_identifier(
                                &task,
                                &collection_identifier
                            )
                            .map(|batch_identifier| {
                                let task_id = *task.id();
                                async move {
                                    if let Some(batch_interval) =
                                        Q::to_batch_interval(&batch_identifier)
                                    {
                                        if tx
                                            .interval_has_unaggregated_reports(
                                                &task_id,
                                                batch_interval,
                                            )
                                            .await?
                                        {
                                            return Ok::<_, datastore::Error>(Some(
                                                batch_identifier.clone(),
                                            ));
                                        }
                                    }
                                    Ok(None)
                                }
                            })
                        ),
                    )?;

                    let batches_with_reports: HashSet<_> =
                        batches_with_reports.into_iter().flatten().collect();

                    // Batch size must be validated while handling CollectReq and hence before
                    // creating a collection job.
                    // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.5.6
                    if !task.validate_batch_size(report_count) {
                        return Err(datastore::Error::User(
                            Error::InvalidBatchSize(*task.id(), report_count).into(),
                        ));
                    }

                    // Prepare to update all batches to CLOSING/CLOSED, as well as determining the
                    // initial state of the collection job (which will be START, unless all batches
                    // went to CLOSED, in which case the collection job will start at COLLECTABLE).
                    let mut initial_collection_job_state = CollectionJobState::Collectable;

                    // Rust 1.65's clippy complains about needless collection, but we need to
                    // process through the iterator in order to update
                    // `initial_collection_job_state` as a side-effect, since computing
                    // `initial_collection_job_state` is necessary to compute the collection job we
                    // are writing, and the collection job write occurs concurrently with the batch
                    // writes. This appears to be fixed in Rust 1.69 -- we can likely remove this
                    // annotation after the MSRV reaches that version.
                    #[allow(clippy::needless_collect)]
                    let batches: Vec<_> = batches
                        .into_iter()
                        .flat_map(|(batch_identifier, batch)| {
                            // Determine the batch state we want to write for this batch. If there are
                            // no outstanding aggregation jobs (or no batch has yet been written, which
                            // implies no aggregation jobs have ever existed), and there are no
                            // outstanding unaggregated reports, close immediately. Otherwise, go to
                            // closing to allow the aggregation process to eventually complete & close
                            // the batch.
                            let batch_state = if batch
                                .as_ref()
                                .map_or(0, |b| b.outstanding_aggregation_jobs())
                                == 0
                                && !batches_with_reports.contains(&batch_identifier)
                            {
                                BatchState::Closed
                            } else {
                                initial_collection_job_state = CollectionJobState::Start;
                                BatchState::Closing
                            };

                            match batch {
                                Some(batch) if batch.state() == &BatchState::Open => {
                                    Some((Operation::Update, batch.with_state(batch_state)))
                                }

                                // If no batch has yet been written, write it now.
                                None => Some((
                                    Operation::Put,
                                    Batch::<SEED_SIZE, Q, A>::new(
                                        *task.id(),
                                        batch_identifier,
                                        aggregation_param.as_ref().clone(),
                                        batch_state,
                                        0,
                                        Interval::EMPTY,
                                    ),
                                )),

                                // If the batch exists but is already in a non-Open state, we don't
                                // need to write anything for this batch.
                                _ => None,
                            }
                        })
                        .collect();

                    let collection_job = CollectionJob::<SEED_SIZE, Q, A>::new(
                        *task.id(),
                        collection_job_id,
                        collection_identifier,
                        aggregation_param.as_ref().clone(),
                        initial_collection_job_state,
                    );

                    // Write collection job & batches back to storage.
                    try_join!(
                        tx.put_collection_job(&collection_job),
                        try_join_all(batches.into_iter().map(|(op, batch)| async move {
                            match op {
                                Operation::Put => {
                                    let rslt = tx.put_batch(&batch).await;
                                    if matches!(
                                        rslt,
                                        Err(datastore::Error::MutationTargetAlreadyExists)
                                    ) {
                                        // This codepath can be taken due to a quirk of how the
                                        // Repeatable Read isolation level works. It cannot occur at
                                        // the Serializable isolation level.
                                        //
                                        // For this codepath to be taken, two writers must
                                        // concurrently choose to write the same batch (by task ID,
                                        // batch ID, and aggregation parameter), and this batch must
                                        // not already exist in the datastore.
                                        //
                                        // Both writers will receive `None` from the `get_batch`
                                        // call, and then both will try to `put_batch`. One of the
                                        // writers will succeed. The other will fail with a unique
                                        // constraint violation on (task_id, batch_identifier,
                                        // aggregation_param), since unique constraints are still
                                        // enforced even in the presence of snapshot isolation. This
                                        // unique constraint will be translated to a
                                        // MutationTargetAlreadyExists error.
                                        //
                                        // The failing writer, in this case, can't do anything about
                                        // this problem while in its current transaction: further
                                        // attempts to read the batch will continue to return `None`
                                        // (since all reads in the same transaction are from the
                                        // same snapshot), so it can't update the now-written batch.
                                        // All it can do is give up on this transaction and try
                                        // again, by calling `retry` and returning an error.
                                        tx.retry();
                                    }
                                    rslt
                                }
                                Operation::Update => tx.update_batch(&batch).await,
                            }
                        }))
                    )?;
                    Ok(())
                })
            })
            .await?)
    }

    /// Handle GET requests to a collection job URI obtained from the leader's `/collect` endpoint.
    /// The return value is an encoded `CollectResp<Q>`.
    /// https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.5.1
    #[tracing::instrument(skip(self, datastore, task), fields(task_id = ?task.id()), err)]
    async fn handle_get_collection_job<C: Clock>(
        &self,
        datastore: &Datastore<C>,
        task: Arc<Task>,
        collection_job_id: &CollectionJobId,
    ) -> Result<Option<Vec<u8>>, Error> {
        match task.query_type() {
            task::QueryType::FixedSize { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, _, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_get_collection_job_generic::<
                        VERIFY_KEY_LENGTH,
                        FixedSize,
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
        Q: CollectableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        C: Clock,
    >(
        datastore: &Datastore<C>,
        task: Arc<Task>,
        vdaf: Arc<A>,
        collection_job_id: &CollectionJobId,
    ) -> Result<Option<Vec<u8>>, Error>
    where
        A::AggregationParam: Send + Sync,
        A::AggregateShare: Send + Sync,
    {
        let (collection_job, spanned_interval) = datastore
            .run_tx_with_name("get_collection_job", |tx| {
                let (task, vdaf, collection_job_id) =
                    (Arc::clone(&task), Arc::clone(&vdaf), *collection_job_id);
                Box::pin(async move {
                    let collection_job = tx
                        .get_collection_job::<SEED_SIZE, Q, A>(&vdaf, &collection_job_id)
                        .await?
                        .ok_or_else(|| {
                            datastore::Error::User(
                                Error::UnrecognizedCollectionJob(collection_job_id).into(),
                            )
                        })?;

                    let (batches, _) = try_join!(
                        Q::get_batches_for_collection_identifier(
                            tx,
                            &task,
                            collection_job.batch_identifier(),
                            collection_job.aggregation_parameter()
                        ),
                        Q::acknowledge_collection(tx, task.id(), collection_job.batch_identifier()),
                    )?;

                    // Merge the intervals spanned by the constituent batch aggregations into the
                    // interval spanned by the collection.
                    let mut spanned_interval: Option<Interval> = None;
                    for interval in batches
                        .iter()
                        .map(Batch::<SEED_SIZE, Q, A>::client_timestamp_interval)
                    {
                        match spanned_interval {
                            Some(m) => spanned_interval = Some(m.merge(interval)?),
                            None => spanned_interval = Some(*interval),
                        }
                    }

                    Ok((collection_job, spanned_interval))
                })
            })
            .await?;

        match collection_job.state() {
            CollectionJobState::Start | CollectionJobState::Collectable => {
                debug!(%collection_job_id, task_id = %task.id(), "collection job has not run yet");
                Ok(None)
            }

            CollectionJobState::Finished {
                report_count,
                encrypted_helper_aggregate_share,
                leader_aggregate_share,
            } => {
                let spanned_interval = spanned_interval
                    .ok_or_else(|| {
                        datastore::Error::User(
                            Error::Internal(format!(
                                "collection job {collection_job_id} is finished but spans no time \
                                 interval"
                            ))
                            .into(),
                        )
                    })?
                    .align_to_time_precision(task.time_precision())?;

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
                    task.collector_hpke_config(),
                    &HpkeApplicationInfo::new(
                        &Label::AggregateShare,
                        &Role::Leader,
                        &Role::Collector,
                    ),
                    &leader_aggregate_share.get_encoded(),
                    &AggregateShareAad::new(
                        *collection_job.task_id(),
                        BatchSelector::<Q>::new(collection_job.batch_identifier().clone()),
                    )
                    .get_encoded(),
                )?;

                Ok(Some(
                    Collection::<Q>::new(
                        PartialBatchSelector::new(
                            Q::partial_batch_identifier(collection_job.batch_identifier()).clone(),
                        ),
                        *report_count,
                        spanned_interval,
                        Vec::<HpkeCiphertext>::from([
                            encrypted_leader_aggregate_share,
                            encrypted_helper_aggregate_share.clone(),
                        ]),
                    )
                    .get_encoded(),
                ))
            }

            CollectionJobState::Abandoned => {
                // TODO(#248): decide how to respond for abandoned collection jobs.
                warn!(
                    %collection_job_id,
                    task_id = %task.id(),
                    "Attempting to collect abandoned collection job"
                );
                Ok(None)
            }

            CollectionJobState::Deleted => Err(Error::DeletedCollectionJob(*collection_job_id)),
        }
    }

    #[tracing::instrument(skip(self, datastore, task), fields(task_id = ?task.id()), err)]
    async fn handle_delete_collection_job<C: Clock>(
        &self,
        datastore: &Datastore<C>,
        task: Arc<Task>,
        collection_job_id: &CollectionJobId,
    ) -> Result<(), Error> {
        match task.query_type() {
            task::QueryType::FixedSize { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, _, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_delete_collection_job_generic::<
                        VERIFY_KEY_LENGTH,
                        FixedSize,
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
        Q: CollectableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        C: Clock,
    >(
        datastore: &Datastore<C>,
        task: Arc<Task>,
        vdaf: Arc<A>,
        collection_job_id: &CollectionJobId,
    ) -> Result<(), Error>
    where
        A::AggregationParam: Send + Sync,
        A::AggregateShare: Send + Sync + PartialEq + Eq,
    {
        datastore
            .run_tx_with_name("delete_collection_job", move |tx| {
                let (task, vdaf, collection_job_id) =
                    (Arc::clone(&task), Arc::clone(&vdaf), *collection_job_id);
                Box::pin(async move {
                    let collection_job = tx
                        .get_collection_job::<SEED_SIZE, Q, A>(vdaf.as_ref(), &collection_job_id)
                        .await?
                        .ok_or_else(|| {
                            datastore::Error::User(
                                Error::UnrecognizedCollectionJob(collection_job_id).into(),
                            )
                        })?;
                    Q::acknowledge_collection(tx, task.id(), collection_job.batch_identifier())
                        .await?;
                    if collection_job.state() != &CollectionJobState::Deleted {
                        tx.update_collection_job::<SEED_SIZE, Q, A>(
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

    /// Implements the `/aggregate_share` endpoint for the helper, described in §4.4.4.3
    #[tracing::instrument(
        skip(self, datastore, clock, task, req_bytes),
        fields(task_id = ?task.id()),
        err
    )]
    async fn handle_aggregate_share<C: Clock>(
        &self,
        datastore: &Datastore<C>,
        clock: &C,
        task: Arc<Task>,
        batch_aggregation_shard_count: u64,
        req_bytes: &[u8],
    ) -> Result<AggregateShare, Error> {
        match task.query_type() {
            task::QueryType::FixedSize { .. } => {
                vdaf_ops_dispatch!(self, (vdaf, _, VdafType, VERIFY_KEY_LENGTH) => {
                    Self::handle_aggregate_share_generic::<
                        VERIFY_KEY_LENGTH,
                        FixedSize,
                        VdafType,
                        _,
                    >(datastore, clock, task, Arc::clone(vdaf), req_bytes, batch_aggregation_shard_count)
                    .await
                })
            }
        }
    }

    async fn handle_aggregate_share_generic<
        const SEED_SIZE: usize,
        Q: CollectableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        C: Clock,
    >(
        datastore: &Datastore<C>,
        clock: &C,
        task: Arc<Task>,
        vdaf: Arc<A>,
        req_bytes: &[u8],
        batch_aggregation_shard_count: u64,
    ) -> Result<AggregateShare, Error>
    where
        A::AggregationParam: Send + Sync,
        A::AggregateShare: Send + Sync,
    {
        // Decode request, and verify that it is for the current task. We use an assert to check
        // that the task IDs match as this should be guaranteed by the caller.
        let aggregate_share_req = Arc::new(AggregateShareReq::<Q>::get_decoded(req_bytes)?);

        // §4.4.4.3: check that the batch interval meets the requirements from §4.6
        if !Q::validate_collection_identifier(
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
                Q::to_batch_interval(aggregate_share_req.batch_selector().batch_identifier())
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
            .run_tx_with_name("aggregate_share", |tx| {
                let (task, vdaf, aggregate_share_req) = (
                    Arc::clone(&task),
                    Arc::clone(&vdaf),
                    Arc::clone(&aggregate_share_req),
                );
                Box::pin(async move {
                    // Check if we have already serviced an aggregate share request with these
                    // parameters and serve the cached results if so.
                    let aggregation_param = A::AggregationParam::get_decoded(
                        aggregate_share_req.aggregation_parameter(),
                    )?;
                    let aggregate_share_job = match tx
                        .get_aggregate_share_job(
                            vdaf.as_ref(),
                            task.id(),
                            aggregate_share_req.batch_selector().batch_identifier(),
                            &aggregation_param,
                        )
                        .await?
                    {
                        Some(aggregate_share_job) => {
                            debug!(
                                ?aggregate_share_req,
                                "Serving cached aggregate share job result"
                            );
                            aggregate_share_job
                        }
                        None => {
                            debug!(
                                ?aggregate_share_req,
                                "Cache miss, computing aggregate share job result"
                            );
                            let aggregation_param = A::AggregationParam::get_decoded(
                                aggregate_share_req.aggregation_parameter(),
                            )?;
                            let (batch_aggregations, _) = try_join!(
                                Q::get_batch_aggregations_for_collection_identifier(
                                    tx,
                                    &task,
                                    vdaf.as_ref(),
                                    aggregate_share_req.batch_selector().batch_identifier(),
                                    &aggregation_param
                                ),
                                Q::validate_query_count::<SEED_SIZE, C, A>(
                                    tx,
                                    vdaf.as_ref(),
                                    &task,
                                    aggregate_share_req.batch_selector().batch_identifier(),
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

                            let (helper_aggregate_share, report_count, checksum) =
                                compute_aggregate_share::<SEED_SIZE, Q, A>(
                                    &task,
                                    &batch_aggregations,
                                )
                                .await
                                .map_err(|e| datastore::Error::User(e.into()))?;

                            // Now that we are satisfied that the request is serviceable, we consume
                            // a query by recording the aggregate share request parameters and the
                            // result.
                            let aggregate_share_job = AggregateShareJob::<SEED_SIZE, Q, A>::new(
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
                                    tx.update_batch_aggregation(
                                        &ba.with_state(BatchAggregationState::Collected),
                                    )
                                    .await
                                })),
                                try_join_all(
                                    empty_batch_aggregations
                                        .iter()
                                        .map(|ba| tx.put_batch_aggregation(ba))
                                ),
                            )?;
                            aggregate_share_job
                        }
                    };

                    // §4.4.4.3: verify total report count and the checksum we computed against
                    // those reported by the leader.
                    if aggregate_share_job.report_count() != aggregate_share_req.report_count()
                        || aggregate_share_job.checksum() != aggregate_share_req.checksum()
                    {
                        return Err(datastore::Error::User(
                            Error::BatchMismatch(Box::new(BatchMismatch {
                                task_id: *task.id(),
                                own_checksum: *aggregate_share_job.checksum(),
                                own_report_count: aggregate_share_job.report_count(),
                                peer_checksum: *aggregate_share_req.checksum(),
                                peer_report_count: aggregate_share_req.report_count(),
                            }))
                            .into(),
                        ));
                    }

                    Ok(aggregate_share_job)
                })
            })
            .await?;

        // §4.4.4.3: HPKE encrypt aggregate share to the collector. We store *unencrypted* aggregate
        // shares in the datastore so that we can encrypt cached results to the collector HPKE
        // config valid when the current AggregateShareReq was made, and not whatever was valid at
        // the time the aggregate share was first computed.
        let encrypted_aggregate_share = hpke::seal(
            task.collector_hpke_config(),
            &HpkeApplicationInfo::new(&Label::AggregateShare, &Role::Helper, &Role::Collector),
            &aggregate_share_job.helper_aggregate_share().get_encoded(),
            &AggregateShareAad::new(*task.id(), aggregate_share_req.batch_selector().clone())
                .get_encoded(),
        )?;

        Ok(AggregateShare::new(encrypted_aggregate_share))
    }
}

fn empty_batch_aggregations<
    const SEED_SIZE: usize,
    Q: CollectableQueryType,
    A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
>(
    task: &Task,
    batch_aggregation_shard_count: u64,
    batch_identifier: &Q::BatchIdentifier,
    aggregation_param: &A::AggregationParam,
    batch_aggregations: &[BatchAggregation<SEED_SIZE, Q, A>],
) -> Vec<BatchAggregation<SEED_SIZE, Q, A>> {
    let existing_batch_aggregations: HashSet<_> = batch_aggregations
        .iter()
        .map(|ba| (ba.batch_identifier(), ba.ord()))
        .collect();
    iproduct!(
        Q::batch_identifiers_for_collection_identifier(task, batch_identifier),
        0..batch_aggregation_shard_count
    )
    .filter_map(|(batch_identifier, ord)| {
        if !existing_batch_aggregations.contains(&(&batch_identifier, ord)) {
            Some(BatchAggregation::<SEED_SIZE, Q, A>::new(
                *task.id(),
                batch_identifier,
                aggregation_param.clone(),
                ord,
                BatchAggregationState::Collected,
                None,
                0,
                ReportIdChecksum::default(),
            ))
        } else {
            None
        }
    })
    .collect()
}

/// Convenience method to perform an HTTP request to the helper. This includes common
/// metrics and error handling functionality.
#[tracing::instrument(
    skip(
        http_client,
        url,
        request,
        auth_token,
        http_request_duration_histogram,
    ),
    fields(url = %url),
    err,
)]
async fn send_request_to_helper<T: Encode>(
    http_client: &Client,
    method: Method,
    url: Url,
    route_label: &'static str,
    content_type: &str,
    request: T,
    auth_token: &AuthenticationToken,
    http_request_duration_histogram: &Histogram<f64>,
) -> Result<Bytes, Error> {
    let domain = url.domain().unwrap_or_default().to_string();
    let request_body = request.get_encoded();
    let (auth_header, auth_value) = auth_token.request_authentication();

    let start = Instant::now();
    let response_result = http_client
        .request(method, url)
        .header(CONTENT_TYPE, content_type)
        .header(auth_header, auth_value)
        .body(request_body)
        .send()
        .await;
    let response = match response_result {
        Ok(response) => response,
        Err(error) => {
            http_request_duration_histogram.record(
                &Context::current(),
                start.elapsed().as_secs_f64(),
                &[
                    KeyValue::new("status", "error"),
                    KeyValue::new("domain", domain),
                    KeyValue::new("endpoint", route_label),
                ],
            );
            return Err(error.into());
        }
    };

    let status = response.status();
    if !status.is_success() {
        http_request_duration_histogram.record(
            &Context::current(),
            start.elapsed().as_secs_f64(),
            &[
                KeyValue::new("status", "error"),
                KeyValue::new("domain", domain),
                KeyValue::new("endpoint", route_label),
            ],
        );
        let problem_details = response_to_problem_details(response).await;
        let dap_problem_type = problem_details
            .type_url
            .as_ref()
            .and_then(|str| str.parse::<DapProblemType>().ok());
        return Err(Error::Http {
            problem_details: Box::new(problem_details),
            dap_problem_type,
        });
    }

    match response.bytes().await {
        Ok(response_body) => {
            http_request_duration_histogram.record(
                &Context::current(),
                start.elapsed().as_secs_f64(),
                &[
                    KeyValue::new("status", "success"),
                    KeyValue::new("domain", domain),
                    KeyValue::new("endpoint", route_label),
                ],
            );
            Ok(response_body)
        }
        Err(error) => {
            http_request_duration_histogram.record(
                &Context::current(),
                start.elapsed().as_secs_f64(),
                &[
                    KeyValue::new("status", "error"),
                    KeyValue::new("domain", domain),
                    KeyValue::new("endpoint", route_label),
                ],
            );
            Err(error.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::aggregator::{Aggregator, Config, Error};
    use assert_matches::assert_matches;
    use futures::future::try_join_all;
    use janus_aggregator_core::{
        datastore::{
            test_util::{ephemeral_datastore, EphemeralDatastore},
            Datastore,
        },
        task::{test_util::TaskBuilder, QueryType, Task},
        test_util::noop_meter,
    };
    use janus_core::{
        hpke::{self, HpkeApplicationInfo, Label},
        task::VdafInstance,
        test_util::{dummy_vdaf, install_test_trace_subscriber},
        time::{Clock, MockClock, TimeExt},
    };
    use janus_messages::{
        Duration, Extension, HpkeCiphertext, HpkeConfig, HpkeConfigId, InputShareAad,
        PlaintextInputShare, Report, ReportId, ReportMetadata, ReportShare, Role, TaskId, Time,
    };
    use prio::{
        codec::Encode,
        vdaf::{self, prio3::Prio3Count, Client as _},
    };
    use rand::random;
    use std::{collections::HashSet, iter, sync::Arc, time::Duration as StdDuration};

    pub(super) const DUMMY_VERIFY_KEY_LENGTH: usize = dummy_vdaf::Vdaf::VERIFY_KEY_LENGTH;

    pub(crate) const BATCH_AGGREGATION_SHARD_COUNT: u64 = 32;

    pub(crate) fn default_aggregator_config() -> Config {
        // Enable upload write batching & batch aggregation sharding by default, in hopes that we
        // can shake out any bugs.
        Config {
            max_upload_batch_size: 5,
            max_upload_batch_write_delay: StdDuration::from_millis(100),
            batch_aggregation_shard_count: BATCH_AGGREGATION_SHARD_COUNT,
        }
    }

    pub(super) fn create_report_with_id(
        task: &Task,
        report_timestamp: Time,
        id: ReportId,
    ) -> Report {
        assert_eq!(task.vdaf(), &VdafInstance::Prio3Count);

        let vdaf = Prio3Count::new_count(2).unwrap();
        let hpke_key = task.current_hpke_key();
        let report_metadata = ReportMetadata::new(id, report_timestamp);

        let (public_share, measurements) = vdaf.shard(&1, id.as_ref()).unwrap();

        let associated_data = InputShareAad::new(
            *task.id(),
            report_metadata.clone(),
            public_share.get_encoded(),
        );

        let leader_ciphertext = hpke::seal(
            hpke_key.config(),
            &HpkeApplicationInfo::new(&Label::InputShare, &Role::Client, &Role::Leader),
            &PlaintextInputShare::new(Vec::new(), measurements[0].get_encoded()).get_encoded(),
            &associated_data.get_encoded(),
        )
        .unwrap();
        let helper_ciphertext = hpke::seal(
            hpke_key.config(),
            &HpkeApplicationInfo::new(&Label::InputShare, &Role::Client, &Role::Helper),
            &PlaintextInputShare::new(Vec::new(), measurements[1].get_encoded()).get_encoded(),
            &associated_data.get_encoded(),
        )
        .unwrap();

        Report::new(
            report_metadata,
            public_share.get_encoded(),
            Vec::from([leader_ciphertext, helper_ciphertext]),
        )
    }

    pub(super) fn create_report(task: &Task, report_timestamp: Time) -> Report {
        create_report_with_id(task, report_timestamp, random())
    }

    async fn setup_upload_test(
        cfg: Config,
    ) -> (
        Prio3Count,
        Aggregator<MockClock>,
        MockClock,
        Task,
        Arc<Datastore<MockClock>>,
        EphemeralDatastore,
    ) {
        let clock = MockClock::default();
        let vdaf = Prio3Count::new_count(2).unwrap();
        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Leader,
        )
        .build();

        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        datastore.put_task(&task).await.unwrap();

        let aggregator = Aggregator::new(Arc::clone(&datastore), clock.clone(), &noop_meter(), cfg);

        (
            vdaf,
            aggregator,
            clock,
            task,
            datastore,
            ephemeral_datastore,
        )
    }

    #[tokio::test]
    async fn upload() {
        install_test_trace_subscriber();

        let (vdaf, aggregator, clock, task, datastore, _ephemeral_datastore) =
            setup_upload_test(Config {
                max_upload_batch_size: 1000,
                max_upload_batch_write_delay: StdDuration::from_millis(500),
                ..Default::default()
            })
            .await;
        let report = create_report(&task, clock.now());

        aggregator
            .handle_upload(task.id(), &report.get_encoded())
            .await
            .unwrap();

        let got_report = datastore
            .run_tx(|tx| {
                let (vdaf, task_id, report_id) =
                    (vdaf.clone(), *task.id(), *report.metadata().id());
                Box::pin(async move { tx.get_client_report(&vdaf, &task_id, &report_id).await })
            })
            .await
            .unwrap()
            .unwrap();
        assert_eq!(task.id(), got_report.task_id());
        assert_eq!(report.metadata(), got_report.metadata());

        // Report uploads are idempotent
        aggregator
            .handle_upload(task.id(), &report.get_encoded())
            .await
            .unwrap();

        // Reports may not be mutated
        let mutated_report = create_report_with_id(&task, clock.now(), *report.metadata().id());
        let error = aggregator
            .handle_upload(task.id(), &mutated_report.get_encoded())
            .await
            .unwrap_err();
        assert_matches!(error.as_ref(), Error::ReportRejected(task_id, report_id, timestamp) => {
            assert_eq!(task.id(), task_id);
            assert_eq!(mutated_report.metadata().id(), report_id);
            assert_eq!(mutated_report.metadata().time(), timestamp);
        });
    }

    #[tokio::test]
    async fn upload_batch() {
        install_test_trace_subscriber();

        const BATCH_SIZE: usize = 100;
        let (vdaf, aggregator, clock, task, datastore, _ephemeral_datastore) =
            setup_upload_test(Config {
                max_upload_batch_size: BATCH_SIZE,
                max_upload_batch_write_delay: StdDuration::from_secs(86400),
                ..Default::default()
            })
            .await;

        let reports: Vec<_> = iter::repeat_with(|| create_report(&task, clock.now()))
            .take(BATCH_SIZE)
            .collect();
        let want_report_ids: HashSet<_> = reports.iter().map(|r| *r.metadata().id()).collect();

        let aggregator = Arc::new(aggregator);
        try_join_all(reports.iter().map(|r| {
            let aggregator = Arc::clone(&aggregator);
            let enc = r.get_encoded();
            let task_id = task.id();
            async move { aggregator.handle_upload(task_id, &enc).await }
        }))
        .await
        .unwrap();

        let got_report_ids = datastore
            .run_tx(|tx| {
                let vdaf = vdaf.clone();
                let task = task.clone();
                Box::pin(async move { tx.get_client_reports_for_task(&vdaf, task.id()).await })
            })
            .await
            .unwrap()
            .iter()
            .map(|r| *r.metadata().id())
            .collect();

        assert_eq!(want_report_ids, got_report_ids);
    }

    #[tokio::test]
    async fn upload_wrong_number_of_encrypted_shares() {
        install_test_trace_subscriber();

        let (_, aggregator, clock, task, _, _ephemeral_datastore) =
            setup_upload_test(default_aggregator_config()).await;
        let report = create_report(&task, clock.now());
        let report = Report::new(
            report.metadata().clone(),
            report.public_share().to_vec(),
            Vec::from([report.encrypted_input_shares()[0].clone()]),
        );

        assert_matches!(
            aggregator
                .handle_upload(task.id(), &report.get_encoded())
                .await
                .unwrap_err()
                .as_ref(),
            Error::UnrecognizedMessage(_, _)
        );
    }

    #[tokio::test]
    async fn upload_wrong_hpke_config_id() {
        install_test_trace_subscriber();

        let (_, aggregator, clock, task, _, _ephemeral_datastore) =
            setup_upload_test(default_aggregator_config()).await;
        let report = create_report(&task, clock.now());

        let unused_hpke_config_id = (0..)
            .map(HpkeConfigId::from)
            .find(|id| !task.hpke_keys().contains_key(id))
            .unwrap();

        let report = Report::new(
            report.metadata().clone(),
            report.public_share().to_vec(),
            Vec::from([
                HpkeCiphertext::new(
                    unused_hpke_config_id,
                    report.encrypted_input_shares()[0]
                        .encapsulated_key()
                        .to_vec(),
                    report.encrypted_input_shares()[0].payload().to_vec(),
                ),
                report.encrypted_input_shares()[1].clone(),
            ]),
        );

        assert_matches!(aggregator.handle_upload(task.id(), &report.get_encoded()).await.unwrap_err().as_ref(), Error::OutdatedHpkeConfig(task_id, config_id) => {
            assert_eq!(task.id(), task_id);
            assert_eq!(config_id, &unused_hpke_config_id);
        });
    }

    #[tokio::test]
    async fn upload_report_in_the_future_boundary_condition() {
        install_test_trace_subscriber();

        let (vdaf, aggregator, clock, task, datastore, _ephemeral_datastore) =
            setup_upload_test(default_aggregator_config()).await;
        let report = create_report(&task, clock.now().add(task.tolerable_clock_skew()).unwrap());

        aggregator
            .handle_upload(task.id(), &report.get_encoded())
            .await
            .unwrap();

        let got_report = datastore
            .run_tx(|tx| {
                let (vdaf, task_id, report_id) =
                    (vdaf.clone(), *task.id(), *report.metadata().id());
                Box::pin(async move { tx.get_client_report(&vdaf, &task_id, &report_id).await })
            })
            .await
            .unwrap()
            .unwrap();
        assert_eq!(task.id(), got_report.task_id());
        assert_eq!(report.metadata(), got_report.metadata());
    }

    #[tokio::test]
    async fn upload_report_in_the_future_past_clock_skew() {
        install_test_trace_subscriber();

        let (_, aggregator, clock, task, _, _ephemeral_datastore) =
            setup_upload_test(default_aggregator_config()).await;
        let report = create_report(
            &task,
            clock
                .now()
                .add(task.tolerable_clock_skew())
                .unwrap()
                .add(&Duration::from_seconds(1))
                .unwrap(),
        );

        let upload_error = aggregator
            .handle_upload(task.id(), &report.get_encoded())
            .await
            .unwrap_err();

        assert_matches!(upload_error.as_ref(), Error::ReportTooEarly(task_id, report_id, time) => {
            assert_eq!(task.id(), task_id);
            assert_eq!(report.metadata().id(), report_id);
            assert_eq!(report.metadata().time(), time);
        });
    }

    pub(crate) fn generate_helper_report_share<V: vdaf::Client<16>>(
        task_id: TaskId,
        report_metadata: ReportMetadata,
        cfg: &HpkeConfig,
        public_share: &V::PublicShare,
        extensions: Vec<Extension>,
        input_share: &V::InputShare,
    ) -> ReportShare {
        generate_helper_report_share_for_plaintext(
            report_metadata.clone(),
            cfg,
            public_share.get_encoded(),
            &PlaintextInputShare::new(extensions, input_share.get_encoded()).get_encoded(),
            &InputShareAad::new(task_id, report_metadata, public_share.get_encoded()).get_encoded(),
        )
    }

    pub(super) fn generate_helper_report_share_for_plaintext(
        metadata: ReportMetadata,
        cfg: &HpkeConfig,
        encoded_public_share: Vec<u8>,
        plaintext: &[u8],
        associated_data: &[u8],
    ) -> ReportShare {
        ReportShare::new(
            metadata,
            encoded_public_share,
            hpke::seal(
                cfg,
                &HpkeApplicationInfo::new(&Label::InputShare, &Role::Client, &Role::Helper),
                plaintext,
                associated_data,
            )
            .unwrap(),
        )
    }
}
