//! Implements portions of aggregation job initialization for the Helper.

use crate::{
    aggregator::{
        aggregation_job_writer::WritableReportAggregation, error::handle_ping_pong_error,
        AggregatorMetrics, Error,
    },
    cache::HpkeKeypairCache,
};
use assert_matches::assert_matches;
use janus_aggregator_core::{
    batch_mode::AccumulableBatchMode,
    datastore::models::{AggregationJob, ReportAggregation, ReportAggregationState},
    task::AggregatorTask,
};
use janus_core::{
    hpke::{self, HpkeApplicationInfo, Label},
    time::{Clock, TimeExt as _},
    vdaf::vdaf_application_context,
};
use janus_messages::{
    ExtensionType, InputShareAad, PlaintextInputShare, PrepareResp, PrepareStepResult, ReportError,
    Role,
};
use opentelemetry::{metrics::Counter, KeyValue};
use prio::{
    codec::{Decode as _, Encode, ParameterizedDecode},
    topology::ping_pong::{PingPongState, PingPongTopology as _},
    vdaf,
};
use rayon::iter::{IntoParallelIterator as _, ParallelIterator as _};
use std::{collections::HashMap, panic, sync::Arc};
use tokio::sync::mpsc;
use tracing::{debug, info_span, trace_span, Span};

#[derive(Clone)]
pub struct AggregateInitMetrics {
    /// Counters tracking the number of failures to step client reports through the aggregation
    /// process.
    aggregate_step_failure_counter: Counter<u64>,
}

impl AggregateInitMetrics {
    pub fn new(aggregate_step_failure_counter: Counter<u64>) -> Self {
        Self {
            aggregate_step_failure_counter,
        }
    }
}

impl From<AggregatorMetrics> for AggregateInitMetrics {
    fn from(metrics: AggregatorMetrics) -> Self {
        Self {
            aggregate_step_failure_counter: metrics.aggregate_step_failure_counter,
        }
    }
}

/// Given report aggregations in the `HelperInitProcessing` state, this function computes the next
/// step of the aggregation; the returned [`WritableReportAggregation`]s correspond to the provided
/// report aggregations and will be in the `HelperContinue`, `Finished`, or `Failed` states.
///
/// Only report aggregations in the `HelperInitProcessing` state can be provided. The caller must
/// filter report aggregations which are in other states (e.g. `Failed`) prior to calling this
/// function.
///
/// ### Panics
///
/// Panics if a provided report aggregation is in a state other than `HelperInitProcessing`.
pub async fn compute_helper_aggregate_init<const SEED_SIZE: usize, B, A, C>(
    clock: &C,
    hpke_keypairs: Arc<HpkeKeypairCache>,
    vdaf: Arc<A>,
    metrics: AggregateInitMetrics,
    task: Arc<AggregatorTask>,
    aggregation_job: Arc<AggregationJob<SEED_SIZE, B, A>>,
    report_aggregations: Vec<ReportAggregation<SEED_SIZE, A>>,
) -> Result<Vec<WritableReportAggregation<SEED_SIZE, A>>, Error>
where
    B: AccumulableBatchMode,
    A: vdaf::Aggregator<SEED_SIZE, 16> + 'static + Send + Sync,
    C: Clock,
    A::AggregationParam: Send + Sync + PartialEq + Eq,
    A::AggregateShare: Send + Sync,
    A::InputShare: Send + Sync,
    A::PrepareMessage: Send + Sync + PartialEq,
    A::PrepareShare: Send + Sync + PartialEq,
    for<'a> A::PrepareState: Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)> + PartialEq,
    A::PublicShare: Send + Sync,
    A::OutputShare: Send + Sync + PartialEq,
{
    let verify_key = task.vdaf_verify_key()?;
    let report_aggregation_count = report_aggregations.len();
    let report_deadline = clock
        .now()
        .add(task.tolerable_clock_skew())
        .map_err(Error::from)?;

    // Shutdown on cancellation: if this request is cancelled, the `receiver` will be dropped. This
    // will cause any attempts to send on `sender` to return a `SendError`, which will be returned
    // from the function passed to `try_for_each`; `try_for_each` will terminate early on receiving
    // an error.
    let (sender, mut receiver) = mpsc::unbounded_channel();
    let producer_task = tokio::task::spawn_blocking({
        let parent_span = Span::current();
        let hpke_keypairs = Arc::clone(&hpke_keypairs);
        let vdaf = Arc::clone(&vdaf);
        let task = Arc::clone(&task);
        let metrics = metrics.clone();
        let aggregation_job = Arc::clone(&aggregation_job);

        move || {
            let span =
                info_span!(parent: parent_span, "compute_helper_aggregate_init threadpool task");
            let ctx = vdaf_application_context(task.id());

            report_aggregations
                    .into_par_iter()
                    .try_for_each(|report_aggregation| {
                        let _entered = span.enter();

                        // Assert safety: this function should only be called with report
                        // aggregations in the HelperInitProcessing state.
                        let (prepare_init, require_taskbind_extension) = assert_matches!(
                            report_aggregation.state(),
                            ReportAggregationState::HelperInitProcessing {
                                prepare_init,
                                require_taskbind_extension,
                            } => (prepare_init, *require_taskbind_extension)
                        );

                        // If decryption fails, then the aggregator MUST fail with error `hpke-decrypt-error`. (§4.4.2.2)
                        let hpke_keypair = hpke_keypairs.keypair(
                            prepare_init
                                .report_share()
                                .encrypted_input_share()
                                .config_id(),
                        ).ok_or_else(|| {
                            debug!(
                                config_id = %prepare_init.report_share().encrypted_input_share().config_id(),
                                "Helper encrypted input share references unknown HPKE config ID"
                            );
                            metrics
                                .aggregate_step_failure_counter
                                .add(1, &[KeyValue::new("type", "unknown_hpke_config_id")]);
                            ReportError::HpkeUnknownConfigId
                        });

                        let plaintext = hpke_keypair.and_then(|hpke_keypair| {
                            let input_share_aad = InputShareAad::new(
                                *task.id(),
                                prepare_init.report_share().metadata().clone(),
                                prepare_init.report_share().public_share().to_vec(),
                            )
                            .get_encoded()
                            .map_err(|err| {
                                debug!(
                                    task_id = %task.id(),
                                    report_id = ?prepare_init.report_share().metadata().id(),
                                    ?err,
                                    "Couldn't encode input share AAD"
                                );
                                metrics.aggregate_step_failure_counter.add(
                                    1,
                                    &[KeyValue::new("type", "input_share_aad_encode_failure")],
                                );
                                // HpkeDecryptError isn't strictly accurate, but given that this
                                // fallible encoding is part of the HPKE decryption process, I think
                                // this is as close as we can get to a meaningful error signal.
                                ReportError::HpkeDecryptError
                            })?;

                            hpke::open(
                                &hpke_keypair,
                                &HpkeApplicationInfo::new(
                                    &Label::InputShare,
                                    &Role::Client,
                                    &Role::Helper,
                                ),
                                prepare_init.report_share().encrypted_input_share(),
                                &input_share_aad,
                            )
                            .map_err(|error| {
                                debug!(
                                    task_id = %task.id(),
                                    report_id = ?prepare_init.report_share().metadata().id(),
                                    ?error,
                                    "Couldn't decrypt helper's report share"
                                );
                                metrics
                                    .aggregate_step_failure_counter
                                    .add(1, &[KeyValue::new("type", "decrypt_failure")]);
                                ReportError::HpkeDecryptError
                            })
                        });

                        let plaintext_input_share = plaintext.and_then(|plaintext| {
                            let plaintext_input_share = PlaintextInputShare::get_decoded(&plaintext)
                                .map_err(|error| {
                                    debug!(
                                        task_id = %task.id(),
                                        report_id = ?prepare_init.report_share().metadata().id(),
                                        ?error, "Couldn't decode helper's plaintext input share",
                                    );
                                    metrics.aggregate_step_failure_counter.add(
                                        1,
                                        &[KeyValue::new(
                                            "type",
                                            "plaintext_input_share_decode_failure",
                                        )],
                                    );
                                    ReportError::InvalidMessage
                                })?;

                            // Build map of extension type to extension data, checking for duplicates.
                            let mut extensions = HashMap::new();
                            if !plaintext_input_share.private_extensions().iter().chain(prepare_init.report_share().metadata().public_extensions()).all(|extension| {
                                extensions
                                    .insert(*extension.extension_type(), extension.extension_data())
                                    .is_none()
                            }) {
                                debug!(
                                    task_id = %task.id(),
                                    report_id = ?prepare_init.report_share().metadata().id(),
                                    "Received report share with duplicate extensions",
                                );
                                metrics
                                    .aggregate_step_failure_counter
                                    .add(1, &[KeyValue::new("type", "duplicate_extension")]);
                                return Err(ReportError::InvalidMessage);
                            }

                            if require_taskbind_extension {
                                let valid_taskbind_extension_present = extensions
                                    .get(&ExtensionType::Taskbind)
                                    .map(|data| data.is_empty())
                                    .unwrap_or(false);
                                if !valid_taskbind_extension_present {
                                    debug!(
                                        task_id = %task.id(),
                                        report_id = ?prepare_init.report_share().metadata().id(),
                                        "Taskprov task received report with missing or malformed \
                                        taskbind extension",
                                    );
                                    metrics.aggregate_step_failure_counter.add(
                                        1,
                                        &[KeyValue::new(
                                            "type",
                                            "missing_or_malformed_taskbind_extension",
                                        )],
                                    );
                                    return Err(ReportError::InvalidMessage);
                                }
                            } else if extensions.contains_key(&ExtensionType::Taskbind) {
                                // taskprov not enabled, but the taskbind extension is present.
                                debug!(
                                    task_id = %task.id(),
                                    report_id = ?prepare_init.report_share().metadata().id(),
                                    "Non-taskprov task received report with unexpected taskbind \
                                    extension",
                                );
                                metrics
                                    .aggregate_step_failure_counter
                                    .add(1, &[KeyValue::new("type", "unexpected_taskbind_extension")]);
                                return Err(ReportError::InvalidMessage);
                            }

                            Ok(plaintext_input_share)
                        });

                        let input_share = plaintext_input_share.and_then(|plaintext_input_share| {
                            A::InputShare::get_decoded_with_param(
                                &(&vdaf, Role::Helper.index().unwrap()),
                                plaintext_input_share.payload(),
                            )
                            .map_err(|error| {
                                debug!(
                                    task_id = %task.id(),
                                    report_id = ?prepare_init.report_share().metadata().id(),
                                    ?error, "Couldn't decode helper's input share",
                                );
                                metrics
                                    .aggregate_step_failure_counter
                                    .add(1, &[KeyValue::new("type", "input_share_decode_failure")]);
                                ReportError::InvalidMessage
                            })
                        });

                        let public_share = A::PublicShare::get_decoded_with_param(
                            &vdaf,
                            prepare_init.report_share().public_share(),
                        )
                        .map_err(|error| {
                            debug!(
                                task_id = %task.id(),
                                report_id = ?prepare_init.report_share().metadata().id(),
                                ?error, "Couldn't decode public share",
                            );
                            metrics
                                .aggregate_step_failure_counter
                                .add(1, &[KeyValue::new("type", "public_share_decode_failure")]);
                            ReportError::InvalidMessage
                        });

                        let shares =
                            input_share.and_then(|input_share| Ok((public_share?, input_share)));

                        // Reject reports from too far in the future.
                        let shares = shares.and_then(|shares| {
                            if prepare_init
                                .report_share()
                                .metadata()
                                .time()
                                .is_after(&report_deadline)
                            {
                                return Err(ReportError::ReportTooEarly);
                            }
                            Ok(shares)
                        });

                        // Next, the aggregator runs the preparation-state initialization algorithm for the VDAF
                        // associated with the task and computes the first state transition. [...] If either
                        // step fails, then the aggregator MUST fail with error `vdaf-prep-error`. (§4.4.2.2)
                        let init_rslt = shares.and_then(|(public_share, input_share)| {
                            trace_span!("VDAF preparation (helper initialization)").in_scope(|| {
                                vdaf.helper_initialized(
                                    verify_key.as_bytes(),
                                    &ctx,
                                    aggregation_job.aggregation_parameter(),
                                    /* report ID is used as VDAF nonce */
                                    prepare_init.report_share().metadata().id().as_ref(),
                                    &public_share,
                                    &input_share,
                                    prepare_init.message(),
                                )
                                .and_then(|transition| transition.evaluate(&ctx, &vdaf))
                                .map_err(|error| {
                                    handle_ping_pong_error(
                                        task.id(),
                                        Role::Helper,
                                        prepare_init.report_share().metadata().id(),
                                        error,
                                        &metrics.aggregate_step_failure_counter,
                                    )
                                })
                            })
                        });

                        let (report_aggregation_state, prepare_step_result, output_share) =
                            match init_rslt {
                                Ok((PingPongState::Continued(prepare_state), outgoing_message)) => {
                                    // Helper is not finished. Await the next message from the Leader to advance to
                                    // the next step.
                                    (
                                        ReportAggregationState::HelperContinue { prepare_state },
                                        PrepareStepResult::Continue {
                                            message: outgoing_message,
                                        },
                                        None,
                                    )
                                }
                                Ok((PingPongState::Finished(output_share), outgoing_message)) => (
                                    ReportAggregationState::Finished,
                                    PrepareStepResult::Continue {
                                        message: outgoing_message,
                                    },
                                    Some(output_share),
                                ),
                                Err(report_error) => (
                                    ReportAggregationState::Failed { report_error },
                                    PrepareStepResult::Reject(report_error),
                                    None,
                                ),
                            };

                        let report_id = *prepare_init.report_share().metadata().id();
                        sender.send(WritableReportAggregation::new(
                                report_aggregation
                                    .with_last_prep_resp(
                                        Some(PrepareResp::new(
                                            report_id,
                                            prepare_step_result,
                                        ))
                                    )
                                    .with_state(report_aggregation_state),
                                output_share
                            )
                        )
                    })
        }
    });

    let mut report_aggregations = Vec::with_capacity(report_aggregation_count);
    while receiver.recv_many(&mut report_aggregations, 10).await > 0 {}

    // Await the producer task to resume any panics that may have occurred, and to ensure the
    // producer task is completely done (e.g. all of its memory is released). The only other errors
    // that can occur are: a `JoinError` indicating cancellation, which is impossible because we do
    // not cancel the task; and a `SendError`, which can only happen if this future is cancelled (in
    // which case we will not run this code at all).
    let _ = producer_task.await.map_err(|join_error| {
        if let Ok(reason) = join_error.try_into_panic() {
            panic::resume_unwind(reason);
        }
    });
    assert_eq!(report_aggregations.len(), report_aggregation_count);

    Ok(report_aggregations)
}

#[cfg(feature = "test-util")]
#[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
pub mod test_util {
    use crate::aggregator::test_util::generate_helper_report_share;
    use janus_aggregator_core::task::{test_util::Task, AggregatorTask};
    use janus_core::{
        test_util::{run_vdaf, VdafTranscript},
        time::{Clock, MockClock, TimeExt as _},
    };
    use janus_messages::{
        batch_mode::{self},
        AggregationJobId, AggregationJobInitializeReq, Extension, HpkeConfig, PrepareInit,
        ReportMetadata, ReportShare,
    };
    use prio::{
        codec::Encode,
        vdaf::{self},
    };
    use rand::random;
    use trillium::{Handler, KnownHeaderName};
    use trillium_testing::{prelude::put, TestConn};

    #[derive(Clone)]
    pub struct PrepareInitGenerator<const VERIFY_KEY_SIZE: usize, V>
    where
        V: vdaf::Vdaf,
    {
        clock: MockClock,
        task: AggregatorTask,
        vdaf: V,
        aggregation_param: V::AggregationParam,
        hpke_config: HpkeConfig,
        private_extensions: Vec<Extension>,
    }

    impl<const VERIFY_KEY_SIZE: usize, V> PrepareInitGenerator<VERIFY_KEY_SIZE, V>
    where
        V: vdaf::Vdaf + vdaf::Aggregator<VERIFY_KEY_SIZE, 16> + vdaf::Client<16>,
    {
        pub fn new(
            clock: MockClock,
            task: AggregatorTask,
            hpke_config: HpkeConfig,
            vdaf: V,
            aggregation_param: V::AggregationParam,
        ) -> Self {
            Self {
                clock,
                task,
                vdaf,
                aggregation_param,
                hpke_config,
                private_extensions: Vec::new(),
            }
        }

        pub fn with_private_extensions(mut self, extensions: Vec<Extension>) -> Self {
            self.private_extensions = extensions;
            self
        }

        pub fn next(
            &self,
            measurement: &V::Measurement,
        ) -> (PrepareInit, VdafTranscript<VERIFY_KEY_SIZE, V>) {
            self.next_with_metadata(
                ReportMetadata::new(
                    random(),
                    self.clock
                        .now()
                        .to_batch_interval_start(self.task.time_precision())
                        .unwrap(),
                    Vec::new(),
                ),
                measurement,
            )
        }

        pub fn next_with_metadata(
            &self,
            report_metadata: ReportMetadata,
            measurement: &V::Measurement,
        ) -> (PrepareInit, VdafTranscript<VERIFY_KEY_SIZE, V>) {
            let (report_share, transcript) =
                self.next_report_share_with_metadata(report_metadata, measurement);
            (
                PrepareInit::new(
                    report_share,
                    transcript.leader_prepare_transitions[0].message.clone(),
                ),
                transcript,
            )
        }

        pub fn next_report_share(
            &self,
            measurement: &V::Measurement,
        ) -> (ReportShare, VdafTranscript<VERIFY_KEY_SIZE, V>) {
            self.next_report_share_with_metadata(
                ReportMetadata::new(
                    random(),
                    self.clock
                        .now()
                        .to_batch_interval_start(self.task.time_precision())
                        .unwrap(),
                    Vec::new(),
                ),
                measurement,
            )
        }

        pub fn next_report_share_with_metadata(
            &self,
            report_metadata: ReportMetadata,
            measurement: &V::Measurement,
        ) -> (ReportShare, VdafTranscript<VERIFY_KEY_SIZE, V>) {
            let transcript = run_vdaf(
                &self.vdaf,
                self.task.id(),
                self.task.vdaf_verify_key().unwrap().as_bytes(),
                &self.aggregation_param,
                report_metadata.id(),
                measurement,
            );
            let report_share = generate_helper_report_share::<V>(
                *self.task.id(),
                report_metadata,
                &self.hpke_config,
                &transcript.public_share,
                self.private_extensions.clone(),
                &transcript.helper_input_share,
            );
            (report_share, transcript)
        }
    }

    pub async fn put_aggregation_job<B: batch_mode::BatchMode>(
        task: &Task,
        aggregation_job_id: &AggregationJobId,
        aggregation_job: &AggregationJobInitializeReq<B>,
        handler: &impl Handler,
    ) -> TestConn {
        let (header, value) = task.aggregator_auth_token().request_authentication();

        put(task
            .aggregation_job_uri(aggregation_job_id, None)
            .unwrap()
            .path())
        .with_request_header(header, value)
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregationJobInitializeReq::<B>::MEDIA_TYPE,
        )
        .with_request_body(aggregation_job.get_encoded().unwrap())
        .run_async(handler)
        .await
    }
}

#[cfg(test)]
mod tests {
    use crate::aggregator::{
        aggregation_job_init::test_util::{put_aggregation_job, PrepareInitGenerator},
        http_handlers::{
            test_util::{decode_response_body, take_problem_details},
            AggregatorHandlerBuilder,
        },
        Config,
    };
    use assert_matches::assert_matches;
    use http::StatusCode;
    use janus_aggregator_core::{
        datastore::test_util::{ephemeral_datastore, EphemeralDatastore},
        task::{
            test_util::{Task, TaskBuilder},
            AggregationMode, BatchMode,
        },
        test_util::noop_meter,
    };
    use janus_core::{
        auth_tokens::{AuthenticationToken, DAP_AUTH_HEADER},
        test_util::{install_test_trace_subscriber, runtime::TestRuntime},
        time::{Clock, MockClock, TimeExt as _},
        vdaf::VdafInstance,
    };
    use janus_messages::{
        batch_mode::TimeInterval, AggregationJobId, AggregationJobInitializeReq,
        AggregationJobResp, Duration, Extension, ExtensionType, PartialBatchSelector, PrepareResp,
        PrepareStepResult, ReportError, ReportMetadata,
    };
    use prio::{
        codec::Encode,
        vdaf::{self, dummy},
    };
    use rand::random;
    use serde_json::json;
    use std::sync::Arc;
    use trillium::{Handler, KnownHeaderName, Status};
    use trillium_testing::prelude::put;

    pub(super) struct AggregationJobInitTestCase<
        const VERIFY_KEY_SIZE: usize,
        V: vdaf::Aggregator<VERIFY_KEY_SIZE, 16>,
    > {
        pub(super) clock: MockClock,
        pub(super) task: Task,
        pub(super) prepare_init_generator: PrepareInitGenerator<VERIFY_KEY_SIZE, V>,
        pub(super) aggregation_job_id: AggregationJobId,
        pub(super) aggregation_job_init_req: AggregationJobInitializeReq<TimeInterval>,
        aggregation_job_init_resp: Option<AggregationJobResp>,
        pub(super) aggregation_param: V::AggregationParam,
        pub(super) handler: Box<dyn Handler>,
        _ephemeral_datastore: EphemeralDatastore,
    }

    pub(super) async fn setup_aggregate_init_test() -> AggregationJobInitTestCase<0, dummy::Vdaf> {
        setup_aggregate_init_test_for_vdaf(
            dummy::Vdaf::new(1),
            VdafInstance::Fake { rounds: 1 },
            dummy::AggregationParam(0),
            0,
        )
        .await
    }

    async fn setup_multi_step_aggregate_init_test() -> AggregationJobInitTestCase<0, dummy::Vdaf> {
        setup_aggregate_init_test_for_vdaf(
            dummy::Vdaf::new(2),
            VdafInstance::Fake { rounds: 2 },
            dummy::AggregationParam(7),
            13,
        )
        .await
    }

    async fn setup_aggregate_init_test_for_vdaf<
        const VERIFY_KEY_SIZE: usize,
        V: vdaf::Aggregator<VERIFY_KEY_SIZE, 16> + vdaf::Client<16>,
    >(
        vdaf: V,
        vdaf_instance: VdafInstance,
        aggregation_param: V::AggregationParam,
        measurement: V::Measurement,
    ) -> AggregationJobInitTestCase<VERIFY_KEY_SIZE, V> {
        let mut test_case = setup_aggregate_init_test_without_sending_request(
            vdaf,
            vdaf_instance,
            aggregation_param,
            measurement,
            AuthenticationToken::Bearer(random()),
        )
        .await;

        let mut response = put_aggregation_job(
            &test_case.task,
            &test_case.aggregation_job_id,
            &test_case.aggregation_job_init_req,
            &test_case.handler,
        )
        .await;
        assert_eq!(response.status(), Some(Status::Created));

        let aggregation_job_resp: AggregationJobResp = decode_response_body(&mut response).await;
        let prepare_resps = assert_matches!(
            &aggregation_job_resp,
            AggregationJobResp::Finished { prepare_resps } => prepare_resps
        );
        assert_eq!(
            prepare_resps.len(),
            test_case.aggregation_job_init_req.prepare_inits().len(),
        );
        assert_matches!(
            prepare_resps[0].result(),
            &PrepareStepResult::Continue { .. }
        );

        test_case.aggregation_job_init_resp = Some(aggregation_job_resp);
        test_case
    }

    async fn setup_aggregate_init_test_without_sending_request<
        const VERIFY_KEY_SIZE: usize,
        V: vdaf::Aggregator<VERIFY_KEY_SIZE, 16> + vdaf::Client<16>,
    >(
        vdaf: V,
        vdaf_instance: VdafInstance,
        aggregation_param: V::AggregationParam,
        measurement: V::Measurement,
        auth_token: AuthenticationToken,
    ) -> AggregationJobInitTestCase<VERIFY_KEY_SIZE, V> {
        install_test_trace_subscriber();

        let task = TaskBuilder::new(
            BatchMode::TimeInterval,
            AggregationMode::Synchronous,
            vdaf_instance,
        )
        .with_aggregator_auth_token(auth_token)
        .build();
        let helper_task = task.helper_view().unwrap();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        datastore.put_aggregator_task(&helper_task).await.unwrap();
        let keypair = datastore.put_hpke_key().await.unwrap();

        let handler = AggregatorHandlerBuilder::new(
            Arc::clone(&datastore),
            clock.clone(),
            TestRuntime::default(),
            &noop_meter(),
            Config::default(),
        )
        .await
        .unwrap()
        .build()
        .unwrap();

        let prepare_init_generator = PrepareInitGenerator::new(
            clock.clone(),
            helper_task.clone(),
            keypair.config().clone(),
            vdaf,
            aggregation_param.clone(),
        );

        let prepare_inits = Vec::from([
            prepare_init_generator.next(&measurement).0,
            prepare_init_generator.next(&measurement).0,
        ]);

        let aggregation_job_id = random();
        let aggregation_job_init_req = AggregationJobInitializeReq::new(
            aggregation_param.get_encoded().unwrap(),
            PartialBatchSelector::new_time_interval(),
            prepare_inits.clone(),
        );

        AggregationJobInitTestCase {
            clock,
            task,
            prepare_init_generator,
            aggregation_job_id,
            aggregation_job_init_req,
            aggregation_job_init_resp: None,
            aggregation_param,
            handler: Box::new(handler),
            _ephemeral_datastore: ephemeral_datastore,
        }
    }

    #[tokio::test]
    async fn aggregation_job_init_authorization_dap_auth_token() {
        let test_case = setup_aggregate_init_test_without_sending_request(
            dummy::Vdaf::new(1),
            VdafInstance::Fake { rounds: 1 },
            dummy::AggregationParam(0),
            0,
            AuthenticationToken::DapAuth(random()),
        )
        .await;

        let (auth_header, auth_value) = test_case
            .task
            .aggregator_auth_token()
            .request_authentication();

        let response = put(test_case
            .task
            .aggregation_job_uri(&test_case.aggregation_job_id, None)
            .unwrap()
            .path())
        .with_request_header(auth_header, auth_value)
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
        )
        .with_request_body(test_case.aggregation_job_init_req.get_encoded().unwrap())
        .run_async(&test_case.handler)
        .await;

        assert_eq!(response.status(), Some(Status::Created));
    }

    #[rstest::rstest]
    #[case::not_bearer_token("wrong kind of token")]
    #[case::not_base64("Bearer: ")]
    #[tokio::test]
    async fn aggregation_job_init_malformed_authorization_header(
        #[case] header_value: &'static str,
    ) {
        let test_case = setup_aggregate_init_test_without_sending_request(
            dummy::Vdaf::new(1),
            VdafInstance::Fake { rounds: 1 },
            dummy::AggregationParam(0),
            0,
            AuthenticationToken::Bearer(random()),
        )
        .await;

        let response = put(test_case
            .task
            .aggregation_job_uri(&test_case.aggregation_job_id, None)
            .unwrap()
            .path())
        // Authenticate using a malformed "Authorization: Bearer <token>" header and a `DAP-Auth-Token`
        // header. The presence of the former should cause an error despite the latter being present and
        // well formed.
        .with_request_header(KnownHeaderName::Authorization, header_value.to_string())
        .with_request_header(
            DAP_AUTH_HEADER,
            test_case.task.aggregator_auth_token().as_ref().to_owned(),
        )
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
        )
        .with_request_body(test_case.aggregation_job_init_req.get_encoded().unwrap())
        .run_async(&test_case.handler)
        .await;

        assert_eq!(response.status(), Some(Status::Forbidden));
    }

    #[tokio::test]
    async fn aggregation_job_init_unexpected_taskbind_extension() {
        let test_case = setup_aggregate_init_test_without_sending_request(
            dummy::Vdaf::new(1),
            VdafInstance::Fake { rounds: 1 },
            dummy::AggregationParam(0),
            0,
            random(),
        )
        .await;

        let prepare_init = test_case
            .prepare_init_generator
            .clone()
            .with_private_extensions(Vec::from([Extension::new(
                ExtensionType::Taskbind,
                Vec::new(),
            )]))
            .next(&0)
            .0;
        let report_id = *prepare_init.report_share().metadata().id();
        let aggregation_job_init_req = AggregationJobInitializeReq::new(
            dummy::AggregationParam(1).get_encoded().unwrap(),
            PartialBatchSelector::new_time_interval(),
            Vec::from([prepare_init]),
        );

        let mut response = put_aggregation_job(
            &test_case.task,
            &test_case.aggregation_job_id,
            &aggregation_job_init_req,
            &test_case.handler,
        )
        .await;
        assert_eq!(response.status(), Some(Status::Created));

        let want_aggregation_job_resp = AggregationJobResp::Finished {
            prepare_resps: Vec::from([PrepareResp::new(
                report_id,
                PrepareStepResult::Reject(ReportError::InvalidMessage),
            )]),
        };
        let got_aggregation_job_resp: AggregationJobResp =
            decode_response_body(&mut response).await;
        assert_eq!(want_aggregation_job_resp, got_aggregation_job_resp);
    }

    #[tokio::test]
    async fn aggregation_job_mutation_aggregation_job() {
        let test_case = setup_aggregate_init_test().await;

        // Put the aggregation job again, but with a different aggregation parameter.
        let mutated_aggregation_job_init_req = AggregationJobInitializeReq::new(
            dummy::AggregationParam(1).get_encoded().unwrap(),
            PartialBatchSelector::new_time_interval(),
            test_case.aggregation_job_init_req.prepare_inits().to_vec(),
        );

        let response = put_aggregation_job(
            &test_case.task,
            &test_case.aggregation_job_id,
            &mutated_aggregation_job_init_req,
            &test_case.handler,
        )
        .await;
        assert_eq!(response.status(), Some(Status::Conflict));
    }

    #[tokio::test]
    async fn aggregation_job_mutation_report_shares() {
        let test_case = setup_aggregate_init_test().await;

        let prepare_inits = test_case.aggregation_job_init_req.prepare_inits();

        // Put the aggregation job again, mutating the associated report shares' metadata such that
        // uniqueness constraints on client_reports are violated
        for mutated_prepare_inits in [
            // Omit a report share that was included previously
            Vec::from(&prepare_inits[0..prepare_inits.len() - 1]),
            // Include a different report share than was included previously
            [
                &prepare_inits[0..prepare_inits.len() - 1],
                &[test_case.prepare_init_generator.next(&0).0],
            ]
            .concat(),
            // Include an extra report share than was included previously
            [
                prepare_inits,
                &[test_case.prepare_init_generator.next(&0).0],
            ]
            .concat(),
            // Reverse the order of the reports
            prepare_inits.iter().rev().cloned().collect(),
        ] {
            let mutated_aggregation_job_init_req = AggregationJobInitializeReq::new(
                test_case.aggregation_param.get_encoded().unwrap(),
                PartialBatchSelector::new_time_interval(),
                mutated_prepare_inits,
            );
            let response = put_aggregation_job(
                &test_case.task,
                &test_case.aggregation_job_id,
                &mutated_aggregation_job_init_req,
                &test_case.handler,
            )
            .await;
            assert_eq!(response.status(), Some(Status::Conflict));
        }
    }

    #[tokio::test]
    async fn aggregation_job_mutation_report_aggregations() {
        // We set up a multi-step VDAF in this test so that the aggregation job won't finish on the
        // first step.
        let test_case = setup_multi_step_aggregate_init_test().await;

        // Generate some new reports using the existing reports' metadata, but varying the measurement
        // values such that the prepare state computed during aggregation initializaton won't match the
        // first aggregation job.
        let mutated_prepare_inits = test_case
            .aggregation_job_init_req
            .prepare_inits()
            .iter()
            .map(|s| {
                test_case
                    .prepare_init_generator
                    .next_with_metadata(s.report_share().metadata().clone(), &13)
                    .0
            })
            .collect();

        let mutated_aggregation_job_init_req = AggregationJobInitializeReq::new(
            test_case.aggregation_param.get_encoded().unwrap(),
            PartialBatchSelector::new_time_interval(),
            mutated_prepare_inits,
        );

        let response = put_aggregation_job(
            &test_case.task,
            &test_case.aggregation_job_id,
            &mutated_aggregation_job_init_req,
            &test_case.handler,
        )
        .await;
        assert_eq!(response.status(), Some(Status::Conflict));
    }

    #[tokio::test]
    async fn aggregation_job_intolerable_clock_skew() {
        let mut test_case = setup_aggregate_init_test_without_sending_request(
            dummy::Vdaf::new(1),
            VdafInstance::Fake { rounds: 1 },
            dummy::AggregationParam(0),
            0,
            AuthenticationToken::Bearer(random()),
        )
        .await;

        test_case.aggregation_job_init_req = AggregationJobInitializeReq::new(
            test_case.aggregation_param.get_encoded().unwrap(),
            PartialBatchSelector::new_time_interval(),
            Vec::from([
                // Barely tolerable.
                test_case
                    .prepare_init_generator
                    .next_with_metadata(
                        ReportMetadata::new(
                            random(),
                            test_case
                                .clock
                                .now()
                                .add(test_case.task.tolerable_clock_skew())
                                .unwrap(),
                            Vec::new(),
                        ),
                        &0,
                    )
                    .0,
                // Barely intolerable.
                test_case
                    .prepare_init_generator
                    .next_with_metadata(
                        ReportMetadata::new(
                            random(),
                            test_case
                                .clock
                                .now()
                                .add(test_case.task.tolerable_clock_skew())
                                .unwrap()
                                .add(&Duration::from_seconds(1))
                                .unwrap(),
                            Vec::new(),
                        ),
                        &0,
                    )
                    .0,
            ]),
        );

        let mut response = put_aggregation_job(
            &test_case.task,
            &test_case.aggregation_job_id,
            &test_case.aggregation_job_init_req,
            &test_case.handler,
        )
        .await;
        assert_eq!(response.status(), Some(Status::Created));

        let aggregation_job_resp: AggregationJobResp = decode_response_body(&mut response).await;
        let prepare_resps = assert_matches!(
            aggregation_job_resp,
            AggregationJobResp::Finished { prepare_resps } => prepare_resps
        );
        assert_eq!(
            prepare_resps.len(),
            test_case.aggregation_job_init_req.prepare_inits().len(),
        );
        assert_matches!(
            prepare_resps[0].result(),
            &PrepareStepResult::Continue { .. }
        );
        assert_matches!(
            prepare_resps[1].result(),
            &PrepareStepResult::Reject(ReportError::ReportTooEarly)
        );
    }

    #[tokio::test]
    async fn aggregation_job_init_two_step_vdaf_idempotence() {
        // We set up a multi-step VDAF in this test so that the aggregation job won't finish on the
        // first step.
        let test_case = setup_multi_step_aggregate_init_test().await;

        // Send the aggregation job init request again. We should get an identical response back.
        let mut response = put_aggregation_job(
            &test_case.task,
            &test_case.aggregation_job_id,
            &test_case.aggregation_job_init_req,
            &test_case.handler,
        )
        .await;

        let aggregation_job_resp: AggregationJobResp = decode_response_body(&mut response).await;
        assert_eq!(
            aggregation_job_resp,
            test_case.aggregation_job_init_resp.unwrap(),
        );
    }

    #[tokio::test]
    async fn aggregation_job_init_wrong_query() {
        let test_case = setup_aggregate_init_test().await;

        // setup_aggregate_init_test sets up a task with a time interval query. We send a
        // leader-selected query which should yield an error.
        let wrong_query = AggregationJobInitializeReq::new(
            test_case.aggregation_param.get_encoded().unwrap(),
            PartialBatchSelector::new_leader_selected(random()),
            test_case.aggregation_job_init_req.prepare_inits().to_vec(),
        );

        let (header, value) = test_case
            .task
            .aggregator_auth_token()
            .request_authentication();

        let mut response = put(test_case
            .task
            .aggregation_job_uri(&random(), None)
            .unwrap()
            .path())
        .with_request_header(header, value)
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
        )
        .with_request_body(wrong_query.get_encoded().unwrap())
        .run_async(&test_case.handler)
        .await;
        assert_eq!(
            take_problem_details(&mut response).await,
            json!({
                "status": StatusCode::BAD_REQUEST.as_u16(),
                "type": "urn:ietf:params:ppm:dap:error:invalidMessage",
                "title": "The message type for a response was incorrect or the payload was malformed.",
            }),
        );
    }
}
