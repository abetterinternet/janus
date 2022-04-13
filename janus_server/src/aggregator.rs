//! Common functionality for PPM aggregators
use crate::{
    datastore::{
        self,
        models::{AggregationJob, AggregationJobState, ReportAggregation, ReportAggregationState},
        Datastore,
    },
    hpke::HpkeRecipient,
    message::{
        AggregateReq,
        AggregateReqBody::{AggregateContinueReq, AggregateInitReq},
        AggregateResp, AggregationJobId, AuthenticatedDecodeError, AuthenticatedEncoder,
        AuthenticatedRequestDecoder, HpkeConfigId, Nonce, Report, ReportShare, Role, TaskId,
        Transition, TransitionError, TransitionTypeSpecificData,
    },
    time::Clock,
};
use bytes::Bytes;
use chrono::Duration;
use futures::future;
use http::{header::CACHE_CONTROL, StatusCode};
use prio::{
    codec::{Decode, Encode, ParameterizedDecode},
    vdaf::{self, PrepareTransition},
};
use ring::hmac;
use std::{
    collections::HashSet, convert::Infallible, future::Future, io::Cursor, net::SocketAddr,
    ops::Sub, pin::Pin, sync::Arc,
};
use tracing::{error, warn};
use warp::{
    filters::BoxedFilter,
    reply::{self, Response},
    trace, Filter, Rejection, Reply,
};

/// Errors returned by functions and methods in this module
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An invalid configuration was passed.
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(&'static str),
    /// Error decoding an incoming message.
    #[error("message decoding failed: {0}")]
    MessageDecode(#[from] prio::codec::CodecError),
    /// Corresponds to `staleReport`, §3.1
    #[error("stale report: {0} {1:?}")]
    StaleReport(Nonce, TaskId),
    /// Corresponds to `unrecognizedMessage`, §3.1
    #[error("unrecognized message: {0} {1:?}")]
    UnrecognizedMessage(&'static str, TaskId),
    /// Corresponds to `unrecognizedTask`, §3.1
    #[error("unrecognized task: {0:?}")]
    UnrecognizedTask(TaskId),
    /// Corresponds to `outdatedHpkeConfig`, §3.1
    #[error("outdated HPKE config: {0} {1:?}")]
    OutdatedHpkeConfig(HpkeConfigId, TaskId),
    /// A report was rejected becuase the timestamp is too far in the future,
    /// §4.3.4.
    // TODO(timg): define an error type in §3.1 and clarify language on
    // rejecting future reports
    #[error("report from the future: {0} {1:?}")]
    ReportFromTheFuture(Nonce, TaskId),
    /// Corresponds to `invalidHmac`, §3.1
    #[error("invalid HMAC tag: {0:?}")]
    InvalidHmac(TaskId),
    /// An error from the datastore.
    #[error("datastore error: {0}")]
    Datastore(datastore::Error),
    /// An error representing a generic internal aggregation error; intended for "impossible"
    /// conditions.
    #[error("internal aggregator error: {0}")]
    Internal(String),
}

// This From implementation ensures that we don't end up with e.g.
// Error::Datastore(datastore::Error::User(Error::...)) by automatically unwrapping to the internal
// aggregator error if converting a datastore::Error::User that contains an Error. Other
// datastore::Error values are wrapped in Error::Datastore unchanged.
impl From<datastore::Error> for Error {
    fn from(err: datastore::Error) -> Self {
        match err {
            datastore::Error::User(err) => match err.downcast::<Error>() {
                Ok(err) => *err,
                Err(err) => Error::Datastore(datastore::Error::User(err)),
            },
            _ => Error::Datastore(err),
        }
    }
}

/// A PPM aggregator.
// TODO: refactor Aggregator to be non-task-specific (look up task-specific data based on task ID)
// TODO: refactor Aggregator to perform indepedent batched operations (e.g. report handling in
//       Aggregate requests) using a parallelized library like Rayon.
// TODO: refactor Aggregator to support multiple HPKE configs, switch from storing an HpkeRecipient
//       to storing Vec<(HpkeConfig, HpkePrivateKey>).
#[derive(Clone, derivative::Derivative)]
#[derivative(Debug)]
pub struct Aggregator<A: vdaf::Aggregator, C: Clock>
where
    for<'a> &'a A::AggregateShare: Into<Vec<u8>>,
{
    /// The VDAF in use.
    vdaf: A,
    /// The datastore used for durable storage.
    #[derivative(Debug = "ignore")]
    datastore: Arc<Datastore>,
    /// The clock to use to sample time.
    clock: C,
    /// How much clock skew to allow between client and aggregator. Reports from
    /// farther than this duration into the future will be rejected.
    tolerable_clock_skew: Duration,
    /// Role of this aggregator.
    role: Role,
    /// The verify parameter for the task.
    verify_param: A::VerifyParam,
    /// Used to decrypt reports received by this aggregator.
    // TODO: Aggregators should have multiple generations of HPKE config
    // available to decrypt tardy reports
    report_recipient: HpkeRecipient,
    /// The key used to authenticate aggregation messages for this task.
    agg_auth_key: hmac::Key,
}

impl<A: vdaf::Aggregator, C: Clock> Aggregator<A, C>
where
    A: 'static + Send + Sync,
    A::AggregationParam: Send + Sync,
    for<'a> &'a A::AggregateShare: Into<Vec<u8>>,
    A::PrepareStep: Send + Sync + Encode + ParameterizedDecode<A::VerifyParam>,
    A::PrepareMessage: Send + Sync,
    A::OutputShare: Send + Sync + for<'a> TryFrom<&'a [u8]>,
    for<'a> &'a A::OutputShare: Into<Vec<u8>>,
    A::VerifyParam: Send + Sync,
{
    /// Create a new aggregator. `report_recipient` is used to decrypt reports
    /// received by this aggregator.
    fn new(
        vdaf: A,
        datastore: Arc<Datastore>,
        clock: C,
        tolerable_clock_skew: Duration,
        role: Role,
        verify_param: A::VerifyParam,
        report_recipient: HpkeRecipient,
        agg_auth_key: hmac::Key,
    ) -> Result<Self, Error> {
        if tolerable_clock_skew < Duration::zero() {
            return Err(Error::InvalidConfiguration(
                "tolerable clock skew must be non-negative",
            ));
        }

        Ok(Self {
            vdaf,
            datastore,
            clock,
            tolerable_clock_skew,
            role,
            verify_param,
            report_recipient,
            agg_auth_key,
        })
    }

    /// Implements the `/upload` endpoint for the leader, described in §4.2 of
    /// draft-gpew-priv-ppm.
    async fn handle_upload(&self, report: &Report) -> Result<(), Error> {
        // §4.2.2 The leader's report is the first one
        if report.encrypted_input_shares.len() != 2 {
            warn!(
                share_count = report.encrypted_input_shares.len(),
                "unexpected number of encrypted shares in report"
            );
            return Err(Error::UnrecognizedMessage(
                "unexpected number of encrypted shares in report",
                report.task_id,
            ));
        }
        let leader_report = &report.encrypted_input_shares[0];

        // §4.2.2: verify that the report's HPKE config ID is known
        if leader_report.config_id != self.report_recipient.config().id {
            warn!(
                config_id = ?leader_report.config_id,
                "unknown HPKE config ID"
            );
            return Err(Error::OutdatedHpkeConfig(
                leader_report.config_id,
                report.task_id,
            ));
        }

        let now = self.clock.now();

        // §4.2.4: reject reports from too far in the future
        if report.nonce.time.as_naive_date_time().sub(now) > self.tolerable_clock_skew {
            warn!(?report.task_id, ?report.nonce, "report timestamp exceeds tolerable clock skew");
            return Err(Error::ReportFromTheFuture(report.nonce, report.task_id));
        }

        // Check that we can decrypt the report. This isn't required by the spec
        // but this exercises HPKE decryption and saves us the trouble of
        // storing reports we can't use. We don't inform the client if this
        // fails.
        if let Err(error) = self.report_recipient.open(
            leader_report,
            &Report::associated_data(report.nonce, &report.extensions),
        ) {
            warn!(?report.task_id, ?report.nonce, ?error, "report decryption failed");
            return Ok(());
        }

        self.datastore
            .run_tx(|tx| {
                let report = report.clone();
                Box::pin(async move {
                    // §4.2.2 and 4.3.2.2: reject reports whose nonce has been seen before
                    match tx.get_client_report(report.task_id, report.nonce).await {
                        Ok(_) => {
                            warn!(?report.task_id, ?report.nonce, "report replayed");
                            // TODO (issue #34): change this error type.
                            return Err(datastore::Error::User(
                                Error::StaleReport(report.nonce, report.task_id).into(),
                            ));
                        }

                        Err(datastore::Error::NotFound) => (), // happy path

                        Err(err) => return Err(err),
                    };

                    // TODO: reject with `staleReport` reports whose timestamps fall in a
                    // batch interval that has already been collected (§4.3.2). We don't
                    // support collection so we can't implement this requirement yet.

                    // Store the report.
                    tx.put_client_report(&report).await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Implements the `/aggregate` endpoint for the helper, described in §4.4.4.1 & §4.4.4.2 of
    /// draft-gpew-priv-ppm.
    async fn handle_aggregate(&self, req: AggregateReq) -> Result<AggregateResp, Error> {
        match req.body {
            AggregateInitReq { agg_param, seq } => {
                self.handle_aggregate_init(req.task_id, req.job_id, agg_param, seq)
                    .await
            }
            AggregateContinueReq { seq } => {
                self.handle_aggregate_continue(req.task_id, req.job_id, seq)
                    .await
            }
        }
    }

    /// Implements the aggregate initialization request portion of the `/aggregate` endpoint for the
    /// helper, described in §4.4.4.1 of draft-gpew-priv-ppm.
    async fn handle_aggregate_init(
        &self,
        task_id: TaskId,
        job_id: AggregationJobId,
        agg_param: Vec<u8>,
        report_shares: Vec<ReportShare>,
    ) -> Result<AggregateResp, Error> {
        // If two ReportShare messages have the same nonce, then the helper MUST abort with
        // error "unrecognizedMessage". (§4.4.4.1)
        let mut seen_nonces = HashSet::with_capacity(report_shares.len());
        for share in &report_shares {
            if !seen_nonces.insert(share.nonce) {
                return Err(Error::UnrecognizedMessage(
                    "aggregate request contains duplicate nonce",
                    task_id,
                ));
            }
        }

        // Decrypt shares & prepare initialization states. (§4.4.4.1)
        // TODO(brandon): reject reports that are "too old" with `report-dropped`.
        // TODO(brandon): reject reports in batches that have completed an aggregate-share request with `batch-collected`.
        struct ReportShareData<A: vdaf::Aggregator>
        where
            for<'a> &'a A::AggregateShare: Into<Vec<u8>>,
        {
            report_share: ReportShare,
            trans_data: TransitionTypeSpecificData,
            agg_state: ReportAggregationState<A>,
        }
        let mut saw_continue = false;
        let mut saw_finish = false;
        let mut report_share_data = Vec::new();
        let agg_param = A::AggregationParam::get_decoded(&agg_param)?;
        for report_share in report_shares {
            // TODO(brandon): once we have multiple config IDs in use, reject reports with an unknown config ID with `HpkeUnknownConfigId`.

            // If decryption fails, then the aggregator MUST fail with error `hpke-decrypt-error`. (§4.4.2.2)
            let plaintext = self
                .report_recipient
                .open(
                    &report_share.encrypted_input_share,
                    &Report::associated_data(report_share.nonce, &report_share.extensions),
                )
                .map_err(|err| {
                    warn!(
                        ?task_id,
                        nonce = %report_share.nonce,
                        %err,
                        "Couldn't decrypt report share"
                    );
                    TransitionError::HpkeDecryptError
                });

            // `vdaf-prep-error` probably isn't the right code, but there is no better one & we
            // don't want to fail the entire aggregation job with an UnrecognizedMessage error
            // because a single client sent bad data.
            // TODO: agree on/standardize an error code for "client report data can't be decoded" & use it here
            let input_share = plaintext.and_then(|plaintext| {
                A::InputShare::get_decoded_with_param(&self.verify_param, &plaintext).map_err(
                    |err| {
                        warn!(?task_id, nonce = %report_share.nonce, %err, "Couldn't decode input share from report share");
                        TransitionError::VdafPrepError
                    },
                )
            });

            // Next, the aggregator runs the preparation-state initialization algorithm for the VDAF
            // associated with the task and computes the first state transition. [...] If either
            // step fails, then the aggregator MUST fail with error `vdaf-prep-error`. (§4.4.2.2)
            let step = input_share.and_then(|input_share| {
                self.vdaf
                    .prepare_init(
                        &self.verify_param,
                        &agg_param,
                        &report_share.nonce.get_encoded(),
                        &input_share,
                    )
                    .map_err(|err| {
                        warn!(?task_id, nonce = %report_share.nonce, %err, "Couldn't prepare_init report share");
                        TransitionError::VdafPrepError
                    })
            });
            let prep_trans = step.map(|step| self.vdaf.prepare_step(step, None));

            report_share_data.push(match prep_trans {
                Ok(PrepareTransition::Continue(prep_step, prep_msg)) => {
                    saw_continue = true;
                    ReportShareData {
                        report_share,
                        trans_data: TransitionTypeSpecificData::Continued {
                            payload: prep_msg.get_encoded(),
                        },
                        agg_state: ReportAggregationState::<A>::Waiting(prep_step),
                    }
                }

                Ok(PrepareTransition::Finish(output_share)) => {
                    saw_finish = true;
                    ReportShareData {
                        report_share,
                        trans_data: TransitionTypeSpecificData::Finished,
                        agg_state: ReportAggregationState::<A>::Finished(output_share),
                    }
                }

                Ok(PrepareTransition::Fail(err)) => {
                    warn!(?task_id, nonce = %report_share.nonce, %err, "Couldn't prepare_step report share");
                    ReportShareData {
                        report_share,
                        trans_data: TransitionTypeSpecificData::Failed {
                            error: TransitionError::VdafPrepError,
                        },
                        agg_state: ReportAggregationState::<A>::Failed(TransitionError::VdafPrepError),
                    }
                },

                Err(err) => ReportShareData {
                    report_share,
                    trans_data: TransitionTypeSpecificData::Failed { error: err },
                    agg_state: ReportAggregationState::<A>::Failed(err),
                },
            });
        }

        // Store data to datastore.
        let aggregation_job_state = match (saw_continue, saw_finish) {
            (false, false) => AggregationJobState::Finished, // everything failed, or there were no reports
            (true, false) => AggregationJobState::InProgress,
            (false, true) => AggregationJobState::Finished,
            (true, true) => {
                return Err(Error::Internal(
                    "VDAF took an inconsistent number of rounds to reach Finish state".to_string(),
                ))
            }
        };
        let aggregation_job = Arc::new(AggregationJob::<A> {
            aggregation_job_id: job_id,
            task_id,
            aggregation_param: agg_param,
            state: aggregation_job_state,
        });
        let report_share_data = Arc::new(report_share_data);
        self.datastore
            .run_tx(|tx| {
                let aggregation_job = aggregation_job.clone();
                let report_share_data = report_share_data.clone();
                Box::pin(async move {
                    // Write aggregation job.
                    tx.put_aggregation_job(&aggregation_job).await?;

                    for (ord, share_data) in report_share_data.as_ref().iter().enumerate() {
                        // Write client report & report aggregation.
                        tx.put_report_share(task_id, &share_data.report_share)
                            .await?;
                        tx.put_report_aggregation(&ReportAggregation::<A> {
                            aggregation_job_id: job_id,
                            task_id,
                            nonce: share_data.report_share.nonce,
                            ord: ord as i64,
                            state: share_data.agg_state.clone(),
                        })
                        .await?;
                    }
                    Ok(())
                })
            })
            .await?;

        // Construct response and return.
        Ok(AggregateResp {
            seq: report_share_data
                .as_ref()
                .iter()
                .map(|d| Transition {
                    nonce: d.report_share.nonce,
                    trans_data: d.trans_data.clone(),
                })
                .collect(),
        })
    }

    async fn handle_aggregate_continue(
        &self,
        task_id: TaskId,
        job_id: AggregationJobId,
        transitions: Vec<Transition>,
    ) -> Result<AggregateResp, Error> {
        let vdaf = Arc::new(self.vdaf.clone());
        let verify_param = Arc::new(self.verify_param.clone());
        let transitions = Arc::new(transitions);

        // TODO(brandon): don't hold DB transaction open while computing VDAF updates?
        // TODO(brandon): don't do O(n) network round-trips (where n is the number of transitions)
        Ok(self
            .datastore
            .run_tx(|tx| {
                let vdaf = vdaf.clone();
                let verify_param = verify_param.clone();
                let transitions = transitions.clone();

                Box::pin(async move {
                    // Read existing state.
                    let (mut aggregation_job, report_aggregations) = future::try_join(
                        tx.get_aggregation_job::<A>(task_id, job_id),
                        tx.get_report_aggregations_for_aggregation_job::<A>(
                            &verify_param,
                            task_id,
                            job_id,
                        ),
                    )
                    .await?;

                    // Handle each transition in the request.
                    let mut report_aggregations = report_aggregations.into_iter();
                    let (mut saw_continue, mut saw_finish) = (false, false);
                    let mut response_transitions = Vec::new();
                    for transition in transitions.iter() {
                        // Match transition received from leader to stored report aggregation, and
                        // extract the stored preparation step.
                        let mut report_aggregation = loop {
                            let mut report_agg = report_aggregations.next().ok_or_else(|| {
                                warn!(?task_id, ?job_id, nonce = %transition.nonce, "Leader sent unexpected, duplicate, or out-of-order transitions");
                                datastore::Error::User(Box::new(Error::UnrecognizedMessage(
                                    "leader sent unexpected, duplicate, or out-of-order transitions",
                                    task_id,
                                )))
                            })?;
                            if report_agg.nonce != transition.nonce {
                                // This report was omitted by the leader because of a prior failure.
                                // Note that the report was dropped (if it's not already in an error
                                // state) and continue.
                                if matches!(report_agg.state, ReportAggregationState::Waiting(_)) {
                                    report_agg.state = ReportAggregationState::Failed(TransitionError::ReportDropped);
                                    tx.update_report_aggregation(&report_agg).await?;
                                }
                                continue;
                            }
                            break report_agg;
                        };
                        let prep_step =
                            match report_aggregation.state {
                                ReportAggregationState::Waiting(prep_step) => prep_step,
                                _ => {
                                    warn!(?task_id, ?job_id, nonce = %transition.nonce, "Leader sent transition for non-WAITING report aggregation");
                                    return Err(datastore::Error::User(Box::new(
                                        Error::UnrecognizedMessage(
                                            "leader sent transition for non-WAITING report aggregation",
                                            task_id,
                                        ),
                                    )));
                                },
                            };

                        // Parse preparation message out of transition received from leader.
                        let prep_msg = match &transition.trans_data {
                            TransitionTypeSpecificData::Continued { payload } => {
                                A::PrepareMessage::decode_with_param(
                                    &prep_step,
                                    &mut Cursor::new(payload),
                                )?
                            }
                            _ => {
                                // TODO(brandon): should we record a state change in this case?
                                warn!(?task_id, ?job_id, nonce = %transition.nonce, "Leader sent non-Continued transition");
                                return Err(datastore::Error::User(Box::new(
                                    Error::UnrecognizedMessage(
                                        "leader sent non-Continued transition",
                                        task_id,
                                    ),
                                )));
                            }
                        };

                        // Compute the next transition, prepare to respond & update DB.
                        let prep_trans = vdaf.prepare_step(prep_step, Some(prep_msg));
                        match prep_trans {
                            PrepareTransition::Continue(prep_step, prep_msg) => {
                                saw_continue = true;
                                report_aggregation.state =
                                    ReportAggregationState::Waiting(prep_step);
                                response_transitions.push(Transition {
                                    nonce: transition.nonce,
                                    trans_data: TransitionTypeSpecificData::Continued {
                                        payload: prep_msg.get_encoded(),
                                    },
                                })
                            }

                            PrepareTransition::Finish(output_share) => {
                                saw_finish = true;
                                report_aggregation.state =
                                    ReportAggregationState::Finished(output_share);
                                response_transitions.push(Transition {
                                    nonce: transition.nonce,
                                    trans_data: TransitionTypeSpecificData::Finished,
                                });
                            }

                            PrepareTransition::Fail(err) => {
                                warn!(?task_id, ?job_id, nonce = %transition.nonce, %err, "Prepare step failed");
                                report_aggregation.state =
                                    ReportAggregationState::Failed(TransitionError::VdafPrepError);
                                response_transitions.push(Transition {
                                    nonce: transition.nonce,
                                    trans_data: TransitionTypeSpecificData::Failed {
                                        error: TransitionError::VdafPrepError,
                                    },
                                })
                            }
                        }

                        tx.update_report_aggregation(&report_aggregation).await?;
                    }

                    for mut report_agg in report_aggregations {
                        // This report was omitted by the leader because of a prior failure.
                        // Note that the report was dropped (if it's not already in an error state)
                        // and continue.
                        if matches!(report_agg.state, ReportAggregationState::Waiting(_)) {
                            report_agg.state = ReportAggregationState::Failed(TransitionError::ReportDropped);
                            tx.update_report_aggregation(&report_agg).await?;
                        }
                    }

                    aggregation_job.state = match (saw_continue, saw_finish) {
                        (false, false) => AggregationJobState::Finished, // everything failed, or there were no reports
                        (true, false) => AggregationJobState::InProgress,
                        (false, true) => AggregationJobState::Finished,
                        (true, true) => {
                            return Err(datastore::Error::User(Box::new(Error::Internal(
                                "VDAF took an inconsistent number of rounds to reach Finish state"
                                    .to_string(),
                            ))))
                        }
                    };
                    tx.update_aggregation_job(&aggregation_job).await?;

                    Ok(AggregateResp {
                        seq: response_transitions,
                    })
                })
            })
            .await?)
    }
}

/// Injects a clone of the provided value into the warp filter, making it
/// available to the filter's map() or and_then() handler.
fn with_cloned_value<T>(value: T) -> impl Filter<Extract = (T,), Error = Infallible> + Clone
where
    T: Clone + Sync + Send,
{
    warp::any().map(move || value.clone())
}

fn with_decoded_body<T: Decode + Send + Sync>(
) -> impl Filter<Extract = (Result<T, Error>,), Error = Rejection> + Clone {
    warp::body::bytes().map(|body: Bytes| T::get_decoded(&body).map_err(Error::from))
}

fn with_authenticated_body<T, KeyFn>(
    key_fn: KeyFn,
) -> impl Filter<Extract = (Result<(hmac::Key, T), Error>,), Error = Rejection> + Clone
where
    T: Decode + Send + Sync,
    KeyFn: Fn(&TaskId) -> Pin<Box<dyn Future<Output = Result<hmac::Key, Error>> + Send + Sync>>
        + Clone
        + Send
        + Sync,
{
    warp::body::bytes().then(move |body: Bytes| {
        let key_fn = key_fn.clone();
        async move {
            // TODO(brandon): avoid copying the body here (make AuthenticatedRequestDecoder operate on Bytes or &[u8] or AsRef<[u8]>, most likely)
            let decoder: AuthenticatedRequestDecoder<T> =
                AuthenticatedRequestDecoder::new(Vec::from(body.as_ref())).map_err(Error::from)?;
            let task_id = decoder.task_id();
            let key = key_fn(&task_id).await?;
            let decoded_body: T = decoder.decode(&key).map_err(|err| match err {
                AuthenticatedDecodeError::InvalidHmac => Error::InvalidHmac(task_id),
                AuthenticatedDecodeError::Codec(err) => Error::MessageDecode(err),
            })?;
            Ok((key, decoded_body))
        }
    })
}

/// Representation of the different problem types defined in Table 1 in §3.1.
enum PpmProblemType {
    UnrecognizedMessage,
    UnrecognizedTask,
    OutdatedConfig,
    StaleReport,
    InvalidHmac,
}

impl PpmProblemType {
    /// Returns the problem type URI for a particular kind of error.
    fn type_uri(&self) -> &'static str {
        match self {
            PpmProblemType::UnrecognizedMessage => "urn:ietf:params:ppm:error:unrecognizedMessage",
            PpmProblemType::UnrecognizedTask => "urn:ietf:params:ppm:error:unrecognizedTask",
            PpmProblemType::OutdatedConfig => "urn:ietf:params:ppm:error:outdatedConfig",
            PpmProblemType::StaleReport => "urn:ietf:params:ppm:error:staleReport",
            PpmProblemType::InvalidHmac => "urn:ietf:params:ppm:error:invalidHmac",
        }
    }

    /// Returns a human-readable summary of a problem type.
    fn description(&self) -> &'static str {
        match self {
            PpmProblemType::UnrecognizedMessage => {
                "The message type for a response was incorrect or the payload was malformed."
            }
            PpmProblemType::UnrecognizedTask => {
                "An endpoint received a message with an unknown task ID."
            }
            PpmProblemType::OutdatedConfig => {
                "The message was generated using an outdated configuration."
            }
            PpmProblemType::StaleReport => {
                "Report could not be processed because it arrived too late."
            }
            PpmProblemType::InvalidHmac => "The aggregate message's HMAC was not valid.",
        }
    }
}

/// The media type for problem details formatted as a JSON document, per RFC 7807.
static PROBLEM_DETAILS_JSON_MEDIA_TYPE: &str = "application/problem+json";

/// Construct an error response in accordance with §3.1.
//
// TODO (issue abetterinternet/ppm-specification#209): The handling of the instance, title,
// detail, and taskid fields are subject to change.
fn build_problem_details_response(error_type: PpmProblemType, task_id: TaskId) -> Response {
    // So far, 400 Bad Request seems to be the appropriate choice for each defined problem type.
    let status = StatusCode::BAD_REQUEST;
    warp::reply::with_status(
        warp::reply::with_header(
            warp::reply::json(&serde_json::json!({
                "type": error_type.type_uri(),
                "title": error_type.description(),
                "status": status.as_u16(),
                "detail": error_type.description(),
                // The base URI is either "[leader]/upload", "[aggregator]/aggregate",
                // "[helper]/aggregate_share", or "[leader]/collect". Relative URLs are allowed in
                // the instance member, thus ".." will always refer to the aggregator's endpoint,
                // as required by §3.1.
                "instance": "..",
                "taskid": base64::encode(task_id.as_bytes()),
            })),
            http::header::CONTENT_TYPE,
            PROBLEM_DETAILS_JSON_MEDIA_TYPE,
        ),
        status,
    )
    .into_response()
}

/// Produces a closure that will transform applicable errors into a problem details JSON object.
/// (See RFC 7807) The returned closure is meant to be used in a warp `map` filter.
fn error_handler<R: Reply>() -> impl Fn(Result<R, Error>) -> warp::reply::Response + Clone {
    |result| {
        if let Err(error) = &result {
            error!(%error);
        }
        match result {
            Ok(reply) => reply.into_response(),
            Err(Error::InvalidConfiguration(_)) => {
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            }
            Err(Error::MessageDecode(_)) => StatusCode::BAD_REQUEST.into_response(),
            Err(Error::StaleReport(_, task_id)) => {
                build_problem_details_response(PpmProblemType::StaleReport, task_id)
            }
            Err(Error::UnrecognizedMessage(_, task_id)) => {
                build_problem_details_response(PpmProblemType::UnrecognizedMessage, task_id)
            }
            Err(Error::UnrecognizedTask(task_id)) => {
                build_problem_details_response(PpmProblemType::UnrecognizedTask, task_id)
            }
            Err(Error::OutdatedHpkeConfig(_, task_id)) => {
                build_problem_details_response(PpmProblemType::OutdatedConfig, task_id)
            }
            Err(Error::ReportFromTheFuture(_, _)) => {
                // TODO: build a problem details document once an error type is defined for reports
                // with timestamps too far in the future.
                StatusCode::BAD_REQUEST.into_response()
            }
            Err(Error::InvalidHmac(task_id)) => {
                build_problem_details_response(PpmProblemType::InvalidHmac, task_id)
            }
            Err(Error::Datastore(_)) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
            Err(Error::Internal(_)) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        }
    }
}

/// Constructs a Warp filter with endpoints common to all aggregators.
fn aggregator_filter<A, C>(
    vdaf: A,
    datastore: Arc<Datastore>,
    clock: C,
    tolerable_clock_skew: Duration,
    role: Role,
    verify_param: A::VerifyParam,
    hpke_recipient: HpkeRecipient,
    agg_auth_key: hmac::Key,
) -> Result<BoxedFilter<(impl Reply,)>, Error>
where
    A: 'static + vdaf::Aggregator + Send + Sync,
    A::AggregationParam: Send + Sync,
    for<'a> &'a A::AggregateShare: Into<Vec<u8>>,
    A::VerifyParam: Send + Sync,
    A::PrepareStep: Send + Sync + Encode + ParameterizedDecode<A::VerifyParam>,
    A::PrepareMessage: Send + Sync,
    A::OutputShare: Send + Sync + for<'a> TryFrom<&'a [u8]>,
    for<'a> &'a A::OutputShare: Into<Vec<u8>>,
    C: 'static + Clock,
{
    if !role.is_aggregator() {
        return Err(Error::InvalidConfiguration("role is not an aggregator"));
    }

    let hpke_config_encoded = hpke_recipient.config().get_encoded();

    let aggregator = Arc::new(Aggregator::new(
        vdaf,
        datastore,
        clock,
        tolerable_clock_skew,
        role,
        verify_param,
        hpke_recipient,
        agg_auth_key,
    )?);

    let hpke_config_endpoint = warp::path("hpke_config")
        .and(warp::get())
        .map(move || {
            reply::with_header(
                reply::with_status(hpke_config_encoded.clone(), StatusCode::OK),
                CACHE_CONTROL,
                "max-age=86400",
            )
        })
        .with(trace::named("hpke_config"));

    let upload_endpoint = warp::path("upload")
        .and(warp::post())
        .and(with_cloned_value(aggregator.clone()))
        .and_then(|aggregator: Arc<Aggregator<A, C>>| async {
            // Only the leader supports upload
            if aggregator.role != Role::Leader {
                return Err(warp::reject::not_found());
            }
            Ok(aggregator)
        })
        .and(with_decoded_body())
        .then(
            |aggregator: Arc<Aggregator<A, C>>, report_res: Result<Report, Error>| async move {
                aggregator.handle_upload(&report_res?).await?;
                Ok(StatusCode::OK)
            },
        )
        .map(error_handler())
        .with(trace::named("upload"));

    let aggregate_endpoint = warp::path("aggregate")
        .and(warp::post())
        .and(with_cloned_value(aggregator.clone()))
        .and_then(|aggregator: Arc<Aggregator<A, C>>| async {
            // Only the helper supports /aggregate.
            if aggregator.role != Role::Helper {
                return Err(warp::reject::not_found());
            }
            Ok(aggregator)
        })
        .and(with_authenticated_body(move |_task_id| {
            let aggregator = aggregator.clone();
            Box::pin(async move { Ok(aggregator.agg_auth_key.clone()) })
        }))
        .then(
            |aggregator: Arc<Aggregator<A, C>>,
             req_rslt: Result<(hmac::Key, AggregateReq), Error>| async move {
                let (key, req) = req_rslt?;
                let resp = aggregator.handle_aggregate(req).await?;
                let resp_bytes = AuthenticatedEncoder::new(resp).encode(&key);
                Ok(reply::with_status(resp_bytes, StatusCode::OK))
            },
        )
        .map(error_handler())
        .with(trace::named("aggregate"));

    Ok(hpke_config_endpoint
        .or(upload_endpoint)
        .or(aggregate_endpoint)
        .boxed())
}

/// Construct a PPM aggregator server, listening on the provided [`SocketAddr`].
/// If the `SocketAddr`'s `port` is 0, an ephemeral port is used. Returns a
/// `SocketAddr` representing the address and port the server are listening on
/// and a future that can be `await`ed to begin serving requests.
pub fn aggregator_server<A, C>(
    vdaf: A,
    datastore: Arc<Datastore>,
    clock: C,
    tolerable_clock_skew: Duration,
    role: Role,
    verify_param: A::VerifyParam,
    hpke_recipient: HpkeRecipient,
    agg_auth_key: hmac::Key,
    listen_address: SocketAddr,
) -> Result<(SocketAddr, impl Future<Output = ()> + 'static), Error>
where
    A: 'static + vdaf::Aggregator + Send + Sync,
    A::AggregationParam: Send + Sync,
    for<'a> &'a A::AggregateShare: Into<Vec<u8>>,
    A::VerifyParam: Send + Sync,
    A::PrepareStep: Send + Sync + Encode + ParameterizedDecode<A::VerifyParam>,
    A::PrepareMessage: Send + Sync,
    A::OutputShare: Send + Sync + for<'a> TryFrom<&'a [u8]>,
    for<'a> &'a A::OutputShare: Into<Vec<u8>>,
    C: 'static + Clock,
{
    Ok(warp::serve(aggregator_filter(
        vdaf,
        datastore,
        clock,
        tolerable_clock_skew,
        role,
        verify_param,
        hpke_recipient,
        agg_auth_key,
    )?)
    .bind_ephemeral(listen_address))
}

#[cfg(test)]
pub(crate) mod test_util {
    pub mod fake {
        use prio::vdaf::{self, Aggregatable, PrepareTransition, VdafError};
        use std::convert::Infallible;
        use std::fmt::Debug;
        use std::sync::Arc;

        #[derive(Clone)]
        pub struct Vdaf {
            prep_init_fn: Arc<dyn Fn() -> Result<(), VdafError> + 'static + Send + Sync>,
            prep_step_fn:
                Arc<dyn Fn() -> PrepareTransition<(), (), OutputShare> + 'static + Send + Sync>,
        }

        impl Debug for Vdaf {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("Vdaf")
                    .field("prep_init_result", &"[omitted]")
                    .field("prep_step_result", &"[omitted]")
                    .finish()
            }
        }

        impl Vdaf {
            pub fn new() -> Self {
                Vdaf {
                    prep_init_fn: Arc::new(|| -> Result<(), VdafError> { Ok(()) }),
                    prep_step_fn: Arc::new(|| -> PrepareTransition<(), (), OutputShare> {
                        PrepareTransition::Finish(OutputShare())
                    }),
                }
            }

            pub fn with_prep_init_fn<F: Fn() -> Result<(), VdafError>>(mut self, f: F) -> Self
            where
                F: 'static + Send + Sync,
            {
                self.prep_init_fn = Arc::new(f);
                self
            }

            pub fn with_prep_step_fn<F: Fn() -> PrepareTransition<(), (), OutputShare>>(
                mut self,
                f: F,
            ) -> Self
            where
                F: 'static + Send + Sync,
            {
                self.prep_step_fn = Arc::new(f);
                self
            }
        }

        impl vdaf::Vdaf for Vdaf {
            type Measurement = ();
            type AggregateResult = ();
            type AggregationParam = ();
            type PublicParam = ();
            type VerifyParam = ();
            type InputShare = ();
            type OutputShare = OutputShare;
            type AggregateShare = AggregateShare;

            fn setup(&self) -> Result<(Self::PublicParam, Vec<Self::VerifyParam>), VdafError> {
                Ok(((), vec![(), ()]))
            }

            fn num_aggregators(&self) -> usize {
                2
            }
        }

        impl vdaf::Aggregator for Vdaf {
            type PrepareStep = ();
            type PrepareMessage = ();

            fn prepare_init(
                &self,
                _: &Self::VerifyParam,
                _: &Self::AggregationParam,
                _: &[u8],
                _: &Self::InputShare,
            ) -> Result<Self::PrepareStep, VdafError> {
                (self.prep_init_fn)()
            }

            fn prepare_preprocess<M: IntoIterator<Item = Self::PrepareMessage>>(
                &self,
                _: M,
            ) -> Result<Self::PrepareMessage, VdafError> {
                Ok(())
            }

            fn prepare_step(
                &self,
                _: Self::PrepareStep,
                _: Option<Self::PrepareMessage>,
            ) -> PrepareTransition<Self::PrepareStep, Self::PrepareMessage, Self::OutputShare>
            {
                (self.prep_step_fn)()
            }

            fn aggregate<M: IntoIterator<Item = Self::OutputShare>>(
                &self,
                _: &Self::AggregationParam,
                _: M,
            ) -> Result<Self::AggregateShare, VdafError> {
                Ok(AggregateShare())
            }
        }

        impl vdaf::Client for Vdaf {
            fn shard(
                &self,
                _: &Self::PublicParam,
                _: &Self::Measurement,
            ) -> Result<Vec<Self::InputShare>, VdafError> {
                Ok(vec![(), ()])
            }
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        pub struct OutputShare();

        impl TryFrom<&[u8]> for OutputShare {
            type Error = Infallible;

            fn try_from(_: &[u8]) -> Result<Self, Self::Error> {
                Ok(Self())
            }
        }

        impl From<&OutputShare> for Vec<u8> {
            fn from(_: &OutputShare) -> Self {
                Self::new()
            }
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        pub struct AggregateShare();

        impl Aggregatable for AggregateShare {
            type OutputShare = OutputShare;

            fn merge(&mut self, _: &Self) -> Result<(), VdafError> {
                Ok(())
            }

            fn accumulate(&mut self, _: &Self::OutputShare) -> Result<(), VdafError> {
                Ok(())
            }
        }

        impl From<OutputShare> for AggregateShare {
            fn from(_: OutputShare) -> Self {
                Self()
            }
        }

        impl TryFrom<&[u8]> for AggregateShare {
            type Error = Infallible;

            fn try_from(_: &[u8]) -> Result<Self, Self::Error> {
                Ok(Self())
            }
        }

        impl From<&AggregateShare> for Vec<u8> {
            fn from(_: &AggregateShare) -> Self {
                Self::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        aggregator::test_util::fake,
        datastore::test_util::{ephemeral_datastore, DbHandle},
        hpke::{test_util::generate_hpke_config_and_private_key, HpkeSender, Label},
        message::{AuthenticatedResponseDecoder, HpkeCiphertext, HpkeConfig, TaskId, Time},
        task::{test_util::new_dummy_task_parameters, Vdaf},
        time::tests::MockClock,
        trace::test_util::install_test_trace_subscriber,
    };
    use assert_matches::assert_matches;
    use http::Method;
    use prio::{
        codec::Decode,
        vdaf::prio3::Prio3Aes128Count,
        vdaf::{Vdaf as VdafTrait, VdafError},
    };
    use rand::{thread_rng, Rng};
    use ring::{hmac::HMAC_SHA256, rand::SystemRandom};
    use std::io::Cursor;
    use warp::reply::Reply;

    type PrepareTransition<V> = vdaf::PrepareTransition<
        <V as vdaf::Aggregator>::PrepareStep,
        <V as vdaf::Aggregator>::PrepareMessage,
        <V as vdaf::Vdaf>::OutputShare,
    >;

    #[tokio::test]
    async fn invalid_role() {
        install_test_trace_subscriber();

        let vdaf = Prio3Aes128Count::new(2).unwrap();
        let verify_param = vdaf.setup().unwrap().1.first().unwrap().clone();
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let datastore = Arc::new(datastore);
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            TaskId::random(),
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Leader,
            hpke_private_key,
        );

        for invalid_role in [Role::Collector, Role::Client] {
            assert_matches!(
                aggregator_filter(
                    vdaf.clone(),
                    datastore.clone(),
                    MockClock::default(),
                    Duration::minutes(10),
                    invalid_role,
                    verify_param.clone(),
                    hpke_recipient.clone(),
                    generate_hmac_key(),
                ),
                Err(Error::InvalidConfiguration(_))
            );
        }
    }

    #[tokio::test]
    async fn invalid_clock_skew() {
        install_test_trace_subscriber();

        let vdaf = Prio3Aes128Count::new(2).unwrap();
        let verify_param = vdaf.setup().unwrap().1.first().unwrap().clone();
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            TaskId::random(),
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Leader,
            hpke_private_key,
        );

        assert_matches!(
            Aggregator::new(
                vdaf,
                Arc::new(datastore),
                MockClock::default(),
                Duration::minutes(-10),
                Role::Leader,
                verify_param,
                hpke_recipient,
                generate_hmac_key(),
            ),
            Err(Error::InvalidConfiguration(_))
        );
    }

    #[tokio::test]
    async fn hpke_config() {
        install_test_trace_subscriber();

        let task_id = TaskId::random();
        let vdaf = Prio3Aes128Count::new(2).unwrap();
        let verify_param = vdaf.setup().unwrap().1.first().unwrap().clone();
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            task_id,
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Leader,
            hpke_private_key,
        );

        let response = warp::test::request()
            .path("/hpke_config")
            .method("GET")
            .filter(
                &aggregator_filter(
                    vdaf,
                    Arc::new(datastore),
                    MockClock::default(),
                    Duration::minutes(10),
                    Role::Leader,
                    verify_param,
                    hpke_recipient.clone(),
                    generate_hmac_key(),
                )
                .unwrap(),
            )
            .await
            .unwrap()
            .into_response();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(CACHE_CONTROL).unwrap(),
            "max-age=86400"
        );

        let bytes = hyper::body::to_bytes(response.into_body()).await.unwrap();
        let hpke_config = HpkeConfig::decode(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(&hpke_config, hpke_recipient.config());
        let sender = HpkeSender::from_recipient(&hpke_recipient);

        let message = b"this is a message";
        let associated_data = b"some associated data";

        let ciphertext = sender.seal(message, associated_data).unwrap();

        let plaintext = hpke_recipient.open(&ciphertext, associated_data).unwrap();
        assert_eq!(&plaintext, message);
    }

    async fn setup_report(
        datastore: &Datastore,
        clock: &MockClock,
        skew: Duration,
    ) -> (HpkeRecipient, Report) {
        let task_id = TaskId::random();

        datastore
            .run_tx(|tx| {
                let task_parameters =
                    new_dummy_task_parameters(task_id, Vdaf::Prio3Aes128Count, Role::Leader);
                Box::pin(async move { tx.put_task(&task_parameters).await })
            })
            .await
            .unwrap();

        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            task_id,
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Leader,
            hpke_private_key,
        );

        let report_time = clock.now() - skew;

        let nonce = Nonce {
            time: Time(report_time.timestamp() as u64),
            rand: 0,
        };
        let extensions = vec![];
        let associated_data = Report::associated_data(nonce, &extensions);
        let message = b"this is a message";

        let leader_sender = HpkeSender::from_recipient(&hpke_recipient);
        let leader_ciphertext = leader_sender.seal(message, &associated_data).unwrap();

        let helper_sender = HpkeSender::from_recipient(&hpke_recipient);
        let helper_ciphertext = helper_sender.seal(message, &associated_data).unwrap();

        let report = Report {
            task_id,
            nonce,
            extensions,
            encrypted_input_shares: vec![leader_ciphertext, helper_ciphertext],
        };

        (hpke_recipient, report)
    }

    /// Convenience method to handle interaction with `warp::test` for typical PPM requests.
    async fn drive_filter(
        method: Method,
        path: &str,
        body: &[u8],
        filter: &BoxedFilter<(impl Reply + 'static,)>,
    ) -> Result<Response, Rejection> {
        warp::test::request()
            .method(method.as_str())
            .path(path)
            .body(body)
            .filter(filter)
            .await
            .map(|reply| reply.into_response())
    }

    #[tokio::test]
    async fn upload_filter() {
        install_test_trace_subscriber();

        let vdaf = Prio3Aes128Count::new(2).unwrap();
        let verify_param = vdaf.setup().unwrap().1.first().unwrap().clone();
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let clock = MockClock::default();
        let skew = Duration::minutes(10);

        let (report_recipient, report) = setup_report(&datastore, &clock, skew).await;
        let filter = aggregator_filter(
            vdaf,
            Arc::new(datastore),
            clock,
            skew,
            Role::Leader,
            verify_param,
            report_recipient,
            generate_hmac_key(),
        )
        .unwrap();

        let response = drive_filter(Method::POST, "/upload", &report.get_encoded(), &filter)
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(hyper::body::to_bytes(response.into_body())
            .await
            .unwrap()
            .is_empty());

        // should reject duplicate reports with the staleReport type.
        // TODO (issue #34): change this error type.
        let response = drive_filter(Method::POST, "/upload", &report.get_encoded(), &filter)
            .await
            .unwrap();
        let (part, body) = response.into_parts();
        assert!(!part.status.is_success());
        let bytes = hyper::body::to_bytes(body).await.unwrap();
        let problem_details: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            problem_details,
            serde_json::json!({
                "status": 400,
                "type": "urn:ietf:params:ppm:error:staleReport",
                "title": "Report could not be processed because it arrived too late.",
                "detail": "Report could not be processed because it arrived too late.",
                "instance": "..",
                "taskid": base64::encode(report.task_id.as_bytes()),
            })
        );
        assert_eq!(
            problem_details
                .as_object()
                .unwrap()
                .get("status")
                .unwrap()
                .as_u64()
                .unwrap(),
            part.status.as_u16() as u64
        );

        // should reject a report with only one share with the unrecognizedMessage type.
        let mut bad_report = report.clone();
        bad_report.encrypted_input_shares.truncate(1);
        let response = drive_filter(Method::POST, "/upload", &bad_report.get_encoded(), &filter)
            .await
            .unwrap();
        let (part, body) = response.into_parts();
        assert!(!part.status.is_success());
        let bytes = hyper::body::to_bytes(body).await.unwrap();
        let problem_details: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            problem_details,
            serde_json::json!({
                "status": 400,
                "type": "urn:ietf:params:ppm:error:unrecognizedMessage",
                "title": "The message type for a response was incorrect or the payload was malformed.",
                "detail": "The message type for a response was incorrect or the payload was malformed.",
                "instance": "..",
                "taskid": base64::encode(report.task_id.as_bytes()),
            })
        );
        assert_eq!(
            problem_details
                .as_object()
                .unwrap()
                .get("status")
                .unwrap()
                .as_u64()
                .unwrap(),
            part.status.as_u16() as u64
        );

        // should reject a report using the wrong HPKE config for the leader, and reply with
        // the error type outdatedConfig.
        let mut bad_report = report.clone();
        bad_report.encrypted_input_shares[0].config_id = HpkeConfigId(101);
        let response = drive_filter(Method::POST, "/upload", &bad_report.get_encoded(), &filter)
            .await
            .unwrap();
        let (part, body) = response.into_parts();
        assert!(!part.status.is_success());
        let bytes = hyper::body::to_bytes(body).await.unwrap();
        let problem_details: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            problem_details,
            serde_json::json!({
                "status": 400,
                "type": "urn:ietf:params:ppm:error:outdatedConfig",
                "title": "The message was generated using an outdated configuration.",
                "detail": "The message was generated using an outdated configuration.",
                "instance": "..",
                "taskid": base64::encode(report.task_id.as_bytes()),
            })
        );
        assert_eq!(
            problem_details
                .as_object()
                .unwrap()
                .get("status")
                .unwrap()
                .as_u64()
                .unwrap(),
            part.status.as_u16() as u64
        );

        // reports from the future should be rejected.
        let mut bad_report = report.clone();
        bad_report.nonce.time = Time::from_naive_date_time(
            MockClock::default().now() + Duration::minutes(10) + Duration::seconds(1),
        );
        let response = drive_filter(Method::POST, "/upload", &bad_report.get_encoded(), &filter)
            .await
            .unwrap();
        assert!(!response.status().is_success());
        // TODO: update this test once an error type has been defined, and validate the problem
        // details.
        assert_eq!(response.status().as_u16(), 400);
    }

    // Helper should not expose /upload endpoint
    #[tokio::test]
    async fn upload_filter_helper() {
        install_test_trace_subscriber();

        let vdaf = Prio3Aes128Count::new(2).unwrap();
        let verify_param = vdaf.setup().unwrap().1.first().unwrap().clone();
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let clock = MockClock::default();
        let skew = Duration::minutes(10);

        let (report_recipient, report) = setup_report(&datastore, &clock, skew).await;

        let filter = aggregator_filter(
            vdaf,
            Arc::new(datastore),
            clock,
            skew,
            Role::Helper,
            verify_param,
            report_recipient,
            generate_hmac_key(),
        )
        .unwrap();

        let result = warp::test::request()
            .method("POST")
            .path("/upload")
            .body(report.get_encoded())
            .filter(&filter)
            .await;

        // We can't use `Result::unwrap_err` or `assert_matches!` here because
        //  `impl Reply` is not `Debug`
        if let Err(rejection) = result {
            assert!(rejection.is_not_found());
        } else {
            panic!("should get rejection");
        }
    }

    async fn setup_upload_test(
        skew: Duration,
    ) -> (
        Aggregator<Prio3Aes128Count, MockClock>,
        Report,
        Arc<Datastore>,
        DbHandle,
    ) {
        let vdaf = Prio3Aes128Count::new(2).unwrap();
        let verify_param = vdaf.setup().unwrap().1.first().unwrap().clone();
        let (datastore, db_handle) = ephemeral_datastore().await;
        let datastore = Arc::new(datastore);
        let clock = MockClock::default();
        let (report_recipient, report) = setup_report(&datastore, &clock, skew).await;

        let aggregator = Aggregator::new(
            vdaf,
            datastore.clone(),
            clock,
            skew,
            Role::Leader,
            verify_param,
            report_recipient,
            generate_hmac_key(),
        )
        .unwrap();

        (aggregator, report, datastore, db_handle)
    }

    #[tokio::test]
    async fn upload() {
        install_test_trace_subscriber();

        let skew = Duration::minutes(10);
        let (aggregator, report, datastore, _db_handle) = setup_upload_test(skew).await;

        aggregator.handle_upload(&report).await.unwrap();

        let got_report = datastore
            .run_tx(|tx| {
                Box::pin(async move { tx.get_client_report(report.task_id, report.nonce).await })
            })
            .await
            .unwrap();
        assert_eq!(report, got_report);

        // should reject duplicate reports.
        // TODO (issue #34): change this error type.
        assert_matches!(aggregator.handle_upload(&report).await, Err(Error::StaleReport(stale_nonce, task_id)) => {
            assert_eq!(task_id, report.task_id);
            assert_eq!(report.nonce, stale_nonce);
        });
    }

    #[tokio::test]
    async fn upload_wrong_number_of_encrypted_shares() {
        install_test_trace_subscriber();

        let skew = Duration::minutes(10);
        let (aggregator, mut report, _, _db_handle) = setup_upload_test(skew).await;

        report.encrypted_input_shares = vec![report.encrypted_input_shares[0].clone()];

        assert_matches!(
            aggregator.handle_upload(&report).await,
            Err(Error::UnrecognizedMessage(_, _))
        );
    }

    #[tokio::test]
    async fn upload_wrong_hpke_config_id() {
        install_test_trace_subscriber();

        let skew = Duration::minutes(10);
        let (aggregator, mut report, _, _db_handle) = setup_upload_test(skew).await;

        report.encrypted_input_shares[0].config_id = HpkeConfigId(101);

        assert_matches!(aggregator.handle_upload(&report).await, Err(Error::OutdatedHpkeConfig(config_id, task_id)) => {
            assert_eq!(task_id, report.task_id);
            assert_eq!(config_id, HpkeConfigId(101));
        });
    }

    fn reencrypt_report(report: Report, hpke_recipient: &HpkeRecipient) -> Report {
        let associated_data = Report::associated_data(report.nonce, &report.extensions);
        let message = b"this is a message";

        let leader_sender = HpkeSender::from_recipient(hpke_recipient);
        let leader_ciphertext = leader_sender.seal(message, &associated_data).unwrap();

        let helper_sender = HpkeSender::from_recipient(hpke_recipient);
        let helper_ciphertext = helper_sender.seal(message, &associated_data).unwrap();

        Report {
            task_id: report.task_id,
            nonce: report.nonce,
            extensions: report.extensions,
            encrypted_input_shares: vec![leader_ciphertext, helper_ciphertext],
        }
    }

    #[tokio::test]
    async fn upload_report_in_the_future() {
        install_test_trace_subscriber();

        let skew = Duration::minutes(10);
        let (aggregator, mut report, datastore, _db_handle) = setup_upload_test(skew).await;

        // Boundary condition
        report.nonce.time = Time::from_naive_date_time(aggregator.clock.now() + skew);
        let mut report = reencrypt_report(report, &aggregator.report_recipient);
        aggregator.handle_upload(&report).await.unwrap();

        let got_report = datastore
            .run_tx(|tx| {
                Box::pin(async move { tx.get_client_report(report.task_id, report.nonce).await })
            })
            .await
            .unwrap();
        assert_eq!(report, got_report);

        // Just past the clock skew
        report.nonce.time =
            Time::from_naive_date_time(aggregator.clock.now() + skew + Duration::seconds(1));
        let report = reencrypt_report(report, &aggregator.report_recipient);
        assert_matches!(aggregator.handle_upload(&report).await, Err(Error::ReportFromTheFuture(nonce, task_id)) => {
            assert_eq!(task_id, report.task_id);
            assert_eq!(report.nonce, nonce);
        });
    }

    #[tokio::test]
    async fn aggregate_leader() {
        install_test_trace_subscriber();

        let task_id = TaskId::random();
        let vdaf = Prio3Aes128Count::new(2).unwrap();
        let verify_param = vdaf.setup().unwrap().1.remove(1);
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let clock = MockClock::default();
        let skew = Duration::minutes(10);
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            task_id,
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Leader,
            hpke_private_key,
        );
        let hmac_key = generate_hmac_key();

        let request = AggregateReq {
            task_id: TaskId::random(),
            job_id: AggregationJobId::random(),
            body: AggregateInitReq {
                agg_param: Vec::new(),
                seq: Vec::new(),
            },
        };

        let filter = aggregator_filter(
            vdaf,
            Arc::new(datastore),
            clock,
            skew,
            Role::Leader,
            verify_param,
            hpke_recipient,
            hmac_key.clone(),
        )
        .unwrap();

        let result = warp::test::request()
            .method("POST")
            .path("/aggregate")
            .body(AuthenticatedEncoder::new(request).encode(&hmac_key))
            .filter(&filter)
            .await;

        // We can't use `Result::unwrap_err` or `assert_matches!` here because
        //  `impl Reply` is not `Debug`
        if let Err(rejection) = result {
            assert!(rejection.is_not_found());
        } else {
            panic!("Should get rejection");
        }
    }

    #[tokio::test]
    async fn aggregate_wrong_agg_auth_key() {
        install_test_trace_subscriber();

        let task_id = TaskId::random();
        let vdaf = Prio3Aes128Count::new(2).unwrap();
        let verify_param = vdaf.setup().unwrap().1.remove(1);
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let clock = MockClock::default();
        let skew = Duration::minutes(10);
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            task_id,
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Helper,
            hpke_private_key,
        );

        let request = AggregateReq {
            task_id,
            job_id: AggregationJobId::random(),
            body: AggregateInitReq {
                agg_param: Vec::new(),
                seq: Vec::new(),
            },
        };

        let filter = aggregator_filter(
            vdaf,
            Arc::new(datastore),
            clock,
            skew,
            Role::Helper,
            verify_param,
            hpke_recipient,
            generate_hmac_key(),
        )
        .unwrap();

        let (parts, body) = warp::test::request()
            .method("POST")
            .path("/aggregate")
            .body(AuthenticatedEncoder::new(request).encode(&generate_hmac_key()))
            .filter(&filter)
            .await
            .unwrap()
            .into_response()
            .into_parts();

        let want_status = 400;
        let problem_details: serde_json::Value =
            serde_json::from_slice(&hyper::body::to_bytes(body).await.unwrap()).unwrap();
        assert_eq!(
            problem_details,
            serde_json::json!({
                "status": want_status,
                "type": "urn:ietf:params:ppm:error:invalidHmac",
                "title": "The aggregate message's HMAC was not valid.",
                "detail": "The aggregate message's HMAC was not valid.",
                "instance": "..",
                "taskid": base64::encode(task_id.as_bytes()),
            })
        );
        assert_eq!(want_status, parts.status.as_u16());
    }

    #[tokio::test]
    async fn aggregate_init() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        let task_id = TaskId::random();
        let vdaf = Prio3Aes128Count::new(2).unwrap();
        let (public_param, verify_params) = vdaf.setup().unwrap();
        let aggregation_param = ();
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let clock = MockClock::default();
        let skew = Duration::minutes(10);
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            task_id,
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Helper,
            hpke_private_key,
        );
        let hmac_key = generate_hmac_key();

        datastore
            .run_tx(|tx| {
                Box::pin(async move {
                    tx.put_task(&new_dummy_task_parameters(
                        task_id,
                        Vdaf::Prio3Aes128Count,
                        Role::Helper,
                    ))
                    .await
                })
            })
            .await
            .unwrap();

        // report_share_0 is a "happy path" report.
        let nonce_0 = generate_nonce(&clock);
        let input_share = run_vdaf(
            &vdaf,
            &public_param,
            &verify_params,
            &aggregation_param,
            nonce_0,
            &0,
        )
        .input_shares
        .remove(1);
        let report_share_0 = generate_helper_report_share::<Prio3Aes128Count>(
            task_id,
            nonce_0,
            hpke_recipient.config(),
            &input_share,
        );

        // report_share_1 fails decryption.
        let mut report_share_1 = report_share_0.clone();
        report_share_1.nonce = generate_nonce(&clock);
        report_share_1.encrypted_input_share.payload[0] ^= 0xFF;

        // report_share_2 fails decoding.
        let nonce = generate_nonce(&clock);
        let mut input_share_bytes = input_share.get_encoded();
        input_share_bytes.push(0); // can no longer be decoded.
        let associated_data = Report::associated_data(nonce, &[]);
        let report_share_2 = generate_helper_report_share_for_plaintext(
            task_id,
            nonce,
            hpke_recipient.config(),
            &input_share_bytes,
            &associated_data,
        );

        let request = AggregateReq {
            task_id,
            job_id: AggregationJobId::random(),
            body: AggregateInitReq {
                agg_param: Vec::new(),
                seq: vec![
                    report_share_0.clone(),
                    report_share_1.clone(),
                    report_share_2.clone(),
                ],
            },
        };

        // Create aggregator filter, send request, and parse response.
        let filter = aggregator_filter(
            vdaf,
            Arc::new(datastore),
            clock,
            skew,
            Role::Helper,
            verify_params[1].clone(),
            hpke_recipient,
            hmac_key.clone(),
        )
        .unwrap();

        let mut response = warp::test::request()
            .method("POST")
            .path("/aggregate")
            .body(AuthenticatedEncoder::new(request).encode(&hmac_key))
            .filter(&filter)
            .await
            .unwrap()
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = hyper::body::to_bytes(response.body_mut()).await.unwrap();
        let aggregate_resp: AggregateResp =
            AuthenticatedResponseDecoder::new(Vec::from(body_bytes.as_ref()))
                .unwrap()
                .decode(&hmac_key)
                .unwrap();

        // Validate response.
        assert_eq!(aggregate_resp.seq.len(), 3);

        let transition_0 = aggregate_resp.seq.get(0).unwrap();
        assert_eq!(transition_0.nonce, report_share_0.nonce);
        assert_matches!(
            transition_0.trans_data,
            TransitionTypeSpecificData::Continued { .. }
        );

        let transition_1 = aggregate_resp.seq.get(1).unwrap();
        assert_eq!(transition_1.nonce, report_share_1.nonce);
        assert_matches!(
            transition_1.trans_data,
            TransitionTypeSpecificData::Failed {
                error: TransitionError::HpkeDecryptError
            }
        );

        let transition_2 = aggregate_resp.seq.get(2).unwrap();
        assert_eq!(transition_2.nonce, report_share_2.nonce);
        assert_matches!(
            transition_2.trans_data,
            TransitionTypeSpecificData::Failed {
                error: TransitionError::VdafPrepError
            }
        );
    }

    #[tokio::test]
    async fn aggregate_init_prep_init_failed() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        let task_id = TaskId::random();
        let vdaf = fake::Vdaf::new().with_prep_init_fn(|| -> Result<(), VdafError> {
            Err(VdafError::Uncategorized(
                "PrepInitFailer failed at prep_init".to_string(),
            ))
        });
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let clock = MockClock::default();
        let skew = Duration::minutes(10);
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            task_id,
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Helper,
            hpke_private_key,
        );
        let hmac_key = generate_hmac_key();

        datastore
            .run_tx(|tx| {
                Box::pin(async move {
                    tx.put_task(&new_dummy_task_parameters(
                        task_id,
                        Vdaf::Prio3Aes128Count,
                        Role::Helper,
                    ))
                    .await
                })
            })
            .await
            .unwrap();

        let report_share = generate_helper_report_share::<fake::Vdaf>(
            task_id,
            generate_nonce(&clock),
            hpke_recipient.config(),
            &(),
        );
        let request = AggregateReq {
            task_id,
            job_id: AggregationJobId::random(),
            body: AggregateInitReq {
                agg_param: Vec::new(),
                seq: vec![report_share.clone()],
            },
        };

        // Create aggregator filter, send request, and parse response.
        let filter = aggregator_filter(
            vdaf,
            Arc::new(datastore),
            clock,
            skew,
            Role::Helper,
            (),
            hpke_recipient,
            hmac_key.clone(),
        )
        .unwrap();

        let mut response = warp::test::request()
            .method("POST")
            .path("/aggregate")
            .body(AuthenticatedEncoder::new(request).encode(&hmac_key))
            .filter(&filter)
            .await
            .unwrap()
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = hyper::body::to_bytes(response.body_mut()).await.unwrap();
        let aggregate_resp: AggregateResp =
            AuthenticatedResponseDecoder::new(Vec::from(body_bytes.as_ref()))
                .unwrap()
                .decode(&hmac_key)
                .unwrap();

        // Validate response.
        assert_eq!(aggregate_resp.seq.len(), 1);

        let transition = aggregate_resp.seq.get(0).unwrap();
        assert_eq!(transition.nonce, report_share.nonce);
        assert_matches!(
            transition.trans_data,
            TransitionTypeSpecificData::Failed {
                error: TransitionError::VdafPrepError,
            }
        );
    }

    #[tokio::test]
    async fn aggregate_init_prep_step_failed() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        let task_id = TaskId::random();
        let vdaf = fake::Vdaf::new().with_prep_step_fn(|| -> PrepareTransition<fake::Vdaf> {
            PrepareTransition::<fake::Vdaf>::Fail(VdafError::Uncategorized(
                "VDAF failed at prep_step".to_string(),
            ))
        });
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let clock = MockClock::default();
        let skew = Duration::minutes(10);
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            task_id,
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Helper,
            hpke_private_key,
        );
        let hmac_key = generate_hmac_key();

        datastore
            .run_tx(|tx| {
                Box::pin(async move {
                    tx.put_task(&new_dummy_task_parameters(
                        task_id,
                        Vdaf::Prio3Aes128Count,
                        Role::Helper,
                    ))
                    .await
                })
            })
            .await
            .unwrap();

        let report_share = generate_helper_report_share::<fake::Vdaf>(
            task_id,
            generate_nonce(&clock),
            hpke_recipient.config(),
            &(),
        );
        let request = AggregateReq {
            task_id,
            job_id: AggregationJobId::random(),
            body: AggregateInitReq {
                agg_param: Vec::new(),
                seq: vec![report_share.clone()],
            },
        };

        // Create aggregator filter, send request, and parse response.
        let filter = aggregator_filter(
            vdaf,
            Arc::new(datastore),
            clock,
            skew,
            Role::Helper,
            (),
            hpke_recipient,
            hmac_key.clone(),
        )
        .unwrap();

        let mut response = warp::test::request()
            .method("POST")
            .path("/aggregate")
            .body(AuthenticatedEncoder::new(request).encode(&hmac_key))
            .filter(&filter)
            .await
            .unwrap()
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = hyper::body::to_bytes(response.body_mut()).await.unwrap();
        let aggregate_resp: AggregateResp =
            AuthenticatedResponseDecoder::new(Vec::from(body_bytes.as_ref()))
                .unwrap()
                .decode(&hmac_key)
                .unwrap();

        // Validate response.
        assert_eq!(aggregate_resp.seq.len(), 1);

        let transition = aggregate_resp.seq.get(0).unwrap();
        assert_eq!(transition.nonce, report_share.nonce);
        assert_matches!(
            transition.trans_data,
            TransitionTypeSpecificData::Failed {
                error: TransitionError::VdafPrepError,
            }
        );
    }

    #[tokio::test]
    async fn aggregate_init_duplicated_nonce() {
        install_test_trace_subscriber();

        let task_id = TaskId::random();
        let vdaf = Prio3Aes128Count::new(2).unwrap();
        let verify_param = vdaf.setup().unwrap().1.remove(1);
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let clock = MockClock::default();
        let skew = Duration::minutes(10);
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            task_id,
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Helper,
            hpke_private_key,
        );
        let hmac_key = generate_hmac_key();

        let report_share = ReportShare {
            nonce: Nonce {
                time: Time(54321),
                rand: 314,
            },
            extensions: Vec::new(),
            encrypted_input_share: HpkeCiphertext {
                // bogus, but we never get far enough to notice
                config_id: HpkeConfigId(42),
                encapsulated_context: Vec::from("012345"),
                payload: Vec::from("543210"),
            },
        };

        let request = AggregateReq {
            task_id,
            job_id: AggregationJobId::random(),
            body: AggregateInitReq {
                agg_param: Vec::new(),
                seq: vec![report_share.clone(), report_share],
            },
        };

        let filter = aggregator_filter(
            vdaf,
            Arc::new(datastore),
            clock,
            skew,
            Role::Helper,
            verify_param,
            hpke_recipient,
            hmac_key.clone(),
        )
        .unwrap();

        let (parts, body) = warp::test::request()
            .method("POST")
            .path("/aggregate")
            .body(AuthenticatedEncoder::new(request).encode(&hmac_key))
            .filter(&filter)
            .await
            .unwrap()
            .into_response()
            .into_parts();

        let want_status = 400;
        let problem_details: serde_json::Value =
            serde_json::from_slice(&hyper::body::to_bytes(body).await.unwrap()).unwrap();
        assert_eq!(
            problem_details,
            serde_json::json!({
                "status": want_status,
                "type": "urn:ietf:params:ppm:error:unrecognizedMessage",
                "title": "The message type for a response was incorrect or the payload was malformed.",
                "detail": "The message type for a response was incorrect or the payload was malformed.",
                "instance": "..",
                "taskid": base64::encode(task_id.as_bytes()),
            })
        );
        assert_eq!(want_status, parts.status.as_u16());
    }

    #[tokio::test]
    async fn aggregate_continue() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        let task_id = TaskId::random();
        let aggregation_job_id = AggregationJobId::random();
        let vdaf = Prio3Aes128Count::new(2).unwrap();
        let (public_param, verify_params) = vdaf.setup().unwrap();
        let aggregation_param = ();
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let datastore = Arc::new(datastore);
        let clock = MockClock::default();
        let skew = Duration::minutes(10);
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            task_id,
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Helper,
            hpke_private_key,
        );
        let hmac_key = generate_hmac_key();

        // report_share_0 is a "happy path" report.
        let nonce_0 = generate_nonce(&clock);
        let transcript_0 = run_vdaf(
            &vdaf,
            &public_param,
            &verify_params,
            &aggregation_param,
            nonce_0,
            &0,
        );
        let prep_step_0 = assert_matches!(&transcript_0.transitions[1][0], PrepareTransition::<Prio3Aes128Count>::Continue(prep_step, _) => prep_step.clone());
        let out_share_0 = assert_matches!(&transcript_0.transitions[1][1], PrepareTransition::<Prio3Aes128Count>::Finish(out_share) => out_share.clone());
        let prep_msg_0 = transcript_0.messages[0].clone();
        let report_share_0 = generate_helper_report_share::<Prio3Aes128Count>(
            task_id,
            nonce_0,
            hpke_recipient.config(),
            &transcript_0.input_shares[1],
        );

        // report_share_1 is omitted by the leader's request.
        let nonce_1 = generate_nonce(&clock);
        let transcript_1 = run_vdaf(
            &vdaf,
            &public_param,
            &verify_params,
            &aggregation_param,
            nonce_1,
            &0,
        );
        let prep_step_1 = assert_matches!(&transcript_1.transitions[1][0], PrepareTransition::<Prio3Aes128Count>::Continue(prep_step, _) => prep_step.clone());
        let report_share_1 = generate_helper_report_share::<Prio3Aes128Count>(
            task_id,
            nonce_1,
            hpke_recipient.config(),
            &transcript_1.input_shares[1],
        );

        datastore
            .run_tx(|tx| {
                let (report_share_0, report_share_1) =
                    (report_share_0.clone(), report_share_1.clone());
                let (prep_step_0, prep_step_1) = (prep_step_0.clone(), prep_step_1.clone());

                Box::pin(async move {
                    tx.put_task(&new_dummy_task_parameters(
                        task_id,
                        Vdaf::Prio3Aes128Count,
                        Role::Helper,
                    ))
                    .await?;

                    tx.put_report_share(task_id, &report_share_0).await?;
                    tx.put_report_share(task_id, &report_share_1).await?;

                    tx.put_aggregation_job(&AggregationJob::<Prio3Aes128Count> {
                        aggregation_job_id,
                        task_id,
                        aggregation_param: (),
                        state: AggregationJobState::InProgress,
                    })
                    .await?;

                    tx.put_report_aggregation(&ReportAggregation::<Prio3Aes128Count> {
                        aggregation_job_id,
                        task_id,
                        nonce: nonce_0,
                        ord: 0,
                        state: ReportAggregationState::Waiting(prep_step_0),
                    })
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<Prio3Aes128Count> {
                        aggregation_job_id,
                        task_id,
                        nonce: nonce_1,
                        ord: 1,
                        state: ReportAggregationState::Waiting(prep_step_1),
                    })
                    .await
                })
            })
            .await
            .unwrap();

        let request = AggregateReq {
            task_id,
            job_id: aggregation_job_id,
            body: AggregateContinueReq {
                seq: vec![Transition {
                    nonce: nonce_0,
                    trans_data: TransitionTypeSpecificData::Continued {
                        payload: prep_msg_0.get_encoded(),
                    },
                }],
            },
        };

        // Create aggregator filter, send request, and parse response.
        let filter = aggregator_filter(
            vdaf,
            datastore.clone(),
            clock,
            skew,
            Role::Helper,
            verify_params[1].clone(),
            hpke_recipient,
            hmac_key.clone(),
        )
        .unwrap();

        let mut response = warp::test::request()
            .method("POST")
            .path("/aggregate")
            .body(AuthenticatedEncoder::new(request).encode(&hmac_key))
            .filter(&filter)
            .await
            .unwrap()
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = hyper::body::to_bytes(response.body_mut()).await.unwrap();
        let aggregate_resp: AggregateResp =
            AuthenticatedResponseDecoder::new(Vec::from(body_bytes.as_ref()))
                .unwrap()
                .decode(&hmac_key)
                .unwrap();

        // Validate response.
        assert_eq!(
            aggregate_resp,
            AggregateResp {
                seq: vec![Transition {
                    nonce: nonce_0,
                    trans_data: TransitionTypeSpecificData::Finished,
                }]
            }
        );

        // Validate datastore.
        let (aggregation_job, report_aggregations) = datastore
            .run_tx(|tx| {
                let verify_params = verify_params.clone();

                Box::pin(async move {
                    let aggregation_job = tx
                        .get_aggregation_job::<Prio3Aes128Count>(task_id, aggregation_job_id)
                        .await?;
                    let report_aggregations = tx
                        .get_report_aggregations_for_aggregation_job::<Prio3Aes128Count>(
                            &verify_params[1].clone(),
                            task_id,
                            aggregation_job_id,
                        )
                        .await?;
                    Ok((aggregation_job, report_aggregations))
                })
            })
            .await
            .unwrap();

        assert_eq!(
            aggregation_job,
            AggregationJob {
                aggregation_job_id,
                task_id,
                aggregation_param,
                state: AggregationJobState::Finished,
            }
        );
        assert_eq!(
            report_aggregations,
            vec![
                ReportAggregation {
                    aggregation_job_id,
                    task_id,
                    nonce: nonce_0,
                    ord: 0,
                    state: ReportAggregationState::Finished(out_share_0),
                },
                ReportAggregation {
                    aggregation_job_id,
                    task_id,
                    nonce: nonce_1,
                    ord: 1,
                    state: ReportAggregationState::Failed(TransitionError::ReportDropped),
                }
            ]
        );
    }

    #[tokio::test]
    async fn aggregate_continue_leader_sends_non_continue_transition() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        // Prepare parameters.
        let task_id = TaskId::random();
        let aggregation_job_id = AggregationJobId::random();
        let nonce = Nonce {
            time: Time(54321),
            rand: 314,
        };
        let vdaf = fake::Vdaf::new();
        let verify_param = vdaf.setup().unwrap().1.remove(1);
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let datastore = Arc::new(datastore);
        let clock = MockClock::default();
        let skew = Duration::minutes(10);
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            task_id,
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Helper,
            hpke_private_key,
        );
        let hmac_key = generate_hmac_key();

        // Setup datastore.
        datastore
            .run_tx(|tx| {
                Box::pin(async move {
                    tx.put_task(&new_dummy_task_parameters(
                        task_id,
                        Vdaf::Prio3Aes128Count,
                        Role::Helper,
                    ))
                    .await?;
                    tx.put_report_share(
                        task_id,
                        &ReportShare {
                            nonce,
                            extensions: Vec::new(),
                            encrypted_input_share: HpkeCiphertext {
                                config_id: HpkeConfigId(42),
                                encapsulated_context: Vec::from("012345"),
                                payload: Vec::from("543210"),
                            },
                        },
                    )
                    .await?;
                    tx.put_aggregation_job(&AggregationJob::<fake::Vdaf> {
                        aggregation_job_id,
                        task_id,
                        aggregation_param: (),
                        state: AggregationJobState::InProgress,
                    })
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<fake::Vdaf> {
                        aggregation_job_id,
                        task_id,
                        nonce,
                        ord: 0,
                        state: ReportAggregationState::Waiting(()),
                    })
                    .await
                })
            })
            .await
            .unwrap();

        // Make request.
        let request = AggregateReq {
            task_id,
            job_id: aggregation_job_id,
            body: AggregateContinueReq {
                seq: vec![Transition {
                    nonce,
                    trans_data: TransitionTypeSpecificData::Finished,
                }],
            },
        };

        let filter = aggregator_filter(
            vdaf,
            datastore.clone(),
            clock,
            skew,
            Role::Helper,
            verify_param,
            hpke_recipient,
            hmac_key.clone(),
        )
        .unwrap();

        let (parts, body) = warp::test::request()
            .method("POST")
            .path("/aggregate")
            .body(AuthenticatedEncoder::new(request).encode(&hmac_key))
            .filter(&filter)
            .await
            .unwrap()
            .into_response()
            .into_parts();

        // Check that response is as desired.
        assert_eq!(parts.status, StatusCode::BAD_REQUEST);
        let problem_details: serde_json::Value =
            serde_json::from_slice(&hyper::body::to_bytes(body).await.unwrap()).unwrap();
        assert_eq!(
            problem_details,
            serde_json::json!({
                "status": StatusCode::BAD_REQUEST.as_u16(),
                "type": "urn:ietf:params:ppm:error:unrecognizedMessage",
                "title": "The message type for a response was incorrect or the payload was malformed.",
                "detail": "The message type for a response was incorrect or the payload was malformed.",
                "instance": "..",
                "taskid": base64::encode(task_id.as_bytes()),
            })
        );
    }

    #[tokio::test]
    async fn aggregate_continue_prep_step_fails() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        // Prepare parameters.
        let task_id = TaskId::random();
        let aggregation_job_id = AggregationJobId::random();
        let nonce = Nonce {
            time: Time(54321),
            rand: 314,
        };
        let vdaf = fake::Vdaf::new().with_prep_step_fn(|| -> PrepareTransition<fake::Vdaf> {
            PrepareTransition::<fake::Vdaf>::Fail(VdafError::Uncategorized(
                "VDAF failed at prep_step".to_string(),
            ))
        });
        let verify_param = vdaf.setup().unwrap().1.remove(1);
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let datastore = Arc::new(datastore);
        let clock = MockClock::default();
        let skew = Duration::minutes(10);
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            task_id,
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Helper,
            hpke_private_key,
        );
        let hmac_key = generate_hmac_key();

        // Setup datastore.
        datastore
            .run_tx(|tx| {
                Box::pin(async move {
                    tx.put_task(&new_dummy_task_parameters(
                        task_id,
                        Vdaf::Prio3Aes128Count,
                        Role::Helper,
                    ))
                    .await?;
                    tx.put_report_share(
                        task_id,
                        &ReportShare {
                            nonce,
                            extensions: Vec::new(),
                            encrypted_input_share: HpkeCiphertext {
                                config_id: HpkeConfigId(42),
                                encapsulated_context: Vec::from("012345"),
                                payload: Vec::from("543210"),
                            },
                        },
                    )
                    .await?;
                    tx.put_aggregation_job(&AggregationJob::<fake::Vdaf> {
                        aggregation_job_id,
                        task_id,
                        aggregation_param: (),
                        state: AggregationJobState::InProgress,
                    })
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<fake::Vdaf> {
                        aggregation_job_id,
                        task_id,
                        nonce,
                        ord: 0,
                        state: ReportAggregationState::Waiting(()),
                    })
                    .await
                })
            })
            .await
            .unwrap();

        // Make request.
        let request = AggregateReq {
            task_id,
            job_id: aggregation_job_id,
            body: AggregateContinueReq {
                seq: vec![Transition {
                    nonce,
                    trans_data: TransitionTypeSpecificData::Continued {
                        payload: Vec::new(),
                    },
                }],
            },
        };

        let filter = aggregator_filter(
            vdaf,
            datastore.clone(),
            clock,
            skew,
            Role::Helper,
            verify_param,
            hpke_recipient,
            hmac_key.clone(),
        )
        .unwrap();

        let (parts, body) = warp::test::request()
            .method("POST")
            .path("/aggregate")
            .body(AuthenticatedEncoder::new(request).encode(&hmac_key))
            .filter(&filter)
            .await
            .unwrap()
            .into_response()
            .into_parts();

        // Check that response is as desired.
        assert_eq!(parts.status, StatusCode::OK);
        let body_bytes = hyper::body::to_bytes(body).await.unwrap();
        let aggregate_resp: AggregateResp =
            AuthenticatedResponseDecoder::new(Vec::from(body_bytes.as_ref()))
                .unwrap()
                .decode(&hmac_key)
                .unwrap();
        assert_eq!(
            aggregate_resp,
            AggregateResp {
                seq: vec![Transition {
                    nonce,
                    trans_data: TransitionTypeSpecificData::Failed {
                        error: TransitionError::VdafPrepError
                    }
                }]
            }
        );

        // Check datastore state.
        let (aggregation_job, report_aggregation) = datastore
            .run_tx(|tx| {
                Box::pin(async move {
                    let aggregation_job = tx
                        .get_aggregation_job::<fake::Vdaf>(task_id, aggregation_job_id)
                        .await?;
                    let report_aggregation = tx
                        .get_report_aggregation::<fake::Vdaf>(
                            &(),
                            task_id,
                            aggregation_job_id,
                            nonce,
                        )
                        .await?;
                    Ok((aggregation_job, report_aggregation))
                })
            })
            .await
            .unwrap();

        assert_eq!(
            aggregation_job,
            AggregationJob {
                aggregation_job_id,
                task_id,
                aggregation_param: (),
                state: AggregationJobState::Finished,
            }
        );
        assert_eq!(
            report_aggregation,
            ReportAggregation {
                aggregation_job_id,
                task_id,
                nonce,
                ord: 0,
                state: ReportAggregationState::Failed(TransitionError::VdafPrepError),
            }
        );
    }

    #[tokio::test]
    async fn aggregate_continue_unexpected_transition() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        // Prepare parameters.
        let task_id = TaskId::random();
        let aggregation_job_id = AggregationJobId::random();
        let nonce = Nonce {
            time: Time(54321),
            rand: 314,
        };
        let vdaf = fake::Vdaf::new();
        let verify_param = vdaf.setup().unwrap().1.remove(1);
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let clock = MockClock::default();
        let skew = Duration::minutes(10);
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            task_id,
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Helper,
            hpke_private_key,
        );
        let hmac_key = generate_hmac_key();

        // Setup datastore.
        datastore
            .run_tx(|tx| {
                Box::pin(async move {
                    tx.put_task(&new_dummy_task_parameters(
                        task_id,
                        Vdaf::Prio3Aes128Count,
                        Role::Helper,
                    ))
                    .await?;
                    tx.put_report_share(
                        task_id,
                        &ReportShare {
                            nonce,
                            extensions: Vec::new(),
                            encrypted_input_share: HpkeCiphertext {
                                config_id: HpkeConfigId(42),
                                encapsulated_context: Vec::from("012345"),
                                payload: Vec::from("543210"),
                            },
                        },
                    )
                    .await?;
                    tx.put_aggregation_job(&AggregationJob::<fake::Vdaf> {
                        aggregation_job_id,
                        task_id,
                        aggregation_param: (),
                        state: AggregationJobState::InProgress,
                    })
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<fake::Vdaf> {
                        aggregation_job_id,
                        task_id,
                        nonce,
                        ord: 0,
                        state: ReportAggregationState::Waiting(()),
                    })
                    .await
                })
            })
            .await
            .unwrap();

        // Make request.
        let request = AggregateReq {
            task_id,
            job_id: aggregation_job_id,
            body: AggregateContinueReq {
                seq: vec![Transition {
                    nonce: Nonce {
                        time: Time(54321),
                        rand: 315, // not the same as above
                    },
                    trans_data: TransitionTypeSpecificData::Continued {
                        payload: Vec::new(),
                    },
                }],
            },
        };

        let filter = aggregator_filter(
            vdaf,
            Arc::new(datastore),
            clock,
            skew,
            Role::Helper,
            verify_param,
            hpke_recipient,
            hmac_key.clone(),
        )
        .unwrap();

        let (parts, body) = warp::test::request()
            .method("POST")
            .path("/aggregate")
            .body(AuthenticatedEncoder::new(request).encode(&hmac_key))
            .filter(&filter)
            .await
            .unwrap()
            .into_response()
            .into_parts();

        // Check that response is as desired.
        assert_eq!(parts.status, StatusCode::BAD_REQUEST);
        let problem_details: serde_json::Value =
            serde_json::from_slice(&hyper::body::to_bytes(body).await.unwrap()).unwrap();
        assert_eq!(
            problem_details,
            serde_json::json!({
                "status": StatusCode::BAD_REQUEST.as_u16(),
                "type": "urn:ietf:params:ppm:error:unrecognizedMessage",
                "title": "The message type for a response was incorrect or the payload was malformed.",
                "detail": "The message type for a response was incorrect or the payload was malformed.",
                "instance": "..",
                "taskid": base64::encode(task_id.as_bytes()),
            })
        );
    }

    #[tokio::test]
    async fn aggregate_continue_out_of_order_transition() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        // Prepare parameters.
        let task_id = TaskId::random();
        let aggregation_job_id = AggregationJobId::random();
        let nonce_0 = Nonce {
            time: Time(54321),
            rand: 314,
        };
        let nonce_1 = Nonce {
            time: Time(54321),
            rand: 315,
        };

        let vdaf = fake::Vdaf::new();
        let verify_param = vdaf.setup().unwrap().1.remove(1);
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let clock = MockClock::default();
        let skew = Duration::minutes(10);
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            task_id,
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Helper,
            hpke_private_key,
        );
        let hmac_key = generate_hmac_key();

        // Setup datastore.
        datastore
            .run_tx(|tx| {
                Box::pin(async move {
                    tx.put_task(&new_dummy_task_parameters(
                        task_id,
                        Vdaf::Prio3Aes128Count,
                        Role::Helper,
                    ))
                    .await?;

                    tx.put_report_share(
                        task_id,
                        &ReportShare {
                            nonce: nonce_0,
                            extensions: Vec::new(),
                            encrypted_input_share: HpkeCiphertext {
                                config_id: HpkeConfigId(42),
                                encapsulated_context: Vec::from("012345"),
                                payload: Vec::from("543210"),
                            },
                        },
                    )
                    .await?;
                    tx.put_report_share(
                        task_id,
                        &ReportShare {
                            nonce: nonce_1,
                            extensions: Vec::new(),
                            encrypted_input_share: HpkeCiphertext {
                                config_id: HpkeConfigId(42),
                                encapsulated_context: Vec::from("012345"),
                                payload: Vec::from("543210"),
                            },
                        },
                    )
                    .await?;

                    tx.put_aggregation_job(&AggregationJob::<fake::Vdaf> {
                        aggregation_job_id,
                        task_id,
                        aggregation_param: (),
                        state: AggregationJobState::InProgress,
                    })
                    .await?;

                    tx.put_report_aggregation(&ReportAggregation::<fake::Vdaf> {
                        aggregation_job_id,
                        task_id,
                        nonce: nonce_0,
                        ord: 0,
                        state: ReportAggregationState::Waiting(()),
                    })
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<fake::Vdaf> {
                        aggregation_job_id,
                        task_id,
                        nonce: nonce_1,
                        ord: 1,
                        state: ReportAggregationState::Waiting(()),
                    })
                    .await
                })
            })
            .await
            .unwrap();

        // Make request.
        let request = AggregateReq {
            task_id,
            job_id: aggregation_job_id,
            body: AggregateContinueReq {
                seq: vec![
                    // nonces are in opposite order to what was stored in the datastore.
                    Transition {
                        nonce: nonce_1,
                        trans_data: TransitionTypeSpecificData::Continued {
                            payload: Vec::new(),
                        },
                    },
                    Transition {
                        nonce: nonce_0,
                        trans_data: TransitionTypeSpecificData::Continued {
                            payload: Vec::new(),
                        },
                    },
                ],
            },
        };

        let filter = aggregator_filter(
            vdaf,
            Arc::new(datastore),
            clock,
            skew,
            Role::Helper,
            verify_param,
            hpke_recipient,
            hmac_key.clone(),
        )
        .unwrap();

        let (parts, body) = warp::test::request()
            .method("POST")
            .path("/aggregate")
            .body(AuthenticatedEncoder::new(request).encode(&hmac_key))
            .filter(&filter)
            .await
            .unwrap()
            .into_response()
            .into_parts();

        // Check that response is as desired.
        assert_eq!(parts.status, StatusCode::BAD_REQUEST);
        let problem_details: serde_json::Value =
            serde_json::from_slice(&hyper::body::to_bytes(body).await.unwrap()).unwrap();
        assert_eq!(
            problem_details,
            serde_json::json!({
                "status": StatusCode::BAD_REQUEST.as_u16(),
                "type": "urn:ietf:params:ppm:error:unrecognizedMessage",
                "title": "The message type for a response was incorrect or the payload was malformed.",
                "detail": "The message type for a response was incorrect or the payload was malformed.",
                "instance": "..",
                "taskid": base64::encode(task_id.as_bytes()),
            })
        );
    }

    #[tokio::test]
    async fn aggregate_continue_for_non_waiting_aggregation() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        // Prepare parameters.
        let task_id = TaskId::random();
        let aggregation_job_id = AggregationJobId::random();
        let nonce = Nonce {
            time: Time(54321),
            rand: 314,
        };
        let vdaf = fake::Vdaf::new();
        let verify_param = vdaf.setup().unwrap().1.remove(1);
        let (datastore, _db_handle) = ephemeral_datastore().await;
        let clock = MockClock::default();
        let skew = Duration::minutes(10);
        let (hpke_config, hpke_private_key) = generate_hpke_config_and_private_key();
        let hpke_recipient = HpkeRecipient::new(
            task_id,
            hpke_config,
            Label::InputShare,
            Role::Client,
            Role::Helper,
            hpke_private_key,
        );
        let hmac_key = generate_hmac_key();

        // Setup datastore.
        datastore
            .run_tx(|tx| {
                Box::pin(async move {
                    tx.put_task(&new_dummy_task_parameters(
                        task_id,
                        Vdaf::Prio3Aes128Count,
                        Role::Helper,
                    ))
                    .await?;
                    tx.put_report_share(
                        task_id,
                        &ReportShare {
                            nonce,
                            extensions: Vec::new(),
                            encrypted_input_share: HpkeCiphertext {
                                config_id: HpkeConfigId(42),
                                encapsulated_context: Vec::from("012345"),
                                payload: Vec::from("543210"),
                            },
                        },
                    )
                    .await?;
                    tx.put_aggregation_job(&AggregationJob::<fake::Vdaf> {
                        aggregation_job_id,
                        task_id,
                        aggregation_param: (),
                        state: AggregationJobState::InProgress,
                    })
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<fake::Vdaf> {
                        aggregation_job_id,
                        task_id,
                        nonce,
                        ord: 0,
                        state: ReportAggregationState::Invalid,
                    })
                    .await
                })
            })
            .await
            .unwrap();

        // Make request.
        let request = AggregateReq {
            task_id,
            job_id: aggregation_job_id,
            body: AggregateContinueReq {
                seq: vec![Transition {
                    nonce: Nonce {
                        time: Time(54321),
                        rand: 314,
                    },
                    trans_data: TransitionTypeSpecificData::Continued {
                        payload: Vec::new(),
                    },
                }],
            },
        };

        let filter = aggregator_filter(
            vdaf,
            Arc::new(datastore),
            clock,
            skew,
            Role::Helper,
            verify_param,
            hpke_recipient,
            hmac_key.clone(),
        )
        .unwrap();

        let (parts, body) = warp::test::request()
            .method("POST")
            .path("/aggregate")
            .body(AuthenticatedEncoder::new(request).encode(&hmac_key))
            .filter(&filter)
            .await
            .unwrap()
            .into_response()
            .into_parts();

        // Check that response is as desired.
        assert_eq!(parts.status, StatusCode::BAD_REQUEST);
        let problem_details: serde_json::Value =
            serde_json::from_slice(&hyper::body::to_bytes(body).await.unwrap()).unwrap();
        assert_eq!(
            problem_details,
            serde_json::json!({
                "status": StatusCode::BAD_REQUEST.as_u16(),
                "type": "urn:ietf:params:ppm:error:unrecognizedMessage",
                "title": "The message type for a response was incorrect or the payload was malformed.",
                "detail": "The message type for a response was incorrect or the payload was malformed.",
                "instance": "..",
                "taskid": base64::encode(task_id.as_bytes()),
            })
        );
    }

    fn generate_nonce<C: Clock>(clock: &C) -> Nonce {
        Nonce {
            time: Time::from_naive_date_time(clock.now()),
            rand: thread_rng().gen(),
        }
    }

    /// A transcript of a VDAF run. All fields are indexed by participant index (in PPM terminology,
    /// index 0 = leader, index 1 = helper).
    struct VdafTranscript<V: vdaf::Aggregator>
    where
        for<'a> &'a V::AggregateShare: Into<Vec<u8>>,
    {
        input_shares: Vec<V::InputShare>,
        transitions: Vec<Vec<PrepareTransition<V>>>,
        messages: Vec<V::PrepareMessage>,
    }

    // run_vdaf runs a VDAF state machine from sharding through to generating an output share,
    // returning a "transcript" of all states & messages.
    fn run_vdaf<V: vdaf::Aggregator + vdaf::Client>(
        vdaf: &V,
        public_param: &V::PublicParam,
        verify_params: &[V::VerifyParam],
        aggregation_param: &V::AggregationParam,
        nonce: Nonce,
        measurement: &V::Measurement,
    ) -> VdafTranscript<V>
    where
        for<'a> &'a V::AggregateShare: Into<Vec<u8>>,
    {
        // Shard inputs into input shares, and initialize the initial PrepareTransitions.
        let input_shares = vdaf.shard(public_param, measurement).unwrap();
        let mut prep_trans: Vec<Vec<PrepareTransition<V>>> = input_shares
            .iter()
            .enumerate()
            .map(|(idx, input_share)| {
                let prep_step = vdaf.prepare_init(
                    &verify_params[idx],
                    aggregation_param,
                    &nonce.get_encoded(),
                    input_share,
                )?;
                let prep_trans = vdaf.prepare_step(prep_step, None);
                Ok(vec![prep_trans])
            })
            .collect::<Result<Vec<Vec<PrepareTransition<V>>>, VdafError>>()
            .unwrap();
        let mut combined_prep_msgs = Vec::new();

        // Repeatedly step the VDAF until we reach a terminal state.
        loop {
            // Gather messages from last round & combine them into next round's message; if any
            // participants have reached a terminal state (Finish or Fail), we are done.
            let mut prep_msgs = Vec::new();
            for pts in &prep_trans {
                match pts.last().unwrap() {
                    PrepareTransition::<V>::Continue(_, prep_msg) => {
                        prep_msgs.push(prep_msg.clone())
                    }
                    _ => {
                        return VdafTranscript {
                            input_shares,
                            transitions: prep_trans,
                            messages: combined_prep_msgs,
                        }
                    }
                }
            }
            let combined_prep_msg = vdaf.prepare_preprocess(prep_msgs).unwrap();
            combined_prep_msgs.push(combined_prep_msg.clone());

            // Compute each participant's next transition.
            for pts in &mut prep_trans {
                let prep_step = assert_matches!(pts.last().unwrap(), PrepareTransition::<V>::Continue(prep_step, _) => prep_step).clone();
                pts.push(vdaf.prepare_step(prep_step, Some(combined_prep_msg.clone())));
            }
        }
    }

    fn generate_helper_report_share<V: vdaf::Client>(
        task_id: TaskId,
        nonce: Nonce,
        cfg: &HpkeConfig,
        input_share: &V::InputShare,
    ) -> ReportShare
    where
        for<'a> &'a V::AggregateShare: Into<Vec<u8>>,
    {
        let associated_data = Report::associated_data(nonce, &[]);
        generate_helper_report_share_for_plaintext(
            task_id,
            nonce,
            cfg,
            &input_share.get_encoded(),
            &associated_data,
        )
    }

    fn generate_helper_report_share_for_plaintext(
        task_id: TaskId,
        nonce: Nonce,
        cfg: &HpkeConfig,
        plaintext: &[u8],
        associated_data: &[u8],
    ) -> ReportShare {
        let helper_sender = HpkeSender::new(
            task_id,
            cfg.clone(),
            Label::InputShare,
            Role::Client,
            Role::Helper,
        );
        let encrypted_input_share = helper_sender.seal(plaintext, associated_data).unwrap();
        ReportShare {
            nonce,
            extensions: Vec::new(),
            encrypted_input_share,
        }
    }

    fn generate_hmac_key() -> hmac::Key {
        hmac::Key::generate(HMAC_SHA256, &SystemRandom::new()).unwrap()
    }
}
