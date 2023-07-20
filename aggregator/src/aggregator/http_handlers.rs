use super::{Aggregator, Config, Error};
use crate::aggregator::problem_details::ProblemDetailsConnExt;
use async_trait::async_trait;
use janus_aggregator_core::{datastore::Datastore, instrumented};
use janus_core::{
    http::extract_bearer_token,
    task::{AuthenticationToken, DapAuthToken, DAP_AUTH_HEADER},
    time::Clock,
};
use janus_messages::{
    problem_type::DapProblemType, query_type::FixedSize, AggregateShare, AggregateShareReq,
    AggregationJobContinueReq, AggregationJobId, AggregationJobInitializeReq, AggregationJobResp,
    Collection, CollectionJobId, CollectionReq, HpkeConfigList, Report, TaskId,
};
use opentelemetry::{
    metrics::{Counter, Meter},
    Context, KeyValue,
};
use prio::codec::Encode;
use routefinder::Captures;
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration as StdDuration;
use tracing::warn;
use trillium::{Conn, Handler, KnownHeaderName, Status};
use trillium_api::{api, State};
use trillium_caching_headers::CacheControlDirective;
use trillium_opentelemetry::metrics;
use trillium_router::{Router, RouterConnExt};

/// Newtype holding a textual error code, to be stored in a Trillium connection's state.
struct ErrorCode(&'static str);

#[async_trait]
impl Handler for Error {
    async fn run(&self, mut conn: Conn) -> Conn {
        let error_code = self.error_code();
        conn.set_state(ErrorCode(error_code));
        warn!(error_code, error=?self, "Error handling endpoint");
        match self {
            Error::InvalidConfiguration(_) => conn.with_status(Status::InternalServerError),
            Error::MessageDecode(_) => {
                conn.with_problem_details(DapProblemType::UnrecognizedMessage, None)
            }
            Error::ReportRejected(task_id, _, _) => {
                conn.with_problem_details(DapProblemType::ReportRejected, Some(task_id))
            }
            Error::UnrecognizedMessage(task_id, _) => {
                conn.with_problem_details(DapProblemType::UnrecognizedMessage, task_id.as_ref())
            }
            Error::RoundMismatch { task_id, .. } => {
                conn.with_problem_details(DapProblemType::RoundMismatch, Some(task_id))
            }
            Error::UnrecognizedTask(task_id) => {
                conn.with_problem_details(DapProblemType::UnrecognizedTask, Some(task_id))
            }
            Error::MissingTaskId => conn.with_problem_details(DapProblemType::MissingTaskId, None),
            Error::UnrecognizedAggregationJob(task_id, _) => {
                conn.with_problem_details(DapProblemType::UnrecognizedAggregationJob, Some(task_id))
            }
            Error::DeletedCollectionJob(_) => conn.with_status(Status::NoContent),
            Error::UnrecognizedCollectionJob(_) => conn.with_status(Status::NotFound),
            Error::OutdatedHpkeConfig(task_id, _) => {
                conn.with_problem_details(DapProblemType::OutdatedConfig, Some(task_id))
            }
            Error::ReportTooEarly(task_id, _, _) => {
                conn.with_problem_details(DapProblemType::ReportTooEarly, Some(task_id))
            }
            Error::UnauthorizedRequest(task_id) => {
                conn.with_problem_details(DapProblemType::UnauthorizedRequest, Some(task_id))
            }
            Error::InvalidBatchSize(task_id, _) => {
                conn.with_problem_details(DapProblemType::InvalidBatchSize, Some(task_id))
            }
            Error::BatchInvalid(task_id, _) => {
                conn.with_problem_details(DapProblemType::BatchInvalid, Some(task_id))
            }
            Error::BatchOverlap(task_id, _) => {
                conn.with_problem_details(DapProblemType::BatchOverlap, Some(task_id))
            }
            Error::BatchMismatch(inner) => {
                conn.with_problem_details(DapProblemType::BatchMismatch, Some(&inner.task_id))
            }
            Error::BatchQueriedTooManyTimes(task_id, _) => {
                conn.with_problem_details(DapProblemType::BatchQueriedTooManyTimes, Some(task_id))
            }
            Error::Hpke(_)
            | Error::Datastore(_)
            | Error::Vdaf(_)
            | Error::Internal(_)
            | Error::Url(_)
            | Error::Message(_)
            | Error::HttpClient(_)
            | Error::Http { .. }
            | Error::TaskParameters(_) => conn.with_status(Status::InternalServerError),
            Error::AggregateShareRequestRejected(_, _) => conn.with_status(Status::BadRequest),
            Error::EmptyAggregation(task_id) => {
                conn.with_problem_details(DapProblemType::UnrecognizedMessage, Some(task_id))
            }
            Error::ForbiddenMutation { .. } => conn.with_status(Status::Conflict),
            Error::BadRequest(_) => conn.with_status(Status::BadRequest),
        }
    }
}

/// The number of seconds we send in the Access-Control-Max-Age header. This determines for how
/// long clients will cache the results of CORS preflight requests. Of popular browsers, Mozilla
/// Firefox has the highest Max-Age cap, at 24 hours, so we use that. Our CORS preflight handlers
/// are tightly scoped to relevant endpoints, and our CORS settings are unlikely to change.
/// See: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Max-Age
const CORS_PREFLIGHT_CACHE_AGE: u32 = 24 * 60 * 60;

/// Wrapper around a type that implements [`Encode`]. It acts as a Trillium handler, encoding the
/// inner object and sending it as the response body, setting the Content-Type header to the
/// provided media type, and setting the status to 200.
struct EncodedBody<T> {
    object: T,
    media_type: &'static str,
}

impl<T> EncodedBody<T>
where
    T: Encode,
{
    fn new(object: T, media_type: &'static str) -> Self {
        Self { object, media_type }
    }
}

#[async_trait]
impl<T> Handler for EncodedBody<T>
where
    T: Encode + Sync + Send + 'static,
{
    async fn run(&self, conn: Conn) -> Conn {
        conn.with_header(KnownHeaderName::ContentType, self.media_type)
            .ok(self.object.get_encoded())
    }
}

/// A Trillium handler that checks for state set when sending an error response, and updates an
/// OpenTelemetry counter accordingly.
struct StatusCounter(Counter<u64>);

impl StatusCounter {
    fn new(meter: &Meter) -> Self {
        Self(
            meter
                .u64_counter("janus_aggregator_responses")
                .with_description(
                    "Count of requests handled by the aggregator, by method, route, and response status.",
                )
                .init(),
        )
    }
}

#[async_trait]
impl Handler for StatusCounter {
    async fn run(&self, conn: Conn) -> Conn {
        conn
    }

    async fn before_send(&self, conn: Conn) -> Conn {
        // Check for the error code set by the Error handler implementation.
        let error_code_opt = conn.state::<ErrorCode>().map(|error_code| error_code.0);
        let error_code = if let Some(status) = conn.status() {
            if status.is_client_error() || status.is_server_error() {
                error_code_opt.unwrap_or("unknown")
            } else {
                // Set the label to an empty string on success.
                ""
            }
        } else {
            // No status is set, it will fall back to 404.
            error_code_opt.unwrap_or("unknown")
        };
        // Fetch the method.
        let method = conn.method().as_str();
        // Check for the route set by the router.
        let route = conn
            .route()
            .map(ToString::to_string)
            .unwrap_or_else(|| "unknown".to_owned());
        self.0.add(
            &Context::current(),
            1,
            &[
                KeyValue::new("method", method),
                KeyValue::new("route", route),
                KeyValue::new("error_code", error_code),
            ],
        );
        conn
    }
}

pub(crate) static AGGREGATION_JOB_ROUTE: &str =
    "tasks/:task_id/aggregation_jobs/:aggregation_job_id";
pub(crate) static COLLECTION_JOB_ROUTE: &str = "tasks/:task_id/collection_jobs/:collection_job_id";
pub(crate) static AGGREGATE_SHARES_ROUTE: &str = "tasks/:task_id/aggregate_shares";

/// Constructs a Trillium handler for the aggregator.
pub fn aggregator_handler<C: Clock>(
    datastore: Arc<Datastore<C>>,
    clock: C,
    meter: &Meter,
    cfg: Config,
) -> Result<impl Handler, Error> {
    let aggregator = Arc::new(Aggregator::new(datastore, clock, meter, cfg));

    Ok((
        State(aggregator),
        metrics("janus_aggregator").with_route(|conn| conn.route().map(ToString::to_string)),
        Router::new()
            .without_options_handling()
            .get("hpke_config", instrumented(api(hpke_config::<C>)))
            .with_route(
                trillium::Method::Options,
                "hpke_config",
                hpke_config_cors_preflight,
            )
            .put("tasks/:task_id/reports", instrumented(api(upload::<C>)))
            .with_route(
                trillium::Method::Options,
                "tasks/:task_id/reports",
                upload_cors_preflight,
            )
            .put(
                AGGREGATION_JOB_ROUTE,
                instrumented(api(aggregation_jobs_put::<C>)),
            )
            .post(
                AGGREGATION_JOB_ROUTE,
                instrumented(api(aggregation_jobs_post::<C>)),
            )
            .put(
                COLLECTION_JOB_ROUTE,
                instrumented(api(collection_jobs_put::<C>)),
            )
            .post(
                COLLECTION_JOB_ROUTE,
                instrumented(api(collection_jobs_post::<C>)),
            )
            .delete(
                COLLECTION_JOB_ROUTE,
                instrumented(api(collection_jobs_delete::<C>)),
            )
            .post(
                AGGREGATE_SHARES_ROUTE,
                instrumented(api(aggregate_shares::<C>)),
            ),
        StatusCounter::new(meter),
    ))
}

/// Deserialization helper struct to extract a "task_id" parameter from a query string.
#[derive(Deserialize)]
struct HpkeConfigQuery {
    /// The optional "task_id" parameter, in base64url-encoded form.
    #[serde(default)]
    task_id: Option<String>,
}

/// API handler for the "/hpke_config" GET endpoint.
async fn hpke_config<C: Clock>(
    conn: &mut Conn,
    State(aggregator): State<Arc<Aggregator<C>>>,
) -> Result<(CacheControlDirective, EncodedBody<HpkeConfigList>), Error> {
    let query = serde_urlencoded::from_str::<HpkeConfigQuery>(conn.querystring())
        .map_err(|err| Error::BadRequest(format!("couldn't parse query string: {err}")))?;
    let hpke_config_list = aggregator
        .handle_hpke_config(query.task_id.as_ref().map(AsRef::as_ref))
        .await?;

    // Handle CORS, if the request header is present.
    if let Some(origin) = conn.request_headers().get(KnownHeaderName::Origin) {
        // Unconditionally allow CORS requests from all origins.
        let origin = origin.clone();
        conn.headers_mut()
            .insert(KnownHeaderName::AccessControlAllowOrigin, origin);
    }

    Ok((
        CacheControlDirective::MaxAge(StdDuration::from_secs(86400)),
        EncodedBody::new(hpke_config_list, HpkeConfigList::MEDIA_TYPE),
    ))
}

/// Handler for CORS preflight requests to "/hpke_config".
async fn hpke_config_cors_preflight(mut conn: Conn) -> Conn {
    conn.headers_mut().insert(KnownHeaderName::Allow, "GET");
    if let Some(origin) = conn.request_headers().get(KnownHeaderName::Origin) {
        let origin = origin.clone();
        let request_headers = conn.headers_mut();
        request_headers.insert(KnownHeaderName::AccessControlAllowOrigin, origin);
        request_headers.insert(KnownHeaderName::AccessControlAllowMethods, "GET");
        request_headers.insert(
            KnownHeaderName::AccessControlMaxAge,
            format!("{CORS_PREFLIGHT_CACHE_AGE}"),
        );
    }
    conn.set_status(Status::Ok);
    conn
}

/// API handler for the "/tasks/.../reports" PUT endpoint.
async fn upload<C: Clock>(
    conn: &mut Conn,
    (State(aggregator), State(captures), body): (
        State<Arc<Aggregator<C>>>,
        State<Captures<'static, 'static>>,
        Vec<u8>,
    ),
) -> Result<Status, Arc<Error>> {
    validate_content_type(conn, Report::MEDIA_TYPE).map_err(Arc::new)?;

    let task_id = parse_task_id(&captures).map_err(Arc::new)?;
    aggregator.handle_upload(&task_id, &body).await?;

    // Handle CORS, if the request header is present.
    if let Some(origin) = conn.request_headers().get(KnownHeaderName::Origin) {
        // Unconditionally allow CORS requests from all origins.
        let origin = origin.clone();
        conn.headers_mut()
            .insert(KnownHeaderName::AccessControlAllowOrigin, origin);
    }

    Ok(Status::Ok)
}

/// Handler for CORS preflight requests to "/tasks/.../reports".
async fn upload_cors_preflight(mut conn: Conn) -> Conn {
    conn.headers_mut().insert(KnownHeaderName::Allow, "PUT");
    if let Some(origin) = conn.request_headers().get(KnownHeaderName::Origin) {
        let origin = origin.clone();
        let request_headers = conn.headers_mut();
        request_headers.insert(KnownHeaderName::AccessControlAllowOrigin, origin);
        request_headers.insert(KnownHeaderName::AccessControlAllowMethods, "PUT");
        request_headers.insert(KnownHeaderName::AccessControlAllowHeaders, "content-type");
        request_headers.insert(
            KnownHeaderName::AccessControlMaxAge,
            format!("{CORS_PREFLIGHT_CACHE_AGE}"),
        );
    }
    conn.set_status(Status::Ok);
    conn
}

/// API handler for the "/tasks/.../aggregation_jobs/..." PUT endpoint.
async fn aggregation_jobs_put<C: Clock>(
    conn: &mut Conn,
    (State(aggregator), State(captures), body): (
        State<Arc<Aggregator<C>>>,
        State<Captures<'static, 'static>>,
        Vec<u8>,
    ),
) -> Result<EncodedBody<AggregationJobResp>, Error> {
    validate_content_type(conn, AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE)?;

    let task_id = parse_task_id(&captures)?;
    let aggregation_job_id = parse_aggregation_job_id(&captures)?;
    let auth_token = parse_auth_token(&task_id, conn)?;
    let response = aggregator
        .handle_aggregate_init(&task_id, &aggregation_job_id, &body, auth_token)
        .await?;

    Ok(EncodedBody::new(response, AggregationJobResp::MEDIA_TYPE))
}

/// API handler for the "/tasks/.../aggregation_jobs/..." POST endpoint.
async fn aggregation_jobs_post<C: Clock>(
    conn: &mut Conn,
    (State(aggregator), State(captures), body): (
        State<Arc<Aggregator<C>>>,
        State<Captures<'static, 'static>>,
        Vec<u8>,
    ),
) -> Result<EncodedBody<AggregationJobResp>, Error> {
    validate_content_type(conn, AggregationJobContinueReq::MEDIA_TYPE)?;

    let task_id = parse_task_id(&captures)?;
    let aggregation_job_id = parse_aggregation_job_id(&captures)?;
    let auth_token = parse_auth_token(&task_id, conn)?;
    let response = aggregator
        .handle_aggregate_continue(&task_id, &aggregation_job_id, &body, auth_token)
        .await?;

    Ok(EncodedBody::new(response, AggregationJobResp::MEDIA_TYPE))
}

/// API handler for the "/tasks/.../collection_jobs/..." PUT endpoint.
async fn collection_jobs_put<C: Clock>(
    conn: &mut Conn,
    (State(aggregator), State(captures), body): (
        State<Arc<Aggregator<C>>>,
        State<Captures<'static, 'static>>,
        Vec<u8>,
    ),
) -> Result<Status, Error> {
    validate_content_type(conn, CollectionReq::<FixedSize>::MEDIA_TYPE)?;

    let task_id = parse_task_id(&captures)?;
    let collection_job_id = parse_collection_job_id(&captures)?;
    let auth_token = parse_auth_token(&task_id, conn)?;
    aggregator
        .handle_create_collection_job(&task_id, &collection_job_id, &body, auth_token)
        .await?;

    Ok(Status::Created)
}

/// API handler for the "/tasks/.../collection_jobs/..." POST endpoint.
async fn collection_jobs_post<C: Clock>(
    conn: &mut Conn,
    (State(aggregator), State(captures)): (
        State<Arc<Aggregator<C>>>,
        State<Captures<'static, 'static>>,
    ),
) -> Result<(), Error> {
    let task_id = parse_task_id(&captures)?;
    let collection_job_id = parse_collection_job_id(&captures)?;
    let auth_token = parse_auth_token(&task_id, conn)?;
    let response_opt = aggregator
        .handle_get_collection_job(&task_id, &collection_job_id, auth_token)
        .await?;
    match response_opt {
        Some(response_bytes) => {
            conn.headers_mut().insert(
                KnownHeaderName::ContentType,
                Collection::<FixedSize>::MEDIA_TYPE,
            );
            conn.set_status(Status::Ok);
            conn.set_body(response_bytes);
        }
        None => conn.set_status(Status::Accepted),
    }
    Ok(())
}

/// API handler for the "/tasks/.../collection_jobs/..." DELETE endpoint.
async fn collection_jobs_delete<C: Clock>(
    conn: &mut Conn,
    (State(aggregator), State(captures)): (
        State<Arc<Aggregator<C>>>,
        State<Captures<'static, 'static>>,
    ),
) -> Result<Status, Error> {
    let task_id = parse_task_id(&captures)?;
    let collection_job_id = parse_collection_job_id(&captures)?;
    let auth_token = parse_auth_token(&task_id, conn)?;
    aggregator
        .handle_delete_collection_job(&task_id, &collection_job_id, auth_token)
        .await?;
    Ok(Status::NoContent)
}

/// API handler for the "/tasks/.../aggregate_shares" POST endpoint.
async fn aggregate_shares<C: Clock>(
    conn: &mut Conn,
    (State(aggregator), State(captures), body): (
        State<Arc<Aggregator<C>>>,
        State<Captures<'static, 'static>>,
        Vec<u8>,
    ),
) -> Result<EncodedBody<AggregateShare>, Error> {
    validate_content_type(conn, AggregateShareReq::<FixedSize>::MEDIA_TYPE)?;

    let task_id = parse_task_id(&captures)?;
    let auth_token = parse_auth_token(&task_id, conn)?;
    let share = aggregator
        .handle_aggregate_share(&task_id, &body, auth_token)
        .await?;

    Ok(EncodedBody::new(share, AggregateShare::MEDIA_TYPE))
}

/// Check the request's Content-Type header, and return an error if it is missing or not equal to
/// the expected value.
fn validate_content_type(conn: &Conn, expected_media_type: &'static str) -> Result<(), Error> {
    if let Some(content_type) = conn.request_headers().get(KnownHeaderName::ContentType) {
        if content_type != expected_media_type {
            Err(Error::BadRequest(format!(
                "wrong Content-Type header: {content_type}"
            )))
        } else {
            Ok(())
        }
    } else {
        Err(Error::BadRequest("no Content-Type header".to_owned()))
    }
}

/// Parse a [`TaskId`] from the "task_id" parameter in a set of path parameter [`Captures`].
fn parse_task_id(captures: &Captures) -> Result<TaskId, Error> {
    let encoded = captures
        .get("task_id")
        .ok_or_else(|| Error::Internal("task_id parameter is missing from captures".to_string()))?;
    encoded
        .parse()
        .map_err(|_| Error::BadRequest("invalid TaskId".to_owned()))
}

/// Parse an [`AggregationJobId`] from the "aggregation_job_id" parameter in a set of path parameter
/// [`Captures`].
fn parse_aggregation_job_id(captures: &Captures) -> Result<AggregationJobId, Error> {
    let encoded = captures.get("aggregation_job_id").ok_or_else(|| {
        Error::Internal("aggregation_job_id parameter is missing from captures".to_string())
    })?;
    encoded
        .parse()
        .map_err(|_| Error::BadRequest("invalid AggregationJobId".to_owned()))
}

/// Parse an [`CollectionJobId`] from the "collection_job_id" parameter in a set of path parameter
/// [`Captures`].
fn parse_collection_job_id(captures: &Captures) -> Result<CollectionJobId, Error> {
    let encoded = captures.get("collection_job_id").ok_or_else(|| {
        Error::Internal("collection_job_id parameter is missing from captures".to_string())
    })?;
    encoded
        .parse()
        .map_err(|_| Error::BadRequest("invalid CollectionJobId".to_owned()))
}

/// Get an [`AuthenticationToken`] from the request.
fn parse_auth_token(task_id: &TaskId, conn: &Conn) -> Result<Option<AuthenticationToken>, Error> {
    // Prefer a bearer token, then fall back to DAP-Auth-Token
    if let Some(bearer_token) =
        extract_bearer_token(conn).map_err(|_| Error::UnauthorizedRequest(*task_id))?
    {
        return Ok(Some(AuthenticationToken::Bearer(bearer_token)));
    }

    conn.request_headers()
        .get(DAP_AUTH_HEADER)
        .map(|value| {
            DapAuthToken::try_from(value.as_ref().to_vec())
                .map(AuthenticationToken::DapAuth)
                .map_err(|e| Error::BadRequest(format!("bad DAP-Auth-Token header: {e}")))
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use crate::aggregator::{
        aggregate_init_tests::{put_aggregation_job, setup_aggregate_init_test},
        aggregation_job_continue::test_util::{
            post_aggregation_job_and_decode, post_aggregation_job_expecting_error,
        },
        collection_job_tests::setup_collection_job_test_case,
        empty_batch_aggregations,
        http_handlers::aggregator_handler,
        tests::{
            create_report, create_report_with_id, default_aggregator_config,
            generate_helper_report_share, generate_helper_report_share_for_plaintext,
            BATCH_AGGREGATION_SHARD_COUNT, DUMMY_VERIFY_KEY_LENGTH,
        },
    };
    use assert_matches::assert_matches;
    use futures::future::try_join_all;
    use janus_aggregator_core::{
        datastore::{
            models::{
                AggregateShareJob, AggregationJob, AggregationJobState, Batch, BatchAggregation,
                BatchAggregationState, BatchState, CollectionJob, CollectionJobState,
                ReportAggregation, ReportAggregationState,
            },
            test_util::ephemeral_datastore,
        },
        query_type::CollectableQueryType,
        task::{test_util::TaskBuilder, QueryType, VerifyKey},
        test_util::noop_meter,
    };
    use janus_core::{
        hpke::{
            self, test_util::generate_test_hpke_config_and_private_key, HpkeApplicationInfo, Label,
        },
        report_id::ReportIdChecksumExt,
        task::{AuthenticationToken, VdafInstance, PRIO3_VERIFY_KEY_LENGTH},
        test_util::{dummy_vdaf, install_test_trace_subscriber, run_vdaf},
        time::{Clock, DurationExt, MockClock, TimeExt},
    };
    use janus_messages::{
        query_type::FixedSize, AggregateShare as AggregateShareMessage, AggregateShareAad,
        AggregateShareReq, AggregationJobContinueReq, AggregationJobId,
        AggregationJobInitializeReq, AggregationJobResp, AggregationJobRound, BatchSelector,
        Collection, CollectionJobId, CollectionReq, Duration, Extension, ExtensionType,
        FixedSizeQuery, HpkeCiphertext, HpkeConfigId, HpkeConfigList, InputShareAad, Interval,
        PartialBatchSelector, PrepareStep, PrepareStepResult, Query, Report, ReportId,
        ReportIdChecksum, ReportMetadata, ReportShare, ReportShareError, Role, TaskId, Time,
    };
    use prio::{
        codec::{Decode, Encode},
        field::Field64,
        vdaf::{
            prio3::{Prio3, Prio3Count},
            AggregateShare, Aggregator, OutputShare,
        },
    };
    use rand::random;
    use serde_json::json;
    use std::{borrow::Cow, io::Cursor, sync::Arc};
    use trillium::{KnownHeaderName, Status};
    use trillium_testing::{
        assert_headers,
        prelude::{delete, get, post, put},
        TestConn,
    };

    #[tokio::test]
    async fn hpke_config() {
        install_test_trace_subscriber();

        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Leader,
        )
        .build();
        let unknown_task_id: TaskId = random();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = ephemeral_datastore.datastore(clock.clone()).await;

        datastore.put_task(&task).await.unwrap();

        let want_hpke_key = task.current_hpke_key().clone();

        let handler = aggregator_handler(
            Arc::new(datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        // No task ID provided
        let mut test_conn = get("/hpke_config").run_async(&handler).await;
        assert_eq!(test_conn.status(), Some(Status::BadRequest));
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": 400u16,
                "type": "urn:ietf:params:ppm:dap:error:missingTaskID",
                "title": "HPKE configuration was requested without specifying a task ID.",
            })
        );

        // Unknown task ID provided
        let mut test_conn = get(&format!("/hpke_config?task_id={unknown_task_id}"))
            .run_async(&handler)
            .await;
        // Expected status and problem type should be per the protocol
        // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.3.1
        assert_eq!(test_conn.status(), Some(Status::BadRequest));
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": 400u16,
                "type": "urn:ietf:params:ppm:dap:error:unrecognizedTask",
                "title": "An endpoint received a message with an unknown task ID.",
                "taskid": format!("{unknown_task_id}"),
            })
        );

        // Recognized task ID provided
        let mut test_conn = get(&format!("/hpke_config?task_id={}", task.id()))
            .run_async(&handler)
            .await;

        assert_eq!(test_conn.status(), Some(Status::Ok));
        assert_headers!(
            &test_conn,
            "cache-control" => "max-age=86400",
            "content-type" => (HpkeConfigList::MEDIA_TYPE),
        );

        let bytes = take_response_body(&mut test_conn).await;
        let hpke_config_list = HpkeConfigList::decode(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(
            hpke_config_list.hpke_configs(),
            &[want_hpke_key.config().clone()]
        );

        let application_info =
            HpkeApplicationInfo::new(&Label::InputShare, &Role::Client, &Role::Leader);
        let message = b"this is a message";
        let associated_data = b"some associated data";

        let ciphertext = hpke::seal(
            &hpke_config_list.hpke_configs()[0],
            &application_info,
            message,
            associated_data,
        )
        .unwrap();
        let plaintext = hpke::open(
            want_hpke_key.config(),
            want_hpke_key.private_key(),
            &application_info,
            &ciphertext,
            associated_data,
        )
        .unwrap();
        assert_eq!(&plaintext, message);
    }

    #[tokio::test]
    async fn hpke_config_cors_headers() {
        install_test_trace_subscriber();

        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Leader,
        )
        .build();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = ephemeral_datastore.datastore(clock.clone()).await;

        datastore.put_task(&task).await.unwrap();

        let handler = aggregator_handler(
            Arc::new(datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        // Check for appropriate CORS headers in response to a preflight request.
        let test_conn = TestConn::build(
            trillium::Method::Options,
            &format!("/hpke_config?task_id={}", task.id()),
            (),
        )
        .with_request_header(KnownHeaderName::Origin, "https://example.com/")
        .with_request_header(KnownHeaderName::AccessControlRequestMethod, "GET")
        .run_async(&handler)
        .await;
        assert!(test_conn.status().unwrap().is_success());
        assert_headers!(
            &test_conn,
            "access-control-allow-origin" => "https://example.com/",
            "access-control-allow-methods"=> "GET",
            "access-control-max-age"=> "86400",
        );

        // Check for appropriate CORS headers with a simple GET request.
        let test_conn = get(&format!("/hpke_config?task_id={}", task.id()))
            .with_request_header(KnownHeaderName::Origin, "https://example.com/")
            .run_async(&handler)
            .await;
        assert!(test_conn.status().unwrap().is_success());
        assert_headers!(
            &test_conn,
            "access-control-allow-origin" => "https://example.com/",
        );
    }

    #[tokio::test]
    async fn upload_handler() {
        async fn check_response(
            test_conn: &mut TestConn,
            desired_status: Status,
            desired_type: &str,
            desired_title: &str,
            desired_task_id: &TaskId,
        ) {
            assert_eq!(test_conn.status(), Some(desired_status));
            assert_eq!(
                take_problem_details(test_conn).await,
                json!({
                    "status": desired_status as u16,
                    "type": format!("urn:ietf:params:ppm:dap:error:{desired_type}"),
                    "title": desired_title,
                    "taskid": format!("{desired_task_id}"),
                }),
            )
        }

        install_test_trace_subscriber();

        const REPORT_EXPIRY_AGE: u64 = 1_000_000;
        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Leader,
        )
        .with_report_expiry_age(Some(Duration::from_seconds(REPORT_EXPIRY_AGE)))
        .build();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        datastore.put_task(&task).await.unwrap();
        let report = create_report(&task, clock.now());
        let handler = aggregator_handler(
            Arc::clone(&datastore),
            clock.clone(),
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        // Upload a report. Do this twice to prove that PUT is idempotent.
        for _ in 0..2 {
            let mut test_conn = put(task.report_upload_uri().unwrap().path())
                .with_request_header(KnownHeaderName::ContentType, Report::MEDIA_TYPE)
                .with_request_body(report.get_encoded())
                .run_async(&handler)
                .await;

            assert_eq!(test_conn.status(), Some(Status::Ok));
            assert!(test_conn.take_response_body().is_none());
        }

        let accepted_report_id = report.metadata().id();

        // Verify that new reports using an existing report ID are rejected with reportRejected
        let duplicate_id_report = create_report_with_id(&task, clock.now(), *accepted_report_id);
        let mut test_conn = put(task.report_upload_uri().unwrap().path())
            .with_request_header(KnownHeaderName::ContentType, Report::MEDIA_TYPE)
            .with_request_body(duplicate_id_report.get_encoded())
            .run_async(&handler)
            .await;
        check_response(
            &mut test_conn,
            Status::BadRequest,
            "reportRejected",
            "Report could not be processed.",
            task.id(),
        )
        .await;

        // Verify that reports older than the report expiry age are rejected with the reportRejected
        // error type.
        let gc_eligible_report = Report::new(
            ReportMetadata::new(
                random(),
                clock
                    .now()
                    .sub(&Duration::from_seconds(REPORT_EXPIRY_AGE + 30000))
                    .unwrap(),
            ),
            report.public_share().to_vec(),
            report.encrypted_input_shares().to_vec(),
        );
        let mut test_conn = put(task.report_upload_uri().unwrap().path())
            .with_request_header(KnownHeaderName::ContentType, Report::MEDIA_TYPE)
            .with_request_body(gc_eligible_report.get_encoded())
            .run_async(&handler)
            .await;
        check_response(
            &mut test_conn,
            Status::BadRequest,
            "reportRejected",
            "Report could not be processed.",
            task.id(),
        )
        .await;

        // should reject a report with only one share with the unrecognizedMessage type.
        let bad_report = Report::new(
            report.metadata().clone(),
            report.public_share().to_vec(),
            Vec::from([report.encrypted_input_shares()[0].clone()]),
        );
        let mut test_conn = put(task.report_upload_uri().unwrap().path())
            .with_request_header(KnownHeaderName::ContentType, Report::MEDIA_TYPE)
            .with_request_body(bad_report.get_encoded())
            .run_async(&handler)
            .await;
        check_response(
            &mut test_conn,
            Status::BadRequest,
            "unrecognizedMessage",
            "The message type for a response was incorrect or the payload was malformed.",
            task.id(),
        )
        .await;

        // should reject a report using the wrong HPKE config for the leader, and reply with
        // the error type outdatedConfig.
        let unused_hpke_config_id = (0..)
            .map(HpkeConfigId::from)
            .find(|id| !task.hpke_keys().contains_key(id))
            .unwrap();
        let bad_report = Report::new(
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
        let mut test_conn = put(task.report_upload_uri().unwrap().path())
            .with_request_header(KnownHeaderName::ContentType, Report::MEDIA_TYPE)
            .with_request_body(bad_report.get_encoded())
            .run_async(&handler)
            .await;
        check_response(
            &mut test_conn,
            Status::BadRequest,
            "outdatedConfig",
            "The message was generated using an outdated configuration.",
            task.id(),
        )
        .await;

        // Reports from the future should be rejected.
        let bad_report_time = clock
            .now()
            .add(&Duration::from_minutes(10).unwrap())
            .unwrap()
            .add(&Duration::from_seconds(1))
            .unwrap();
        let bad_report = Report::new(
            ReportMetadata::new(*report.metadata().id(), bad_report_time),
            report.public_share().to_vec(),
            report.encrypted_input_shares().to_vec(),
        );
        let mut test_conn = put(task.report_upload_uri().unwrap().path())
            .with_request_header(KnownHeaderName::ContentType, Report::MEDIA_TYPE)
            .with_request_body(bad_report.get_encoded())
            .run_async(&handler)
            .await;
        check_response(
            &mut test_conn,
            Status::BadRequest,
            "reportTooEarly",
            "Report could not be processed because it arrived too early.",
            task.id(),
        )
        .await;

        // Reports with timestamps past the task's expiration should be rejected.
        let task_expire_soon = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Leader,
        )
        .with_task_expiration(Some(clock.now().add(&Duration::from_seconds(60)).unwrap()))
        .build();
        datastore.put_task(&task_expire_soon).await.unwrap();
        let report_2 = create_report(
            &task_expire_soon,
            clock.now().add(&Duration::from_seconds(120)).unwrap(),
        );
        let mut test_conn = put(task_expire_soon.report_upload_uri().unwrap().path())
            .with_request_header(KnownHeaderName::ContentType, Report::MEDIA_TYPE)
            .with_request_body(report_2.get_encoded())
            .run_async(&handler)
            .await;
        check_response(
            &mut test_conn,
            Status::BadRequest,
            "reportRejected",
            "Report could not be processed.",
            task_expire_soon.id(),
        )
        .await;

        // Check for appropriate CORS headers in response to a preflight request.
        let test_conn = TestConn::build(
            trillium::Method::Options,
            task.report_upload_uri().unwrap().path(),
            (),
        )
        .with_request_header(KnownHeaderName::Origin, "https://example.com/")
        .with_request_header(KnownHeaderName::AccessControlRequestMethod, "PUT")
        .with_request_header(KnownHeaderName::AccessControlRequestHeaders, "content-type")
        .run_async(&handler)
        .await;
        assert!(test_conn.status().unwrap().is_success());
        assert_headers!(
            &test_conn,
            "access-control-allow-origin" => "https://example.com/",
            "access-control-allow-methods"=> "PUT",
            "access-control-allow-headers" => "content-type",
            "access-control-max-age"=> "86400",
        );

        // Check for appropriate CORS headers in response to the main request.
        let test_conn = put(task.report_upload_uri().unwrap().path())
            .with_request_header(KnownHeaderName::Origin, "https://example.com/")
            .with_request_header(KnownHeaderName::ContentType, Report::MEDIA_TYPE)
            .with_request_body(
                Report::new(
                    ReportMetadata::new(
                        random(),
                        clock
                            .now()
                            .to_batch_interval_start(task.time_precision())
                            .unwrap(),
                    ),
                    report.public_share().to_vec(),
                    report.encrypted_input_shares().to_vec(),
                )
                .get_encoded(),
            )
            .run_async(&handler)
            .await;
        assert!(test_conn.status().unwrap().is_success());
        assert_headers!(
            &test_conn,
            "access-control-allow-origin" => "https://example.com/"
        );
    }

    // Helper should not expose /upload endpoint
    #[tokio::test]
    async fn upload_handler_helper() {
        install_test_trace_subscriber();

        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = ephemeral_datastore.datastore(clock.clone()).await;

        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Helper,
        )
        .build();
        datastore.put_task(&task).await.unwrap();
        let report = create_report(&task, clock.now());

        let handler = aggregator_handler(
            Arc::new(datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        let mut test_conn = put(task.report_upload_uri().unwrap().path())
            .with_request_header(KnownHeaderName::ContentType, Report::MEDIA_TYPE)
            .with_request_body(report.get_encoded())
            .run_async(&handler)
            .await;

        assert!(!test_conn.status().unwrap().is_success());
        let problem_details = take_problem_details(&mut test_conn).await;
        assert_eq!(
            problem_details,
            json!({
                "status": 400,
                "type": "urn:ietf:params:ppm:dap:error:unrecognizedTask",
                "title": "An endpoint received a message with an unknown task ID.",
                "taskid": format!("{}", task.id()),
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
            test_conn.status().unwrap() as u16 as u64
        );
    }

    #[tokio::test]
    async fn aggregate_leader() {
        install_test_trace_subscriber();

        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Leader,
        )
        .build();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = ephemeral_datastore.datastore(clock.clone()).await;

        datastore.put_task(&task).await.unwrap();

        let request = AggregationJobInitializeReq::new(
            Vec::new(),
            PartialBatchSelector::new_fixed_size(random()),
            Vec::new(),
        );

        let handler = aggregator_handler(
            Arc::new(datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();
        let aggregation_job_id: AggregationJobId = random();

        let mut test_conn =
            put_aggregation_job(&task, &aggregation_job_id, &request, &handler).await;
        assert!(!test_conn.status().unwrap().is_success());

        let problem_details = take_problem_details(&mut test_conn).await;
        assert_eq!(
            problem_details,
            json!({
                "status": 400,
                "type": "urn:ietf:params:ppm:dap:error:unrecognizedTask",
                "title": "An endpoint received a message with an unknown task ID.",
                "taskid": format!("{}", task.id()),
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
            test_conn.status().unwrap() as u16 as u64
        );

        // Check that CORS headers don't bleed over to other routes.
        assert!(test_conn
            .response_headers()
            .get("access-control-allow-origin")
            .is_none());
        assert!(test_conn
            .response_headers()
            .get("access-control-allow-methods")
            .is_none());
        assert!(test_conn
            .response_headers()
            .get("access-control-max-age")
            .is_none());

        let test_conn = TestConn::build(
            trillium::Method::Options,
            task.aggregation_job_uri(&aggregation_job_id)
                .unwrap()
                .path(),
            (),
        )
        .with_request_header(KnownHeaderName::Origin, "https://example.com/")
        .with_request_header(KnownHeaderName::AccessControlRequestMethod, "PUT")
        .run_async(&handler)
        .await;
        assert!(test_conn
            .response_headers()
            .get(KnownHeaderName::AccessControlAllowMethods)
            .is_none());
    }

    #[tokio::test]
    async fn aggregate_wrong_agg_auth_token() {
        install_test_trace_subscriber();

        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Helper,
        )
        .build();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = ephemeral_datastore.datastore(clock.clone()).await;

        datastore.put_task(&task).await.unwrap();

        let request = AggregationJobInitializeReq::new(
            Vec::new(),
            PartialBatchSelector::new_fixed_size(random()),
            Vec::new(),
        );

        let handler = aggregator_handler(
            Arc::new(datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();
        let aggregation_job_id: AggregationJobId = random();

        let wrong_token_value = random();

        // Send the right token, but the wrong format: we find a DapAuth token in the task's
        // aggregator tokens and convert it to an equivalent Bearer token, which should be rejected.
        let wrong_token_format = task
            .aggregator_auth_tokens()
            .iter()
            .find(|token| matches!(token, AuthenticationToken::DapAuth(_)))
            .map(|token| AuthenticationToken::Bearer(token.as_ref().to_vec()))
            .unwrap();

        for auth_token in [Some(wrong_token_value), Some(wrong_token_format), None] {
            let mut test_conn = put(task
                .aggregation_job_uri(&aggregation_job_id)
                .unwrap()
                .path())
            .with_request_header(
                KnownHeaderName::ContentType,
                AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_request_body(request.get_encoded());

            if let Some(auth_token) = auth_token {
                let (auth_header, auth_value) = auth_token.request_authentication();
                test_conn = test_conn.with_request_header(auth_header, auth_value);
            }

            let mut test_conn = test_conn.run_async(&handler).await;

            let want_status = 400;
            assert_eq!(
                take_problem_details(&mut test_conn).await,
                json!({
                    "status": want_status,
                    "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
                    "title": "The request's authorization is not valid.",
                    "taskid": format!("{}", task.id()),
                })
            );
            assert_eq!(want_status, test_conn.status().unwrap() as u16);
        }
    }

    #[tokio::test]
    // Silence the unit_arg lint so that we can work with dummy_vdaf::Vdaf::InputShare values (whose
    // type is ()).
    #[allow(clippy::unit_arg)]
    async fn aggregate_init() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Helper,
        )
        .build();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        let vdaf = dummy_vdaf::Vdaf::new();
        let verify_key: VerifyKey<0> = task.primary_vdaf_verify_key().unwrap();
        let hpke_key = task.current_hpke_key();

        // report_share_0 is a "happy path" report.
        let report_metadata_0 = ReportMetadata::new(
            random(),
            clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &dummy_vdaf::AggregationParam(0),
            report_metadata_0.id(),
            &(),
        );
        let report_share_0 = generate_helper_report_share::<dummy_vdaf::Vdaf>(
            *task.id(),
            report_metadata_0,
            hpke_key.config(),
            &transcript.public_share,
            Vec::new(),
            &transcript.input_shares[1],
        );

        // report_share_1 fails decryption.
        let report_metadata_1 = ReportMetadata::new(
            random(),
            clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &dummy_vdaf::AggregationParam(0),
            report_metadata_1.id(),
            &(),
        );
        let report_share_1 = generate_helper_report_share::<dummy_vdaf::Vdaf>(
            *task.id(),
            report_metadata_1.clone(),
            hpke_key.config(),
            &transcript.public_share,
            Vec::new(),
            &transcript.input_shares[1],
        );
        let encrypted_input_share = report_share_1.encrypted_input_share();
        let mut corrupted_payload = encrypted_input_share.payload().to_vec();
        corrupted_payload[0] ^= 0xFF;
        let corrupted_input_share = HpkeCiphertext::new(
            *encrypted_input_share.config_id(),
            encrypted_input_share.encapsulated_key().to_vec(),
            corrupted_payload,
        );
        let encoded_public_share = transcript.public_share.get_encoded();
        let report_share_1 = ReportShare::new(
            report_metadata_1,
            encoded_public_share.clone(),
            corrupted_input_share,
        );

        // report_share_2 fails decoding due to an issue with the input share.
        let report_metadata_2 = ReportMetadata::new(
            random(),
            clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &dummy_vdaf::AggregationParam(0),
            report_metadata_2.id(),
            &(),
        );
        let mut input_share_bytes = transcript.input_shares[1].get_encoded();
        input_share_bytes.push(0); // can no longer be decoded.
        let report_share_2 = generate_helper_report_share_for_plaintext(
            report_metadata_2.clone(),
            hpke_key.config(),
            encoded_public_share.clone(),
            &input_share_bytes,
            &InputShareAad::new(*task.id(), report_metadata_2, encoded_public_share).get_encoded(),
        );

        // report_share_3 has an unknown HPKE config ID.
        let report_metadata_3 = ReportMetadata::new(
            random(),
            clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &dummy_vdaf::AggregationParam(0),
            report_metadata_3.id(),
            &(),
        );
        let wrong_hpke_config = loop {
            let hpke_config = generate_test_hpke_config_and_private_key().config().clone();
            if task.hpke_keys().contains_key(hpke_config.id()) {
                continue;
            }
            break hpke_config;
        };
        let report_share_3 = generate_helper_report_share::<dummy_vdaf::Vdaf>(
            *task.id(),
            report_metadata_3,
            &wrong_hpke_config,
            &transcript.public_share,
            Vec::new(),
            &transcript.input_shares[1],
        );

        // report_share_4 has already been aggregated in another aggregation job, with the same
        // aggregation parameter.
        let report_metadata_4 = ReportMetadata::new(
            random(),
            clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &dummy_vdaf::AggregationParam(0),
            report_metadata_4.id(),
            &(),
        );
        let report_share_4 = generate_helper_report_share::<dummy_vdaf::Vdaf>(
            *task.id(),
            report_metadata_4,
            hpke_key.config(),
            &transcript.public_share,
            Vec::new(),
            &transcript.input_shares[1],
        );

        // report_share_5 fails decoding due to an issue with the public share.
        let public_share_5 = Vec::from([0]);
        let report_metadata_5 = ReportMetadata::new(
            random(),
            clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &dummy_vdaf::AggregationParam(0),
            report_metadata_5.id(),
            &(),
        );
        let report_share_5 = generate_helper_report_share_for_plaintext(
            report_metadata_5.clone(),
            hpke_key.config(),
            public_share_5.clone(),
            &transcript.input_shares[1].get_encoded(),
            &InputShareAad::new(*task.id(), report_metadata_5, public_share_5).get_encoded(),
        );

        // report_share_6 fails due to having repeated extensions.
        let report_metadata_6 = ReportMetadata::new(
            random(),
            clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &dummy_vdaf::AggregationParam(0),
            report_metadata_6.id(),
            &(),
        );
        let report_share_6 = generate_helper_report_share::<dummy_vdaf::Vdaf>(
            *task.id(),
            report_metadata_6,
            hpke_key.config(),
            &transcript.public_share,
            Vec::from([
                Extension::new(ExtensionType::Tbd, Vec::new()),
                Extension::new(ExtensionType::Tbd, Vec::new()),
            ]),
            &transcript.input_shares[0],
        );

        // report_share_7 has already been aggregated in another aggregation job, with a different
        // aggregation parameter.
        let report_metadata_7 = ReportMetadata::new(
            random(),
            clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &dummy_vdaf::AggregationParam(1),
            report_metadata_7.id(),
            &(),
        );
        let report_share_7 = generate_helper_report_share::<dummy_vdaf::Vdaf>(
            *task.id(),
            report_metadata_7,
            hpke_key.config(),
            &transcript.public_share,
            Vec::new(),
            &transcript.input_shares[1],
        );

        let (conflicting_aggregation_job, non_conflicting_aggregation_job) = datastore
            .run_tx(|tx| {
                let (task, report_share_4, report_share_7) =
                    (task.clone(), report_share_4.clone(), report_share_7.clone());
                Box::pin(async move {
                    tx.put_task(&task).await?;

                    // report_share_4 and report_share_7 are already in the datastore as they were
                    // referenced by existing aggregation jobs.
                    tx.put_report_share(task.id(), &report_share_4).await?;
                    tx.put_report_share(task.id(), &report_share_7).await?;

                    // Put in an aggregation job and report aggregation for report_share_4. It uses
                    // the same aggregation parameter as the aggregation job this test will later
                    // add and so should cause report_share_4 to fail to prepare.
                    let conflicting_aggregation_job = AggregationJob::new(
                        *task.id(),
                        random(),
                        dummy_vdaf::AggregationParam(0),
                        random(),
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    );
                    tx.put_aggregation_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &conflicting_aggregation_job,
                    )
                    .await
                    .unwrap();
                    tx.put_report_aggregation::<0, dummy_vdaf::Vdaf>(&ReportAggregation::new(
                        *task.id(),
                        *conflicting_aggregation_job.id(),
                        *report_share_4.metadata().id(),
                        *report_share_4.metadata().time(),
                        0,
                        None,
                        ReportAggregationState::Start,
                    ))
                    .await
                    .unwrap();

                    // Put in an aggregation job and report aggregation for report_share_7, using a
                    // a different aggregation parameter. As the aggregation parameter differs,
                    // report_share_7 should prepare successfully in the aggregation job we'll PUT
                    // later.
                    let non_conflicting_aggregation_job = AggregationJob::new(
                        *task.id(),
                        random(),
                        dummy_vdaf::AggregationParam(1),
                        random(),
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    );
                    tx.put_aggregation_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &non_conflicting_aggregation_job,
                    )
                    .await
                    .unwrap();
                    tx.put_report_aggregation::<0, dummy_vdaf::Vdaf>(&ReportAggregation::new(
                        *task.id(),
                        *non_conflicting_aggregation_job.id(),
                        *report_share_7.metadata().id(),
                        *report_share_7.metadata().time(),
                        0,
                        None,
                        ReportAggregationState::Start,
                    ))
                    .await
                    .unwrap();

                    Ok((conflicting_aggregation_job, non_conflicting_aggregation_job))
                })
            })
            .await
            .unwrap();

        let batch_id = random();
        let request = AggregationJobInitializeReq::new(
            dummy_vdaf::AggregationParam(0).get_encoded(),
            PartialBatchSelector::new_fixed_size(batch_id),
            Vec::from([
                report_share_0.clone(),
                report_share_1.clone(),
                report_share_2.clone(),
                report_share_3.clone(),
                report_share_4.clone(),
                report_share_5.clone(),
                report_share_6.clone(),
                report_share_7.clone(),
            ]),
        );

        // Create aggregator handler, send request, and parse response. Do this twice to prove that
        // the request is idempotent.
        let handler = aggregator_handler(
            Arc::clone(&datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();
        let aggregation_job_id: AggregationJobId = random();

        for _ in 0..2 {
            let mut test_conn =
                put_aggregation_job(&task, &aggregation_job_id, &request, &handler).await;
            assert_eq!(test_conn.status(), Some(Status::Ok));
            assert_headers!(
                &test_conn,
                "content-type" => (AggregationJobResp::MEDIA_TYPE)
            );
            let body_bytes = take_response_body(&mut test_conn).await;
            let aggregate_resp = AggregationJobResp::get_decoded(&body_bytes).unwrap();

            // Validate response.
            assert_eq!(aggregate_resp.prepare_steps().len(), 8);

            let prepare_step_0 = aggregate_resp.prepare_steps().get(0).unwrap();
            assert_eq!(prepare_step_0.report_id(), report_share_0.metadata().id());
            assert_matches!(prepare_step_0.result(), &PrepareStepResult::Continued(..));

            let prepare_step_1 = aggregate_resp.prepare_steps().get(1).unwrap();
            assert_eq!(prepare_step_1.report_id(), report_share_1.metadata().id());
            assert_matches!(
                prepare_step_1.result(),
                &PrepareStepResult::Failed(ReportShareError::HpkeDecryptError)
            );

            let prepare_step_2 = aggregate_resp.prepare_steps().get(2).unwrap();
            assert_eq!(prepare_step_2.report_id(), report_share_2.metadata().id());
            assert_matches!(
                prepare_step_2.result(),
                &PrepareStepResult::Failed(ReportShareError::UnrecognizedMessage)
            );

            let prepare_step_3 = aggregate_resp.prepare_steps().get(3).unwrap();
            assert_eq!(prepare_step_3.report_id(), report_share_3.metadata().id());
            assert_matches!(
                prepare_step_3.result(),
                &PrepareStepResult::Failed(ReportShareError::HpkeUnknownConfigId)
            );

            let prepare_step_4 = aggregate_resp.prepare_steps().get(4).unwrap();
            assert_eq!(prepare_step_4.report_id(), report_share_4.metadata().id());
            assert_eq!(
                prepare_step_4.result(),
                &PrepareStepResult::Failed(ReportShareError::ReportReplayed)
            );

            let prepare_step_5 = aggregate_resp.prepare_steps().get(5).unwrap();
            assert_eq!(prepare_step_5.report_id(), report_share_5.metadata().id());
            assert_eq!(
                prepare_step_5.result(),
                &PrepareStepResult::Failed(ReportShareError::UnrecognizedMessage),
            );

            let prepare_step_6 = aggregate_resp.prepare_steps().get(6).unwrap();
            assert_eq!(prepare_step_6.report_id(), report_share_6.metadata().id());
            assert_eq!(
                prepare_step_6.result(),
                &PrepareStepResult::Failed(ReportShareError::UnrecognizedMessage),
            );

            let prepare_step_7 = aggregate_resp.prepare_steps().get(7).unwrap();
            assert_eq!(prepare_step_7.report_id(), report_share_7.metadata().id());
            assert_matches!(prepare_step_7.result(), &PrepareStepResult::Continued(..));

            // Check aggregation job in datastore.
            let aggregation_jobs = datastore
                .run_tx(|tx| {
                    let task = task.clone();
                    Box::pin(async move {
                        tx.get_aggregation_jobs_for_task::<0, FixedSize, dummy_vdaf::Vdaf>(
                            task.id(),
                        )
                        .await
                    })
                })
                .await
                .unwrap();

            assert_eq!(aggregation_jobs.len(), 3);

            let mut saw_conflicting_aggregation_job = false;
            let mut saw_non_conflicting_aggregation_job = false;
            let mut saw_new_aggregation_job = false;
            for aggregation_job in aggregation_jobs {
                if aggregation_job.eq(&conflicting_aggregation_job) {
                    saw_conflicting_aggregation_job = true;
                } else if aggregation_job.eq(&non_conflicting_aggregation_job) {
                    saw_non_conflicting_aggregation_job = true;
                } else if aggregation_job.task_id().eq(task.id())
                    && aggregation_job.id().eq(&aggregation_job_id)
                    && aggregation_job.partial_batch_identifier().eq(&batch_id)
                    && aggregation_job.state().eq(&AggregationJobState::InProgress)
                {
                    saw_new_aggregation_job = true;
                }
            }

            assert!(saw_conflicting_aggregation_job);
            assert!(saw_non_conflicting_aggregation_job);
            assert!(saw_new_aggregation_job);
        }
    }

    #[allow(clippy::unit_arg)]
    #[tokio::test]
    async fn aggregate_init_change_report_timestamp() {
        let test_case = setup_aggregate_init_test().await;

        let other_aggregation_parameter = dummy_vdaf::AggregationParam(1);
        assert_ne!(test_case.aggregation_param, other_aggregation_parameter);

        // This report has the same ID as the previous one, but a different timestamp.
        let mutated_timestamp_report_metadata = ReportMetadata::new(
            *test_case.report_shares[0].metadata().id(),
            test_case
                .clock
                .now()
                .add(test_case.task.time_precision())
                .unwrap(),
        );
        let mutated_timestamp_report_share = test_case
            .report_share_generator
            .next_with_metadata(mutated_timestamp_report_metadata)
            .0;

        // Send another aggregate job re-using the same report ID but with a different timestamp. It
        // should be flagged as a replay.
        let request = AggregationJobInitializeReq::new(
            other_aggregation_parameter.get_encoded(),
            PartialBatchSelector::new_fixed_size(random()),
            Vec::from([mutated_timestamp_report_share.clone()]),
        );

        let mut test_conn =
            put_aggregation_job(&test_case.task, &random(), &request, &test_case.handler).await;
        assert_eq!(test_conn.status(), Some(Status::Ok));
        let body_bytes = take_response_body(&mut test_conn).await;
        let aggregate_resp = AggregationJobResp::get_decoded(&body_bytes).unwrap();

        assert_eq!(aggregate_resp.prepare_steps().len(), 1);

        let prepare_step = aggregate_resp.prepare_steps().get(0).unwrap();
        assert_eq!(
            prepare_step.report_id(),
            mutated_timestamp_report_share.metadata().id()
        );
        assert_matches!(
            prepare_step.result(),
            &PrepareStepResult::Failed(ReportShareError::ReportReplayed)
        );

        // The attempt to mutate the report share timestamp should not cause any change in the
        // datastore.
        let client_reports = test_case
            .datastore
            .run_tx(|tx| {
                let task_id = *test_case.task.id();
                Box::pin(async move {
                    let reports = tx.get_report_metadatas_for_task(&task_id).await.unwrap();

                    Ok(reports)
                })
            })
            .await
            .unwrap();
        assert_eq!(client_reports.len(), 2);
        assert_eq!(&client_reports[0], test_case.report_shares[0].metadata());
        assert_eq!(&client_reports[1], test_case.report_shares[1].metadata());
    }

    #[tokio::test]
    async fn aggregate_init_prep_init_failed() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::FakeFailsPrepInit,
            Role::Helper,
        )
        .build();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = ephemeral_datastore.datastore(clock.clone()).await;
        let hpke_key = task.current_hpke_key();

        datastore.put_task(&task).await.unwrap();

        let report_share = generate_helper_report_share::<dummy_vdaf::Vdaf>(
            *task.id(),
            ReportMetadata::new(
                random(),
                clock
                    .now()
                    .to_batch_interval_start(task.time_precision())
                    .unwrap(),
            ),
            hpke_key.config(),
            &(),
            Vec::new(),
            &dummy_vdaf::InputShare::default(),
        );
        let request = AggregationJobInitializeReq::new(
            dummy_vdaf::AggregationParam(0).get_encoded(),
            PartialBatchSelector::new_fixed_size(random()),
            Vec::from([report_share.clone()]),
        );

        // Create aggregator handler, send request, and parse response.
        let handler = aggregator_handler(
            Arc::new(datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();
        let aggregation_job_id: AggregationJobId = random();

        let mut test_conn =
            put_aggregation_job(&task, &aggregation_job_id, &request, &handler).await;
        assert_eq!(test_conn.status(), Some(Status::Ok));
        assert_headers!(
            &test_conn,
            "content-type" => (AggregationJobResp::MEDIA_TYPE)
        );
        let body_bytes = take_response_body(&mut test_conn).await;
        let aggregate_resp = AggregationJobResp::get_decoded(&body_bytes).unwrap();

        // Validate response.
        assert_eq!(aggregate_resp.prepare_steps().len(), 1);

        let prepare_step = aggregate_resp.prepare_steps().get(0).unwrap();
        assert_eq!(prepare_step.report_id(), report_share.metadata().id());
        assert_matches!(
            prepare_step.result(),
            &PrepareStepResult::Failed(ReportShareError::VdafPrepError)
        );
    }

    #[tokio::test]
    async fn aggregate_init_prep_step_failed() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::FakeFailsPrepInit,
            Role::Helper,
        )
        .build();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = ephemeral_datastore.datastore(clock.clone()).await;
        let hpke_key = task.current_hpke_key();

        datastore.put_task(&task).await.unwrap();

        let report_share = generate_helper_report_share::<dummy_vdaf::Vdaf>(
            *task.id(),
            ReportMetadata::new(
                random(),
                clock
                    .now()
                    .to_batch_interval_start(task.time_precision())
                    .unwrap(),
            ),
            hpke_key.config(),
            &(),
            Vec::new(),
            &dummy_vdaf::InputShare::default(),
        );
        let request = AggregationJobInitializeReq::new(
            dummy_vdaf::AggregationParam(0).get_encoded(),
            PartialBatchSelector::new_fixed_size(random()),
            Vec::from([report_share.clone()]),
        );

        // Create aggregator filter, send request, and parse response.
        let handler = aggregator_handler(
            Arc::new(datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();
        let aggregation_job_id: AggregationJobId = random();

        let mut test_conn =
            put_aggregation_job(&task, &aggregation_job_id, &request, &handler).await;
        assert_eq!(test_conn.status(), Some(Status::Ok));
        assert_headers!(
            &test_conn,
            "content-type" => (AggregationJobResp::MEDIA_TYPE)
        );
        let body_bytes = take_response_body(&mut test_conn).await;
        let aggregate_resp = AggregationJobResp::get_decoded(&body_bytes).unwrap();

        // Validate response.
        assert_eq!(aggregate_resp.prepare_steps().len(), 1);

        let prepare_step = aggregate_resp.prepare_steps().get(0).unwrap();
        assert_eq!(prepare_step.report_id(), report_share.metadata().id());
        assert_matches!(
            prepare_step.result(),
            &PrepareStepResult::Failed(ReportShareError::VdafPrepError)
        );
    }

    #[tokio::test]
    async fn aggregate_init_duplicated_report_id() {
        install_test_trace_subscriber();

        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::FakeFailsPrepInit,
            Role::Helper,
        )
        .build();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = ephemeral_datastore.datastore(clock.clone()).await;

        datastore.put_task(&task).await.unwrap();

        let report_share = ReportShare::new(
            ReportMetadata::new(
                ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
                Time::from_seconds_since_epoch(54321),
            ),
            Vec::from("PUBLIC"),
            HpkeCiphertext::new(
                // bogus, but we never get far enough to notice
                HpkeConfigId::from(42),
                Vec::from("012345"),
                Vec::from("543210"),
            ),
        );

        let request = AggregationJobInitializeReq::new(
            dummy_vdaf::AggregationParam(0).get_encoded(),
            PartialBatchSelector::new_fixed_size(random()),
            Vec::from([report_share.clone(), report_share]),
        );

        let handler = aggregator_handler(
            Arc::new(datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();
        let aggregation_job_id: AggregationJobId = random();

        let mut test_conn =
            put_aggregation_job(&task, &aggregation_job_id, &request, &handler).await;

        let want_status = 400;
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": want_status,
                "type": "urn:ietf:params:ppm:dap:error:unrecognizedMessage",
                "title": "The message type for a response was incorrect or the payload was malformed.",
                "taskid": format!("{}", task.id()),
            })
        );
        assert_eq!(want_status, test_conn.status().unwrap());
    }

    #[tokio::test]
    async fn aggregate_continue() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        let aggregation_job_id = random();
        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Helper,
        )
        .build();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        let vdaf = Arc::new(Prio3::new_count(2).unwrap());
        let verify_key: VerifyKey<PRIO3_VERIFY_KEY_LENGTH> =
            task.primary_vdaf_verify_key().unwrap();
        let hpke_key = task.current_hpke_key();

        // report_share_0 is a "happy path" report.
        let report_metadata_0 = ReportMetadata::new(
            random(),
            clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript_0 = run_vdaf(
            vdaf.as_ref(),
            verify_key.as_bytes(),
            &(),
            report_metadata_0.id(),
            &0,
        );
        let (prep_state_0, _) = transcript_0.helper_prep_state(0);
        let prep_msg_0 = transcript_0.prepare_messages[0].clone();
        let report_share_0 = generate_helper_report_share::<Prio3Count>(
            *task.id(),
            report_metadata_0.clone(),
            hpke_key.config(),
            &transcript_0.public_share,
            Vec::new(),
            &transcript_0.input_shares[1],
        );

        // report_share_1 is omitted by the leader's request.
        let report_metadata_1 = ReportMetadata::new(
            random(),
            clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript_1 = run_vdaf(
            vdaf.as_ref(),
            verify_key.as_bytes(),
            &(),
            report_metadata_1.id(),
            &0,
        );

        let (prep_state_1, _) = transcript_1.helper_prep_state(0);
        let report_share_1 = generate_helper_report_share::<Prio3Count>(
            *task.id(),
            report_metadata_1.clone(),
            hpke_key.config(),
            &transcript_1.public_share,
            Vec::new(),
            &transcript_1.input_shares[1],
        );

        let batch_id = random();
        datastore
            .run_tx(|tx| {
                let task = task.clone();
                let (report_share_0, report_share_1) =
                    (report_share_0.clone(), report_share_1.clone());
                let (prep_state_0, prep_state_1) = (prep_state_0.clone(), prep_state_1.clone());
                let (report_metadata_0, report_metadata_1) =
                    (report_metadata_0.clone(), report_metadata_1.clone());

                Box::pin(async move {
                    tx.put_task(&task).await?;

                    tx.put_report_share(task.id(), &report_share_0).await?;
                    tx.put_report_share(task.id(), &report_share_1).await?;

                    tx.put_aggregation_job(&AggregationJob::<
                        PRIO3_VERIFY_KEY_LENGTH,
                        FixedSize,
                        Prio3Count,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        (),
                        batch_id,
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    ))
                    .await?;

                    tx.put_report_aggregation::<PRIO3_VERIFY_KEY_LENGTH, Prio3Count>(
                        &ReportAggregation::new(
                            *task.id(),
                            aggregation_job_id,
                            *report_metadata_0.id(),
                            *report_metadata_0.time(),
                            0,
                            None,
                            ReportAggregationState::Waiting(prep_state_0, None),
                        ),
                    )
                    .await?;
                    tx.put_report_aggregation::<PRIO3_VERIFY_KEY_LENGTH, Prio3Count>(
                        &ReportAggregation::new(
                            *task.id(),
                            aggregation_job_id,
                            *report_metadata_1.id(),
                            *report_metadata_1.time(),
                            1,
                            None,
                            ReportAggregationState::Waiting(prep_state_1, None),
                        ),
                    )
                    .await?;

                    tx.put_aggregate_share_job::<PRIO3_VERIFY_KEY_LENGTH, FixedSize, Prio3Count>(
                        &AggregateShareJob::new(
                            *task.id(),
                            batch_id,
                            (),
                            AggregateShare::from(OutputShare::from(Vec::from([Field64::from(7)]))),
                            0,
                            ReportIdChecksum::default(),
                        ),
                    )
                    .await
                })
            })
            .await
            .unwrap();

        let request = AggregationJobContinueReq::new(
            AggregationJobRound::from(1),
            Vec::from([PrepareStep::new(
                *report_metadata_0.id(),
                PrepareStepResult::Continued(prep_msg_0.get_encoded()),
            )]),
        );

        // Create aggregator handler, send request, and parse response.
        let handler = aggregator_handler(
            Arc::clone(&datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        let aggregate_resp =
            post_aggregation_job_and_decode(&task, &aggregation_job_id, &request, &handler).await;

        // Validate response.
        assert_eq!(
            aggregate_resp,
            AggregationJobResp::new(Vec::from([PrepareStep::new(
                *report_metadata_0.id(),
                PrepareStepResult::Finished
            ),]))
        );

        // Validate datastore.
        let (aggregation_job, report_aggregations) = datastore
            .run_tx(|tx| {
                let (vdaf, task) = (Arc::clone(&vdaf), task.clone());
                Box::pin(async move {
                    let aggregation_job = tx
                        .get_aggregation_job::<PRIO3_VERIFY_KEY_LENGTH, FixedSize, Prio3Count>(
                            task.id(),
                            &aggregation_job_id,
                        )
                        .await
                        .unwrap()
                        .unwrap();
                    let report_aggregations = tx
                        .get_report_aggregations_for_aggregation_job(
                            vdaf.as_ref(),
                            &Role::Helper,
                            task.id(),
                            &aggregation_job_id,
                        )
                        .await
                        .unwrap();
                    Ok((aggregation_job, report_aggregations))
                })
            })
            .await
            .unwrap();

        assert_eq!(
            aggregation_job,
            AggregationJob::new(
                *task.id(),
                aggregation_job_id,
                (),
                batch_id,
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Finished,
                AggregationJobRound::from(1),
            )
            .with_last_continue_request_hash(aggregation_job.last_continue_request_hash().unwrap())
        );
        assert_eq!(
            report_aggregations,
            Vec::from([
                ReportAggregation::new(
                    *task.id(),
                    aggregation_job_id,
                    *report_metadata_0.id(),
                    *report_metadata_0.time(),
                    0,
                    Some(PrepareStep::new(
                        *report_metadata_0.id(),
                        PrepareStepResult::Finished
                    )),
                    ReportAggregationState::Finished,
                ),
                ReportAggregation::new(
                    *task.id(),
                    aggregation_job_id,
                    *report_metadata_1.id(),
                    *report_metadata_1.time(),
                    1,
                    None,
                    ReportAggregationState::Failed(ReportShareError::ReportDropped),
                ),
            ])
        );
    }

    #[tokio::test]
    async fn aggregate_continue_accumulate_batch_aggregation() {
        install_test_trace_subscriber();

        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Helper,
        )
        .build();
        let aggregation_job_id_0 = random();
        let aggregation_job_id_1 = random();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(MockClock::default()).await);
        let first_batch_interval_clock = MockClock::default();
        let second_batch_interval_clock = MockClock::new(
            first_batch_interval_clock
                .now()
                .add(task.time_precision())
                .unwrap(),
        );

        let vdaf = Prio3::new_count(2).unwrap();
        let verify_key: VerifyKey<PRIO3_VERIFY_KEY_LENGTH> =
            task.primary_vdaf_verify_key().unwrap();
        let hpke_key = task.current_hpke_key();

        // report_share_0 is a "happy path" report.
        let report_metadata_0 = ReportMetadata::new(
            random(),
            first_batch_interval_clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript_0 = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &(),
            report_metadata_0.id(),
            &0,
        );
        let (prep_state_0, _) = transcript_0.helper_prep_state(0);
        let out_share_0 = transcript_0.output_share(Role::Helper);
        let prep_msg_0 = transcript_0.prepare_messages[0].clone();
        let report_share_0 = generate_helper_report_share::<Prio3Count>(
            *task.id(),
            report_metadata_0.clone(),
            hpke_key.config(),
            &transcript_0.public_share,
            Vec::new(),
            &transcript_0.input_shares[1],
        );

        // report_share_1 is another "happy path" report to exercise in-memory accumulation of
        // output shares
        let report_metadata_1 = ReportMetadata::new(
            random(),
            first_batch_interval_clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript_1 = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &(),
            report_metadata_1.id(),
            &0,
        );
        let (prep_state_1, _) = transcript_1.helper_prep_state(0);
        let out_share_1 = transcript_1.output_share(Role::Helper);
        let prep_msg_1 = transcript_1.prepare_messages[0].clone();
        let report_share_1 = generate_helper_report_share::<Prio3Count>(
            *task.id(),
            report_metadata_1.clone(),
            hpke_key.config(),
            &transcript_1.public_share,
            Vec::new(),
            &transcript_1.input_shares[1],
        );

        // report_share_2 aggregates successfully, but into a distinct batch aggregation which has
        // already been collected.
        let report_metadata_2 = ReportMetadata::new(
            random(),
            second_batch_interval_clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript_2 = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &(),
            report_metadata_2.id(),
            &0,
        );
        let (prep_state_2, _) = transcript_2.helper_prep_state(0);
        let out_share_2 = transcript_2.output_share(Role::Helper);
        let prep_msg_2 = transcript_2.prepare_messages[0].clone();
        let report_share_2 = generate_helper_report_share::<Prio3Count>(
            *task.id(),
            report_metadata_2.clone(),
            hpke_key.config(),
            &transcript_2.public_share,
            Vec::new(),
            &transcript_2.input_shares[1],
        );

        let first_batch_id = random();
        let second_batch_id = random();
        let second_batch_want_batch_aggregations =
            empty_batch_aggregations::<PRIO3_VERIFY_KEY_LENGTH, FixedSize, Prio3Count>(
                &task,
                BATCH_AGGREGATION_SHARD_COUNT,
                &second_batch_id,
                &(),
                &[],
            );

        datastore
            .run_tx(|tx| {
                let task = task.clone();
                let (report_share_0, report_share_1, report_share_2) = (
                    report_share_0.clone(),
                    report_share_1.clone(),
                    report_share_2.clone(),
                );
                let (prep_state_0, prep_state_1, prep_state_2) = (
                    prep_state_0.clone(),
                    prep_state_1.clone(),
                    prep_state_2.clone(),
                );
                let (report_metadata_0, report_metadata_1, report_metadata_2) = (
                    report_metadata_0.clone(),
                    report_metadata_1.clone(),
                    report_metadata_2.clone(),
                );
                let second_batch_want_batch_aggregations =
                    second_batch_want_batch_aggregations.clone();

                Box::pin(async move {
                    tx.put_task(&task).await?;

                    tx.put_report_share(task.id(), &report_share_0).await?;
                    tx.put_report_share(task.id(), &report_share_1).await?;
                    tx.put_report_share(task.id(), &report_share_2).await?;

                    tx.put_aggregation_job(&AggregationJob::<
                        PRIO3_VERIFY_KEY_LENGTH,
                        FixedSize,
                        Prio3Count,
                    >::new(
                        *task.id(),
                        aggregation_job_id_0,
                        (),
                        first_batch_id,
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    ))
                    .await?;

                    tx.put_report_aggregation(&ReportAggregation::<
                        PRIO3_VERIFY_KEY_LENGTH,
                        Prio3Count,
                    >::new(
                        *task.id(),
                        aggregation_job_id_0,
                        *report_metadata_0.id(),
                        *report_metadata_0.time(),
                        0,
                        None,
                        ReportAggregationState::Waiting(prep_state_0, None),
                    ))
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<
                        PRIO3_VERIFY_KEY_LENGTH,
                        Prio3Count,
                    >::new(
                        *task.id(),
                        aggregation_job_id_0,
                        *report_metadata_1.id(),
                        *report_metadata_1.time(),
                        1,
                        None,
                        ReportAggregationState::Waiting(prep_state_1, None),
                    ))
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<
                        PRIO3_VERIFY_KEY_LENGTH,
                        Prio3Count,
                    >::new(
                        *task.id(),
                        aggregation_job_id_0,
                        *report_metadata_2.id(),
                        *report_metadata_2.time(),
                        2,
                        None,
                        ReportAggregationState::Waiting(prep_state_2, None),
                    ))
                    .await?;

                    try_join_all(
                        second_batch_want_batch_aggregations
                            .iter()
                            .map(|ba| tx.put_batch_aggregation(ba)),
                    )
                    .await?;

                    Ok(())
                })
            })
            .await
            .unwrap();

        let request = AggregationJobContinueReq::new(
            AggregationJobRound::from(1),
            Vec::from([
                PrepareStep::new(
                    *report_metadata_0.id(),
                    PrepareStepResult::Continued(prep_msg_0.get_encoded()),
                ),
                PrepareStep::new(
                    *report_metadata_1.id(),
                    PrepareStepResult::Continued(prep_msg_1.get_encoded()),
                ),
                PrepareStep::new(
                    *report_metadata_2.id(),
                    PrepareStepResult::Continued(prep_msg_2.get_encoded()),
                ),
            ]),
        );

        // Create aggregator handler, send request, and parse response.
        let handler = aggregator_handler(
            Arc::clone(&datastore),
            first_batch_interval_clock.clone(),
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        let _ =
            post_aggregation_job_and_decode(&task, &aggregation_job_id_0, &request, &handler).await;

        // Map the batch aggregation ordinal value to 0, as it may vary due to sharding.
        let first_batch_got_batch_aggregations: Vec<_> = datastore
            .run_tx(|tx| {
                let (task, vdaf) = (task.clone(), vdaf.clone());
                Box::pin(async move {
                    FixedSize::get_batch_aggregations_for_collection_identifier::<
                        PRIO3_VERIFY_KEY_LENGTH,
                        Prio3Count,
                        _,
                    >(tx, &task, &vdaf, &first_batch_id, &())
                    .await
                })
            })
            .await
            .unwrap()
            .into_iter()
            .map(|agg| {
                BatchAggregation::<PRIO3_VERIFY_KEY_LENGTH, FixedSize, Prio3Count>::new(
                    *agg.task_id(),
                    *agg.batch_identifier(),
                    (),
                    0,
                    BatchAggregationState::Aggregating,
                    agg.aggregate_share().cloned(),
                    agg.report_count(),
                    *agg.checksum(),
                )
            })
            .collect();

        let aggregate_share = vdaf
            .aggregate(
                &(),
                [
                    out_share_0.clone(),
                    out_share_1.clone(),
                    out_share_2.clone(),
                ],
            )
            .unwrap();
        let checksum = ReportIdChecksum::for_report_id(report_metadata_0.id())
            .updated_with(report_metadata_1.id())
            .updated_with(report_metadata_2.id());

        assert_eq!(
            first_batch_got_batch_aggregations,
            Vec::from([BatchAggregation::new(
                *task.id(),
                first_batch_id,
                (),
                0,
                BatchAggregationState::Aggregating,
                Some(aggregate_share),
                3,
                checksum,
            ),])
        );

        let second_batch_got_batch_aggregations = datastore
            .run_tx(|tx| {
                let (task, vdaf) = (task.clone(), vdaf.clone());
                Box::pin(async move {
                    FixedSize::get_batch_aggregations_for_collection_identifier::<
                        PRIO3_VERIFY_KEY_LENGTH,
                        Prio3Count,
                        _,
                    >(tx, &task, &vdaf, &second_batch_id, &())
                    .await
                })
            })
            .await
            .unwrap();
        assert_eq!(
            second_batch_got_batch_aggregations,
            second_batch_want_batch_aggregations
        );

        // Aggregate some more reports, which should get accumulated into the
        // batch_aggregations rows created earlier.
        // report_share_3 gets aggreated into the first batch interval.
        let report_metadata_3 = ReportMetadata::new(
            random(),
            first_batch_interval_clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript_3 = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &(),
            report_metadata_3.id(),
            &0,
        );
        let (prep_state_3, _) = transcript_3.helper_prep_state(0);
        let out_share_3 = transcript_3.output_share(Role::Helper);
        let prep_msg_3 = transcript_3.prepare_messages[0].clone();
        let report_share_3 = generate_helper_report_share::<Prio3Count>(
            *task.id(),
            report_metadata_3.clone(),
            hpke_key.config(),
            &transcript_3.public_share,
            Vec::new(),
            &transcript_3.input_shares[1],
        );

        // report_share_4 gets aggregated into the second batch interval (which has already been
        // collected).
        let report_metadata_4 = ReportMetadata::new(
            random(),
            second_batch_interval_clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript_4 = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &(),
            report_metadata_4.id(),
            &0,
        );
        let (prep_state_4, _) = transcript_4.helper_prep_state(0);
        let out_share_4 = transcript_4.output_share(Role::Helper);
        let prep_msg_4 = transcript_4.prepare_messages[0].clone();
        let report_share_4 = generate_helper_report_share::<Prio3Count>(
            *task.id(),
            report_metadata_4.clone(),
            hpke_key.config(),
            &transcript_4.public_share,
            Vec::new(),
            &transcript_4.input_shares[1],
        );

        // report_share_5 also gets aggregated into the second batch interval (which has already
        // been collected).
        let report_metadata_5 = ReportMetadata::new(
            random(),
            second_batch_interval_clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let transcript_5 = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &(),
            report_metadata_5.id(),
            &0,
        );
        let (prep_state_5, _) = transcript_5.helper_prep_state(0);
        let out_share_5 = transcript_5.output_share(Role::Helper);
        let prep_msg_5 = transcript_5.prepare_messages[0].clone();
        let report_share_5 = generate_helper_report_share::<Prio3Count>(
            *task.id(),
            report_metadata_5.clone(),
            hpke_key.config(),
            &transcript_5.public_share,
            Vec::new(),
            &transcript_5.input_shares[1],
        );

        datastore
            .run_tx(|tx| {
                let task = task.clone();
                let (report_share_3, report_share_4, report_share_5) = (
                    report_share_3.clone(),
                    report_share_4.clone(),
                    report_share_5.clone(),
                );
                let (prep_state_3, prep_state_4, prep_state_5) = (
                    prep_state_3.clone(),
                    prep_state_4.clone(),
                    prep_state_5.clone(),
                );
                let (report_metadata_3, report_metadata_4, report_metadata_5) = (
                    report_metadata_3.clone(),
                    report_metadata_4.clone(),
                    report_metadata_5.clone(),
                );

                Box::pin(async move {
                    tx.put_report_share(task.id(), &report_share_3).await?;
                    tx.put_report_share(task.id(), &report_share_4).await?;
                    tx.put_report_share(task.id(), &report_share_5).await?;

                    tx.put_aggregation_job(&AggregationJob::<
                        PRIO3_VERIFY_KEY_LENGTH,
                        FixedSize,
                        Prio3Count,
                    >::new(
                        *task.id(),
                        aggregation_job_id_1,
                        (),
                        first_batch_id,
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    ))
                    .await?;

                    tx.put_report_aggregation(&ReportAggregation::<
                        PRIO3_VERIFY_KEY_LENGTH,
                        Prio3Count,
                    >::new(
                        *task.id(),
                        aggregation_job_id_1,
                        *report_metadata_3.id(),
                        *report_metadata_3.time(),
                        3,
                        None,
                        ReportAggregationState::Waiting(prep_state_3, None),
                    ))
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<
                        PRIO3_VERIFY_KEY_LENGTH,
                        Prio3Count,
                    >::new(
                        *task.id(),
                        aggregation_job_id_1,
                        *report_metadata_4.id(),
                        *report_metadata_4.time(),
                        4,
                        None,
                        ReportAggregationState::Waiting(prep_state_4, None),
                    ))
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<
                        PRIO3_VERIFY_KEY_LENGTH,
                        Prio3Count,
                    >::new(
                        *task.id(),
                        aggregation_job_id_1,
                        *report_metadata_5.id(),
                        *report_metadata_5.time(),
                        5,
                        None,
                        ReportAggregationState::Waiting(prep_state_5, None),
                    ))
                    .await?;

                    Ok(())
                })
            })
            .await
            .unwrap();

        let request = AggregationJobContinueReq::new(
            AggregationJobRound::from(1),
            Vec::from([
                PrepareStep::new(
                    *report_metadata_3.id(),
                    PrepareStepResult::Continued(prep_msg_3.get_encoded()),
                ),
                PrepareStep::new(
                    *report_metadata_4.id(),
                    PrepareStepResult::Continued(prep_msg_4.get_encoded()),
                ),
                PrepareStep::new(
                    *report_metadata_5.id(),
                    PrepareStepResult::Continued(prep_msg_5.get_encoded()),
                ),
            ]),
        );

        // Create aggregator handler, send request, and parse response.
        let handler = aggregator_handler(
            Arc::clone(&datastore),
            first_batch_interval_clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        let _ =
            post_aggregation_job_and_decode(&task, &aggregation_job_id_1, &request, &handler).await;

        // Map the batch aggregation ordinal value to 0, as it may vary due to sharding, and merge
        // batch aggregations over the same interval. (the task & aggregation parameter will always
        // be the same)
        let merged_first_batch_aggregation = datastore
            .run_tx(|tx| {
                let (task, vdaf) = (task.clone(), vdaf.clone());
                Box::pin(async move {
                    FixedSize::get_batch_aggregations_for_collection_identifier::<
                        PRIO3_VERIFY_KEY_LENGTH,
                        Prio3Count,
                        _,
                    >(tx, &task, &vdaf, &first_batch_id, &())
                    .await
                })
            })
            .await
            .unwrap()
            .into_iter()
            .map(|agg| {
                BatchAggregation::<PRIO3_VERIFY_KEY_LENGTH, FixedSize, Prio3Count>::new(
                    *agg.task_id(),
                    *agg.batch_identifier(),
                    (),
                    0,
                    BatchAggregationState::Aggregating,
                    agg.aggregate_share().cloned(),
                    agg.report_count(),
                    *agg.checksum(),
                )
            })
            .reduce(|left, right| left.merged_with(&right).unwrap())
            .unwrap();

        let first_aggregate_share = vdaf
            .aggregate(
                &(),
                [
                    out_share_0,
                    out_share_1,
                    out_share_2,
                    out_share_3,
                    out_share_4,
                    out_share_5,
                ]
                .into_iter()
                .cloned(),
            )
            .unwrap();
        let first_checksum = ReportIdChecksum::for_report_id(report_metadata_0.id())
            .updated_with(report_metadata_1.id())
            .updated_with(report_metadata_2.id())
            .updated_with(report_metadata_3.id())
            .updated_with(report_metadata_4.id())
            .updated_with(report_metadata_5.id());

        assert_eq!(
            merged_first_batch_aggregation,
            BatchAggregation::new(
                *task.id(),
                first_batch_id,
                (),
                0,
                BatchAggregationState::Aggregating,
                Some(first_aggregate_share),
                6,
                first_checksum,
            ),
        );

        let second_batch_got_batch_aggregations = datastore
            .run_tx(|tx| {
                let (task, vdaf) = (task.clone(), vdaf.clone());
                Box::pin(async move {
                    FixedSize::get_batch_aggregations_for_collection_identifier::<
                        PRIO3_VERIFY_KEY_LENGTH,
                        Prio3Count,
                        _,
                    >(tx, &task, &vdaf, &second_batch_id, &())
                    .await
                })
            })
            .await
            .unwrap();
        assert_eq!(
            second_batch_got_batch_aggregations,
            second_batch_want_batch_aggregations
        );
    }

    #[tokio::test]
    async fn aggregate_continue_leader_sends_non_continue_transition() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        // Prepare parameters.
        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Helper,
        )
        .build();
        let aggregation_job_id = random();
        let report_metadata = ReportMetadata::new(
            ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            Time::from_seconds_since_epoch(54321),
        );
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        // Setup datastore.
        datastore
            .run_tx(|tx| {
                let (task, report_metadata) = (task.clone(), report_metadata.clone());
                Box::pin(async move {
                    tx.put_task(&task).await?;
                    tx.put_report_share(
                        task.id(),
                        &ReportShare::new(
                            report_metadata.clone(),
                            Vec::from("Public Share"),
                            HpkeCiphertext::new(
                                HpkeConfigId::from(42),
                                Vec::from("012345"),
                                Vec::from("543210"),
                            ),
                        ),
                    )
                    .await?;

                    tx.put_aggregation_job(&AggregationJob::<
                        DUMMY_VERIFY_KEY_LENGTH,
                        FixedSize,
                        dummy_vdaf::Vdaf,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        dummy_vdaf::AggregationParam(0),
                        random(),
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    ))
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<
                        DUMMY_VERIFY_KEY_LENGTH,
                        dummy_vdaf::Vdaf,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        *report_metadata.id(),
                        *report_metadata.time(),
                        0,
                        None,
                        ReportAggregationState::Waiting(dummy_vdaf::PrepareState::default(), None),
                    ))
                    .await
                })
            })
            .await
            .unwrap();

        // Make request.
        let request = AggregationJobContinueReq::new(
            AggregationJobRound::from(1),
            Vec::from([PrepareStep::new(
                *report_metadata.id(),
                PrepareStepResult::Finished,
            )]),
        );

        let handler = aggregator_handler(
            Arc::clone(&datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        post_aggregation_job_expecting_error(
            &task,
            &aggregation_job_id,
            &request,
            &handler,
            Status::BadRequest,
            "urn:ietf:params:ppm:dap:error:unrecognizedMessage",
            "The message type for a response was incorrect or the payload was malformed.",
        )
        .await;
    }

    #[tokio::test]
    async fn aggregate_continue_prep_step_fails() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        // Prepare parameters.
        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::FakeFailsPrepStep,
            Role::Helper,
        )
        .build();
        let aggregation_job_id = random();
        let report_metadata = ReportMetadata::new(
            ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            Time::from_seconds_since_epoch(54321),
        );
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        // Setup datastore.
        let batch_id = random();
        datastore
            .run_tx(|tx| {
                let (task, report_metadata) = (task.clone(), report_metadata.clone());

                Box::pin(async move {
                    tx.put_task(&task).await?;
                    tx.put_report_share(
                        task.id(),
                        &ReportShare::new(
                            report_metadata.clone(),
                            Vec::from("public share"),
                            HpkeCiphertext::new(
                                HpkeConfigId::from(42),
                                Vec::from("012345"),
                                Vec::from("543210"),
                            ),
                        ),
                    )
                    .await?;
                    tx.put_aggregation_job(&AggregationJob::<
                        DUMMY_VERIFY_KEY_LENGTH,
                        FixedSize,
                        dummy_vdaf::Vdaf,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        dummy_vdaf::AggregationParam(0),
                        batch_id,
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    ))
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<
                        DUMMY_VERIFY_KEY_LENGTH,
                        dummy_vdaf::Vdaf,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        *report_metadata.id(),
                        *report_metadata.time(),
                        0,
                        None,
                        ReportAggregationState::Waiting(dummy_vdaf::PrepareState::default(), None),
                    ))
                    .await
                })
            })
            .await
            .unwrap();

        // Make request.
        let request = AggregationJobContinueReq::new(
            AggregationJobRound::from(1),
            Vec::from([PrepareStep::new(
                *report_metadata.id(),
                PrepareStepResult::Continued(Vec::new()),
            )]),
        );

        let handler = aggregator_handler(
            Arc::clone(&datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        let aggregate_resp =
            post_aggregation_job_and_decode(&task, &aggregation_job_id, &request, &handler).await;
        assert_eq!(
            aggregate_resp,
            AggregationJobResp::new(Vec::from([PrepareStep::new(
                *report_metadata.id(),
                PrepareStepResult::Failed(ReportShareError::VdafPrepError),
            )]),)
        );

        // Check datastore state.
        let (aggregation_job, report_aggregation) = datastore
            .run_tx(|tx| {
                let (task, report_metadata) = (task.clone(), report_metadata.clone());
                Box::pin(async move {
                    let aggregation_job = tx
                        .get_aggregation_job::<DUMMY_VERIFY_KEY_LENGTH, FixedSize, dummy_vdaf::Vdaf>(
                            task.id(),
                            &aggregation_job_id,
                        )
                        .await.unwrap().unwrap();
                    let report_aggregation = tx
                        .get_report_aggregation(
                            &dummy_vdaf::Vdaf::default(),
                            &Role::Helper,
                            task.id(),
                            &aggregation_job_id,
                            report_metadata.id(),
                        )
                        .await.unwrap().unwrap();
                    Ok((aggregation_job, report_aggregation))
                })
            })
            .await
            .unwrap();

        assert_eq!(
            aggregation_job,
            AggregationJob::new(
                *task.id(),
                aggregation_job_id,
                dummy_vdaf::AggregationParam(0),
                batch_id,
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Finished,
                AggregationJobRound::from(1),
            )
            .with_last_continue_request_hash(aggregation_job.last_continue_request_hash().unwrap())
        );
        assert_eq!(
            report_aggregation,
            ReportAggregation::new(
                *task.id(),
                aggregation_job_id,
                *report_metadata.id(),
                *report_metadata.time(),
                0,
                Some(PrepareStep::new(
                    *report_metadata.id(),
                    PrepareStepResult::Failed(ReportShareError::VdafPrepError)
                )),
                ReportAggregationState::Failed(ReportShareError::VdafPrepError),
            )
        );
    }

    #[tokio::test]
    async fn aggregate_continue_unexpected_transition() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        // Prepare parameters.
        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Helper,
        )
        .build();
        let aggregation_job_id = random();
        let report_metadata = ReportMetadata::new(
            ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            Time::from_seconds_since_epoch(54321),
        );
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = ephemeral_datastore.datastore(clock.clone()).await;

        // Setup datastore.
        datastore
            .run_tx(|tx| {
                let (task, report_metadata) = (task.clone(), report_metadata.clone());

                Box::pin(async move {
                    tx.put_task(&task).await?;
                    tx.put_report_share(
                        task.id(),
                        &ReportShare::new(
                            report_metadata.clone(),
                            Vec::from("PUBLIC"),
                            HpkeCiphertext::new(
                                HpkeConfigId::from(42),
                                Vec::from("012345"),
                                Vec::from("543210"),
                            ),
                        ),
                    )
                    .await?;
                    tx.put_aggregation_job(&AggregationJob::<
                        DUMMY_VERIFY_KEY_LENGTH,
                        FixedSize,
                        dummy_vdaf::Vdaf,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        dummy_vdaf::AggregationParam(0),
                        random(),
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    ))
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<
                        DUMMY_VERIFY_KEY_LENGTH,
                        dummy_vdaf::Vdaf,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        *report_metadata.id(),
                        *report_metadata.time(),
                        0,
                        None,
                        ReportAggregationState::Waiting(dummy_vdaf::PrepareState::default(), None),
                    ))
                    .await
                })
            })
            .await
            .unwrap();

        // Make request.
        let request = AggregationJobContinueReq::new(
            AggregationJobRound::from(1),
            Vec::from([PrepareStep::new(
                ReportId::from(
                    [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], // not the same as above
                ),
                PrepareStepResult::Continued(Vec::new()),
            )]),
        );

        let handler = aggregator_handler(
            Arc::new(datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        post_aggregation_job_expecting_error(
            &task,
            &aggregation_job_id,
            &request,
            &handler,
            Status::BadRequest,
            "urn:ietf:params:ppm:dap:error:unrecognizedMessage",
            "The message type for a response was incorrect or the payload was malformed.",
        )
        .await;
    }

    #[tokio::test]
    async fn aggregate_continue_out_of_order_transition() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        // Prepare parameters.
        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Helper,
        )
        .build();
        let aggregation_job_id = random();
        let report_metadata_0 = ReportMetadata::new(
            ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            Time::from_seconds_since_epoch(54321),
        );
        let report_metadata_1 = ReportMetadata::new(
            ReportId::from([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
            Time::from_seconds_since_epoch(54321),
        );

        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = ephemeral_datastore.datastore(clock.clone()).await;

        // Setup datastore.
        datastore
            .run_tx(|tx| {
                let (task, report_metadata_0, report_metadata_1) = (
                    task.clone(),
                    report_metadata_0.clone(),
                    report_metadata_1.clone(),
                );

                Box::pin(async move {
                    tx.put_task(&task).await?;

                    tx.put_report_share(
                        task.id(),
                        &ReportShare::new(
                            report_metadata_0.clone(),
                            Vec::from("public"),
                            HpkeCiphertext::new(
                                HpkeConfigId::from(42),
                                Vec::from("012345"),
                                Vec::from("543210"),
                            ),
                        ),
                    )
                    .await?;
                    tx.put_report_share(
                        task.id(),
                        &ReportShare::new(
                            report_metadata_1.clone(),
                            Vec::from("public"),
                            HpkeCiphertext::new(
                                HpkeConfigId::from(42),
                                Vec::from("012345"),
                                Vec::from("543210"),
                            ),
                        ),
                    )
                    .await?;

                    tx.put_aggregation_job(&AggregationJob::<
                        DUMMY_VERIFY_KEY_LENGTH,
                        FixedSize,
                        dummy_vdaf::Vdaf,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        dummy_vdaf::AggregationParam(0),
                        random(),
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    ))
                    .await?;

                    tx.put_report_aggregation(&ReportAggregation::<
                        DUMMY_VERIFY_KEY_LENGTH,
                        dummy_vdaf::Vdaf,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        *report_metadata_0.id(),
                        *report_metadata_0.time(),
                        0,
                        None,
                        ReportAggregationState::Waiting(dummy_vdaf::PrepareState::default(), None),
                    ))
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<
                        DUMMY_VERIFY_KEY_LENGTH,
                        dummy_vdaf::Vdaf,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        *report_metadata_1.id(),
                        *report_metadata_1.time(),
                        1,
                        None,
                        ReportAggregationState::Waiting(dummy_vdaf::PrepareState::default(), None),
                    ))
                    .await
                })
            })
            .await
            .unwrap();

        // Make request.
        let request = AggregationJobContinueReq::new(
            AggregationJobRound::from(1),
            Vec::from([
                // Report IDs are in opposite order to what was stored in the datastore.
                PrepareStep::new(
                    *report_metadata_1.id(),
                    PrepareStepResult::Continued(Vec::new()),
                ),
                PrepareStep::new(
                    *report_metadata_0.id(),
                    PrepareStepResult::Continued(Vec::new()),
                ),
            ]),
        );

        let handler = aggregator_handler(
            Arc::new(datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        post_aggregation_job_expecting_error(
            &task,
            &aggregation_job_id,
            &request,
            &handler,
            Status::BadRequest,
            "urn:ietf:params:ppm:dap:error:unrecognizedMessage",
            "The message type for a response was incorrect or the payload was malformed.",
        )
        .await;
    }

    #[tokio::test]
    async fn aggregate_continue_for_non_waiting_aggregation() {
        // Prepare datastore & request.
        install_test_trace_subscriber();

        // Prepare parameters.
        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Helper,
        )
        .build();
        let aggregation_job_id = random();
        let report_metadata = ReportMetadata::new(
            ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            Time::from_seconds_since_epoch(54321),
        );

        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = ephemeral_datastore.datastore(clock.clone()).await;

        // Setup datastore.
        datastore
            .run_tx(|tx| {
                let (task, report_metadata) = (task.clone(), report_metadata.clone());
                Box::pin(async move {
                    tx.put_task(&task).await?;
                    tx.put_report_share(
                        task.id(),
                        &ReportShare::new(
                            report_metadata.clone(),
                            Vec::from("public share"),
                            HpkeCiphertext::new(
                                HpkeConfigId::from(42),
                                Vec::from("012345"),
                                Vec::from("543210"),
                            ),
                        ),
                    )
                    .await?;
                    tx.put_aggregation_job(&AggregationJob::<
                        DUMMY_VERIFY_KEY_LENGTH,
                        FixedSize,
                        dummy_vdaf::Vdaf,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        dummy_vdaf::AggregationParam(0),
                        random(),
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    ))
                    .await?;
                    tx.put_report_aggregation(&ReportAggregation::<
                        DUMMY_VERIFY_KEY_LENGTH,
                        dummy_vdaf::Vdaf,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        *report_metadata.id(),
                        *report_metadata.time(),
                        0,
                        None,
                        ReportAggregationState::Failed(ReportShareError::VdafPrepError),
                    ))
                    .await
                })
            })
            .await
            .unwrap();

        // Make request.
        let request = AggregationJobContinueReq::new(
            AggregationJobRound::from(1),
            Vec::from([PrepareStep::new(
                ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
                PrepareStepResult::Continued(Vec::new()),
            )]),
        );

        let handler = aggregator_handler(
            Arc::new(datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        post_aggregation_job_expecting_error(
            &task,
            &aggregation_job_id,
            &request,
            &handler,
            Status::BadRequest,
            "urn:ietf:params:ppm:dap:error:unrecognizedMessage",
            "The message type for a response was incorrect or the payload was malformed.",
        )
        .await;
    }

    #[tokio::test]
    async fn collection_job_put_request_to_helper() {
        let test_case = setup_collection_job_test_case(
            Role::Helper,
            QueryType::FixedSize { max_batch_size: 10 },
        )
        .await;

        let collection_job_id: CollectionJobId = random();
        let request = CollectionReq::new(
            Query::new_fixed_size(FixedSizeQuery::CurrentBatch),
            dummy_vdaf::AggregationParam::default().get_encoded(),
        );

        let mut test_conn = test_case
            .put_collection_job_with_auth_token(&collection_job_id, &request, Some(&random()))
            .await;

        assert_eq!(test_conn.status(), Some(Status::BadRequest));
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": Status::BadRequest as u16,
                "type": "urn:ietf:params:ppm:dap:error:unrecognizedTask",
                "title": "An endpoint received a message with an unknown task ID.",
                "taskid": format!("{}", test_case.task.id()),
            })
        );
    }

    #[tokio::test]
    async fn collection_job_put_request_invalid_aggregation_parameter() {
        let test_case = setup_collection_job_test_case(
            Role::Leader,
            QueryType::FixedSize { max_batch_size: 10 },
        )
        .await;

        let collection_job_id: CollectionJobId = random();
        let request = CollectionReq::new(
            Query::new_fixed_size(FixedSizeQuery::CurrentBatch),
            // dummy_vdaf::AggregationParam is a tuple struct wrapping a u8, so this is not a valid
            // encoding of an aggregation parameter.
            Vec::from([0u8, 0u8]),
        );

        let mut test_conn = test_case
            .put_collection_job(&collection_job_id, &request)
            .await;

        // Collect request will be rejected because the aggregation parameter can't be decoded
        assert_eq!(test_conn.status(), Some(Status::BadRequest));
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": Status::BadRequest as u16,
                "type": "urn:ietf:params:ppm:dap:error:unrecognizedMessage",
                "title": "The message type for a response was incorrect or the payload was malformed.",
            })
        );
    }

    #[tokio::test]
    async fn collection_job_put_request_invalid_batch_size() {
        install_test_trace_subscriber();

        // Prepare parameters.
        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .with_min_batch_size(1)
        .build();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = ephemeral_datastore.datastore(clock.clone()).await;

        datastore.put_task(&task).await.unwrap();

        let handler = aggregator_handler(
            Arc::new(datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        let collection_job_id: CollectionJobId = random();
        let request = CollectionReq::new(
            Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id: random() }),
            dummy_vdaf::AggregationParam::default().get_encoded(),
        );

        let mut test_conn = put(task.collection_job_uri(&collection_job_id).unwrap().path())
            .with_request_header(
                "DAP-Auth-Token",
                task.primary_collector_auth_token().as_ref().to_owned(),
            )
            .with_request_header(
                KnownHeaderName::ContentType,
                CollectionReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_request_body(request.get_encoded())
            .run_async(&handler)
            .await;

        // Collect request will be rejected because there are no reports in the batch interval
        assert_eq!(test_conn.status(), Some(Status::BadRequest));
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": Status::BadRequest as u16,
                "type": "urn:ietf:params:ppm:dap:error:invalidBatchSize",
                "title": "The number of reports included in the batch is invalid.",
                "taskid": format!("{}", task.id()),
            })
        );
    }

    #[tokio::test]
    async fn collection_job_put_request_unauthenticated() {
        let test_case = setup_collection_job_test_case(
            Role::Leader,
            QueryType::FixedSize { max_batch_size: 10 },
        )
        .await;

        let collection_job_id: CollectionJobId = random();
        let req = CollectionReq::new(
            Query::new_fixed_size(FixedSizeQuery::CurrentBatch),
            dummy_vdaf::AggregationParam::default().get_encoded(),
        );

        // Incorrect authentication token.
        let mut test_conn = test_case
            .put_collection_job_with_auth_token(&collection_job_id, &req, Some(&random()))
            .await;

        let want_status = Status::BadRequest;
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": want_status as u16,
                "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
                "title": "The request's authorization is not valid.",
                "taskid": format!("{}", test_case.task.id()),
            })
        );
        assert_eq!(want_status, test_conn.status().unwrap());

        // Aggregator authentication token.
        let mut test_conn = test_case
            .put_collection_job_with_auth_token(
                &collection_job_id,
                &req,
                Some(test_case.task.primary_aggregator_auth_token()),
            )
            .await;

        let want_status = Status::BadRequest;
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": want_status as u16,
                "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
                "title": "The request's authorization is not valid.",
                "taskid": format!("{}", test_case.task.id()),
            })
        );
        assert_eq!(want_status, test_conn.status().unwrap());

        // Missing authentication token.
        let mut test_conn = test_case
            .put_collection_job_with_auth_token(&collection_job_id, &req, None)
            .await;

        let want_status = Status::BadRequest;
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": want_status as u16,
                "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
                "title": "The request's authorization is not valid.",
                "taskid": format!("{}", test_case.task.id()),
            })
        );
        assert_eq!(want_status, test_conn.status().unwrap());
    }

    #[tokio::test]
    async fn collection_job_post_request_unauthenticated_collection_jobs() {
        let test_case = setup_collection_job_test_case(
            Role::Leader,
            QueryType::FixedSize { max_batch_size: 10 },
        )
        .await;

        let collection_job_id: CollectionJobId = random();
        let request = CollectionReq::new(
            Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id: random() }),
            dummy_vdaf::AggregationParam::default().get_encoded(),
        );

        let test_conn = test_case
            .put_collection_job(&collection_job_id, &request)
            .await;

        assert_eq!(test_conn.status().unwrap(), Status::Created);

        // Incorrect authentication token.
        let mut test_conn = test_case
            .post_collection_job_with_auth_token(&collection_job_id, Some(&random()))
            .await;

        let want_status = Status::BadRequest;
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": want_status as u16,
                "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
                "title": "The request's authorization is not valid.",
                "taskid": format!("{}", test_case.task.id()),
            })
        );
        assert_eq!(want_status, test_conn.status().unwrap());

        // Aggregator authentication token.
        let mut test_conn = test_case
            .post_collection_job_with_auth_token(
                &collection_job_id,
                Some(test_case.task.primary_aggregator_auth_token()),
            )
            .await;

        let want_status = Status::BadRequest;
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": want_status as u16,
                "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
                "title": "The request's authorization is not valid.",
                "taskid": format!("{}", test_case.task.id()),
            })
        );
        assert_eq!(want_status, test_conn.status().unwrap());

        // Missing authentication token.
        let mut test_conn = test_case
            .post_collection_job_with_auth_token(&collection_job_id, None)
            .await;

        let want_status = Status::BadRequest;
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": want_status as u16,
                "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
                "title": "The request's authorization is not valid.",
                "taskid": format!("{}", test_case.task.id()),
            })
        );
        assert_eq!(want_status, test_conn.status().unwrap());
    }

    #[tokio::test]
    async fn collection_job_success() {
        let test_case = setup_collection_job_test_case(
            Role::Leader,
            QueryType::FixedSize { max_batch_size: 10 },
        )
        .await;

        let batch_id = random();
        let client_timestamp_interval = Interval::new(
            Time::from_seconds_since_epoch(0),
            *test_case.task.time_precision(),
        )
        .unwrap();
        let aggregation_param = dummy_vdaf::AggregationParam::default();
        let leader_aggregate_share = dummy_vdaf::AggregateShare(0);
        let helper_aggregate_share = dummy_vdaf::AggregateShare(1);

        let collection_job_id: CollectionJobId = random();
        let request = CollectionReq::new(
            Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
            aggregation_param.get_encoded(),
        );

        test_case
            .datastore
            .run_tx(|tx| {
                let task_id = *test_case.task.id();

                Box::pin(async move {
                    tx.put_batch(&Batch::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                        task_id,
                        batch_id,
                        aggregation_param,
                        BatchState::Open,
                        1,
                        client_timestamp_interval,
                    ))
                    .await?;
                    Ok(())
                })
            })
            .await
            .unwrap();

        let test_conn = test_case
            .put_collection_job(&collection_job_id, &request)
            .await;

        let want_collection_job = CollectionJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
            *test_case.task.id(),
            collection_job_id,
            batch_id,
            aggregation_param,
            CollectionJobState::Start,
        );
        let want_batches = Vec::from([Batch::<0, FixedSize, dummy_vdaf::Vdaf>::new(
            *test_case.task.id(),
            batch_id,
            aggregation_param,
            BatchState::Closing,
            1,
            client_timestamp_interval,
        )]);

        let (got_collection_job, got_batches) = test_case
            .datastore
            .run_tx(|tx| {
                let task_id = *test_case.task.id();

                Box::pin(async move {
                    let got_collection_job = tx
                        .get_collection_job(&dummy_vdaf::Vdaf::new(), &collection_job_id)
                        .await?
                        .unwrap();
                    let got_batches = tx.get_batches_for_task(&task_id).await?;
                    Ok((got_collection_job, got_batches))
                })
            })
            .await
            .unwrap();

        assert_eq!(want_collection_job, got_collection_job);
        assert_eq!(want_batches, got_batches);

        assert_eq!(test_conn.status(), Some(Status::Created));

        let test_conn = test_case.post_collection_job(&collection_job_id).await;
        assert_eq!(test_conn.status(), Some(Status::Accepted));

        // Update the collection job with the aggregate shares and some aggregation jobs. collection
        // job should now be complete.
        test_case
            .datastore
            .run_tx(|tx| {
                let task = test_case.task.clone();
                let helper_aggregate_share_bytes = helper_aggregate_share.get_encoded();
                Box::pin(async move {
                    let encrypted_helper_aggregate_share = hpke::seal(
                        task.collector_hpke_config(),
                        &HpkeApplicationInfo::new(
                            &Label::AggregateShare,
                            &Role::Helper,
                            &Role::Collector,
                        ),
                        &helper_aggregate_share_bytes,
                        &AggregateShareAad::new(
                            *task.id(),
                            BatchSelector::new_fixed_size(batch_id),
                        )
                        .get_encoded(),
                    )
                    .unwrap();

                    let collection_job = tx
                        .get_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                            &dummy_vdaf::Vdaf::new(),
                            &collection_job_id,
                        )
                        .await
                        .unwrap()
                        .unwrap()
                        .with_state(CollectionJobState::Finished {
                            report_count: 12,
                            encrypted_helper_aggregate_share,
                            leader_aggregate_share,
                        });

                    tx.update_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(&collection_job)
                        .await
                        .unwrap();
                    Ok(())
                })
            })
            .await
            .unwrap();

        let mut test_conn = test_case.post_collection_job(&collection_job_id).await;

        assert_eq!(test_conn.status(), Some(Status::Ok));
        assert_headers!(
            &test_conn,
            "content-type" => (Collection::<FixedSize>::MEDIA_TYPE)
        );
        let body_bytes = take_response_body(&mut test_conn).await;
        let collect_resp = Collection::<FixedSize>::get_decoded(body_bytes.as_ref()).unwrap();

        assert_eq!(collect_resp.report_count(), 12);
        assert_eq!(collect_resp.interval(), &client_timestamp_interval);
        assert_eq!(collect_resp.encrypted_aggregate_shares().len(), 2);

        let decrypted_leader_aggregate_share = hpke::open(
            test_case.task.collector_hpke_config(),
            test_case.collector_hpke_keypair.private_key(),
            &HpkeApplicationInfo::new(&Label::AggregateShare, &Role::Leader, &Role::Collector),
            &collect_resp.encrypted_aggregate_shares()[0],
            &AggregateShareAad::new(
                *test_case.task.id(),
                BatchSelector::new_fixed_size(batch_id),
            )
            .get_encoded(),
        )
        .unwrap();
        assert_eq!(
            leader_aggregate_share,
            dummy_vdaf::AggregateShare::get_decoded(decrypted_leader_aggregate_share.as_ref())
                .unwrap()
        );

        let decrypted_helper_aggregate_share = hpke::open(
            test_case.task.collector_hpke_config(),
            test_case.collector_hpke_keypair.private_key(),
            &HpkeApplicationInfo::new(&Label::AggregateShare, &Role::Helper, &Role::Collector),
            &collect_resp.encrypted_aggregate_shares()[1],
            &AggregateShareAad::new(
                *test_case.task.id(),
                BatchSelector::new_fixed_size(batch_id),
            )
            .get_encoded(),
        )
        .unwrap();
        assert_eq!(
            helper_aggregate_share,
            dummy_vdaf::AggregateShare::get_decoded(decrypted_helper_aggregate_share.as_ref())
                .unwrap()
        );
    }

    #[tokio::test]
    async fn collection_job_post_request_no_such_collection_job() {
        let test_case = setup_collection_job_test_case(
            Role::Leader,
            QueryType::FixedSize { max_batch_size: 10 },
        )
        .await;

        let no_such_collection_job_id: CollectionJobId = random();

        let test_conn = post(&format!(
            "/tasks/{}/collection_jobs/{no_such_collection_job_id}",
            test_case.task.id()
        ))
        .with_request_header(
            "DAP-Auth-Token",
            test_case
                .task
                .primary_collector_auth_token()
                .as_ref()
                .to_owned(),
        )
        .run_async(&test_case.handler)
        .await;
        assert_eq!(test_conn.status(), Some(Status::NotFound));
    }

    #[tokio::test]
    async fn collection_job_put_request_batch_queried_too_many_times() {
        let test_case = setup_collection_job_test_case(
            Role::Leader,
            QueryType::FixedSize { max_batch_size: 10 },
        )
        .await;
        let batch_id = random();

        test_case
            .datastore
            .run_tx(|tx| {
                let task = test_case.task.clone();
                Box::pin(async move {
                    tx.put_batch_aggregation(&BatchAggregation::<
                        DUMMY_VERIFY_KEY_LENGTH,
                        FixedSize,
                        dummy_vdaf::Vdaf,
                    >::new(
                        *task.id(),
                        batch_id,
                        dummy_vdaf::AggregationParam(0),
                        0,
                        BatchAggregationState::Aggregating,
                        Some(dummy_vdaf::AggregateShare(0)),
                        10,
                        ReportIdChecksum::get_decoded(&[2; 32]).unwrap(),
                    ))
                    .await
                })
            })
            .await
            .unwrap();

        // Sending this request will consume a query for [0, time_precision).
        let request = CollectionReq::new(
            Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
            dummy_vdaf::AggregationParam(0).get_encoded(),
        );

        let test_conn = test_case.put_collection_job(&random(), &request).await;

        assert_eq!(test_conn.status(), Some(Status::Created));

        // This request will not be allowed due to the query count already being consumed.
        let invalid_request = CollectionReq::new(
            Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
            dummy_vdaf::AggregationParam(1).get_encoded(),
        );

        let mut test_conn = test_case
            .put_collection_job(&random(), &invalid_request)
            .await;
        assert_eq!(test_conn.status(), Some(Status::BadRequest));
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": Status::BadRequest as u16,
                "type": "urn:ietf:params:ppm:dap:error:batchQueriedTooManyTimes",
                "title": "The batch described by the query has been queried too many times.",
                "taskid": format!("{}", test_case.task.id()),
            })
        );
    }

    #[tokio::test]
    async fn delete_collection_job() {
        let test_case = setup_collection_job_test_case(
            Role::Leader,
            QueryType::FixedSize { max_batch_size: 10 },
        )
        .await;
        let batch_id = random();

        let collection_job_id: CollectionJobId = random();

        // Try to delete a collection job that doesn't exist
        let test_conn = delete(
            test_case
                .task
                .collection_job_uri(&collection_job_id)
                .unwrap()
                .path(),
        )
        .with_request_header(
            "DAP-Auth-Token",
            test_case
                .task
                .primary_collector_auth_token()
                .as_ref()
                .to_owned(),
        )
        .run_async(&test_case.handler)
        .await;
        assert_eq!(test_conn.status(), Some(Status::NotFound));

        // Create a collection job
        let request = CollectionReq::new(
            Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
            dummy_vdaf::AggregationParam::default().get_encoded(),
        );

        let test_conn = test_case
            .put_collection_job(&collection_job_id, &request)
            .await;

        assert_eq!(test_conn.status(), Some(Status::Created));

        // Cancel the job
        let test_conn = delete(
            test_case
                .task
                .collection_job_uri(&collection_job_id)
                .unwrap()
                .path(),
        )
        .with_request_header(
            "DAP-Auth-Token",
            test_case
                .task
                .primary_collector_auth_token()
                .as_ref()
                .to_owned(),
        )
        .run_async(&test_case.handler)
        .await;
        assert_eq!(test_conn.status(), Some(Status::NoContent));

        // Get the job again
        let test_conn = test_case.post_collection_job(&collection_job_id).await;
        assert_eq!(test_conn.status(), Some(Status::NoContent));
    }

    #[tokio::test]
    async fn aggregate_share_request_to_leader() {
        install_test_trace_subscriber();

        // Prepare parameters.
        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .build();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = ephemeral_datastore.datastore(clock.clone()).await;

        datastore.put_task(&task).await.unwrap();

        let handler = aggregator_handler(
            Arc::new(datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        let request = AggregateShareReq::new(
            BatchSelector::new_fixed_size(random()),
            Vec::new(),
            0,
            ReportIdChecksum::default(),
        );

        let mut test_conn = post(task.aggregate_shares_uri().unwrap().path())
            .with_request_header(
                "DAP-Auth-Token",
                task.primary_aggregator_auth_token().as_ref().to_owned(),
            )
            .with_request_header(
                KnownHeaderName::ContentType,
                AggregateShareReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_request_body(request.get_encoded())
            .run_async(&handler)
            .await;

        assert_eq!(test_conn.status(), Some(Status::BadRequest));
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": Status::BadRequest as u16,
                "type": "urn:ietf:params:ppm:dap:error:unrecognizedTask",
                "title": "An endpoint received a message with an unknown task ID.",
                "taskid": format!("{}", task.id()),
            })
        );
    }

    #[tokio::test]
    async fn aggregate_share_request() {
        install_test_trace_subscriber();

        let collector_hpke_keypair = generate_test_hpke_config_and_private_key();
        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Helper,
        )
        .with_max_batch_query_count(1)
        .with_time_precision(Duration::from_seconds(500))
        .with_min_batch_size(10)
        .with_collector_hpke_config(collector_hpke_keypair.config().clone())
        .build();

        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        datastore.put_task(&task).await.unwrap();

        let handler = aggregator_handler(
            Arc::clone(&datastore),
            clock,
            &noop_meter(),
            default_aggregator_config(),
        )
        .unwrap();

        // There are no batch aggregations in the datastore yet
        let request = AggregateShareReq::new(
            BatchSelector::new_fixed_size(random()),
            dummy_vdaf::AggregationParam(0).get_encoded(),
            0,
            ReportIdChecksum::default(),
        );

        let mut test_conn = post(task.aggregate_shares_uri().unwrap().path())
            .with_request_header(
                "DAP-Auth-Token",
                task.primary_aggregator_auth_token().as_ref().to_owned(),
            )
            .with_request_header(
                KnownHeaderName::ContentType,
                AggregateShareReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_request_body(request.get_encoded())
            .run_async(&handler)
            .await;

        assert_eq!(test_conn.status(), Some(Status::BadRequest));
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": Status::BadRequest as u16,
                "type": "urn:ietf:params:ppm:dap:error:invalidBatchSize",
                "title": "The number of reports included in the batch is invalid.",
                "taskid": format!("{}", task.id()),
            })
        );

        let batch_id_1 = random();
        let batch_id_2 = random();

        // Put some batch aggregations in the DB.
        datastore
            .run_tx(|tx| {
                let task = task.clone();
                Box::pin(async move {
                    for aggregation_param in [
                        dummy_vdaf::AggregationParam(0),
                        dummy_vdaf::AggregationParam(1),
                    ] {
                        tx.put_batch_aggregation(&BatchAggregation::<
                            DUMMY_VERIFY_KEY_LENGTH,
                            FixedSize,
                            dummy_vdaf::Vdaf,
                        >::new(
                            *task.id(),
                            batch_id_1,
                            aggregation_param,
                            0,
                            BatchAggregationState::Aggregating,
                            Some(dummy_vdaf::AggregateShare(64)),
                            5,
                            ReportIdChecksum::get_decoded(&[3; 32]).unwrap(),
                        ))
                        .await?;

                        tx.put_batch_aggregation(&BatchAggregation::<
                            DUMMY_VERIFY_KEY_LENGTH,
                            FixedSize,
                            dummy_vdaf::Vdaf,
                        >::new(
                            *task.id(),
                            batch_id_2,
                            aggregation_param,
                            0,
                            BatchAggregationState::Aggregating,
                            Some(dummy_vdaf::AggregateShare(128)),
                            10,
                            ReportIdChecksum::get_decoded(&[2; 32]).unwrap(),
                        ))
                        .await?;
                    }

                    Ok(())
                })
            })
            .await
            .unwrap();

        // Specified batch includes too few reports.
        let request = AggregateShareReq::new(
            BatchSelector::new_fixed_size(batch_id_1),
            dummy_vdaf::AggregationParam(0).get_encoded(),
            5,
            ReportIdChecksum::default(),
        );
        let mut test_conn = post(task.aggregate_shares_uri().unwrap().path())
            .with_request_header(
                "DAP-Auth-Token",
                task.primary_aggregator_auth_token().as_ref().to_owned(),
            )
            .with_request_header(
                KnownHeaderName::ContentType,
                AggregateShareReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_request_body(request.get_encoded())
            .run_async(&handler)
            .await;

        assert_eq!(test_conn.status(), Some(Status::BadRequest));
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": Status::BadRequest as u16,
                "type": "urn:ietf:params:ppm:dap:error:invalidBatchSize",
                "title": "The number of reports included in the batch is invalid.",
                "taskid": format!("{}", task.id()),
            })
        );

        // Make requests that will fail because the checksum or report counts don't match.
        for misaligned_request in [
            // Checksum doesn't match.
            AggregateShareReq::new(
                BatchSelector::new_fixed_size(batch_id_2),
                dummy_vdaf::AggregationParam(0).get_encoded(),
                10,
                ReportIdChecksum::get_decoded(&[3; 32]).unwrap(),
            ),
            // Report count doesn't match.
            AggregateShareReq::new(
                BatchSelector::new_fixed_size(batch_id_2),
                dummy_vdaf::AggregationParam(0).get_encoded(),
                20,
                ReportIdChecksum::get_decoded(&[2; 32]).unwrap(),
            ),
        ] {
            let mut test_conn = post(task.aggregate_shares_uri().unwrap().path())
                .with_request_header(
                    "DAP-Auth-Token",
                    task.primary_aggregator_auth_token().as_ref().to_owned(),
                )
                .with_request_header(
                    KnownHeaderName::ContentType,
                    AggregateShareReq::<FixedSize>::MEDIA_TYPE,
                )
                .with_request_body(misaligned_request.get_encoded())
                .run_async(&handler)
                .await;

            assert_eq!(test_conn.status(), Some(Status::BadRequest));
            assert_eq!(
                take_problem_details(&mut test_conn).await,
                json!({
                    "status": Status::BadRequest as u16,
                    "type": "urn:ietf:params:ppm:dap:error:batchMismatch",
                    "title": "Leader and helper disagree on reports aggregated in a batch.",
                    "taskid": format!("{}", task.id()),
                })
            );
        }

        // Valid request: checksum and report count are good.
        // Request the aggregate share multiple times. If the request parameters don't change,
        // then there is no query count violation and all requests should succeed.
        for iteration in 0..3 {
            let request = AggregateShareReq::new(
                BatchSelector::new_fixed_size(batch_id_2),
                dummy_vdaf::AggregationParam(0).get_encoded(),
                10,
                ReportIdChecksum::get_decoded(&[2; 32]).unwrap(),
            );
            let expected_result = dummy_vdaf::AggregateShare(128);

            let mut test_conn = post(task.aggregate_shares_uri().unwrap().path())
                .with_request_header(
                    "DAP-Auth-Token",
                    task.primary_aggregator_auth_token().as_ref().to_owned(),
                )
                .with_request_header(
                    KnownHeaderName::ContentType,
                    AggregateShareReq::<FixedSize>::MEDIA_TYPE,
                )
                .with_request_body(request.get_encoded())
                .run_async(&handler)
                .await;

            assert_eq!(
                test_conn.status(),
                Some(Status::Ok),
                "iteration: {iteration}"
            );
            assert_headers!(
                &test_conn,
                "content-type" => (AggregateShareMessage::MEDIA_TYPE)
            );
            let body_bytes = take_response_body(&mut test_conn).await;
            let aggregate_share_resp = AggregateShareMessage::get_decoded(&body_bytes).unwrap();

            let aggregate_share = hpke::open(
                collector_hpke_keypair.config(),
                collector_hpke_keypair.private_key(),
                &HpkeApplicationInfo::new(&Label::AggregateShare, &Role::Helper, &Role::Collector),
                aggregate_share_resp.encrypted_aggregate_share(),
                &AggregateShareAad::new(*task.id(), request.batch_selector().clone()).get_encoded(),
            )
            .unwrap();

            // Should get the sum over the first and second aggregate shares
            let decoded_aggregate_share =
                dummy_vdaf::AggregateShare::get_decoded(aggregate_share.as_ref()).unwrap();
            assert_eq!(
                decoded_aggregate_share, expected_result,
                "iteration: {iteration}"
            );
        }

        // Previous sequence of aggregate share requests should have consumed the available queries
        // for all the batches. Further requests for any batches will cause query count violations.
        let query_count_violation_request = AggregateShareReq::new(
            BatchSelector::new_fixed_size(batch_id_2),
            dummy_vdaf::AggregationParam(1).get_encoded(),
            10,
            ReportIdChecksum::get_decoded(&[2; 32]).unwrap(),
        );
        let mut test_conn = post(task.aggregate_shares_uri().unwrap().path())
            .with_request_header(
                "DAP-Auth-Token",
                task.primary_aggregator_auth_token().as_ref().to_owned(),
            )
            .with_request_header(
                KnownHeaderName::ContentType,
                AggregateShareReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_request_body(query_count_violation_request.get_encoded())
            .run_async(&handler)
            .await;
        assert_eq!(test_conn.status(), Some(Status::BadRequest));
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": Status::BadRequest as u16,
                "type": "urn:ietf:params:ppm:dap:error:batchQueriedTooManyTimes",
                "title": "The batch described by the query has been queried too many times.",
                "taskid": format!("{}", task.id()),
            })
        );
    }

    async fn take_response_body(test_conn: &mut TestConn) -> Cow<'_, [u8]> {
        test_conn
            .take_response_body()
            .unwrap()
            .into_bytes()
            .await
            .unwrap()
    }

    async fn take_problem_details(test_conn: &mut TestConn) -> serde_json::Value {
        serde_json::from_slice(&take_response_body(test_conn).await).unwrap()
    }
}
