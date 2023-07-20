//! This crate implements the Janus Aggregator API.

use crate::models::{GetTaskIdsResp, PostTaskReq};
use async_trait::async_trait;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use janus_aggregator_core::{
    datastore::{self, Datastore},
    instrumented,
    task::Task,
    SecretBytes,
};
use janus_core::{
    hpke::generate_hpke_config_and_private_key,
    http::extract_bearer_token,
    task::{AuthenticationToken, DapAuthToken},
    time::Clock,
};
use janus_messages::{Duration, HpkeAeadId, HpkeKdfId, HpkeKemId, Role, TaskId};
use models::{GetTaskMetricsResp, TaskResp};
use querystring::querify;
use rand::random;
use ring::{
    constant_time,
    digest::{digest, SHA256},
};
use std::{str::FromStr, sync::Arc, unreachable};
use tracing::{error, warn};
use trillium::{
    Conn, Handler,
    KnownHeaderName::{Accept, ContentType},
    Status,
    Status::{NotAcceptable, UnsupportedMediaType},
};
use trillium_api::{api, Halt, Json, State};
use trillium_opentelemetry::metrics;
use trillium_router::{Router, RouterConnExt};
use url::Url;

/// Represents the configuration for an instance of the Aggregator API.
#[derive(Clone)]
pub struct Config {
    pub auth_tokens: Vec<SecretBytes>,
}

/// Content type
const CONTENT_TYPE: &str = "application/vnd.janus.aggregator+json;version=0.1";

struct ReplaceMimeTypes;

#[trillium::async_trait]
impl Handler for ReplaceMimeTypes {
    async fn run(&self, mut conn: Conn) -> Conn {
        // Content-Type should either be the versioned API, or nothing for e.g. GET or DELETE
        // requests (no response body)
        let request_headers = conn.inner_mut().request_headers_mut();
        if let Some(CONTENT_TYPE) | None = request_headers.get_str(ContentType) {
            request_headers.insert(ContentType, "application/json");
        } else {
            return conn.with_status(UnsupportedMediaType).halt();
        }

        // Accept should always be the versioned API
        if Some(CONTENT_TYPE) == request_headers.get_str(Accept) {
            request_headers.insert(Accept, "application/json");
        } else {
            return conn.with_status(NotAcceptable).halt();
        }

        conn
    }

    async fn before_send(&self, conn: Conn) -> Conn {
        // API responses should always have versioned API content type
        conn.with_header(ContentType, CONTENT_TYPE)
    }
}

/// Returns a new handler for an instance of the aggregator API, backed by the given datastore,
/// according to the given configuration.
pub fn aggregator_api_handler<C: Clock>(ds: Arc<Datastore<C>>, cfg: Config) -> impl Handler {
    (
        // State used by endpoint handlers.
        State(ds),
        State(Arc::new(cfg)),
        // Metrics.
        metrics("janus_aggregator").with_route(|conn| conn.route().map(ToString::to_string)),
        // Authorization check.
        api(auth_check),
        // Check content type and accept headers
        ReplaceMimeTypes,
        // Main functionality router.
        Router::new()
            .get("/task_ids", instrumented(api(get_task_ids::<C>)))
            .post("/tasks", instrumented(api(post_task::<C>)))
            .get("/tasks/:task_id", instrumented(api(get_task::<C>)))
            .delete("/tasks/:task_id", instrumented(api(delete_task::<C>)))
            .get(
                "/tasks/:task_id/metrics",
                instrumented(api(get_task_metrics::<C>)),
            ),
    )
}

async fn auth_check(conn: &mut Conn, State(cfg): State<Arc<Config>>) -> impl Handler {
    let bearer_token = match extract_bearer_token(conn) {
        Ok(Some(t)) => t,
        _ => {
            return Some((Status::Unauthorized, Halt));
        }
    };

    if cfg.auth_tokens.iter().any(|key| {
        constant_time::verify_slices_are_equal(bearer_token.as_ref(), key.as_ref()).is_ok()
    }) {
        // Authorization succeeds.
        return None;
    }

    // Authorization fails.
    Some((Status::Unauthorized, Halt))
}

async fn get_task_ids<C: Clock>(
    conn: &mut Conn,
    State(ds): State<Arc<Datastore<C>>>,
) -> Result<impl Handler, Status> {
    const PAGINATION_TOKEN_KEY: &str = "pagination_token";
    let lower_bound = querify(conn.querystring())
        .into_iter()
        .find(|&(k, _)| k == PAGINATION_TOKEN_KEY)
        .map(|(_, v)| TaskId::from_str(v))
        .transpose()
        .map_err(|err| {
            warn!(err = ?err, "Couldn't parse pagination_token");
            Status::BadRequest
        })?;

    let task_ids = ds
        .run_tx_with_name("get_task_ids", |tx| {
            Box::pin(async move { tx.get_task_ids(lower_bound).await })
        })
        .await
        .map_err(|err| {
            error!(err = %err, "Database transaction error");
            Status::InternalServerError
        })?;
    let pagination_token = task_ids.last().cloned();

    Ok((
        Json(GetTaskIdsResp {
            task_ids,
            pagination_token,
        }),
        Halt,
    ))
}

/// A simple error type that holds a message and an HTTP status. This can be used as a [`Handler`].
struct Error {
    message: String,
    status: Status,
}

impl Error {
    fn new(message: String, status: Status) -> Self {
        Self { message, status }
    }
}

#[async_trait]
impl Handler for Error {
    async fn run(&self, conn: Conn) -> Conn {
        conn.with_body(self.message.clone())
            .with_status(self.status)
            .halt()
    }
}

async fn post_task<C: Clock>(
    _: &mut Conn,
    (State(ds), Json(req)): (State<Arc<Datastore<C>>>, Json<PostTaskReq>),
) -> Result<impl Handler, Error> {
    // We have to resolve impedance mismatches between the aggregator API's view of a task and
    // `aggregator_core::task::Task`. For now, we deal with this in code, but someday the two
    // representations will be harmonized.
    // https://github.com/divviup/janus/issues/1524

    if !matches!(req.role, Role::Leader | Role::Helper) {
        return Err(Error::new(
            format!("invalid role {}", req.role),
            Status::BadRequest,
        ));
    }

    // struct `aggregator_core::task::Task` expects to get two aggregator endpoint URLs, but only
    // the one for the peer aggregator is in the incoming request (or for that matter, is ever used
    // by Janus), so we insert a fake URL for "self".
    // unwrap safety: this fake URL is valid
    let fake_aggregator_url = Url::parse("http://never-used.example.com").unwrap();
    let aggregator_endpoints = match req.role {
        Role::Leader => Vec::from([fake_aggregator_url, req.peer_aggregator_endpoint]),
        Role::Helper => Vec::from([req.peer_aggregator_endpoint, fake_aggregator_url]),
        _ => unreachable!(),
    };

    let vdaf_verify_key_bytes = URL_SAFE_NO_PAD
        .decode(&req.vdaf_verify_key)
        .map_err(|err| {
            Error::new(
                format!("Invalid base64 value for vdaf_verify_key: {err}"),
                Status::BadRequest,
            )
        })?;
    if vdaf_verify_key_bytes.len() != req.vdaf.verify_key_length() {
        return Err(Error::new(
            format!(
                "Wrong VDAF verify key length, expected {}, got {}",
                req.vdaf.verify_key_length(),
                vdaf_verify_key_bytes.len()
            ),
            Status::BadRequest,
        ));
    }

    // DAP recommends deriving the task ID from the VDAF verify key. We deterministically obtain a
    // 32 byte task ID by taking SHA-256(VDAF verify key).
    // https://datatracker.ietf.org/doc/html/draft-ietf-ppm-dap-04#name-verification-key-requiremen
    let task_id = TaskId::try_from(digest(&SHA256, &vdaf_verify_key_bytes).as_ref())
        .map_err(|err| Error::new(err.to_string(), Status::InternalServerError))?;

    let vdaf_verify_keys = Vec::from([SecretBytes::new(vdaf_verify_key_bytes)]);

    let aggregator_auth_tokens = match req.role {
        Role::Leader => {
            let encoded = req.aggregator_auth_token.as_ref().ok_or_else(|| {
                Error::new(
                    "aggregator acting in leader role must be provided an aggregator auth token"
                        .to_string(),
                    Status::BadRequest,
                )
            })?;
            let token_bytes = URL_SAFE_NO_PAD.decode(encoded).map_err(|err| {
                Error::new(
                    format!("Invalid base64 value for aggregator_auth_token: {err}"),
                    Status::BadRequest,
                )
            })?;
            Vec::from([
                // TODO(#472): Each token in the PostTaskReq should indicate whether it is a bearer
                // token or a DAP-Auth-Token. For now, assume the latter.
                DapAuthToken::try_from(token_bytes)
                    .map(AuthenticationToken::DapAuth)
                    .map_err(|_| {
                        Error::new(
                            "Invalid HTTP header value in aggregator_auth_token".to_string(),
                            Status::BadRequest,
                        )
                    })?,
            ])
        }

        Role::Helper => {
            if req.aggregator_auth_token.is_some() {
                return Err(Error::new(
                    "aggregator acting in helper role cannot be given an aggregator auth token"
                        .to_string(),
                    Status::BadRequest,
                ));
            }
            // TODO(#472): switch to generating bearer tokens by default
            Vec::from([AuthenticationToken::DapAuth(random())])
        }

        _ => unreachable!(),
    };

    let collector_auth_tokens = match req.role {
        // TODO(#472): switch to generating bearer tokens by default
        Role::Leader => Vec::from([AuthenticationToken::DapAuth(random())]),
        Role::Helper => Vec::new(),
        _ => unreachable!(),
    };
    let hpke_keys = Vec::from([generate_hpke_config_and_private_key(
        random(),
        HpkeKemId::X25519HkdfSha256,
        HpkeKdfId::HkdfSha256,
        HpkeAeadId::Aes128Gcm,
    )]);

    let task = Arc::new(
        Task::new(
            task_id,
            aggregator_endpoints,
            /* query_type */ req.query_type,
            /* vdaf */ req.vdaf,
            /* role */ req.role,
            vdaf_verify_keys,
            /* max_batch_query_count */ req.max_batch_query_count,
            /* task_expiration */ req.task_expiration,
            /* report_expiry_age */
            Some(Duration::from_seconds(3600 * 24 * 7 * 2)), // 2 weeks
            /* min_batch_size */ req.min_batch_size,
            /* time_precision */ req.time_precision,
            /* tolerable_clock_skew */
            Duration::from_seconds(60), // 1 minute,
            /* collector_hpke_config */ req.collector_hpke_config,
            aggregator_auth_tokens,
            collector_auth_tokens,
            hpke_keys,
        )
        .map_err(|err| {
            Error::new(
                format!("Error constructing task: {err}"),
                Status::BadRequest,
            )
        })?,
    );

    ds.run_tx_with_name("post_task", |tx| {
        let task = Arc::clone(&task);
        Box::pin(async move { tx.put_task(&task).await })
    })
    .await
    .map_err(|err| {
        error!(err = %err, "Database transaction error");
        Error::new(
            "Error storing task".to_string(),
            Status::InternalServerError,
        )
    })?;

    Ok(Json(TaskResp::try_from(task.as_ref()).map_err(|err| {
        Error::new(err.to_string(), Status::InternalServerError)
    })?))
}

async fn get_task<C: Clock>(
    conn: &mut Conn,
    State(ds): State<Arc<Datastore<C>>>,
) -> Result<impl Handler, Status> {
    let task_id = conn.task_id_param()?;

    let task = ds
        .run_tx_with_name("get_task", |tx| {
            Box::pin(async move { tx.get_task(&task_id).await })
        })
        .await
        .map_err(|err| {
            error!(err = %err, "Database transaction error");
            Status::InternalServerError
        })?
        .ok_or(Status::NotFound)?;

    Ok(Json(TaskResp::try_from(&task).map_err(|err| {
        error!(err = %err, "Error converting task to TaskResp");
        Status::InternalServerError
    })?))
}

async fn delete_task<C: Clock>(
    conn: &mut Conn,
    State(ds): State<Arc<Datastore<C>>>,
) -> Result<impl Handler, Status> {
    let task_id = conn.task_id_param()?;

    ds.run_tx_with_name("delete_task", |tx| {
        Box::pin(async move { tx.delete_task(&task_id).await })
    })
    .await
    .map_err(|err| match err {
        datastore::Error::MutationTargetNotFound => Status::NotFound,
        _ => {
            error!(err = %err, "Database transaction error");
            Status::InternalServerError
        }
    })?;

    Ok(Status::NoContent)
}

async fn get_task_metrics<C: Clock>(
    conn: &mut Conn,
    State(ds): State<Arc<Datastore<C>>>,
) -> Result<impl Handler, Status> {
    let task_id = conn.task_id_param()?;

    let (reports, report_aggregations) = ds
        .run_tx_with_name("get_task_metrics", |tx| {
            Box::pin(async move { tx.get_task_metrics(&task_id).await })
        })
        .await
        .map_err(|err| {
            error!(err = %err, "Database transaction error");
            Status::InternalServerError
        })?
        .ok_or(Status::NotFound)?;

    Ok(Json(GetTaskMetricsResp {
        reports,
        report_aggregations,
    }))
}

mod models {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    use janus_aggregator_core::task::{QueryType, Task};
    use janus_core::task::VdafInstance;
    use janus_messages::{Duration, HpkeConfig, Role, TaskId, Time};
    use serde::{Deserialize, Serialize};
    use url::Url;

    #[derive(Serialize)]
    pub(crate) struct GetTaskIdsResp {
        pub(crate) task_ids: Vec<TaskId>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub(crate) pagination_token: Option<TaskId>,
    }

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    pub(crate) struct PostTaskReq {
        /// URL relative to which this task's peer aggregator's DAP API can be found. The peer
        /// aggregator plays the DAP role opposite to the one in the `role` field.
        pub(crate) peer_aggregator_endpoint: Url,
        /// DAP query type for this task.
        pub(crate) query_type: QueryType,
        /// The VDAF being run by this task.
        pub(crate) vdaf: VdafInstance,
        /// The role that this aggregator will play in this task.
        pub(crate) role: Role,
        /// The VDAF verification key used for this DAP task, as Base64 encoded bytes. Task ID is
        /// derived from the verify key.
        pub(crate) vdaf_verify_key: String,
        /// The maximum number of times a given batch may be collected.
        pub(crate) max_batch_query_count: u64,
        /// The time after which the task is considered invalid.
        pub(crate) task_expiration: Option<Time>,
        /// The minimum number of reports in a batch to allow it to be collected.
        pub(crate) min_batch_size: u64,
        /// The duration to which clients should round their reported timestamps, as seconds since
        /// the UNIX epoch.
        pub(crate) time_precision: Duration,
        /// HPKE configuration for the collector.
        pub(crate) collector_hpke_config: HpkeConfig,
        /// If this aggregator is the leader, this is the bearer token to use to authenticate
        /// requests to the helper, as Base64 encoded bytes. If this aggregator is the helper, the
        /// value is `None`.
        pub(crate) aggregator_auth_token: Option<String>,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    pub(crate) struct TaskResp {
        /// ID of the DAP Task.
        pub(crate) task_id: TaskId,
        /// URL relative to which this task's peer aggregator's DAP API can be found. The peer
        /// aggregator plays the DAP role opposite to the one in the `role` field.
        pub(crate) peer_aggregator_endpoint: Url,
        /// DAP query type for this task.
        pub(crate) query_type: QueryType,
        /// The VDAF being run by this task.
        pub(crate) vdaf: VdafInstance,
        /// The role that this aggregator will play in this task.
        pub(crate) role: Role,
        /// The VDAF verification key used for this DAP task, as Base64 encoded bytes. Task ID is
        /// derived from the verify key.
        pub(crate) vdaf_verify_key: String,
        /// The maximum number of times a given batch may be collected.
        pub(crate) max_batch_query_count: u64,
        /// The time after which the task is considered invalid.
        pub(crate) task_expiration: Option<Time>,
        /// The age after which a report is considered to be "expired" and will be considered a
        /// candidate for garbage collection.
        pub(crate) report_expiry_age: Option<Duration>,
        /// The minimum number of reports in a batch to allow it to be collected.
        pub(crate) min_batch_size: u64,
        /// The duration to which clients should round their reported timestamps.
        pub(crate) time_precision: Duration,
        /// How much clock skew to allow between client and aggregator. Reports from
        /// farther than this duration into the future will be rejected.
        pub(crate) tolerable_clock_skew: Duration,
        /// The authentication token for inter-aggregator communication in this task, as Base64
        /// encoded bytes.
        /// If `role` is Leader, this token is used by the aggregator to authenticate requests to
        /// the Helper. If `role` is Helper, this token is used by the aggregator to authenticate
        /// requests from the Leader.
        // TODO(#1509): This field will have to change as Janus helpers will only store a salted
        // hash of aggregator auth tokens.
        pub(crate) aggregator_auth_token: String,
        /// The authentication token used by the task's Collector to authenticate to the Leader, as
        /// Base64 encoded bytes.
        /// `Some` if `role` is Leader, `None` otherwise.
        // TODO(#1509) This field will have to change as Janus leaders will only store a salted hash
        // of collector auth tokens.
        pub(crate) collector_auth_token: Option<String>,
        /// HPKE configuration used by the collector to decrypt aggregate shares.
        pub(crate) collector_hpke_config: HpkeConfig,
        /// HPKE configuration(s) used by this aggregator to decrypt report shares.
        pub(crate) aggregator_hpke_configs: Vec<HpkeConfig>,
    }

    impl TryFrom<&Task> for TaskResp {
        type Error = &'static str;

        fn try_from(task: &Task) -> Result<Self, Self::Error> {
            // We have to resolve impedance mismatches between the aggregator API's view of a task
            // and `aggregator_core::task::Task`. For now, we deal with this in code, but someday
            // the two representations will be harmonized.
            // https://github.com/divviup/janus/issues/1524

            // Return the aggregator endpoint URL for the role opposite our own
            let peer_aggregator_endpoint = task.aggregator_endpoints()[match task.role() {
                Role::Leader => 1,
                Role::Helper => 0,
                _ => return Err("illegal aggregator role in task"),
            }]
            .clone();

            if task.vdaf_verify_keys().len() != 1 {
                return Err("illegal number of VDAF verify keys in task");
            }

            if task.aggregator_auth_tokens().len() != 1 {
                return Err("illegal number of aggregator auth tokens in task");
            }

            let collector_auth_token = match task.role() {
                Role::Leader => {
                    if task.collector_auth_tokens().len() != 1 {
                        return Err("illegal number of collector auth tokens in task");
                    } else {
                        Some(URL_SAFE_NO_PAD.encode(task.collector_auth_tokens()[0].as_ref()))
                    }
                }
                Role::Helper => None,
                _ => return Err("illegal aggregator role in task"),
            };

            let mut aggregator_hpke_configs: Vec<_> = task
                .hpke_keys()
                .values()
                .map(|keypair| keypair.config().clone())
                .collect();
            aggregator_hpke_configs.sort_by_key(|config| *config.id());

            Ok(Self {
                task_id: *task.id(),
                peer_aggregator_endpoint,
                query_type: *task.query_type(),
                vdaf: task.vdaf().clone(),
                role: *task.role(),
                vdaf_verify_key: URL_SAFE_NO_PAD.encode(task.vdaf_verify_keys()[0].as_ref()),
                max_batch_query_count: task.max_batch_query_count(),
                task_expiration: task.task_expiration().copied(),
                report_expiry_age: task.report_expiry_age().cloned(),
                min_batch_size: task.min_batch_size(),
                time_precision: *task.time_precision(),
                tolerable_clock_skew: *task.tolerable_clock_skew(),
                aggregator_auth_token: URL_SAFE_NO_PAD
                    .encode(task.aggregator_auth_tokens()[0].as_ref()),
                collector_auth_token,
                collector_hpke_config: task.collector_hpke_config().clone(),
                aggregator_hpke_configs,
            })
        }
    }

    #[derive(Serialize)]
    pub(crate) struct GetTaskMetricsResp {
        pub(crate) reports: u64,
        pub(crate) report_aggregations: u64,
    }
}

trait ConnExt {
    fn task_id_param(&self) -> Result<TaskId, Status>;
}

impl ConnExt for Conn {
    fn task_id_param(&self) -> Result<TaskId, Status> {
        TaskId::from_str(self.param("task_id").ok_or_else(|| {
            error!("No task_id parameter");
            Status::InternalServerError
        })?)
        .map_err(|err| {
            warn!(err = ?err, "Couldn't parse task_id parameter");
            Status::BadRequest
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        aggregator_api_handler,
        models::{GetTaskIdsResp, GetTaskMetricsResp, PostTaskReq, TaskResp},
        Config, CONTENT_TYPE,
    };
    use base64::{
        engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
        Engine,
    };
    use futures::future::try_join_all;
    use janus_aggregator_core::{
        datastore::{
            models::{
                AggregationJob, AggregationJobState, LeaderStoredReport, ReportAggregation,
                ReportAggregationState,
            },
            test_util::{ephemeral_datastore, EphemeralDatastore},
            Datastore,
        },
        task::{test_util::TaskBuilder, QueryType, Task},
        SecretBytes,
    };
    use janus_core::{
        hpke::{generate_hpke_config_and_private_key, HpkeKeypair, HpkePrivateKey},
        task::{AuthenticationToken, DapAuthToken, VdafInstance},
        test_util::{
            dummy_vdaf::{self, AggregationParam},
            install_test_trace_subscriber,
        },
        time::MockClock,
    };
    use janus_messages::{
        query_type::FixedSize, AggregationJobRound, Duration, HpkeAeadId, HpkeConfig, HpkeConfigId,
        HpkeKdfId, HpkeKemId, HpkePublicKey, Interval, Role, TaskId, Time,
    };
    use rand::{distributions::Standard, random, thread_rng, Rng};
    use serde_test::{assert_ser_tokens, assert_tokens, Token};
    use std::{iter, sync::Arc};
    use trillium::{Handler, Status};
    use trillium_testing::{
        assert_response, assert_status,
        prelude::{delete, get, post},
    };

    const AUTH_TOKEN: &str = "auth_token";

    async fn setup_api_test() -> (impl Handler, EphemeralDatastore, Arc<Datastore<MockClock>>) {
        install_test_trace_subscriber();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(MockClock::default()).await);
        let handler = aggregator_api_handler(
            Arc::clone(&datastore),
            Config {
                auth_tokens: Vec::from([SecretBytes::new(AUTH_TOKEN.as_bytes().to_vec())]),
            },
        );

        (handler, ephemeral_datastore, datastore)
    }

    #[tokio::test]
    async fn get_task_ids() {
        // Setup: write a few tasks to the datastore.
        let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

        let mut task_ids: Vec<_> = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    let tasks: Vec<_> = iter::repeat_with(|| {
                        TaskBuilder::new(
                            QueryType::FixedSize { max_batch_size: 10 },
                            VdafInstance::Fake,
                            Role::Leader,
                        )
                        .build()
                    })
                    .take(10)
                    .collect();

                    try_join_all(tasks.iter().map(|task| tx.put_task(task))).await?;

                    Ok(tasks.into_iter().map(|task| *task.id()).collect())
                })
            })
            .await
            .unwrap();
        task_ids.sort();

        fn response_for(task_ids: &[TaskId]) -> String {
            serde_json::to_string(&GetTaskIdsResp {
                task_ids: task_ids.to_vec(),
                pagination_token: task_ids.last().cloned(),
            })
            .unwrap()
        }

        // Verify: we can get the task IDs we wrote back from the API.
        assert_response!(
            get("/task_ids")
                .with_request_header(
                    "Authorization",
                    format!("Bearer {}", STANDARD.encode(AUTH_TOKEN))
                )
                .with_request_header("Accept", CONTENT_TYPE)
                .run_async(&handler)
                .await,
            Status::Ok,
            response_for(&task_ids),
        );

        // Verify: the lower_bound is respected, if specified.
        assert_response!(
            get(&format!(
                "/task_ids?pagination_token={}",
                task_ids.first().unwrap()
            ))
            .with_request_header(
                "Authorization",
                format!("Bearer {}", STANDARD.encode(AUTH_TOKEN))
            )
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
            Status::Ok,
            response_for(&task_ids[1..]),
        );

        // Verify: if the lower bound is large enough, nothing is returned.
        // (also verifies the "last" response will not include a pagination token)
        assert_response!(
            get(&format!(
                "/task_ids?pagination_token={}",
                task_ids.last().unwrap()
            ))
            .with_request_header(
                "Authorization",
                format!("Bearer {}", STANDARD.encode(AUTH_TOKEN))
            )
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
            Status::Ok,
            response_for(&[]),
        );

        // Verify: unauthorized requests are denied appropriately.
        assert_response!(
            get("/task_ids")
                .with_request_header("Accept", CONTENT_TYPE)
                .run_async(&handler)
                .await,
            Status::Unauthorized,
            "",
        );

        // Verify: requests without the Accept header are denied.
        assert_response!(
            get("/task_ids")
                .with_request_header(
                    "Authorization",
                    format!("Bearer {}", STANDARD.encode(AUTH_TOKEN))
                )
                .run_async(&handler)
                .await,
            Status::NotAcceptable,
            ""
        );
    }

    #[tokio::test]
    async fn post_task_bad_role() {
        // Setup: create a datastore & handler.
        let (handler, _ephemeral_datastore, _) = setup_api_test().await;

        let vdaf_verify_key =
            SecretBytes::new(thread_rng().sample_iter(Standard).take(16).collect());
        let aggregator_auth_token = AuthenticationToken::DapAuth(random());

        let req = PostTaskReq {
            peer_aggregator_endpoint: "http://aggregator.endpoint".try_into().unwrap(),
            query_type: QueryType::FixedSize { max_batch_size: 10 },
            vdaf: VdafInstance::Prio3Count,
            role: Role::Collector,
            vdaf_verify_key: URL_SAFE_NO_PAD.encode(&vdaf_verify_key),
            max_batch_query_count: 12,
            task_expiration: Some(Time::from_seconds_since_epoch(12345)),
            min_batch_size: 223,
            time_precision: Duration::from_seconds(62),
            collector_hpke_config: generate_hpke_config_and_private_key(
                random(),
                HpkeKemId::X25519HkdfSha256,
                HpkeKdfId::HkdfSha256,
                HpkeAeadId::Aes128Gcm,
            )
            .config()
            .clone(),
            aggregator_auth_token: Some(URL_SAFE_NO_PAD.encode(&aggregator_auth_token)),
        };
        assert_response!(
            post("/tasks")
                .with_request_body(serde_json::to_vec(&req).unwrap())
                .with_request_header(
                    "Authorization",
                    format!("Bearer {}", STANDARD.encode(AUTH_TOKEN))
                )
                .with_request_header("Accept", CONTENT_TYPE)
                .with_request_header("Content-Type", CONTENT_TYPE)
                .run_async(&handler)
                .await,
            Status::BadRequest
        );
    }

    #[tokio::test]
    async fn post_task_unauthorized() {
        // Setup: create a datastore & handler.
        let (handler, _ephemeral_datastore, _) = setup_api_test().await;

        let vdaf_verify_key =
            SecretBytes::new(thread_rng().sample_iter(Standard).take(16).collect());
        let aggregator_auth_token = AuthenticationToken::DapAuth(random());

        let req = PostTaskReq {
            peer_aggregator_endpoint: "http://aggregator.endpoint".try_into().unwrap(),
            query_type: QueryType::FixedSize { max_batch_size: 10 },
            vdaf: VdafInstance::Prio3Count,
            role: Role::Helper,
            vdaf_verify_key: URL_SAFE_NO_PAD.encode(&vdaf_verify_key),
            max_batch_query_count: 12,
            task_expiration: Some(Time::from_seconds_since_epoch(12345)),
            min_batch_size: 223,
            time_precision: Duration::from_seconds(62),
            collector_hpke_config: generate_hpke_config_and_private_key(
                random(),
                HpkeKemId::X25519HkdfSha256,
                HpkeKdfId::HkdfSha256,
                HpkeAeadId::Aes128Gcm,
            )
            .config()
            .clone(),
            aggregator_auth_token: Some(URL_SAFE_NO_PAD.encode(&aggregator_auth_token)),
        };
        assert_response!(
            post("/tasks")
                .with_request_body(serde_json::to_vec(&req).unwrap())
                .with_request_header("Accept", CONTENT_TYPE)
                .with_request_header("Content-Type", CONTENT_TYPE)
                // no Authorization header
                .run_async(&handler)
                .await,
            Status::Unauthorized
        );
    }

    /// Test the POST /tasks endpoint, with a helper task with no optional fields defined
    #[tokio::test]
    async fn post_task_helper_no_optional_fields() {
        // Setup: create a datastore & handler.
        let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

        let vdaf_verify_key =
            SecretBytes::new(thread_rng().sample_iter(Standard).take(16).collect());

        // Verify: posting a task creates a new task which matches the request.
        let req = PostTaskReq {
            peer_aggregator_endpoint: "http://aggregator.endpoint".try_into().unwrap(),
            query_type: QueryType::FixedSize {
                max_batch_size: 224,
            },
            vdaf: VdafInstance::Prio3Count,
            role: Role::Helper,
            vdaf_verify_key: URL_SAFE_NO_PAD.encode(&vdaf_verify_key),
            max_batch_query_count: 12,
            task_expiration: Some(Time::from_seconds_since_epoch(12345)),
            min_batch_size: 223,
            time_precision: Duration::from_seconds(62),
            collector_hpke_config: generate_hpke_config_and_private_key(
                random(),
                HpkeKemId::X25519HkdfSha256,
                HpkeKdfId::HkdfSha256,
                HpkeAeadId::Aes128Gcm,
            )
            .config()
            .clone(),
            aggregator_auth_token: None,
        };
        let mut conn = post("/tasks")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header(
                "Authorization",
                format!("Bearer {}", STANDARD.encode(AUTH_TOKEN)),
            )
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await;
        assert_status!(conn, Status::Ok);
        let got_task_resp: TaskResp = serde_json::from_slice(
            &conn
                .take_response_body()
                .unwrap()
                .into_bytes()
                .await
                .unwrap(),
        )
        .unwrap();

        let got_task = ds
            .run_tx(|tx| {
                let got_task_resp = got_task_resp.clone();
                Box::pin(async move { tx.get_task(&got_task_resp.task_id).await })
            })
            .await
            .unwrap()
            .expect("task was not created");

        // Verify that the task written to the datastore matches the request...
        assert_eq!(
            // The other aggregator endpoint in the datastore task is fake
            req.peer_aggregator_endpoint,
            got_task.aggregator_endpoints()[0]
        );
        assert_eq!(&req.query_type, got_task.query_type());
        assert_eq!(&req.vdaf, got_task.vdaf());
        assert_eq!(&req.role, got_task.role());
        assert_eq!(req.max_batch_query_count, got_task.max_batch_query_count());
        assert_eq!(req.task_expiration.as_ref(), got_task.task_expiration());
        assert_eq!(req.min_batch_size, got_task.min_batch_size());
        assert_eq!(&req.time_precision, got_task.time_precision());
        assert_eq!(1, got_task.aggregator_auth_tokens().len());
        assert_eq!(&req.collector_hpke_config, got_task.collector_hpke_config());

        // ...and the response.
        assert_eq!(got_task_resp, TaskResp::try_from(&got_task).unwrap());
    }

    #[tokio::test]
    async fn post_task_helper_with_aggregator_auth_token() {
        // Setup: create a datastore & handler.
        let (handler, _ephemeral_datastore, _) = setup_api_test().await;

        let vdaf_verify_key =
            SecretBytes::new(thread_rng().sample_iter(Standard).take(16).collect());
        let aggregator_auth_token = AuthenticationToken::DapAuth(random());

        // Verify: posting a task with role = helper and an aggregator auth token fails
        let req = PostTaskReq {
            peer_aggregator_endpoint: "http://aggregator.endpoint".try_into().unwrap(),
            query_type: QueryType::FixedSize { max_batch_size: 10 },
            vdaf: VdafInstance::Prio3Count,
            role: Role::Helper,
            vdaf_verify_key: URL_SAFE_NO_PAD.encode(&vdaf_verify_key),
            max_batch_query_count: 12,
            task_expiration: Some(Time::from_seconds_since_epoch(12345)),
            min_batch_size: 223,
            time_precision: Duration::from_seconds(62),
            collector_hpke_config: generate_hpke_config_and_private_key(
                random(),
                HpkeKemId::X25519HkdfSha256,
                HpkeKdfId::HkdfSha256,
                HpkeAeadId::Aes128Gcm,
            )
            .config()
            .clone(),
            aggregator_auth_token: Some(URL_SAFE_NO_PAD.encode(&aggregator_auth_token)),
        };
        assert_response!(
            post("/tasks")
                .with_request_body(serde_json::to_vec(&req).unwrap())
                .with_request_header(
                    "Authorization",
                    format!("Bearer {}", STANDARD.encode(AUTH_TOKEN)),
                )
                .with_request_header("Accept", CONTENT_TYPE)
                .with_request_header("Content-Type", CONTENT_TYPE)
                .run_async(&handler)
                .await,
            Status::BadRequest
        );
    }

    /// Test the POST /tasks endpoint, with a leader task with all of the optional fields provided.
    #[tokio::test]
    async fn post_task_leader_all_optional_fields() {
        // Setup: create a datastore & handler.
        let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

        let vdaf_verify_key =
            SecretBytes::new(thread_rng().sample_iter(Standard).take(16).collect());
        let aggregator_auth_token = AuthenticationToken::DapAuth(random());

        // Verify: posting a task creates a new task which matches the request.
        let req = PostTaskReq {
            peer_aggregator_endpoint: "http://aggregator.endpoint".try_into().unwrap(),
            query_type: QueryType::FixedSize {
                max_batch_size: 224,
            },
            vdaf: VdafInstance::Prio3Count,
            role: Role::Leader,
            vdaf_verify_key: URL_SAFE_NO_PAD.encode(&vdaf_verify_key),
            max_batch_query_count: 12,
            task_expiration: Some(Time::from_seconds_since_epoch(12345)),
            min_batch_size: 223,
            time_precision: Duration::from_seconds(62),
            collector_hpke_config: generate_hpke_config_and_private_key(
                random(),
                HpkeKemId::X25519HkdfSha256,
                HpkeKdfId::HkdfSha256,
                HpkeAeadId::Aes128Gcm,
            )
            .config()
            .clone(),
            aggregator_auth_token: Some(URL_SAFE_NO_PAD.encode(&aggregator_auth_token)),
        };
        let mut conn = post("/tasks")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header(
                "Authorization",
                format!("Bearer {}", STANDARD.encode(AUTH_TOKEN)),
            )
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await;
        assert_status!(conn, Status::Ok);
        let got_task_resp: TaskResp = serde_json::from_slice(
            &conn
                .take_response_body()
                .unwrap()
                .into_bytes()
                .await
                .unwrap(),
        )
        .unwrap();

        let got_task = ds
            .run_tx(|tx| {
                let got_task_resp = got_task_resp.clone();
                Box::pin(async move { tx.get_task(&got_task_resp.task_id).await })
            })
            .await
            .unwrap()
            .expect("task was not created");

        // Verify that the task written to the datastore matches the request...
        assert_eq!(
            // The other aggregator endpoint in the datastore task is fake
            req.peer_aggregator_endpoint,
            got_task.aggregator_endpoints()[1]
        );
        assert_eq!(&req.query_type, got_task.query_type());
        assert_eq!(&req.vdaf, got_task.vdaf());
        assert_eq!(&req.role, got_task.role());
        assert_eq!(1, got_task.vdaf_verify_keys().len());
        assert_eq!(
            vdaf_verify_key.as_ref(),
            got_task.vdaf_verify_keys()[0].as_ref()
        );
        assert_eq!(req.max_batch_query_count, got_task.max_batch_query_count());
        assert_eq!(req.task_expiration.as_ref(), got_task.task_expiration());
        assert_eq!(req.min_batch_size, got_task.min_batch_size());
        assert_eq!(&req.time_precision, got_task.time_precision());
        assert_eq!(&req.collector_hpke_config, got_task.collector_hpke_config());
        assert_eq!(1, got_task.aggregator_auth_tokens().len());
        assert_eq!(
            aggregator_auth_token.as_ref(),
            got_task.aggregator_auth_tokens()[0].as_ref()
        );
        assert_eq!(1, got_task.collector_auth_tokens().len());

        // ...and the response.
        assert_eq!(got_task_resp, TaskResp::try_from(&got_task).unwrap());
    }

    /// Test the POST /tasks endpoint, with a leader task with all of the optional fields provided.
    #[tokio::test]
    async fn post_task_leader_no_aggregator_auth_token() {
        // Setup: create a datastore & handler.
        let (handler, _ephemeral_datastore, _) = setup_api_test().await;

        let vdaf_verify_key =
            SecretBytes::new(thread_rng().sample_iter(Standard).take(16).collect());

        // Verify: posting a task with role = Leader and no aggregator auth token fails
        let req = PostTaskReq {
            peer_aggregator_endpoint: "http://aggregator.endpoint".try_into().unwrap(),
            query_type: QueryType::FixedSize { max_batch_size: 10 },
            vdaf: VdafInstance::Prio3Count,
            role: Role::Leader,
            vdaf_verify_key: URL_SAFE_NO_PAD.encode(&vdaf_verify_key),
            max_batch_query_count: 12,
            task_expiration: Some(Time::from_seconds_since_epoch(12345)),
            min_batch_size: 223,
            time_precision: Duration::from_seconds(62),
            collector_hpke_config: generate_hpke_config_and_private_key(
                random(),
                HpkeKemId::X25519HkdfSha256,
                HpkeKdfId::HkdfSha256,
                HpkeAeadId::Aes128Gcm,
            )
            .config()
            .clone(),
            aggregator_auth_token: None,
        };

        assert_response!(
            post("/tasks")
                .with_request_body(serde_json::to_vec(&req).unwrap())
                .with_request_header(
                    "Authorization",
                    format!("Bearer {}", STANDARD.encode(AUTH_TOKEN)),
                )
                .with_request_header("Accept", CONTENT_TYPE)
                .with_request_header("Content-Type", CONTENT_TYPE)
                .run_async(&handler)
                .await,
            Status::BadRequest
        );
    }

    #[tokio::test]
    async fn get_task() {
        // Setup: write a task to the datastore.
        let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .with_aggregator_auth_tokens(Vec::from([random()]))
        .with_collector_auth_tokens(Vec::from([random()]))
        .build();

        ds.run_tx(|tx| {
            let task = task.clone();
            Box::pin(async move {
                tx.put_task(&task).await?;
                Ok(())
            })
        })
        .await
        .unwrap();

        // Verify: getting the task returns the expected result.
        let want_task_resp = TaskResp::try_from(&task).unwrap();
        let mut conn = get(&format!("/tasks/{}", task.id()))
            .with_request_header(
                "Authorization",
                format!("Bearer {}", STANDARD.encode(AUTH_TOKEN)),
            )
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await;
        assert_status!(conn, Status::Ok);
        let got_task_resp = serde_json::from_slice(
            &conn
                .take_response_body()
                .unwrap()
                .into_bytes()
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(want_task_resp, got_task_resp);

        // Verify: getting a nonexistent task returns NotFound.
        assert_response!(
            get(&format!("/tasks/{}", random::<TaskId>()))
                .with_request_header(
                    "Authorization",
                    format!("Bearer {}", STANDARD.encode(AUTH_TOKEN))
                )
                .with_request_header("Accept", CONTENT_TYPE)
                .run_async(&handler)
                .await,
            Status::NotFound,
            "",
        );

        // Verify: unauthorized requests are denied appropriately.
        assert_response!(
            get(&format!("/tasks/{}", task.id()))
                .with_request_header("Accept", CONTENT_TYPE)
                .run_async(&handler)
                .await,
            Status::Unauthorized,
            "",
        );
    }

    #[tokio::test]
    async fn delete_task() {
        // Setup: write a task to the datastore.
        let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

        let task_id = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    let task = TaskBuilder::new(
                        QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Fake,
                        Role::Leader,
                    )
                    .build();

                    tx.put_task(&task).await?;

                    Ok(*task.id())
                })
            })
            .await
            .unwrap();

        // Verify: deleting a task succeeds (and actually deletes the task).
        assert_response!(
            delete(&format!("/tasks/{}", &task_id))
                .with_request_header(
                    "Authorization",
                    format!("Bearer {}", STANDARD.encode(AUTH_TOKEN))
                )
                .with_request_header("Accept", CONTENT_TYPE)
                .run_async(&handler)
                .await,
            Status::NoContent,
            "",
        );

        ds.run_tx(|tx| {
            Box::pin(async move {
                assert_eq!(tx.get_task(&task_id).await.unwrap(), None);
                Ok(())
            })
        })
        .await
        .unwrap();

        // Verify: deleting a task twice returns NotFound.
        assert_response!(
            delete(&format!("/tasks/{}", &task_id))
                .with_request_header(
                    "Authorization",
                    format!("Bearer {}", STANDARD.encode(AUTH_TOKEN))
                )
                .with_request_header("Accept", CONTENT_TYPE)
                .run_async(&handler)
                .await,
            Status::NotFound,
            "",
        );

        // Verify: deleting an arbitrary nonexistent task ID returns NotFound.
        assert_response!(
            delete(&format!("/tasks/{}", &random::<TaskId>()))
                .with_request_header(
                    "Authorization",
                    format!("Bearer {}", STANDARD.encode(AUTH_TOKEN))
                )
                .with_request_header("Accept", CONTENT_TYPE)
                .run_async(&handler)
                .await,
            Status::NotFound,
            "",
        );

        // Verify: unauthorized requests are denied appropriately.
        assert_response!(
            delete(&format!("/tasks/{}", &task_id))
                .with_request_header("Accept", CONTENT_TYPE)
                .run_async(&handler)
                .await,
            Status::Unauthorized,
            ""
        );
    }

    #[tokio::test]
    async fn get_task_metrics() {
        // Setup: write a task, some reports, and some report aggregations to the datastore.
        const REPORT_COUNT: usize = 10;
        const REPORT_AGGREGATION_COUNT: usize = 4;

        let (handler, _ephemeral_datastore, ds) = setup_api_test().await;
        let task_id = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    let task = TaskBuilder::new(
                        QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Fake,
                        Role::Leader,
                    )
                    .build();
                    let task_id = *task.id();
                    tx.put_task(&task).await?;

                    let reports: Vec<_> = iter::repeat_with(|| {
                        LeaderStoredReport::new_dummy(task_id, Time::from_seconds_since_epoch(0))
                    })
                    .take(REPORT_COUNT)
                    .collect();
                    try_join_all(reports.iter().map(|report| async move {
                        tx.put_client_report(&dummy_vdaf::Vdaf::new(), report).await
                    }))
                    .await?;

                    let aggregation_job_id = random();
                    tx.put_aggregation_job(&AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                        task_id,
                        aggregation_job_id,
                        AggregationParam(0),
                        random(),
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    ))
                    .await?;

                    try_join_all(
                        reports
                            .iter()
                            .take(REPORT_AGGREGATION_COUNT)
                            .enumerate()
                            .map(|(ord, report)| async move {
                                tx.put_report_aggregation(
                                    &ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                                        task_id,
                                        aggregation_job_id,
                                        *report.metadata().id(),
                                        *report.metadata().time(),
                                        ord.try_into().unwrap(),
                                        None,
                                        ReportAggregationState::Start,
                                    ),
                                )
                                .await
                            }),
                    )
                    .await?;

                    Ok(task_id)
                })
            })
            .await
            .unwrap();

        // Verify: requesting metrics on a task returns the correct result.
        assert_response!(
            get(&format!("/tasks/{}/metrics", &task_id))
                .with_request_header(
                    "Authorization",
                    format!("Bearer {}", STANDARD.encode(AUTH_TOKEN))
                )
                .with_request_header("Accept", CONTENT_TYPE)
                .run_async(&handler)
                .await,
            Status::Ok,
            serde_json::to_string(&GetTaskMetricsResp {
                reports: REPORT_COUNT.try_into().unwrap(),
                report_aggregations: REPORT_AGGREGATION_COUNT.try_into().unwrap(),
            })
            .unwrap(),
        );

        // Verify: requesting metrics on a nonexistent task returns NotFound.
        assert_response!(
            delete(&format!("/tasks/{}", &random::<TaskId>()))
                .with_request_header(
                    "Authorization",
                    format!("Bearer {}", STANDARD.encode(AUTH_TOKEN))
                )
                .with_request_header("Accept", CONTENT_TYPE)
                .run_async(&handler)
                .await,
            Status::NotFound,
            "",
        );

        // Verify: unauthorized requests are denied appropriately.
        assert_response!(
            get(&format!("/tasks/{}/metrics", &task_id))
                .with_request_header("Accept", CONTENT_TYPE)
                .run_async(&handler)
                .await,
            Status::Unauthorized,
            "",
        );
    }

    #[test]
    fn get_task_ids_resp_serialization() {
        assert_ser_tokens(
            &GetTaskIdsResp {
                task_ids: Vec::from([TaskId::from([0u8; 32])]),
                pagination_token: None,
            },
            &[
                Token::Struct {
                    name: "GetTaskIdsResp",
                    len: 1,
                },
                Token::Str("task_ids"),
                Token::Seq { len: Some(1) },
                Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
                Token::SeqEnd,
                Token::StructEnd,
            ],
        );
        assert_ser_tokens(
            &GetTaskIdsResp {
                task_ids: Vec::from([TaskId::from([0u8; 32])]),
                pagination_token: Some(TaskId::from([0u8; 32])),
            },
            &[
                Token::Struct {
                    name: "GetTaskIdsResp",
                    len: 2,
                },
                Token::Str("task_ids"),
                Token::Seq { len: Some(1) },
                Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
                Token::SeqEnd,
                Token::Str("pagination_token"),
                Token::Some,
                Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
                Token::StructEnd,
            ],
        );
    }

    #[test]
    fn post_task_req_serialization() {
        // helper request with optional fields omitted
        assert_tokens(
            &PostTaskReq {
                peer_aggregator_endpoint: "https://example.com/".parse().unwrap(),
                query_type: QueryType::FixedSize {
                    max_batch_size: 999,
                },
                vdaf: VdafInstance::Prio3CountVec { length: 5 },
                role: Role::Helper,
                vdaf_verify_key: "encoded".to_owned(),
                max_batch_query_count: 1,
                task_expiration: None,
                min_batch_size: 100,
                time_precision: Duration::from_seconds(3600),
                collector_hpke_config: HpkeConfig::new(
                    HpkeConfigId::from(7),
                    HpkeKemId::X25519HkdfSha256,
                    HpkeKdfId::HkdfSha256,
                    HpkeAeadId::Aes128Gcm,
                    HpkePublicKey::from([0u8; 32].to_vec()),
                ),
                aggregator_auth_token: None,
            },
            &[
                Token::Struct {
                    name: "PostTaskReq",
                    len: 11,
                },
                Token::Str("peer_aggregator_endpoint"),
                Token::Str("https://example.com/"),
                Token::Str("query_type"),
                Token::StructVariant {
                    name: "QueryType",
                    variant: "FixedSize",
                    len: 1,
                },
                Token::Str("max_batch_size"),
                Token::U64(999),
                Token::StructVariantEnd,
                Token::Str("vdaf"),
                Token::StructVariant {
                    name: "VdafInstance",
                    variant: "Prio3CountVec",
                    len: 1,
                },
                Token::Str("length"),
                Token::U64(5),
                Token::StructVariantEnd,
                Token::Str("role"),
                Token::UnitVariant {
                    name: "Role",
                    variant: "Helper",
                },
                Token::Str("vdaf_verify_key"),
                Token::Str("encoded"),
                Token::Str("max_batch_query_count"),
                Token::U64(1),
                Token::Str("task_expiration"),
                Token::None,
                Token::Str("min_batch_size"),
                Token::U64(100),
                Token::Str("time_precision"),
                Token::NewtypeStruct { name: "Duration" },
                Token::U64(3600),
                Token::Str("collector_hpke_config"),
                Token::Struct {
                    name: "HpkeConfig",
                    len: 5,
                },
                Token::Str("id"),
                Token::NewtypeStruct {
                    name: "HpkeConfigId",
                },
                Token::U8(7),
                Token::Str("kem_id"),
                Token::UnitVariant {
                    name: "HpkeKemId",
                    variant: "X25519HkdfSha256",
                },
                Token::Str("kdf_id"),
                Token::UnitVariant {
                    name: "HpkeKdfId",
                    variant: "HkdfSha256",
                },
                Token::Str("aead_id"),
                Token::UnitVariant {
                    name: "HpkeAeadId",
                    variant: "Aes128Gcm",
                },
                Token::Str("public_key"),
                Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
                Token::StructEnd,
                Token::Str("aggregator_auth_token"),
                Token::None,
                Token::StructEnd,
            ],
        );

        // leader request with optional fields set
        assert_tokens(
            &PostTaskReq {
                peer_aggregator_endpoint: "https://example.com/".parse().unwrap(),
                query_type: QueryType::FixedSize {
                    max_batch_size: 999,
                },
                vdaf: VdafInstance::Prio3CountVec { length: 5 },
                role: Role::Leader,
                vdaf_verify_key: "encoded".to_owned(),
                max_batch_query_count: 1,
                task_expiration: Some(Time::from_seconds_since_epoch(1000)),
                min_batch_size: 100,
                time_precision: Duration::from_seconds(3600),
                collector_hpke_config: HpkeConfig::new(
                    HpkeConfigId::from(7),
                    HpkeKemId::X25519HkdfSha256,
                    HpkeKdfId::HkdfSha256,
                    HpkeAeadId::Aes128Gcm,
                    HpkePublicKey::from([0u8; 32].to_vec()),
                ),
                aggregator_auth_token: Some("encoded".to_owned()),
            },
            &[
                Token::Struct {
                    name: "PostTaskReq",
                    len: 11,
                },
                Token::Str("peer_aggregator_endpoint"),
                Token::Str("https://example.com/"),
                Token::Str("query_type"),
                Token::StructVariant {
                    name: "QueryType",
                    variant: "FixedSize",
                    len: 1,
                },
                Token::Str("max_batch_size"),
                Token::U64(999),
                Token::StructVariantEnd,
                Token::Str("vdaf"),
                Token::StructVariant {
                    name: "VdafInstance",
                    variant: "Prio3CountVec",
                    len: 1,
                },
                Token::Str("length"),
                Token::U64(5),
                Token::StructVariantEnd,
                Token::Str("role"),
                Token::UnitVariant {
                    name: "Role",
                    variant: "Leader",
                },
                Token::Str("vdaf_verify_key"),
                Token::Str("encoded"),
                Token::Str("max_batch_query_count"),
                Token::U64(1),
                Token::Str("task_expiration"),
                Token::Some,
                Token::NewtypeStruct { name: "Time" },
                Token::U64(1000),
                Token::Str("min_batch_size"),
                Token::U64(100),
                Token::Str("time_precision"),
                Token::NewtypeStruct { name: "Duration" },
                Token::U64(3600),
                Token::Str("collector_hpke_config"),
                Token::Struct {
                    name: "HpkeConfig",
                    len: 5,
                },
                Token::Str("id"),
                Token::NewtypeStruct {
                    name: "HpkeConfigId",
                },
                Token::U8(7),
                Token::Str("kem_id"),
                Token::UnitVariant {
                    name: "HpkeKemId",
                    variant: "X25519HkdfSha256",
                },
                Token::Str("kdf_id"),
                Token::UnitVariant {
                    name: "HpkeKdfId",
                    variant: "HkdfSha256",
                },
                Token::Str("aead_id"),
                Token::UnitVariant {
                    name: "HpkeAeadId",
                    variant: "Aes128Gcm",
                },
                Token::Str("public_key"),
                Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
                Token::StructEnd,
                Token::Str("aggregator_auth_token"),
                Token::Some,
                Token::Str("encoded"),
                Token::StructEnd,
            ],
        );
    }

    #[test]
    fn task_resp_serialization() {
        let task = Task::new(
            TaskId::from([0u8; 32]),
            Vec::from([
                "https://leader.com/".parse().unwrap(),
                "https://helper.com/".parse().unwrap(),
            ]),
            QueryType::FixedSize {
                max_batch_size: 999,
            },
            VdafInstance::Prio3CountVec { length: 5 },
            Role::Leader,
            Vec::from([SecretBytes::new(b"vdaf verify key!".to_vec())]),
            1,
            None,
            None,
            100,
            Duration::from_seconds(3600),
            Duration::from_seconds(60),
            HpkeConfig::new(
                HpkeConfigId::from(7),
                HpkeKemId::X25519HkdfSha256,
                HpkeKdfId::HkdfSha256,
                HpkeAeadId::Aes128Gcm,
                HpkePublicKey::from([0u8; 32].to_vec()),
            ),
            Vec::from([AuthenticationToken::DapAuth(
                DapAuthToken::try_from(b"aggregator-12345678".to_vec()).unwrap(),
            )]),
            Vec::from([AuthenticationToken::DapAuth(
                DapAuthToken::try_from(b"collector-abcdef00".to_vec()).unwrap(),
            )]),
            [(HpkeKeypair::new(
                HpkeConfig::new(
                    HpkeConfigId::from(13),
                    HpkeKemId::X25519HkdfSha256,
                    HpkeKdfId::HkdfSha256,
                    HpkeAeadId::Aes128Gcm,
                    HpkePublicKey::from([0u8; 32].to_vec()),
                ),
                HpkePrivateKey::new(b"unused".to_vec()),
            ))],
        )
        .unwrap();
        assert_tokens(
            &TaskResp::try_from(&task).unwrap(),
            &[
                Token::Struct {
                    name: "TaskResp",
                    len: 16,
                },
                Token::Str("task_id"),
                Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
                Token::Str("peer_aggregator_endpoint"),
                Token::Str("https://helper.com/"),
                Token::Str("query_type"),
                Token::StructVariant {
                    name: "QueryType",
                    variant: "FixedSize",
                    len: 1,
                },
                Token::Str("max_batch_size"),
                Token::U64(999),
                Token::StructVariantEnd,
                Token::Str("vdaf"),
                Token::StructVariant {
                    name: "VdafInstance",
                    variant: "Prio3CountVec",
                    len: 1,
                },
                Token::Str("length"),
                Token::U64(5),
                Token::StructVariantEnd,
                Token::Str("role"),
                Token::UnitVariant {
                    name: "Role",
                    variant: "Leader",
                },
                Token::Str("vdaf_verify_key"),
                Token::Str("dmRhZiB2ZXJpZnkga2V5IQ"),
                Token::Str("max_batch_query_count"),
                Token::U64(1),
                Token::Str("task_expiration"),
                Token::None,
                Token::Str("report_expiry_age"),
                Token::None,
                Token::Str("min_batch_size"),
                Token::U64(100),
                Token::Str("time_precision"),
                Token::NewtypeStruct { name: "Duration" },
                Token::U64(3600),
                Token::Str("tolerable_clock_skew"),
                Token::NewtypeStruct { name: "Duration" },
                Token::U64(60),
                Token::Str("aggregator_auth_token"),
                Token::Str("YWdncmVnYXRvci0xMjM0NTY3OA"),
                Token::Str("collector_auth_token"),
                Token::Some,
                Token::Str("Y29sbGVjdG9yLWFiY2RlZjAw"),
                Token::Str("collector_hpke_config"),
                Token::Struct {
                    name: "HpkeConfig",
                    len: 5,
                },
                Token::Str("id"),
                Token::NewtypeStruct {
                    name: "HpkeConfigId",
                },
                Token::U8(7),
                Token::Str("kem_id"),
                Token::UnitVariant {
                    name: "HpkeKemId",
                    variant: "X25519HkdfSha256",
                },
                Token::Str("kdf_id"),
                Token::UnitVariant {
                    name: "HpkeKdfId",
                    variant: "HkdfSha256",
                },
                Token::Str("aead_id"),
                Token::UnitVariant {
                    name: "HpkeAeadId",
                    variant: "Aes128Gcm",
                },
                Token::Str("public_key"),
                Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
                Token::StructEnd,
                Token::Str("aggregator_hpke_configs"),
                Token::Seq { len: Some(1) },
                Token::Struct {
                    name: "HpkeConfig",
                    len: 5,
                },
                Token::Str("id"),
                Token::NewtypeStruct {
                    name: "HpkeConfigId",
                },
                Token::U8(13),
                Token::Str("kem_id"),
                Token::UnitVariant {
                    name: "HpkeKemId",
                    variant: "X25519HkdfSha256",
                },
                Token::Str("kdf_id"),
                Token::UnitVariant {
                    name: "HpkeKdfId",
                    variant: "HkdfSha256",
                },
                Token::Str("aead_id"),
                Token::UnitVariant {
                    name: "HpkeAeadId",
                    variant: "Aes128Gcm",
                },
                Token::Str("public_key"),
                Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
                Token::StructEnd,
                Token::SeqEnd,
                Token::StructEnd,
            ],
        );
    }

    #[test]
    fn get_task_metrics_resp_serialization() {
        assert_ser_tokens(
            &GetTaskMetricsResp {
                reports: 87,
                report_aggregations: 348,
            },
            &[
                Token::Struct {
                    name: "GetTaskMetricsResp",
                    len: 2,
                },
                Token::Str("reports"),
                Token::U64(87),
                Token::Str("report_aggregations"),
                Token::U64(348),
                Token::StructEnd,
            ],
        )
    }
}
