use anyhow::{anyhow, Context};
use backoff::ExponentialBackoffBuilder;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use clap::{value_parser, Arg, Command};
#[cfg(feature = "fpvec_bounded_l2")]
use fixed::types::extra::{U15, U31, U63};
#[cfg(feature = "fpvec_bounded_l2")]
use fixed::{FixedI16, FixedI32, FixedI64};
use janus_collector::{Collector, CollectorParameters};
use janus_core::task::DapAuthToken;
use janus_core::{
    hpke::HpkeKeypair,
    task::{AuthenticationToken, VdafInstance},
};
use janus_interop_binaries::Keyring;
use janus_interop_binaries::{
    install_tracing_subscriber,
    status::{COMPLETE, ERROR, IN_PROGRESS, SUCCESS},
    ErrorHandler, HpkeConfigRegistry, NumberAsString, VdafObject,
};
use janus_messages::{
    query_type::QueryType, BatchId, FixedSizeQuery, HpkeConfig, PartialBatchSelector, Query, TaskId,
};
#[cfg(feature = "fpvec_bounded_l2")]
use prio::vdaf::prio3::Prio3FixedPointBoundedL2VecSumMultithreaded;
use prio::{
    codec::{Decode, Encode},
    vdaf::{self, prio3::Prio3},
};
use rand::{distributions::Standard, prelude::Distribution, random};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::{
    collections::{hash_map::Entry, HashMap},
    net::Ipv4Addr,
    sync::Arc,
    time::Duration as StdDuration,
};
use tokio::{sync::Mutex, task::JoinHandle};
use trillium::{Conn, Handler};
use trillium_api::{api, Json, State};
use trillium_router::Router;
#[derive(Debug, Deserialize)]
struct AddTaskRequest {
    task_id: String,
    leader: Url,
    vdaf: VdafObject,
    collector_authentication_token: String,
    #[serde(rename = "query_type")]
    _query_type: u8,
}

#[derive(Debug, Serialize)]
struct AddTaskResponse {
    status: &'static str,
    #[serde(default)]
    error: Option<String>,
    collector_hpke_config: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RequestQuery {
    #[serde(rename = "type")]
    query_type: u8,
    subtype: Option<u8>,
    batch_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CollectStartRequest {
    task_id: String,
    agg_param: String,
    query: RequestQuery,
}

#[derive(Debug, Serialize)]
struct CollectStartResponse {
    status: &'static str,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    handle: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CollectPollRequest {
    handle: String,
}

#[derive(Debug, Clone)]
struct CollectResult {
    partial_batch_selector: Option<BatchId>,
    report_count: u64,
    interval_start: i64,
    interval_duration: i64,
    aggregation_result: AggregationResult,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum AggregationResult {
    Number(NumberAsString<u128>),
    NumberVec(Vec<NumberAsString<u128>>),
    #[cfg(feature = "fpvec_bounded_l2")]
    FloatVec(Vec<NumberAsString<f64>>),
}

#[derive(Debug, Serialize)]
struct CollectPollResponse {
    status: &'static str,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    batch_id: Option<String>,
    #[serde(default)]
    report_count: Option<u64>,
    #[serde(default)]
    interval_start: Option<i64>,
    #[serde(default)]
    interval_duration: Option<i64>,
    #[serde(default)]
    result: Option<AggregationResult>,
}

struct TaskState {
    keypair: HpkeKeypair,
    leader_url: Url,
    vdaf: VdafObject,
    auth_token: AuthenticationToken,
}

/// A collection job handle.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Handle(String);

impl Distribution<Handle> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Handle {
        Handle(URL_SAFE_NO_PAD.encode(rng.gen::<[u8; 32]>()))
    }
}

enum CollectionJobState {
    InProgress(Option<JoinHandle<anyhow::Result<CollectResult>>>),
    Completed(CollectResult),
    Error,
}

async fn handle_add_task(
    tasks: &Mutex<HashMap<TaskId, TaskState>>,
    keyring: &Mutex<HpkeConfigRegistry>,
    request: AddTaskRequest,
) -> anyhow::Result<HpkeConfig> {
    let task_id_bytes = URL_SAFE_NO_PAD
        .decode(request.task_id)
        .context("invalid base64url content in \"task_id\"")?;
    let task_id = TaskId::get_decoded(&task_id_bytes).context("invalid length of TaskId")?;

    let mut tasks_guard = tasks.lock().await;
    let entry = tasks_guard.entry(task_id);
    if let Entry::Occupied(_) = &entry {
        return Err(anyhow::anyhow!("cannot add a task with a duplicate ID"));
    }

    let keypair = keyring.lock().await.get_random_keypair();
    let hpke_config = keypair.config().clone();

    let auth_token = AuthenticationToken::DapAuth(
        DapAuthToken::try_from(request.collector_authentication_token.into_bytes())
            .context("invalid header value in \"collector_authentication_token\"")?,
    );

    entry.or_insert(TaskState {
        keypair,
        leader_url: request.leader,
        vdaf: request.vdaf,
        auth_token,
    });

    Ok(hpke_config)
}

async fn handle_collect_generic<V, Q>(
    http_client: &reqwest::Client,
    collector_params: CollectorParameters,
    query: Query<Q>,
    vdaf: V,
    agg_param_encoded: &[u8],
    batch_convert_fn: impl Fn(&PartialBatchSelector<Q>) -> Option<BatchId> + Send + 'static,
    result_convert_fn: impl Fn(&V::AggregateResult) -> AggregationResult + Send + 'static,
) -> anyhow::Result<JoinHandle<anyhow::Result<CollectResult>>>
where
    V: vdaf::Collector + Send + Sync + 'static,
    V::AggregationParam: Send + Sync + 'static,
    Q: QueryType,
{
    let collector = Collector::new(collector_params, vdaf, http_client.clone());
    let agg_param = V::AggregationParam::get_decoded(agg_param_encoded)?;
    let handle = tokio::spawn(async move {
        let collect_result = collector.collect(query, &agg_param).await?;
        let (interval_start, interval_duration) = collect_result.interval();
        Ok(CollectResult {
            partial_batch_selector: batch_convert_fn(collect_result.partial_batch_selector()),
            report_count: collect_result.report_count(),
            interval_start: interval_start.timestamp(),
            interval_duration: interval_duration.num_seconds(),
            aggregation_result: result_convert_fn(collect_result.aggregate_result()),
        })
    });
    Ok(handle)
}

enum ParsedQuery {
    FixedSize(FixedSizeQuery),
}

async fn handle_collection_start(
    http_client: &reqwest::Client,
    tasks: &Mutex<HashMap<TaskId, TaskState>>,
    collection_jobs: &Mutex<HashMap<Handle, CollectionJobState>>,
    request: CollectStartRequest,
) -> anyhow::Result<Handle> {
    let task_id_bytes = URL_SAFE_NO_PAD
        .decode(request.task_id)
        .context("invalid base64url content in \"task_id\"")?;
    let task_id = TaskId::get_decoded(&task_id_bytes).context("invalid length of TaskId")?;
    let agg_param = URL_SAFE_NO_PAD
        .decode(request.agg_param)
        .context("invalid base64url content in \"agg_param\"")?;

    let tasks_guard = tasks.lock().await;
    let task_state = tasks_guard
        .get(&task_id)
        .context("task was not added before being used in a collect request")?;

    let collector_params = CollectorParameters::new(
        task_id,
        task_state.leader_url.clone(),
        task_state.auth_token.clone(),
        task_state.keypair.config().clone(),
        task_state.keypair.private_key().clone(),
    )
    .with_http_request_backoff(
        ExponentialBackoffBuilder::new()
            .with_initial_interval(StdDuration::from_secs(1))
            .with_max_interval(StdDuration::from_secs(1))
            .with_max_elapsed_time(Some(StdDuration::from_secs(60)))
            .build(),
    )
    .with_collect_poll_backoff(
        ExponentialBackoffBuilder::new()
            .with_initial_interval(StdDuration::from_millis(200))
            .with_max_interval(StdDuration::from_secs(1))
            .with_multiplier(1.2)
            .with_max_elapsed_time(Some(StdDuration::from_secs(60)))
            .build(),
    );

    let query = match request.query.query_type {
        2 => match request.query.subtype {
            Some(0) => {
                let batch_id_bytes = URL_SAFE_NO_PAD
                    .decode(request.query.batch_id.context("\"batch_id\" was missing")?)?;
                let batch_id =
                    BatchId::get_decoded(&batch_id_bytes).context("invalid length of BatchId")?;
                ParsedQuery::FixedSize(FixedSizeQuery::ByBatchId { batch_id })
            }
            Some(1) => ParsedQuery::FixedSize(FixedSizeQuery::CurrentBatch),
            None => return Err(anyhow::anyhow!("\"subtype\" was missing")),
            _ => return Err(anyhow::anyhow!("unrecognized \"subtype\" in query")),
        },
        _ => {
            return Err(anyhow::anyhow!(
                "unsupported query type: {}",
                request.query.query_type
            ))
        }
    };

    let vdaf_instance = task_state.vdaf.clone().into();
    let task_handle = match (query, vdaf_instance) {
        (ParsedQuery::FixedSize(fixed_size_query), VdafInstance::Prio3Count {}) => {
            let vdaf = Prio3::new_count(2).context("failed to construct Prio3Count VDAF")?;
            handle_collect_generic(
                http_client,
                collector_params,
                Query::new_fixed_size(fixed_size_query),
                vdaf,
                &agg_param,
                |selector| Some(*selector.batch_id()),
                |result| AggregationResult::Number(NumberAsString((*result).into())),
            )
            .await?
        }

        (ParsedQuery::FixedSize(fixed_size_query), VdafInstance::Prio3CountVec { length }) => {
            let vdaf = Prio3::new_sum_vec_multithreaded(2, 1, length)
                .context("failed to construct Prio3CountVec VDAF")?;
            handle_collect_generic(
                http_client,
                collector_params,
                Query::new_fixed_size(fixed_size_query),
                vdaf,
                &agg_param,
                |selector| Some(*selector.batch_id()),
                |result| {
                    let converted = result.iter().cloned().map(NumberAsString).collect();
                    AggregationResult::NumberVec(converted)
                },
            )
            .await?
        }

        #[cfg(feature = "fpvec_bounded_l2")]
        (
            ParsedQuery::FixedSize(fixed_size_query),
            janus_core::task::VdafInstance::Prio3FixedPoint16BitBoundedL2VecSum { length },
        ) => {
            let vdaf: Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI16<U15>> =
                Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(2, length)
                    .context("failed to construct Prio3FixedPoint16BitBoundedL2VecSum VDAF")?;
            handle_collect_generic(
                http_client,
                collector_params,
                Query::new_fixed_size(fixed_size_query),
                vdaf,
                &agg_param,
                |selector| Some(*selector.batch_id()),
                |result| {
                    let converted = result.iter().cloned().map(NumberAsString).collect();
                    AggregationResult::FloatVec(converted)
                },
            )
            .await?
        }

        #[cfg(feature = "fpvec_bounded_l2")]
        (
            ParsedQuery::FixedSize(fixed_size_query),
            janus_core::task::VdafInstance::Prio3FixedPoint32BitBoundedL2VecSum { length },
        ) => {
            let vdaf: Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI32<U31>> =
                Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(2, length)
                    .context("failed to construct Prio3FixedPoint32BitBoundedL2VecSum VDAF")?;
            handle_collect_generic(
                http_client,
                collector_params,
                Query::new_fixed_size(fixed_size_query),
                vdaf,
                &agg_param,
                |selector| Some(*selector.batch_id()),
                |result| {
                    let converted = result.iter().cloned().map(NumberAsString).collect();
                    AggregationResult::FloatVec(converted)
                },
            )
            .await?
        }

        #[cfg(feature = "fpvec_bounded_l2")]
        (
            ParsedQuery::FixedSize(fixed_size_query),
            janus_core::task::VdafInstance::Prio3FixedPoint64BitBoundedL2VecSum { length },
        ) => {
            let vdaf: Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI64<U63>> =
                Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(2, length)
                    .context("failed to construct Prio3FixedPoint64BitBoundedL2VecSum VDAF")?;
            handle_collect_generic(
                http_client,
                collector_params,
                Query::new_fixed_size(fixed_size_query),
                vdaf,
                &agg_param,
                |selector| Some(*selector.batch_id()),
                |result| {
                    let converted = result.iter().cloned().map(NumberAsString).collect();
                    AggregationResult::FloatVec(converted)
                },
            )
            .await?
        }

        (ParsedQuery::FixedSize(fixed_size_query), VdafInstance::Prio3Sum { bits }) => {
            let vdaf = Prio3::new_sum(2, bits).context("failed to construct Prio3Sum VDAF")?;
            handle_collect_generic(
                http_client,
                collector_params,
                Query::new_fixed_size(fixed_size_query),
                vdaf,
                &agg_param,
                |selector| Some(*selector.batch_id()),
                |result| AggregationResult::Number(NumberAsString(*result)),
            )
            .await?
        }

        (ParsedQuery::FixedSize(fixed_size_query), VdafInstance::Prio3SumVec { bits, length }) => {
            let vdaf = Prio3::new_sum_vec_multithreaded(2, bits, length)
                .context("failed to construct Prio3SumVec VDAF")?;
            handle_collect_generic(
                http_client,
                collector_params,
                Query::new_fixed_size(fixed_size_query),
                vdaf,
                &agg_param,
                |selector| Some(*selector.batch_id()),
                |result| {
                    let converted = result.iter().cloned().map(NumberAsString).collect();
                    AggregationResult::NumberVec(converted)
                },
            )
            .await?
        }

        (ParsedQuery::FixedSize(fixed_size_query), VdafInstance::Prio3Histogram { buckets }) => {
            let vdaf = Prio3::new_histogram(2, &buckets)
                .context("failed to construct Prio3Histogram VDAF")?;
            handle_collect_generic(
                http_client,
                collector_params,
                Query::new_fixed_size(fixed_size_query),
                vdaf,
                &agg_param,
                |selector| Some(*selector.batch_id()),
                |result| {
                    let converted = result.iter().cloned().map(NumberAsString).collect();
                    AggregationResult::NumberVec(converted)
                },
            )
            .await?
        }

        (_, vdaf_instance) => {
            panic!("Unsupported VDAF: {vdaf_instance:?}")
        }
    };

    let mut collection_jobs_guard = collection_jobs.lock().await;
    Ok(loop {
        match collection_jobs_guard.entry(random()) {
            Entry::Occupied(_) => continue,
            entry @ Entry::Vacant(_) => {
                let key = entry.key().clone();
                entry.or_insert(CollectionJobState::InProgress(Some(task_handle)));
                break key;
            }
        }
    })
}

async fn handle_collection_poll(
    collection_jobs: &Mutex<HashMap<Handle, CollectionJobState>>,
    request: CollectPollRequest,
) -> anyhow::Result<Option<CollectResult>> {
    let mut collection_jobs_guard = collection_jobs.lock().await;
    let collection_job_state_entry = collection_jobs_guard.entry(Handle(request.handle.clone()));
    match collection_job_state_entry {
        Entry::Occupied(mut occupied_entry) => match occupied_entry.get_mut() {
            CollectionJobState::InProgress(join_handle_opt) => {
                if join_handle_opt.as_ref().unwrap().is_finished() {
                    // Awaiting on the JoinHandle requires owning it. We take it out of the Option,
                    // and ensure that a different enum variant is stored over it before dropping
                    // the lock on the HashMap.
                    let taken_handle = join_handle_opt.take().unwrap();
                    let task_result = taken_handle.await;
                    let collect_result = match task_result {
                        Ok(collect_result) => collect_result,
                        Err(e) => {
                            occupied_entry.insert(CollectionJobState::Error);
                            return Err(e).context("panic while handling collection");
                        }
                    };
                    match collect_result {
                        Ok(collect_result) => {
                            occupied_entry
                                .insert(CollectionJobState::Completed(collect_result.clone()));
                            Ok(Some(collect_result))
                        }
                        Err(e) => {
                            occupied_entry.insert(CollectionJobState::Error);
                            Err(e)
                        }
                    }
                } else {
                    Ok(None)
                }
            }
            CollectionJobState::Completed(collect_result) => Ok(Some(collect_result.clone())),
            CollectionJobState::Error => Err(anyhow::anyhow!(
                "collection previously resulted in an error"
            )),
        },
        Entry::Vacant(_) => Err(anyhow::anyhow!(
            "did not recognize handle in collection_poll request"
        )),
    }
}

#[derive(Clone)]
struct TaskStateMap(Arc<Mutex<HashMap<TaskId, TaskState>>>);

impl TaskStateMap {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(HashMap::new())))
    }
}

#[derive(Clone)]
struct CollectionJobStateMap(Arc<Mutex<HashMap<Handle, CollectionJobState>>>);

impl CollectionJobStateMap {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(HashMap::new())))
    }
}

fn handler() -> anyhow::Result<impl Handler> {
    let http_client = janus_collector::default_http_client()?;
    let tasks = TaskStateMap::new();
    let collection_jobs = CollectionJobStateMap::new();
    let keyring = Keyring::new();

    let router = Router::new()
        .post("/internal/test/ready", Json(serde_json::json!({})))
        .post(
            "/internal/test/add_task",
            api(
                |_conn: &mut Conn,
                 (State(tasks), State(keyring), Json(request)): (
                    State<TaskStateMap>,
                    State<Keyring>,
                    Json<AddTaskRequest>,
                )| async move {
                    match handle_add_task(&tasks.0, &keyring.0, request).await {
                        Ok(collector_hpke_config) => Json(AddTaskResponse {
                            status: SUCCESS,
                            error: None,
                            collector_hpke_config: Some(
                                URL_SAFE_NO_PAD.encode(collector_hpke_config.get_encoded()),
                            ),
                        }),
                        Err(e) => Json(AddTaskResponse {
                            status: ERROR,
                            error: Some(format!("{e:?}")),
                            collector_hpke_config: None,
                        }),
                    }
                },
            ),
        )
        .post(
            "/internal/test/collection_start",
            api(
                |_conn: &mut Conn,
                 (State(http_client), State(tasks), State(collection_jobs), Json(request)): (
                    State<reqwest::Client>,
                    State<TaskStateMap>,
                    State<CollectionJobStateMap>,
                    Json<CollectStartRequest>,
                )| async move {
                    match handle_collection_start(
                        &http_client,
                        &tasks.0,
                        &collection_jobs.0,
                        request,
                    )
                    .await
                    {
                        Ok(handle) => Json(CollectStartResponse {
                            status: SUCCESS,
                            error: None,
                            handle: Some(handle.0),
                        }),
                        Err(e) => Json(CollectStartResponse {
                            status: ERROR,
                            error: Some(format!("{e:?}")),
                            handle: None,
                        }),
                    }
                },
            ),
        )
        .post(
            "/internal/test/collection_poll",
            api(
                |_conn: &mut Conn,
                 (State(collection_jobs), Json(request)): (
                    State<CollectionJobStateMap>,
                    Json<CollectPollRequest>,
                )| async move {
                    match handle_collection_poll(&collection_jobs.0, request).await {
                        Ok(Some(collect_result)) => Json(CollectPollResponse {
                            status: COMPLETE,
                            error: None,
                            batch_id: collect_result
                                .partial_batch_selector
                                .map(|batch_id| URL_SAFE_NO_PAD.encode(batch_id.as_ref())),
                            report_count: Some(collect_result.report_count),
                            interval_start: Some(collect_result.interval_start),
                            interval_duration: Some(collect_result.interval_duration),
                            result: Some(collect_result.aggregation_result),
                        }),
                        Ok(None) => Json(CollectPollResponse {
                            status: IN_PROGRESS,
                            error: None,
                            batch_id: None,
                            report_count: None,
                            interval_start: None,
                            interval_duration: None,
                            result: None,
                        }),
                        Err(e) => Json(CollectPollResponse {
                            status: ERROR,
                            error: Some(format!("{e:?}")),
                            batch_id: None,
                            report_count: None,
                            interval_start: None,
                            interval_duration: None,
                            result: None,
                        }),
                    }
                },
            ),
        );

    Ok((
        State(http_client),
        State(tasks),
        State(collection_jobs),
        State(keyring),
        router,
        ErrorHandler,
    ))
}

fn app() -> clap::Command {
    Command::new("Janus interoperation test collector").arg(
        Arg::new("port")
            .long("port")
            .short('p')
            .default_value("8080")
            .value_parser(value_parser!(u16))
            .help("Port number to listen on."),
    )
}

fn main() -> anyhow::Result<()> {
    install_tracing_subscriber()?;
    let matches = app().get_matches();
    let port = matches
        .try_get_one::<u16>("port")?
        .ok_or_else(|| anyhow!("port argument missing"))?;
    trillium_tokio::config()
        .with_host(&Ipv4Addr::UNSPECIFIED.to_string())
        .with_port(*port)
        .run(handler()?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::app;

    #[test]
    fn verify_clap_app() {
        app().debug_assert();
    }
}
