use anyhow::Context;
use backoff::ExponentialBackoffBuilder;
use base64::URL_SAFE_NO_PAD;
use clap::{Arg, Command};
use interop_binaries::{
    install_tracing_subscriber,
    status::{COMPLETE, ERROR, IN_PROGRESS, SUCCESS},
    HpkeConfigRegistry, NumberAsString, VdafObject,
};
use janus_collector::{Collector, CollectorParameters};
use janus_core::{
    hpke::HpkePrivateKey,
    message::{Duration, HpkeConfig, Interval, TaskId, Time},
    task::AuthenticationToken,
};
use janus_server::task::VdafInstance;
use prio::{
    codec::{Decode, Encode},
    vdaf::{self, prio3::Prio3},
};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::{
    collections::{hash_map::Entry, HashMap},
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration as StdDuration,
};
use tokio::{spawn, sync::Mutex, task::JoinHandle};
use warp::{hyper::StatusCode, reply::Response, Filter, Reply};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AddTaskRequest {
    task_id: String,
    leader: Url,
    vdaf: VdafObject,
    collector_authentication_token: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AddTaskResponse {
    status: &'static str,
    #[serde(default)]
    error: Option<String>,
    collector_hpke_config: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CollectStartRequest {
    task_id: String,
    agg_param: String,
    batch_interval_start: u64,
    batch_interval_duration: u64,
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

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum AggregationResult {
    Number(NumberAsString<u128>),
    NumberVec(Vec<NumberAsString<u128>>),
}

#[derive(Debug, Serialize)]
struct CollectPollResponse {
    status: &'static str,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    result: Option<AggregationResult>,
}

struct TaskState {
    private_key: HpkePrivateKey,
    hpke_config: HpkeConfig,
    leader_url: Url,
    vdaf: VdafObject,
    auth_token: AuthenticationToken,
}

/// A collect job handle.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Handle(String);

impl Handle {
    fn generate() -> Handle {
        let randomness = rand::random::<[u8; 32]>();
        Handle(base64::encode_config(randomness, URL_SAFE_NO_PAD))
    }
}

enum CollectJobState {
    InProgress(Option<JoinHandle<anyhow::Result<AggregationResult>>>),
    Completed(AggregationResult),
    Error,
}

async fn handle_add_task(
    tasks: &Mutex<HashMap<TaskId, TaskState>>,
    keyring: &Mutex<HpkeConfigRegistry>,
    request: AddTaskRequest,
) -> anyhow::Result<HpkeConfig> {
    let task_id_bytes = base64::decode_config(request.task_id, base64::URL_SAFE_NO_PAD)
        .context("invalid base64url content in \"taskId\"")?;
    let task_id = TaskId::get_decoded(&task_id_bytes).context("invalid length of TaskId")?;

    let mut tasks_guard = tasks.lock().await;
    let entry = tasks_guard.entry(task_id);
    if let Entry::Occupied(_) = &entry {
        return Err(anyhow::anyhow!("cannot add a task with a duplicate ID"));
    }

    let (hpke_config, private_key) = keyring.lock().await.get_random_keypair();

    entry.or_insert(TaskState {
        private_key,
        hpke_config: hpke_config.clone(),
        leader_url: request.leader,
        vdaf: request.vdaf,
        auth_token: AuthenticationToken::from(request.collector_authentication_token.into_bytes()),
    });

    Ok(hpke_config)
}

async fn handle_collect_generic<V>(
    http_client: &reqwest::Client,
    collector_params: CollectorParameters,
    batch_interval: Interval,
    vdaf: V,
    agg_param_encoded: &[u8],
    convert_fn: impl Fn(V::AggregateResult) -> AggregationResult + Send + 'static,
) -> anyhow::Result<JoinHandle<anyhow::Result<AggregationResult>>>
where
    V: vdaf::Collector + Send + Sync + 'static,
    V::AggregationParam: Send + Sync + 'static,
    for<'a> Vec<u8>: From<&'a V::AggregateShare>,
{
    let collector = Collector::new(collector_params, vdaf, http_client);
    let agg_param = V::AggregationParam::get_decoded(agg_param_encoded)?;
    let handle = spawn(async move {
        let vdaf_result = collector.collect(batch_interval, &agg_param).await?;
        let enum_result = convert_fn(vdaf_result);
        Ok(enum_result)
    });
    Ok(handle)
}

async fn handle_collect_start(
    http_client: &reqwest::Client,
    tasks: &Mutex<HashMap<TaskId, TaskState>>,
    collect_jobs: &Mutex<HashMap<Handle, CollectJobState>>,
    request: CollectStartRequest,
) -> anyhow::Result<Handle> {
    let task_id_bytes = base64::decode_config(request.task_id, URL_SAFE_NO_PAD)
        .context("invalid base64url content in \"taskId\"")?;
    let task_id = TaskId::get_decoded(&task_id_bytes).context("invalid length of TaskId")?;
    let agg_param = base64::decode_config(request.agg_param, URL_SAFE_NO_PAD)
        .context("invalid base64url content in \"aggParam\"")?;
    let batch_interval = Interval::new(
        Time::from_seconds_since_epoch(request.batch_interval_start),
        Duration::from_seconds(request.batch_interval_duration),
    )
    .context("invalid batch interval specification")?;

    let tasks_guard = tasks.lock().await;
    let task_state = tasks_guard
        .get(&task_id)
        .context("task was not added before being used in a collect request")?;

    let collector_params = CollectorParameters::new(
        task_id,
        task_state.leader_url.clone(),
        task_state.auth_token.clone(),
        task_state.hpke_config.clone(),
        task_state.private_key.clone(),
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

    let vdaf_instance = task_state.vdaf.clone().into();
    let task_handle = match vdaf_instance {
        VdafInstance::Real(janus_core::task::VdafInstance::Prio3Aes128Count {}) => {
            let vdaf =
                Prio3::new_aes128_count(2).context("failed to construct Prio3Aes128Count VDAF")?;
            handle_collect_generic(
                http_client,
                collector_params,
                batch_interval,
                vdaf,
                &agg_param,
                |result| AggregationResult::Number(NumberAsString(result.into())),
            )
            .await?
        }
        VdafInstance::Real(janus_core::task::VdafInstance::Prio3Aes128CountVec { length }) => {
            let vdaf = Prio3::new_aes128_count_vec_multithreaded(2, length)
                .context("failed to construct Prio3Aes128CountVec VDAF")?;
            handle_collect_generic(
                http_client,
                collector_params,
                batch_interval,
                vdaf,
                &agg_param,
                |result| {
                    let converted = result.into_iter().map(NumberAsString).collect();
                    AggregationResult::NumberVec(converted)
                },
            )
            .await?
        }
        VdafInstance::Real(janus_core::task::VdafInstance::Prio3Aes128Sum { bits }) => {
            let vdaf = Prio3::new_aes128_sum(2, bits)
                .context("failed to construct Prio3Aes128Sum VDAF")?;
            handle_collect_generic(
                http_client,
                collector_params,
                batch_interval,
                vdaf,
                &agg_param,
                |result| AggregationResult::Number(NumberAsString(result)),
            )
            .await?
        }
        VdafInstance::Real(janus_core::task::VdafInstance::Prio3Aes128Histogram {
            ref buckets,
        }) => {
            let vdaf = Prio3::new_aes128_histogram(2, buckets)
                .context("failed to construct Prio3Aes128Histogram VDAF")?;
            handle_collect_generic(
                http_client,
                collector_params,
                batch_interval,
                vdaf,
                &agg_param,
                |result| {
                    let converted = result.into_iter().map(NumberAsString).collect();
                    AggregationResult::NumberVec(converted)
                },
            )
            .await?
        }
        _ => panic!("Unsupported VDAF: {:?}", vdaf_instance),
    };

    let mut collect_jobs_guard = collect_jobs.lock().await;
    let handle = loop {
        let handle = Handle::generate();
        match collect_jobs_guard.entry(handle.clone()) {
            Entry::Occupied(_) => continue,
            entry @ Entry::Vacant(_) => {
                entry.or_insert(CollectJobState::InProgress(Some(task_handle)));
                break handle;
            }
        }
    };

    Ok(handle)
}

async fn handle_collect_poll(
    collect_jobs: &Mutex<HashMap<Handle, CollectJobState>>,
    request: CollectPollRequest,
) -> anyhow::Result<Option<AggregationResult>> {
    let mut collect_jobs_guard = collect_jobs.lock().await;
    let collect_job_state_entry = collect_jobs_guard.entry(Handle(request.handle.clone()));
    match collect_job_state_entry {
        Entry::Occupied(mut occupied_entry) => match occupied_entry.get_mut() {
            CollectJobState::InProgress(join_handle_opt) => {
                if join_handle_opt.as_ref().unwrap().is_finished() {
                    // Awaiting on the JoinHandle requires owning it. We take it out of the Option,
                    // and ensure that a different enum variant is stored over it before dropping
                    // the lock on the HashMap.
                    let taken_handle = join_handle_opt.take().unwrap();
                    let task_result = taken_handle.await;
                    let collect_result = match task_result {
                        Ok(collect_result) => collect_result,
                        Err(e) => {
                            occupied_entry.insert(CollectJobState::Error);
                            return Err(e).context("panic while handling collection");
                        }
                    };
                    match collect_result {
                        Ok(agg_result) => {
                            occupied_entry.insert(CollectJobState::Completed(agg_result.clone()));
                            Ok(Some(agg_result))
                        }
                        Err(e) => {
                            occupied_entry.insert(CollectJobState::Error);
                            Err(e)
                        }
                    }
                } else {
                    Ok(None)
                }
            }
            CollectJobState::Completed(ref agg_result) => Ok(Some(agg_result.clone())),
            CollectJobState::Error => Err(anyhow::anyhow!(
                "collection previously resulted in an error"
            )),
        },
        Entry::Vacant(_) => Err(anyhow::anyhow!(
            "did not recognize handle in collect_poll request"
        )),
    }
}

fn make_filter() -> anyhow::Result<impl Filter<Extract = (Response,)> + Clone> {
    let http_client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let tasks: Arc<Mutex<HashMap<TaskId, TaskState>>> = Arc::new(Mutex::new(HashMap::new()));
    let collect_jobs: Arc<Mutex<HashMap<Handle, CollectJobState>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let keyring = Arc::new(Mutex::new(HpkeConfigRegistry::new()));

    let ready_filter = warp::path!("ready").map(|| {
        warp::reply::with_status(warp::reply::json(&serde_json::json!({})), StatusCode::OK)
            .into_response()
    });
    let add_task_filter = warp::path!("add_task").and(warp::body::json()).then({
        let tasks = Arc::clone(&tasks);
        let keyring = Arc::clone(&keyring);
        move |request: AddTaskRequest| {
            let tasks = Arc::clone(&tasks);
            let keyring = Arc::clone(&keyring);
            async move {
                let response = match handle_add_task(&tasks, &keyring, request).await {
                    Ok(collector_hpke_config) => AddTaskResponse {
                        status: SUCCESS,
                        error: None,
                        collector_hpke_config: Some(base64::encode_config(
                            collector_hpke_config.get_encoded(),
                            URL_SAFE_NO_PAD,
                        )),
                    },
                    Err(e) => AddTaskResponse {
                        status: ERROR,
                        error: Some(format!("{:?}", e)),
                        collector_hpke_config: None,
                    },
                };
                warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
                    .into_response()
            }
        }
    });
    let collect_start_filter =
        warp::path!("collect_start").and(warp::body::json()).then({
            let tasks = Arc::clone(&tasks);
            let collect_jobs = Arc::clone(&collect_jobs);
            move |request: CollectStartRequest| {
                let http_client = http_client.clone();
                let tasks = Arc::clone(&tasks);
                let collect_jobs = Arc::clone(&collect_jobs);
                async move {
                    let response =
                        match handle_collect_start(&http_client, &tasks, &collect_jobs, request)
                            .await
                        {
                            Ok(handle) => CollectStartResponse {
                                status: SUCCESS,
                                error: None,
                                handle: Some(handle.0),
                            },
                            Err(e) => CollectStartResponse {
                                status: ERROR,
                                error: Some(format!("{:?}", e)),
                                handle: None,
                            },
                        };
                    warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
                        .into_response()
                }
            }
        });
    let collect_poll_filter = warp::path!("collect_poll").and(warp::body::json()).then({
        move |request: CollectPollRequest| {
            let collect_jobs = Arc::clone(&collect_jobs);
            async move {
                let response = match handle_collect_poll(&collect_jobs, request).await {
                    Ok(Some(result)) => CollectPollResponse {
                        status: COMPLETE,
                        error: None,
                        result: Some(result),
                    },
                    Ok(None) => CollectPollResponse {
                        status: IN_PROGRESS,
                        error: None,
                        result: None,
                    },
                    Err(e) => CollectPollResponse {
                        status: ERROR,
                        error: Some(format!("{:?}", e)),
                        result: None,
                    },
                };
                warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
                    .into_response()
            }
        }
    });

    Ok(warp::path!("internal" / "test" / ..).and(warp::post()).and(
        ready_filter
            .or(add_task_filter)
            .unify()
            .or(collect_start_filter)
            .unify()
            .or(collect_poll_filter)
            .unify(),
    ))
}

fn app() -> clap::Command<'static> {
    Command::new("Janus interoperation test collector").arg(
        Arg::new("port")
            .long("port")
            .short('p')
            .default_value("8080")
            .help("Port number to listen on."),
    )
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    install_tracing_subscriber()?;
    let matches = app().get_matches();
    let port = matches.value_of_t::<u16>("port")?;
    let filter = make_filter()?;
    let server = warp::serve(filter);
    server
        .bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, port)))
        .await;
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
