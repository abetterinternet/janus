use crate::{
    git_revision,
    models::{
        AggregatorApiConfig, AggregatorRole, DeleteTaskprovPeerAggregatorReq,
        GetTaskAggregationMetricsResp, GetTaskIdsResp, GetTaskUploadMetricsResp, HpkeConfigResp,
        PatchHpkeConfigReq, PatchTaskReq, PostTaskReq, PostTaskprovPeerAggregatorReq,
        PutHpkeConfigReq, SupportedVdaf, TaskResp, TaskprovPeerAggregatorResp,
    },
    Config, ConnExt, Error,
};
use aws_lc_rs::digest::{digest, SHA256};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use janus_aggregator_core::{
    datastore::{self, Datastore},
    task::{AggregatorTask, AggregatorTaskParameters},
    taskprov::PeerAggregator,
    SecretBytes,
};
use janus_core::{auth_tokens::AuthenticationTokenHash, hpke::HpkeKeypair, time::Clock};
use janus_messages::HpkeConfigId;
use janus_messages::{
    batch_mode::Code as SupportedBatchMode, Duration, HpkeAeadId, HpkeKdfId, HpkeKemId, Role,
    TaskId,
};
use querystring::querify;
use rand::random;
use std::{
    str::FromStr,
    sync::{Arc, LazyLock},
    unreachable,
};
use trillium::{Conn, Status};
use trillium_api::{Json, State};

pub(super) async fn get_config(
    _: &mut Conn,
    State(config): State<Arc<Config>>,
) -> Json<AggregatorApiConfig> {
    static VERSION: LazyLock<String> =
        LazyLock::new(|| format!("{}-{}", env!("CARGO_PKG_VERSION"), git_revision()));

    Json(AggregatorApiConfig {
        protocol: "DAP-09",
        dap_url: config.public_dap_url.clone(),
        role: AggregatorRole::Either,
        vdafs: Vec::from([
            SupportedVdaf::Prio3Count,
            SupportedVdaf::Prio3Sum,
            SupportedVdaf::Prio3Histogram,
            SupportedVdaf::Prio3SumVec,
        ]),
        batch_modes: Vec::from([
            SupportedBatchMode::TimeInterval,
            SupportedBatchMode::LeaderSelected,
        ]),
        features: &[
            "TokenHash",
            "UploadMetrics",
            "TimeBucketedLeaderSelected",
            "PureDpDiscreteLaplace",
        ],
        software_name: "Janus",
        software_version: &VERSION,
    })
}

pub(super) async fn get_task_ids<C: Clock>(
    conn: &mut Conn,
    State(ds): State<Arc<Datastore<C>>>,
) -> Result<Json<GetTaskIdsResp>, Error> {
    const PAGINATION_TOKEN_KEY: &str = "pagination_token";
    let lower_bound = querify(conn.querystring())
        .into_iter()
        .find(|&(k, _)| k == PAGINATION_TOKEN_KEY)
        .map(|(_, v)| TaskId::from_str(v))
        .transpose()
        .map_err(|err| Error::BadRequest(format!("Couldn't parse pagination_token: {:?}", err)))?;

    let task_ids = ds
        .run_tx("get_task_ids", |tx| {
            Box::pin(async move { tx.get_task_ids(lower_bound).await })
        })
        .await?;
    let pagination_token = task_ids.last().cloned();

    Ok(Json(GetTaskIdsResp {
        task_ids,
        pagination_token,
    }))
}

pub(super) async fn post_task<C: Clock>(
    _: &mut Conn,
    (State(ds), Json(req)): (State<Arc<Datastore<C>>>, Json<PostTaskReq>),
) -> Result<Json<TaskResp>, Error> {
    if !matches!(req.role, Role::Leader | Role::Helper) {
        return Err(Error::BadRequest(format!("invalid role {}", req.role)));
    }

    let vdaf_verify_key_bytes = URL_SAFE_NO_PAD
        .decode(&req.vdaf_verify_key)
        .map_err(|err| {
            Error::BadRequest(format!("Invalid base64 value for vdaf_verify_key: {err}"))
        })?;
    if vdaf_verify_key_bytes.len() != req.vdaf.verify_key_length() {
        return Err(Error::BadRequest(format!(
            "Wrong VDAF verify key length, expected {}, got {}",
            req.vdaf.verify_key_length(),
            vdaf_verify_key_bytes.len()
        )));
    }

    // DAP recommends deriving the task ID from the VDAF verify key. We deterministically obtain a
    // 32 byte task ID by taking SHA-256(VDAF verify key).
    // https://datatracker.ietf.org/doc/html/draft-ietf-ppm-dap-04#name-verification-key-requiremen
    let task_id = TaskId::try_from(digest(&SHA256, &vdaf_verify_key_bytes).as_ref())
        .map_err(|err| Error::Internal(err.to_string()))?;

    let vdaf_verify_key = SecretBytes::new(vdaf_verify_key_bytes);

    let (aggregator_auth_token, aggregator_parameters) = match req.role {
        Role::Leader => {
            let aggregator_auth_token = req.aggregator_auth_token.ok_or_else(|| {
                Error::BadRequest(
                    "aggregator acting in leader role must be provided an aggregator auth token"
                        .to_string(),
                )
            })?;
            let collector_auth_token_hash = req.collector_auth_token_hash.ok_or_else(|| {
                Error::BadRequest(
                    "aggregator acting in leader role must be provided a collector auth token hash"
                        .to_string(),
                )
            })?;
            (
                None,
                AggregatorTaskParameters::Leader {
                    aggregator_auth_token,
                    collector_auth_token_hash,
                    collector_hpke_config: req.collector_hpke_config,
                },
            )
        }

        Role::Helper => {
            if req.aggregator_auth_token.is_some() {
                return Err(Error::BadRequest(
                    "aggregator acting in helper role cannot be given an aggregator auth token"
                        .to_string(),
                ));
            }

            let aggregator_auth_token = random();
            let aggregator_auth_token_hash = AuthenticationTokenHash::from(&aggregator_auth_token);
            (
                Some(aggregator_auth_token),
                AggregatorTaskParameters::Helper {
                    aggregator_auth_token_hash,
                    collector_hpke_config: req.collector_hpke_config,
                    aggregation_mode: req.aggregation_mode.ok_or_else(|| {
                        Error::BadRequest(
                            "aggregator acting in helper role must be provided an aggregation mode"
                                .to_string(),
                        )
                    })?,
                },
            )
        }

        _ => unreachable!(),
    };

    let task = Arc::new(
        AggregatorTask::new(
            task_id,
            /* peer_aggregator_endpoint */ req.peer_aggregator_endpoint,
            /* batch_mode */ req.batch_mode,
            /* vdaf */ req.vdaf,
            vdaf_verify_key,
            /* task_start */ req.task_start,
            /* task_end */ req.task_end,
            /* report_expiry_age */
            Some(Duration::from_seconds(3600 * 24 * 7 * 2)), // 2 weeks
            /* min_batch_size */ req.min_batch_size,
            /* time_precision */ req.time_precision,
            /* tolerable_clock_skew */
            Duration::from_seconds(60), // 1 minute,
            aggregator_parameters,
        )
        .map_err(|err| Error::BadRequest(format!("Error constructing task: {err}")))?,
    );

    ds.run_tx("post_task", |tx| {
        let task = Arc::clone(&task);
        Box::pin(async move {
            if let Some(existing_task) = tx.get_aggregator_task(task.id()).await? {
            // Check whether the existing task in the DB corresponds to the incoming task, ignoring
            // those fields that are randomly generated.
            if existing_task.peer_aggregator_endpoint() == task.peer_aggregator_endpoint()
                && existing_task.batch_mode() == task.batch_mode()
                && existing_task.vdaf() == task.vdaf()
                && existing_task.opaque_vdaf_verify_key() == task.opaque_vdaf_verify_key()
                && existing_task.role() == task.role()
                && existing_task.task_start() == task.task_start()
                && existing_task.task_end() == task.task_end()
                && existing_task.min_batch_size() == task.min_batch_size()
                && existing_task.time_precision() == task.time_precision()
                && existing_task.collector_hpke_config() == task.collector_hpke_config() {
                    return Ok(())
                }

                let err = Error::Conflict(
                    "task with same VDAF verify key and task ID already exists with different parameters".to_string(),
                );
                return Err(datastore::Error::User(err.into()));
            }

            tx.put_aggregator_task(&task).await
        })
    })
    .await?;

    let mut task_resp =
        TaskResp::try_from(task.as_ref()).map_err(|err| Error::Internal(err.to_string()))?;

    // When creating a new task in the helper, we must put the unhashed aggregator auth token in the
    // response so that divviup-api can later provide it to the leader, but the helper doesn't store
    // the unhashed token and can't later provide it.
    task_resp.aggregator_auth_token = aggregator_auth_token;

    Ok(Json(task_resp))
}

pub(super) async fn get_task<C: Clock>(
    conn: &mut Conn,
    State(ds): State<Arc<Datastore<C>>>,
) -> Result<Json<TaskResp>, Error> {
    let task_id = conn.task_id_param()?;

    let task = ds
        .run_tx("get_task", |tx| {
            Box::pin(async move { tx.get_aggregator_task(&task_id).await })
        })
        .await?
        .ok_or(Error::NotFound)?;

    Ok(Json(
        TaskResp::try_from(&task).map_err(|err| Error::Internal(err.to_string()))?,
    ))
}

pub(super) async fn delete_task<C: Clock>(
    conn: &mut Conn,
    State(ds): State<Arc<Datastore<C>>>,
) -> Result<Status, Error> {
    let task_id = conn.task_id_param()?;
    match ds
        .run_tx("delete_task", |tx| {
            Box::pin(async move { tx.delete_task(&task_id).await })
        })
        .await
    {
        Ok(_) | Err(datastore::Error::MutationTargetNotFound) => Ok(Status::NoContent),
        Err(err) => Err(err.into()),
    }
}

pub(super) async fn patch_task<C: Clock>(
    conn: &mut Conn,
    (State(ds), Json(req)): (State<Arc<Datastore<C>>>, Json<PatchTaskReq>),
) -> Result<Json<TaskResp>, Error> {
    let task_id = conn.task_id_param()?;
    let task = ds
        .run_tx("patch_task", |tx| {
            Box::pin(async move {
                if let Some(task_end) = req.task_end {
                    tx.update_task_end(&task_id, task_end.as_ref()).await?;
                }
                tx.get_aggregator_task(&task_id).await
            })
        })
        .await?
        .ok_or(Error::NotFound)?;

    Ok(Json(
        TaskResp::try_from(&task).map_err(|err| Error::Internal(err.to_string()))?,
    ))
}

pub(super) async fn get_task_upload_metrics<C: Clock>(
    conn: &mut Conn,
    State(ds): State<Arc<Datastore<C>>>,
) -> Result<Json<GetTaskUploadMetricsResp>, Error> {
    let task_id = conn.task_id_param()?;
    Ok(Json(GetTaskUploadMetricsResp(
        ds.run_tx("get_task_upload_metrics", |tx| {
            Box::pin(async move { tx.get_task_upload_counter(&task_id).await })
        })
        .await?
        .ok_or(Error::NotFound)?,
    )))
}

pub(super) async fn get_task_aggregation_metrics<C: Clock>(
    conn: &mut Conn,
    State(ds): State<Arc<Datastore<C>>>,
) -> Result<Json<GetTaskAggregationMetricsResp>, Error> {
    let task_id = conn.task_id_param()?;
    Ok(Json(GetTaskAggregationMetricsResp(
        ds.run_tx("get_task_aggregation_metrics", |tx| {
            Box::pin(async move { tx.get_task_aggregation_counter(&task_id).await })
        })
        .await?
        .ok_or(Error::NotFound)?,
    )))
}

pub(super) async fn get_hpke_configs<C: Clock>(
    _: &mut Conn,
    State(ds): State<Arc<Datastore<C>>>,
) -> Result<Json<Vec<HpkeConfigResp>>, Error> {
    Ok(Json(
        ds.run_tx("get_hpke_configs", |tx| {
            Box::pin(async move { tx.get_hpke_keypairs().await })
        })
        .await?
        .into_iter()
        .map(HpkeConfigResp::from)
        .collect::<Vec<_>>(),
    ))
}

pub(super) async fn get_hpke_config<C: Clock>(
    conn: &mut Conn,
    State(ds): State<Arc<Datastore<C>>>,
) -> Result<Json<HpkeConfigResp>, Error> {
    let config_id = conn.hpke_config_id_param()?;
    Ok(Json(HpkeConfigResp::from(
        ds.run_tx("get_hpke_config", |tx| {
            Box::pin(async move { tx.get_hpke_keypair(&config_id).await })
        })
        .await?
        .ok_or(Error::NotFound)?,
    )))
}

pub(super) async fn put_hpke_config<C: Clock>(
    _: &mut Conn,
    (State(ds), Json(req)): (State<Arc<Datastore<C>>>, Json<PutHpkeConfigReq>),
) -> Result<(Status, Json<HpkeConfigResp>), Error> {
    let existing_keypairs = ds
        .run_tx("put_hpke_config_determine_id", |tx| {
            Box::pin(async move { tx.get_hpke_keypairs().await })
        })
        .await?
        .iter()
        .map(|keypair| u8::from(*keypair.hpke_keypair().config().id()))
        .collect::<Vec<_>>();

    let config_id = HpkeConfigId::from(
        (0..=u8::MAX)
            .find(|i| !existing_keypairs.contains(i))
            .ok_or_else(|| {
                Error::Conflict("All possible IDs for HPKE keys have been taken".to_string())
            })?,
    );
    let keypair = HpkeKeypair::generate(
        config_id,
        req.kem_id.unwrap_or(HpkeKemId::X25519HkdfSha256),
        req.kdf_id.unwrap_or(HpkeKdfId::HkdfSha256),
        req.aead_id.unwrap_or(HpkeAeadId::Aes128Gcm),
    )?;

    let inserted_keypair = ds
        .run_tx("put_hpke_config", |tx| {
            let keypair = keypair.clone();
            Box::pin(async move {
                tx.put_hpke_keypair(&keypair).await?;
                tx.get_hpke_keypair(&config_id).await
            })
        })
        .await?
        .ok_or_else(|| Error::Internal("Newly inserted key disappeared".to_string()))?;

    Ok((
        Status::Created,
        Json(HpkeConfigResp::from(inserted_keypair)),
    ))
}

pub(super) async fn patch_hpke_config<C: Clock>(
    conn: &mut Conn,
    (State(ds), Json(req)): (State<Arc<Datastore<C>>>, Json<PatchHpkeConfigReq>),
) -> Result<Status, Error> {
    let config_id = conn.hpke_config_id_param()?;

    ds.run_tx("patch_hpke_keypair", |tx| {
        Box::pin(async move { tx.set_hpke_keypair_state(&config_id, &req.state).await })
    })
    .await?;

    Ok(Status::Ok)
}

pub(super) async fn delete_hpke_config<C: Clock>(
    conn: &mut Conn,
    State(ds): State<Arc<Datastore<C>>>,
) -> Result<Status, Error> {
    let config_id = conn.hpke_config_id_param()?;
    match ds
        .run_tx("delete_hpke_config", |tx| {
            Box::pin(async move { tx.delete_hpke_keypair(&config_id).await })
        })
        .await
    {
        Ok(_) | Err(datastore::Error::MutationTargetNotFound) => Ok(Status::NoContent),
        Err(err) => Err(err.into()),
    }
}

pub(super) async fn get_taskprov_peer_aggregators<C: Clock>(
    _: &mut Conn,
    State(ds): State<Arc<Datastore<C>>>,
) -> Result<Json<Vec<TaskprovPeerAggregatorResp>>, Error> {
    Ok(Json(
        ds.run_tx("get_taskprov_peer_aggregators", |tx| {
            Box::pin(async move { tx.get_taskprov_peer_aggregators().await })
        })
        .await?
        .into_iter()
        .map(TaskprovPeerAggregatorResp::from)
        .collect::<Vec<_>>(),
    ))
}

/// Inserts a new peer aggregator. Insertion is only supported, attempting to modify an existing
/// peer aggregator will fail.
///
/// TODO(1685): Requiring that we delete an existing peer aggregator before we can change it makes
/// token rotation cumbersome and fragile. Since token rotation is the main use case for updating
/// an existing peer aggregator, we will resolve peer aggregator updates in that issue.
pub(super) async fn post_taskprov_peer_aggregator<C: Clock>(
    _: &mut Conn,
    (State(ds), Json(req)): (
        State<Arc<Datastore<C>>>,
        Json<PostTaskprovPeerAggregatorReq>,
    ),
) -> Result<(Status, Json<TaskprovPeerAggregatorResp>), Error> {
    let to_insert = PeerAggregator::new(
        req.endpoint,
        req.peer_role,
        req.aggregation_mode,
        req.verify_key_init,
        req.collector_hpke_config,
        req.report_expiry_age,
        req.tolerable_clock_skew,
        req.aggregator_auth_tokens,
        req.collector_auth_tokens,
    );

    let inserted = ds
        .run_tx("post_taskprov_peer_aggregator", |tx| {
            let to_insert = to_insert.clone();
            Box::pin(async move {
                tx.put_taskprov_peer_aggregator(&to_insert).await?;
                tx.get_taskprov_peer_aggregator(to_insert.endpoint(), to_insert.peer_role())
                    .await
            })
        })
        .await?
        .map(TaskprovPeerAggregatorResp::from)
        .ok_or_else(|| Error::Internal("Newly inserted peer aggregator disappeared".to_string()))?;

    Ok((Status::Created, Json(inserted)))
}

pub(super) async fn delete_taskprov_peer_aggregator<C: Clock>(
    _: &mut Conn,
    (State(ds), Json(req)): (
        State<Arc<Datastore<C>>>,
        Json<DeleteTaskprovPeerAggregatorReq>,
    ),
) -> Result<Status, Error> {
    let res = ds
        .run_tx("delete_taskprov_peer_aggregator", |tx| {
            let req = req.clone();
            Box::pin(async move {
                tx.delete_taskprov_peer_aggregator(&req.endpoint, &req.peer_role)
                    .await
            })
        })
        .await;
    match res {
        Ok(_) | Err(datastore::Error::MutationTargetNotFound) => Ok(Status::NoContent),
        Err(err) => Err(err.into()),
    }
}
