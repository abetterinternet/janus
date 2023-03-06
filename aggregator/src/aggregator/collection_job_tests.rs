use crate::aggregator::{aggregator_filter, Config};
use http::{header::CONTENT_TYPE, StatusCode};
use hyper::body;
use janus_aggregator_core::{
    datastore::{
        models::{
            AggregationJob, AggregationJobState, BatchAggregation, CollectionJobState,
            LeaderStoredReport, ReportAggregation, ReportAggregationState,
        },
        test_util::{ephemeral_datastore, EphemeralDatastore},
        Datastore,
    },
    task::{test_util::TaskBuilder, QueryType, Task},
};
use janus_core::{
    hpke::{
        self, test_util::generate_test_hpke_config_and_private_key, HpkeApplicationInfo,
        HpkeKeypair, Label,
    },
    task::{AuthenticationToken, VdafInstance},
    test_util::{dummy_vdaf, install_test_trace_subscriber},
    time::{Clock, IntervalExt, MockClock},
};
use janus_messages::{
    query_type::{FixedSize, QueryType as QueryTypeTrait, TimeInterval},
    AggregateShareAad, BatchId, BatchSelector, Collection, CollectionJobId, CollectionReq,
    Duration, FixedSizeQuery, Interval, Query, ReportIdChecksum, Role, Time,
};
use prio::codec::{Decode, Encode};
use rand::random;
use serde_json::json;
use std::sync::Arc;
use warp::{
    filters::BoxedFilter,
    reply::{Reply, Response},
};

pub(crate) struct CollectionJobTestCase<R> {
    pub(super) task: Task,
    clock: MockClock,
    pub(super) collector_hpke_keypair: HpkeKeypair,
    pub(super) filter: BoxedFilter<(R,)>,
    pub(super) datastore: Arc<Datastore<MockClock>>,
    _ephemeral_datastore: EphemeralDatastore,
}

impl<R: Reply + 'static> CollectionJobTestCase<R> {
    pub(super) async fn put_collection_job_with_auth_token<Q: QueryTypeTrait>(
        &self,
        collection_job_id: &CollectionJobId,
        request: &CollectionReq<Q>,
        auth_token: Option<&AuthenticationToken>,
    ) -> Response {
        let mut builder = warp::test::request().method("PUT").path(
            self.task
                .collection_job_uri(collection_job_id)
                .unwrap()
                .path(),
        );
        if let Some(token) = auth_token {
            builder = builder.header("DAP-Auth-Token", token.as_bytes())
        }

        builder
            .header(CONTENT_TYPE, CollectionReq::<TimeInterval>::MEDIA_TYPE)
            .body(request.get_encoded())
            .filter(&self.filter)
            .await
            .unwrap()
            .into_response()
    }

    pub(super) async fn put_collection_job<Q: QueryTypeTrait>(
        &self,
        collection_job_id: &CollectionJobId,
        request: &CollectionReq<Q>,
    ) -> Response {
        self.put_collection_job_with_auth_token(
            collection_job_id,
            request,
            Some(self.task.primary_collector_auth_token()),
        )
        .await
    }

    pub(super) async fn post_collection_job_with_auth_token(
        &self,
        collection_job_id: &CollectionJobId,
        auth_token: Option<&AuthenticationToken>,
    ) -> Response {
        let mut builder = warp::test::request().method("POST").path(
            self.task
                .collection_job_uri(collection_job_id)
                .unwrap()
                .path(),
        );
        if let Some(token) = auth_token {
            builder = builder.header("DAP-Auth-Token", token.as_bytes())
        }
        builder.filter(&self.filter).await.unwrap().into_response()
    }

    pub(super) async fn post_collection_job(
        &self,
        collection_job_id: &CollectionJobId,
    ) -> Response {
        self.post_collection_job_with_auth_token(
            collection_job_id,
            Some(self.task.primary_collector_auth_token()),
        )
        .await
    }
}

pub(crate) async fn setup_collection_job_test_case(
    role: Role,
    query_type: QueryType,
) -> CollectionJobTestCase<impl Reply + 'static> {
    install_test_trace_subscriber();

    let collector_hpke_keypair = generate_test_hpke_config_and_private_key();
    let task = TaskBuilder::new(query_type, VdafInstance::Fake, role)
        .with_collector_hpke_config(collector_hpke_keypair.config().clone())
        .build();
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()));

    datastore.put_task(&task).await.unwrap();

    let filter = aggregator_filter(
        Arc::clone(&datastore),
        clock.clone(),
        Config {
            batch_aggregation_shard_count: 32,
            ..Default::default()
        },
    )
    .unwrap();

    CollectionJobTestCase {
        task,
        clock,
        collector_hpke_keypair,
        filter,
        datastore,
        _ephemeral_datastore: ephemeral_datastore,
    }
}

async fn setup_fixed_size_current_batch_collection_job_test_case() -> (
    CollectionJobTestCase<impl Reply + 'static>,
    BatchId,
    BatchId,
    Interval,
) {
    let test_case =
        setup_collection_job_test_case(Role::Leader, QueryType::FixedSize { max_batch_size: 10 })
            .await;

    // Fill the datastore with the necessary data so that there is are two outstanding batches to be
    // collected.
    let batch_id_1 = random();
    let batch_id_2 = random();
    let time = test_case.clock.now();
    let interval = Interval::new(
        time,
        Duration::from_seconds(test_case.task.time_precision().as_seconds() / 2),
    )
    .unwrap();

    test_case
        .datastore
        .run_tx(|tx| {
            let task = test_case.task.clone();
            Box::pin(async move {
                for batch_id in [batch_id_1, batch_id_2] {
                    let aggregation_job_id = random();
                    tx.put_aggregation_job::<0, FixedSize, dummy_vdaf::Vdaf>(&AggregationJob::new(
                        *task.id(),
                        aggregation_job_id,
                        dummy_vdaf::AggregationParam::default(),
                        batch_id,
                        interval,
                        AggregationJobState::Finished,
                    ))
                    .await
                    .unwrap();

                    for ord in 0..task.min_batch_size() + 1 {
                        let report = LeaderStoredReport::new_dummy(*task.id(), time);
                        tx.put_client_report(&dummy_vdaf::Vdaf::new(), &report)
                            .await
                            .unwrap();

                        tx.put_report_aggregation::<0, dummy_vdaf::Vdaf>(&ReportAggregation::new(
                            *task.id(),
                            aggregation_job_id,
                            *report.metadata().id(),
                            time,
                            ord,
                            ReportAggregationState::Finished(dummy_vdaf::OutputShare()),
                        ))
                        .await
                        .unwrap();
                    }

                    tx.put_batch_aggregation::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &BatchAggregation::new(
                            *task.id(),
                            batch_id,
                            dummy_vdaf::AggregationParam::default(),
                            0,
                            dummy_vdaf::AggregateShare(0),
                            task.min_batch_size() + 1,
                            interval,
                            ReportIdChecksum::default(),
                        ),
                    )
                    .await
                    .unwrap();

                    tx.put_outstanding_batch(task.id(), &batch_id)
                        .await
                        .unwrap();
                }

                Ok(())
            })
        })
        .await
        .unwrap();

    (test_case, batch_id_1, batch_id_2, interval)
}

#[tokio::test]
async fn collection_job_success_fixed_size() {
    // This test drives two current batch collection jobs to completion, verifying that distinct
    // batch IDs are collected each time. Then, we attempt to collect another current batch, which
    // must fail as there are no more outstanding batches.
    let (test_case, batch_id_1, batch_id_2, spanned_interval) =
        setup_fixed_size_current_batch_collection_job_test_case().await;

    let mut saw_batch_id_1 = false;
    let mut saw_batch_id_2 = false;
    let leader_aggregate_share = dummy_vdaf::AggregateShare(0);
    let helper_aggregate_share = dummy_vdaf::AggregateShare(1);
    let request = CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::CurrentBatch),
        dummy_vdaf::AggregationParam::default().get_encoded(),
    );

    for _ in 0..2 {
        let collection_job_id: CollectionJobId = random();

        let response = test_case
            .put_collection_job(&collection_job_id, &request)
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::CREATED);

        let collection_job_response = test_case.post_collection_job(&collection_job_id).await;
        assert_eq!(collection_job_response.status(), StatusCode::ACCEPTED);

        // Update the collection job with the aggregate shares. collection job should now be complete.
        let batch_id = test_case
            .datastore
            .run_tx(|tx| {
                let task = test_case.task.clone();
                let helper_aggregate_share_bytes: Vec<u8> = (&helper_aggregate_share).into();
                Box::pin(async move {
                    let collection_job = tx
                        .get_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(&collection_job_id)
                        .await
                        .unwrap()
                        .unwrap();
                    let batch_id = *collection_job.batch_identifier();

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

                    tx.update_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &collection_job.with_state(CollectionJobState::Finished {
                            report_count: task.min_batch_size() + 1,
                            encrypted_helper_aggregate_share,
                            leader_aggregate_share,
                        }),
                    )
                    .await
                    .unwrap();

                    Ok(batch_id)
                })
            })
            .await
            .unwrap();

        if batch_id.eq(&batch_id_1) {
            saw_batch_id_1 = true;
        } else if batch_id.eq(&batch_id_2) {
            saw_batch_id_2 = true;
        } else {
            panic!("unexpected batch ID");
        }

        let (parts, body) = test_case
            .post_collection_job(&collection_job_id)
            .await
            .into_parts();

        assert_eq!(parts.status, StatusCode::OK);
        assert_eq!(
            parts.headers.get(CONTENT_TYPE).unwrap(),
            Collection::<FixedSize>::MEDIA_TYPE
        );
        let body_bytes = body::to_bytes(body).await.unwrap();
        let collect_resp = Collection::<FixedSize>::get_decoded(body_bytes.as_ref()).unwrap();

        assert_eq!(
            collect_resp.report_count(),
            test_case.task.min_batch_size() + 1
        );
        assert_eq!(
            collect_resp.interval(),
            &spanned_interval
                .align_to_time_precision(test_case.task.time_precision())
                .unwrap(),
        );
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
            dummy_vdaf::AggregateShare::try_from(decrypted_leader_aggregate_share.as_ref())
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
            dummy_vdaf::AggregateShare::try_from(decrypted_helper_aggregate_share.as_ref())
                .unwrap()
        );
    }

    assert!(saw_batch_id_1 && saw_batch_id_2);

    // We have run the two ready batches to completion. Further attempts to collect current batch
    // ought to fail.
    let collection_job_id: CollectionJobId = random();

    let mut response = test_case
        .put_collection_job(&collection_job_id, &request)
        .await
        .into_response();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let problem_details: serde_json::Value =
        serde_json::from_slice(&body::to_bytes(response.body_mut()).await.unwrap()).unwrap();
    assert_eq!(
        problem_details,
        json!({
            "status": StatusCode::BAD_REQUEST.as_u16(),
            "type": "urn:ietf:params:ppm:dap:error:batchInvalid",
            "title": "The batch implied by the query is invalid.",
            "taskid": format!("{}", test_case.task.id()),
        }),
    );
}

#[tokio::test]
async fn collection_job_put_idempotence_time_interval() {
    let test_case = setup_collection_job_test_case(Role::Leader, QueryType::TimeInterval).await;

    let collection_job_id = random();
    let request = CollectionReq::new(
        Query::new_time_interval(
            Interval::new(
                Time::from_seconds_since_epoch(0),
                *test_case.task.time_precision(),
            )
            .unwrap(),
        ),
        dummy_vdaf::AggregationParam::default().get_encoded(),
    );

    for _ in 0..2 {
        let response = test_case
            .put_collection_job(&collection_job_id, &request)
            .await;
        assert_eq!(response.status(), StatusCode::CREATED);
    }

    // There should only be a single collection job despite two successful PUTs
    test_case
        .datastore
        .run_tx(|tx| {
            let task_id = *test_case.task.id();
            Box::pin(async move {
                let collection_jobs = tx
                    .get_collection_jobs_for_task::<0, TimeInterval, dummy_vdaf::Vdaf>(&task_id)
                    .await
                    .unwrap();
                assert_eq!(collection_jobs.len(), 1);
                assert_eq!(collection_jobs[0].collection_job_id(), &collection_job_id);

                Ok(())
            })
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn collection_job_put_idempotence_time_interval_mutate_time_interval() {
    let test_case = setup_collection_job_test_case(Role::Leader, QueryType::TimeInterval).await;

    let collection_job_id = random();
    let request = CollectionReq::new(
        Query::new_time_interval(
            Interval::new(
                Time::from_seconds_since_epoch(0),
                *test_case.task.time_precision(),
            )
            .unwrap(),
        ),
        dummy_vdaf::AggregationParam::default().get_encoded(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &request)
        .await;
    assert_eq!(response.status(), StatusCode::CREATED);

    let mutated_request = CollectionReq::new(
        Query::new_time_interval(
            Interval::new(
                Time::from_seconds_since_epoch(test_case.task.time_precision().as_seconds()),
                *test_case.task.time_precision(),
            )
            .unwrap(),
        ),
        dummy_vdaf::AggregationParam::default().get_encoded(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &mutated_request)
        .await;
    assert_eq!(response.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn collection_job_put_idempotence_time_interval_mutate_aggregation_param() {
    let test_case = setup_collection_job_test_case(Role::Leader, QueryType::TimeInterval).await;

    let collection_job_id = random();
    let request = CollectionReq::new(
        Query::new_time_interval(
            Interval::new(
                Time::from_seconds_since_epoch(0),
                *test_case.task.time_precision(),
            )
            .unwrap(),
        ),
        dummy_vdaf::AggregationParam(0).get_encoded(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &request)
        .await;
    assert_eq!(response.status(), StatusCode::CREATED);

    let mutated_request = CollectionReq::new(
        Query::new_time_interval(
            Interval::new(
                Time::from_seconds_since_epoch(0),
                *test_case.task.time_precision(),
            )
            .unwrap(),
        ),
        dummy_vdaf::AggregationParam(1).get_encoded(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &mutated_request)
        .await;
    assert_eq!(response.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn collection_job_put_idempotence_fixed_size_current_batch() {
    let (test_case, batch_id_1, batch_id_2, _) =
        setup_fixed_size_current_batch_collection_job_test_case().await;

    let collection_job_id = random();
    let request = CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::CurrentBatch),
        dummy_vdaf::AggregationParam(0).get_encoded(),
    );
    let mut seen_batch_id = None;

    for _ in 0..2 {
        let response = test_case
            .put_collection_job(&collection_job_id, &request)
            .await;

        assert_eq!(response.status(), StatusCode::CREATED);

        // Make sure that there is only ever a single collection job, and that it uses the same
        // batch ID after each PUT
        let batch_id = test_case
            .datastore
            .run_tx(|tx| {
                let task_id = *test_case.task.id();
                Box::pin(async move {
                    let collection_jobs = tx
                        .get_collection_jobs_for_task::<0, FixedSize, dummy_vdaf::Vdaf>(&task_id)
                        .await
                        .unwrap();
                    assert_eq!(collection_jobs.len(), 1);
                    assert_eq!(collection_jobs[0].collection_job_id(), &collection_job_id);
                    assert!(
                        collection_jobs[0].batch_identifier().eq(&batch_id_1)
                            || collection_jobs[0].batch_identifier().eq(&batch_id_2)
                    );

                    Ok(*collection_jobs[0].batch_identifier())
                })
            })
            .await
            .unwrap();
        match seen_batch_id {
            None => seen_batch_id = Some(batch_id),
            Some(seen_batch_id) => assert_eq!(seen_batch_id, batch_id),
        }
    }
}

#[tokio::test]
async fn collection_job_put_idempotence_fixed_size_current_batch_mutate_aggregation_param() {
    let (test_case, _, _, _) = setup_fixed_size_current_batch_collection_job_test_case().await;

    let collection_job_id = random();
    let request = CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::CurrentBatch),
        dummy_vdaf::AggregationParam(0).get_encoded(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &request)
        .await;

    assert_eq!(response.status(), StatusCode::CREATED);

    let mutated_request = CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::CurrentBatch),
        dummy_vdaf::AggregationParam(1).get_encoded(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &mutated_request)
        .await;
    assert_eq!(response.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn collection_job_put_idempotence_fixed_size_by_batch_id() {
    let test_case =
        setup_collection_job_test_case(Role::Leader, QueryType::FixedSize { max_batch_size: 10 })
            .await;

    let collection_job_id = random();
    let request = CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::ByBatchId {
            batch_id: BatchId::try_from([1u8; 32]).unwrap(),
        }),
        dummy_vdaf::AggregationParam(0).get_encoded(),
    );

    for _ in 0..2 {
        let response = test_case
            .put_collection_job(&collection_job_id, &request)
            .await;

        assert_eq!(response.status(), StatusCode::CREATED);
    }
}

#[tokio::test]
async fn collection_job_put_idempotence_fixed_size_by_batch_id_mutate_batch_id() {
    let test_case =
        setup_collection_job_test_case(Role::Leader, QueryType::FixedSize { max_batch_size: 10 })
            .await;

    let collection_job_id = random();
    let request = CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::ByBatchId {
            batch_id: BatchId::try_from([1u8; 32]).unwrap(),
        }),
        dummy_vdaf::AggregationParam(0).get_encoded(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &request)
        .await;

    assert_eq!(response.status(), StatusCode::CREATED);

    let mutated_request = CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::ByBatchId {
            batch_id: BatchId::try_from([2u8; 32]).unwrap(),
        }),
        dummy_vdaf::AggregationParam(0).get_encoded(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &mutated_request)
        .await;
    assert_eq!(response.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn collection_job_put_idempotence_fixed_size_by_batch_id_mutate_aggregation_param() {
    let test_case =
        setup_collection_job_test_case(Role::Leader, QueryType::FixedSize { max_batch_size: 10 })
            .await;

    let collection_job_id = random();
    let request = CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::ByBatchId {
            batch_id: BatchId::try_from([1u8; 32]).unwrap(),
        }),
        dummy_vdaf::AggregationParam(0).get_encoded(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &request)
        .await;

    assert_eq!(response.status(), StatusCode::CREATED);

    let mutated_request = CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::ByBatchId {
            batch_id: BatchId::try_from([1u8; 32]).unwrap(),
        }),
        dummy_vdaf::AggregationParam(1).get_encoded(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &mutated_request)
        .await;
    assert_eq!(response.status(), StatusCode::CONFLICT);
}
