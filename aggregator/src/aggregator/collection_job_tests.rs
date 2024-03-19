use crate::aggregator::{
    http_handlers::{
        aggregator_handler,
        test_util::{decode_response_body, take_problem_details},
    },
    test_util::BATCH_AGGREGATION_SHARD_COUNT,
    Config,
};
use http::StatusCode;
use janus_aggregator_core::{
    datastore::{
        models::{
            AggregationJob, AggregationJobState, BatchAggregation, BatchAggregationState,
            CollectionJobState, LeaderStoredReport, ReportAggregation, ReportAggregationState,
        },
        test_util::{ephemeral_datastore, EphemeralDatastore},
        Datastore,
    },
    task::{
        test_util::{Task, TaskBuilder},
        QueryType,
    },
    test_util::noop_meter,
};
use janus_core::{
    auth_tokens::AuthenticationToken,
    hpke::{self, HpkeApplicationInfo, Label},
    test_util::{install_test_trace_subscriber, runtime::TestRuntime},
    time::{Clock, IntervalExt, MockClock},
    vdaf::VdafInstance,
};
use janus_messages::{
    query_type::{FixedSize, QueryType as QueryTypeTrait, TimeInterval},
    AggregateShareAad, AggregationJobStep, BatchId, BatchSelector, Collection, CollectionJobId,
    CollectionReq, FixedSizeQuery, Interval, Query, ReportIdChecksum, Role, Time,
};
use prio::{
    codec::{Decode, Encode},
    vdaf::dummy,
};
use rand::random;
use serde_json::json;
use std::{collections::HashSet, sync::Arc};
use trillium::{Handler, KnownHeaderName, Status};
use trillium_testing::{
    assert_headers,
    prelude::{post, put},
    TestConn,
};

pub(crate) struct CollectionJobTestCase {
    pub(super) task: Task,
    clock: MockClock,
    pub(super) handler: Box<dyn Handler>,
    pub(super) datastore: Arc<Datastore<MockClock>>,
    _ephemeral_datastore: EphemeralDatastore,
}

impl CollectionJobTestCase {
    pub(super) async fn put_collection_job_with_auth_token<Q: QueryTypeTrait>(
        &self,
        collection_job_id: &CollectionJobId,
        request: &CollectionReq<Q>,
        auth_token: Option<&AuthenticationToken>,
    ) -> TestConn {
        let mut test_conn = put(self
            .task
            .collection_job_uri(collection_job_id)
            .unwrap()
            .path());
        if let Some(auth) = auth_token {
            let (header, value) = auth.request_authentication();
            test_conn = test_conn.with_request_header(header, value);
        }

        test_conn
            .with_request_header(
                KnownHeaderName::ContentType,
                CollectionReq::<TimeInterval>::MEDIA_TYPE,
            )
            .with_request_body(request.get_encoded().unwrap())
            .run_async(&self.handler)
            .await
    }

    pub(super) async fn put_collection_job<Q: QueryTypeTrait>(
        &self,
        collection_job_id: &CollectionJobId,
        request: &CollectionReq<Q>,
    ) -> TestConn {
        self.put_collection_job_with_auth_token(
            collection_job_id,
            request,
            Some(self.task.collector_auth_token()),
        )
        .await
    }

    pub(super) async fn post_collection_job_with_auth_token(
        &self,
        collection_job_id: &CollectionJobId,
        auth_token: Option<&AuthenticationToken>,
    ) -> TestConn {
        let mut test_conn = post(
            self.task
                .collection_job_uri(collection_job_id)
                .unwrap()
                .path(),
        );
        if let Some(auth) = auth_token {
            let (header, value) = auth.request_authentication();
            test_conn = test_conn.with_request_header(header, value);
        }
        test_conn.run_async(&self.handler).await
    }

    pub(super) async fn post_collection_job(
        &self,
        collection_job_id: &CollectionJobId,
    ) -> TestConn {
        self.post_collection_job_with_auth_token(
            collection_job_id,
            Some(self.task.collector_auth_token()),
        )
        .await
    }
}

pub(crate) async fn setup_collection_job_test_case(
    role: Role,
    query_type: QueryType,
) -> CollectionJobTestCase {
    install_test_trace_subscriber();

    let task = TaskBuilder::new(query_type, VdafInstance::Fake { rounds: 1 }).build();
    let role_task = task.view_for_role(role).unwrap();
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

    datastore.put_aggregator_task(&role_task).await.unwrap();

    let handler = aggregator_handler(
        Arc::clone(&datastore),
        clock.clone(),
        TestRuntime::default(),
        &noop_meter(),
        Config {
            batch_aggregation_shard_count: BATCH_AGGREGATION_SHARD_COUNT,
            ..Default::default()
        },
    )
    .await
    .unwrap();

    CollectionJobTestCase {
        task,
        clock,
        handler: Box::new(handler),
        datastore,
        _ephemeral_datastore: ephemeral_datastore,
    }
}

async fn setup_fixed_size_current_batch_collection_job_test_case(
) -> (CollectionJobTestCase, BatchId, BatchId, Interval) {
    let test_case = setup_collection_job_test_case(
        Role::Leader,
        QueryType::FixedSize {
            max_batch_size: Some(10),
            batch_time_window_size: None,
        },
    )
    .await;

    // Fill the datastore with the necessary data so that there is are two outstanding batches to be
    // collected.
    let batch_id_1 = random();
    let batch_id_2 = random();
    let time = test_case.clock.now();
    let client_timestamp_interval = Interval::from_time(&time).unwrap();

    test_case
        .datastore
        .run_unnamed_tx(|tx| {
            let task = test_case.task.clone();
            Box::pin(async move {
                for batch_id in [batch_id_1, batch_id_2] {
                    let aggregation_job_id = random();
                    tx.put_aggregation_job::<0, FixedSize, dummy::Vdaf>(&AggregationJob::new(
                        *task.id(),
                        aggregation_job_id,
                        dummy::AggregationParam::default(),
                        batch_id,
                        client_timestamp_interval,
                        AggregationJobState::Finished,
                        AggregationJobStep::from(1),
                    ))
                    .await
                    .unwrap();

                    for ord in 0..task.min_batch_size() + 1 {
                        let report = LeaderStoredReport::new_dummy(*task.id(), time);
                        tx.put_client_report(&dummy::Vdaf::new(1), &report)
                            .await
                            .unwrap();

                        tx.put_report_aggregation::<0, dummy::Vdaf>(&ReportAggregation::new(
                            *task.id(),
                            aggregation_job_id,
                            *report.metadata().id(),
                            time,
                            ord,
                            None,
                            ReportAggregationState::Finished,
                        ))
                        .await
                        .unwrap();
                    }

                    tx.put_batch_aggregation::<0, FixedSize, dummy::Vdaf>(&BatchAggregation::new(
                        *task.id(),
                        batch_id,
                        dummy::AggregationParam::default(),
                        0,
                        client_timestamp_interval,
                        BatchAggregationState::Aggregating {
                            aggregate_share: Some(dummy::AggregateShare(0)),
                            report_count: task.min_batch_size() + 1,
                            checksum: ReportIdChecksum::default(),
                            aggregation_jobs_created: 1,
                            aggregation_jobs_terminated: 1,
                        },
                    ))
                    .await
                    .unwrap();

                    tx.put_outstanding_batch(task.id(), &batch_id, &None)
                        .await
                        .unwrap();
                }

                Ok(())
            })
        })
        .await
        .unwrap();

    (test_case, batch_id_1, batch_id_2, client_timestamp_interval)
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
    let vdaf = dummy::Vdaf::new(1);
    let leader_aggregate_share = dummy::AggregateShare(0);
    let helper_aggregate_share = dummy::AggregateShare(1);
    let aggregation_param = dummy::AggregationParam::default();
    let request = CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::CurrentBatch),
        aggregation_param.get_encoded().unwrap(),
    );

    for _ in 0..2 {
        let collection_job_id: CollectionJobId = random();

        let test_conn = test_case
            .put_collection_job(&collection_job_id, &request)
            .await;
        assert_eq!(test_conn.status(), Some(Status::Created));

        let test_conn = test_case.post_collection_job(&collection_job_id).await;
        assert_eq!(test_conn.status(), Some(Status::Accepted));

        // Update the collection job with the aggregate shares. collection job should now be complete.
        let batch_id = test_case
            .datastore
            .run_unnamed_tx(|tx| {
                let task = test_case.task.clone();
                let vdaf = vdaf.clone();
                let helper_aggregate_share_bytes = helper_aggregate_share.get_encoded().unwrap();
                Box::pin(async move {
                    let collection_job = tx
                        .get_collection_job::<0, FixedSize, dummy::Vdaf>(
                            &vdaf,
                            task.id(),
                            &collection_job_id,
                        )
                        .await
                        .unwrap()
                        .unwrap();
                    let batch_id = *collection_job.batch_identifier();

                    let encrypted_helper_aggregate_share = hpke::seal(
                        task.collector_hpke_keypair().config(),
                        &HpkeApplicationInfo::new(
                            &Label::AggregateShare,
                            &Role::Helper,
                            &Role::Collector,
                        ),
                        &helper_aggregate_share_bytes,
                        &AggregateShareAad::new(
                            *task.id(),
                            aggregation_param.get_encoded().unwrap(),
                            BatchSelector::new_fixed_size(batch_id),
                        )
                        .get_encoded()
                        .unwrap(),
                    )
                    .unwrap();

                    tx.update_collection_job::<0, FixedSize, dummy::Vdaf>(
                        &collection_job.with_state(CollectionJobState::Finished {
                            report_count: task.min_batch_size() + 1,
                            client_timestamp_interval: spanned_interval,
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

        let mut test_conn = test_case.post_collection_job(&collection_job_id).await;
        assert_headers!(&test_conn, "content-type" => (Collection::<FixedSize>::MEDIA_TYPE));

        let collect_resp: Collection<FixedSize> = decode_response_body(&mut test_conn).await;
        assert_eq!(
            collect_resp.report_count(),
            test_case.task.min_batch_size() + 1
        );
        assert_eq!(collect_resp.interval(), &spanned_interval);

        let decrypted_leader_aggregate_share = hpke::open(
            test_case.task.collector_hpke_keypair(),
            &HpkeApplicationInfo::new(&Label::AggregateShare, &Role::Leader, &Role::Collector),
            collect_resp.leader_encrypted_aggregate_share(),
            &AggregateShareAad::new(
                *test_case.task.id(),
                aggregation_param.get_encoded().unwrap(),
                BatchSelector::new_fixed_size(batch_id),
            )
            .get_encoded()
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            leader_aggregate_share,
            dummy::AggregateShare::get_decoded(decrypted_leader_aggregate_share.as_ref()).unwrap()
        );

        let decrypted_helper_aggregate_share = hpke::open(
            test_case.task.collector_hpke_keypair(),
            &HpkeApplicationInfo::new(&Label::AggregateShare, &Role::Helper, &Role::Collector),
            collect_resp.helper_encrypted_aggregate_share(),
            &AggregateShareAad::new(
                *test_case.task.id(),
                aggregation_param.get_encoded().unwrap(),
                BatchSelector::new_fixed_size(batch_id),
            )
            .get_encoded()
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            helper_aggregate_share,
            dummy::AggregateShare::get_decoded(decrypted_helper_aggregate_share.as_ref()).unwrap()
        );
    }

    assert!(saw_batch_id_1 && saw_batch_id_2);

    // We have run the two ready batches to completion. Further attempts to collect current batch
    // ought to fail.
    let collection_job_id: CollectionJobId = random();

    let mut test_conn = test_case
        .put_collection_job(&collection_job_id, &request)
        .await;
    assert_eq!(test_conn.status(), Some(Status::BadRequest));
    assert_eq!(
        take_problem_details(&mut test_conn).await,
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
        dummy::AggregationParam::default().get_encoded().unwrap(),
    );

    for _ in 0..2 {
        let response = test_case
            .put_collection_job(&collection_job_id, &request)
            .await;
        assert_eq!(response.status(), Some(Status::Created));
    }

    // There should only be a single collection job despite two successful PUTs
    test_case
        .datastore
        .run_unnamed_tx(|tx| {
            let task_id = *test_case.task.id();
            let vdaf = dummy::Vdaf::new(1);
            Box::pin(async move {
                let collection_jobs = tx
                    .get_collection_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(&vdaf, &task_id)
                    .await
                    .unwrap();
                assert_eq!(collection_jobs.len(), 1);
                assert_eq!(collection_jobs[0].id(), &collection_job_id);

                Ok(())
            })
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn collection_job_put_idempotence_time_interval_varied_collection_id() {
    // This test sends repeated, identical collection requests with differing collection job IDs and
    // validates that they are accepted. They should be accepted because calculation of the query
    // count for max_batch_query_count testing is based on the number of distinct aggregation
    // parameters that the batch has been collected against.

    let test_case = setup_collection_job_test_case(Role::Leader, QueryType::TimeInterval).await;

    let collection_job_ids = HashSet::from(random::<[CollectionJobId; 2]>());
    let request = CollectionReq::new(
        Query::new_time_interval(
            Interval::new(
                Time::from_seconds_since_epoch(0),
                *test_case.task.time_precision(),
            )
            .unwrap(),
        ),
        dummy::AggregationParam::default().get_encoded().unwrap(),
    );

    for collection_job_id in &collection_job_ids {
        let response = test_case
            .put_collection_job(collection_job_id, &request)
            .await;
        assert_eq!(response.status(), Some(Status::Created));
    }

    test_case
        .datastore
        .run_unnamed_tx(|tx| {
            let task_id = *test_case.task.id();
            let collection_job_ids = collection_job_ids.clone();

            Box::pin(async move {
                let collection_jobs = tx
                    .get_collection_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &dummy::Vdaf::new(1),
                        &task_id,
                    )
                    .await
                    .unwrap();

                assert_eq!(collection_jobs.len(), 2);
                assert_eq!(
                    collection_jobs
                        .into_iter()
                        .map(|job| *job.id())
                        .collect::<HashSet<_>>(),
                    collection_job_ids
                );

                Ok(())
            })
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn collection_job_put_idempotence_fixed_size_varied_collection_id() {
    // This test sends repeated, identical collection requests with differing collection job IDs and
    // validates that they are accepted. They should be accepted because calculation of the query
    // count for max_batch_query_count testing is based on the number of distinct aggregation
    // parameters that the batch has been collected against.

    let (test_case, batch_id, _, _) =
        setup_fixed_size_current_batch_collection_job_test_case().await;

    let collection_job_ids = HashSet::from(random::<[CollectionJobId; 2]>());
    let request = CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
        dummy::AggregationParam::default().get_encoded().unwrap(),
    );

    for collection_job_id in &collection_job_ids {
        let response = test_case
            .put_collection_job(collection_job_id, &request)
            .await;
        assert_eq!(response.status(), Some(Status::Created));
    }

    test_case
        .datastore
        .run_unnamed_tx(|tx| {
            let task_id = *test_case.task.id();
            let collection_job_ids = collection_job_ids.clone();

            Box::pin(async move {
                let collection_jobs = tx
                    .get_collection_jobs_for_task::<0, FixedSize, dummy::Vdaf>(
                        &dummy::Vdaf::new(1),
                        &task_id,
                    )
                    .await
                    .unwrap();

                assert_eq!(collection_jobs.len(), 2);
                assert_eq!(
                    collection_jobs
                        .into_iter()
                        .map(|job| *job.id())
                        .collect::<HashSet<_>>(),
                    collection_job_ids
                );

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
        dummy::AggregationParam::default().get_encoded().unwrap(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &request)
        .await;
    assert_eq!(response.status(), Some(Status::Created));

    let mutated_request = CollectionReq::new(
        Query::new_time_interval(
            Interval::new(
                Time::from_seconds_since_epoch(test_case.task.time_precision().as_seconds()),
                *test_case.task.time_precision(),
            )
            .unwrap(),
        ),
        dummy::AggregationParam::default().get_encoded().unwrap(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &mutated_request)
        .await;
    assert_eq!(response.status(), Some(Status::Conflict));
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
        dummy::AggregationParam(0).get_encoded().unwrap(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &request)
        .await;
    assert_eq!(response.status(), Some(Status::Created));

    let mutated_request = CollectionReq::new(
        Query::new_time_interval(
            Interval::new(
                Time::from_seconds_since_epoch(0),
                *test_case.task.time_precision(),
            )
            .unwrap(),
        ),
        dummy::AggregationParam(1).get_encoded().unwrap(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &mutated_request)
        .await;
    assert_eq!(response.status(), Some(Status::Conflict));
}

#[tokio::test]
async fn collection_job_put_idempotence_fixed_size_current_batch() {
    let (test_case, batch_id_1, batch_id_2, _) =
        setup_fixed_size_current_batch_collection_job_test_case().await;

    let collection_job_id = random();
    let request = CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::CurrentBatch),
        dummy::AggregationParam(0).get_encoded().unwrap(),
    );
    let mut seen_batch_id = None;

    for _ in 0..2 {
        let response = test_case
            .put_collection_job(&collection_job_id, &request)
            .await;

        assert_eq!(response.status(), Some(Status::Created));

        // Make sure that there is only ever a single collection job, and that it uses the same
        // batch ID after each PUT
        let batch_id = test_case
            .datastore
            .run_unnamed_tx(|tx| {
                let task_id = *test_case.task.id();
                Box::pin(async move {
                    let vdaf = dummy::Vdaf::new(1);
                    let collection_jobs = tx
                        .get_collection_jobs_for_task::<0, FixedSize, dummy::Vdaf>(&vdaf, &task_id)
                        .await
                        .unwrap();
                    assert_eq!(collection_jobs.len(), 1);
                    assert_eq!(collection_jobs[0].id(), &collection_job_id);
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
        dummy::AggregationParam(0).get_encoded().unwrap(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &request)
        .await;

    assert_eq!(response.status(), Some(Status::Created));

    let mutated_request = CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::CurrentBatch),
        dummy::AggregationParam(1).get_encoded().unwrap(),
    );

    let response = test_case
        .put_collection_job(&collection_job_id, &mutated_request)
        .await;
    assert_eq!(response.status(), Some(Status::Conflict));
}

#[tokio::test]
async fn collection_job_put_idempotence_fixed_size_current_batch_no_extra_reports() {
    let (test_case, _batch_id_1, _batch_id_2, _) =
        setup_fixed_size_current_batch_collection_job_test_case().await;

    let collection_job_id_1 = random();
    let collection_job_id_2 = random();
    let request = Arc::new(CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::CurrentBatch),
        dummy::AggregationParam(0).get_encoded().unwrap(),
    ));

    // Create the first collection job.
    let response = test_case
        .put_collection_job(&collection_job_id_1, &request)
        .await;
    assert_eq!(response.status(), Some(Status::Created));

    // Fetch the first collection job, to advance the current batch.
    let response = test_case.post_collection_job(&collection_job_id_1).await;
    assert_eq!(response.status(), Some(Status::Accepted));

    // Create the second collection job.
    let response = test_case
        .put_collection_job(&collection_job_id_2, &request)
        .await;
    assert_eq!(response.status(), Some(Status::Created));

    // Fetch the second collection job, to advance the current batch. There are now no outstanding
    // batches left.
    let response = test_case.post_collection_job(&collection_job_id_2).await;
    assert_eq!(response.status(), Some(Status::Accepted));

    // Re-send the collection job creation requests to confirm they are still idempotent.
    let response = test_case
        .put_collection_job(&collection_job_id_1, &request)
        .await;
    assert_eq!(response.status(), Some(Status::Created));
    let response = test_case
        .put_collection_job(&collection_job_id_2, &request)
        .await;
    assert_eq!(response.status(), Some(Status::Created));
}

#[tokio::test]
async fn collection_job_put_idempotence_fixed_size_by_batch_id() {
    let test_case = setup_collection_job_test_case(
        Role::Leader,
        QueryType::FixedSize {
            max_batch_size: Some(10),
            batch_time_window_size: None,
        },
    )
    .await;

    let collection_job_id = random();
    let batch_id = random();

    let request = CollectionReq::new(
        Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
        dummy::AggregationParam(0).get_encoded().unwrap(),
    );

    for _ in 0..2 {
        let response = test_case
            .put_collection_job(&collection_job_id, &request)
            .await;

        assert_eq!(response.status(), Some(Status::Created));
    }
}

#[tokio::test]
async fn collection_job_put_idempotence_fixed_size_by_batch_id_mutate_batch_id() {
    let test_case = setup_collection_job_test_case(
        Role::Leader,
        QueryType::FixedSize {
            max_batch_size: Some(10),
            batch_time_window_size: None,
        },
    )
    .await;

    let collection_job_id = random();
    let first_batch_id = random();
    let second_batch_id = random();

    let response = test_case
        .put_collection_job(
            &collection_job_id,
            &CollectionReq::new(
                Query::new_fixed_size(FixedSizeQuery::ByBatchId {
                    batch_id: first_batch_id,
                }),
                dummy::AggregationParam(0).get_encoded().unwrap(),
            ),
        )
        .await;
    assert_eq!(response.status(), Some(Status::Created));

    let response = test_case
        .put_collection_job(
            &collection_job_id,
            &CollectionReq::new(
                Query::new_fixed_size(FixedSizeQuery::ByBatchId {
                    batch_id: second_batch_id,
                }),
                dummy::AggregationParam(0).get_encoded().unwrap(),
            ),
        )
        .await;
    assert_eq!(response.status(), Some(Status::Conflict));
}

#[tokio::test]
async fn collection_job_put_idempotence_fixed_size_by_batch_id_mutate_aggregation_param() {
    let test_case = setup_collection_job_test_case(
        Role::Leader,
        QueryType::FixedSize {
            max_batch_size: Some(10),
            batch_time_window_size: None,
        },
    )
    .await;

    let collection_job_id = random();
    let batch_id = random();
    let first_aggregation_param = dummy::AggregationParam(0);
    let second_aggregation_param = dummy::AggregationParam(1);

    let response = test_case
        .put_collection_job(
            &collection_job_id,
            &CollectionReq::new(
                Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
                first_aggregation_param.get_encoded().unwrap(),
            ),
        )
        .await;
    assert_eq!(response.status(), Some(Status::Created));

    let response = test_case
        .put_collection_job(
            &collection_job_id,
            &CollectionReq::new(
                Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
                second_aggregation_param.get_encoded().unwrap(),
            ),
        )
        .await;
    assert_eq!(response.status(), Some(Status::Conflict));
}
