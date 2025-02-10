use crate::aggregator::{
    collection_job_tests::setup_collection_job_test_case,
    http_handlers::test_util::{decode_response_body, take_problem_details, HttpHandlerTest},
};
use assert_matches::assert_matches;
use janus_aggregator_core::{
    batch_mode::AccumulableBatchMode,
    datastore::models::{CollectionJob, CollectionJobState},
    task::{test_util::TaskBuilder, AggregationMode, BatchMode},
};
use janus_core::{
    hpke::{self, HpkeApplicationInfo, Label},
    vdaf::VdafInstance,
};
use janus_messages::{
    batch_mode::TimeInterval, AggregateShareAad, BatchSelector, CollectionJobId, CollectionJobReq,
    CollectionJobResp, Duration, Interval, Query, Role, Time,
};
use prio::{
    codec::{Decode, Encode},
    vdaf::dummy,
};
use rand::random;
use serde_json::json;
use trillium::{KnownHeaderName, Status};
use trillium_testing::{
    assert_headers,
    prelude::{delete, get, put},
};

#[tokio::test]
async fn collection_job_put_request_to_helper() {
    let test_case = setup_collection_job_test_case(Role::Helper, BatchMode::TimeInterval).await;
    test_case
        .setup_time_interval_batch(Time::from_seconds_since_epoch(0))
        .await;

    let collection_job_id: CollectionJobId = random();
    let request = CollectionJobReq::new(
        Query::new_time_interval(
            Interval::new(
                Time::from_seconds_since_epoch(0),
                *test_case.task.time_precision(),
            )
            .unwrap(),
        ),
        dummy::AggregationParam::default().get_encoded().unwrap(),
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
async fn collection_job_put_request_invalid_batch_interval() {
    let test_case = setup_collection_job_test_case(Role::Leader, BatchMode::TimeInterval).await;
    test_case
        .setup_time_interval_batch(Time::from_seconds_since_epoch(0))
        .await;

    let collection_job_id: CollectionJobId = random();
    let request = CollectionJobReq::new(
        Query::new_time_interval(
            Interval::new(
                Time::from_seconds_since_epoch(0),
                // Collect request will be rejected because batch interval is too small
                Duration::from_seconds(test_case.task.time_precision().as_seconds() - 1),
            )
            .unwrap(),
        ),
        dummy::AggregationParam::default().get_encoded().unwrap(),
    );

    let mut test_conn = test_case
        .put_collection_job(&collection_job_id, &request)
        .await;

    assert_eq!(test_conn.status(), Some(Status::BadRequest));
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": Status::BadRequest as u16,
            "type": "urn:ietf:params:ppm:dap:error:batchInvalid",
            "title": "The batch implied by the query is invalid.",
            "taskid": format!("{}", test_case.task.id()),
        })
    );
}

#[tokio::test]
async fn collection_job_put_request_invalid_aggregation_parameter() {
    let test_case = setup_collection_job_test_case(Role::Leader, BatchMode::TimeInterval).await;
    test_case
        .setup_time_interval_batch(Time::from_seconds_since_epoch(0))
        .await;

    let collection_job_id: CollectionJobId = random();
    let request = CollectionJobReq::new(
        Query::new_time_interval(
            Interval::new(
                Time::from_seconds_since_epoch(0),
                Duration::from_seconds(test_case.task.time_precision().as_seconds()),
            )
            .unwrap(),
        ),
        // dummy::AggregationParam is a tuple struct wrapping a u8, so this is not a valid
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
            "type": "urn:ietf:params:ppm:dap:error:invalidMessage",
            "title": "The message type for a response was incorrect or the payload was malformed.",
        })
    );
}

#[tokio::test]
async fn collection_job_put_request_invalid_batch_size() {
    let HttpHandlerTest {
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        ..
    } = HttpHandlerTest::new().await;

    // Prepare parameters.
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .with_min_batch_size(1)
    .build();
    let leader_task = task.leader_view().unwrap();
    datastore.put_aggregator_task(&leader_task).await.unwrap();

    let collection_job_id: CollectionJobId = random();
    let request = CollectionJobReq::new(
        Query::new_time_interval(
            Interval::new(
                Time::from_seconds_since_epoch(0),
                Duration::from_seconds(task.time_precision().as_seconds()),
            )
            .unwrap(),
        ),
        dummy::AggregationParam::default().get_encoded().unwrap(),
    );

    let (header, value) = task.collector_auth_token().request_authentication();
    let mut test_conn = put(task.collection_job_uri(&collection_job_id).unwrap().path())
        .with_request_header(header, value)
        .with_request_header(
            KnownHeaderName::ContentType,
            CollectionJobReq::<TimeInterval>::MEDIA_TYPE,
        )
        .with_request_body(request.get_encoded().unwrap())
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
    let test_case = setup_collection_job_test_case(Role::Leader, BatchMode::TimeInterval).await;
    test_case
        .setup_time_interval_batch(Time::from_seconds_since_epoch(0))
        .await;

    let batch_interval = Interval::new(
        Time::from_seconds_since_epoch(0),
        *test_case.task.time_precision(),
    )
    .unwrap();
    let collection_job_id: CollectionJobId = random();
    let req = CollectionJobReq::new(
        Query::new_time_interval(batch_interval),
        dummy::AggregationParam::default().get_encoded().unwrap(),
    );

    // Incorrect authentication token.
    let mut test_conn = test_case
        .put_collection_job_with_auth_token(&collection_job_id, &req, Some(&random()))
        .await;

    let want_status = u16::from(Status::Forbidden);
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": want_status,
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
            Some(test_case.task.aggregator_auth_token()),
        )
        .await;

    let want_status = u16::from(Status::Forbidden);
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": want_status,
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

    let want_status = u16::from(Status::Forbidden);
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": want_status,
            "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
            "title": "The request's authorization is not valid.",
            "taskid": format!("{}", test_case.task.id()),
        })
    );
    assert_eq!(want_status, test_conn.status().unwrap());
}

#[tokio::test]
async fn collection_job_get_request_unauthenticated_collection_jobs() {
    let test_case = setup_collection_job_test_case(Role::Leader, BatchMode::TimeInterval).await;
    test_case
        .setup_time_interval_batch(Time::from_seconds_since_epoch(0))
        .await;

    let batch_interval = Interval::new(
        Time::from_seconds_since_epoch(0),
        *test_case.task.time_precision(),
    )
    .unwrap();

    let collection_job_id: CollectionJobId = random();
    let request = CollectionJobReq::new(
        Query::new_time_interval(batch_interval),
        dummy::AggregationParam::default().get_encoded().unwrap(),
    );

    let test_conn = test_case
        .put_collection_job(&collection_job_id, &request)
        .await;

    assert_eq!(test_conn.status().unwrap(), Status::Created);

    // Incorrect authentication token.
    let mut test_conn = test_case
        .get_collection_job_with_auth_token(&collection_job_id, Some(&random()))
        .await;

    let want_status = u16::from(Status::Forbidden);
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": want_status,
            "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
            "title": "The request's authorization is not valid.",
            "taskid": format!("{}", test_case.task.id()),
        })
    );
    assert_eq!(want_status, test_conn.status().unwrap());

    // Aggregator authentication token.
    let mut test_conn = test_case
        .get_collection_job_with_auth_token(
            &collection_job_id,
            Some(test_case.task.aggregator_auth_token()),
        )
        .await;

    let want_status = u16::from(Status::Forbidden);
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": want_status,
            "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
            "title": "The request's authorization is not valid.",
            "taskid": format!("{}", test_case.task.id()),
        })
    );
    assert_eq!(want_status, test_conn.status().unwrap());

    // Missing authentication token.
    let mut test_conn = test_case
        .get_collection_job_with_auth_token(&collection_job_id, None)
        .await;

    let want_status = u16::from(Status::Forbidden);
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": want_status,
            "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
            "title": "The request's authorization is not valid.",
            "taskid": format!("{}", test_case.task.id()),
        })
    );
    assert_eq!(want_status, test_conn.status().unwrap());
}

#[tokio::test]
async fn collection_job_success_time_interval() {
    let test_case = setup_collection_job_test_case(Role::Leader, BatchMode::TimeInterval).await;
    test_case
        .setup_time_interval_batch(Time::from_seconds_since_epoch(0))
        .await;

    let batch_interval = TimeInterval::to_batch_identifier(
        &test_case.task.leader_view().unwrap(),
        &(),
        &Time::from_seconds_since_epoch(0),
    )
    .unwrap();

    let aggregation_param = dummy::AggregationParam::default();
    let leader_aggregate_share = dummy::AggregateShare(0);
    let helper_aggregate_share = dummy::AggregateShare(1);

    let collection_job_id: CollectionJobId = random();
    let request = CollectionJobReq::new(
        Query::new_time_interval(batch_interval),
        aggregation_param.get_encoded().unwrap(),
    );

    let mut test_conn = test_case
        .put_collection_job(&collection_job_id, &request)
        .await;

    let want_collection_job = CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
        *test_case.task.id(),
        collection_job_id,
        Query::new_time_interval(batch_interval),
        aggregation_param,
        batch_interval,
        CollectionJobState::Start,
    );

    let got_collection_job = test_case
        .datastore
        .run_unnamed_tx(|tx| {
            let task_id = *test_case.task.id();

            Box::pin(async move {
                Ok(tx
                    .get_collection_job(&dummy::Vdaf::new(1), &task_id, &collection_job_id)
                    .await
                    .unwrap()
                    .unwrap())
            })
        })
        .await
        .unwrap();

    assert_eq!(want_collection_job, got_collection_job);

    assert_eq!(test_conn.status(), Some(Status::Created));
    assert_headers!(
        &test_conn,
        "content-type" => (CollectionJobResp::<TimeInterval>::MEDIA_TYPE)
    );
    let collect_resp: CollectionJobResp<TimeInterval> = decode_response_body(&mut test_conn).await;
    assert_eq!(collect_resp, CollectionJobResp::<TimeInterval>::Processing);

    let mut test_conn = test_case.get_collection_job(&collection_job_id).await;
    assert_eq!(test_conn.status(), Some(Status::Ok));
    assert_headers!(
        &test_conn,
        "content-type" => (CollectionJobResp::<TimeInterval>::MEDIA_TYPE)
    );
    let collect_resp: CollectionJobResp<TimeInterval> = decode_response_body(&mut test_conn).await;
    assert_eq!(collect_resp, CollectionJobResp::<TimeInterval>::Processing);

    // Update the collection job with the aggregate shares and some aggregation jobs. collection
    // job should now be complete.
    test_case
        .datastore
        .run_unnamed_tx(|tx| {
            let task = test_case.task.clone();
            let helper_aggregate_share_bytes = helper_aggregate_share.get_encoded().unwrap();
            Box::pin(async move {
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
                        BatchSelector::new_time_interval(batch_interval),
                    )
                    .get_encoded()
                    .unwrap(),
                )
                .unwrap();

                let collection_job = tx
                    .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                        &dummy::Vdaf::new(1),
                        task.id(),
                        &collection_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap()
                    .with_state(CollectionJobState::Finished {
                        report_count: 12,
                        client_timestamp_interval: batch_interval,
                        encrypted_helper_aggregate_share,
                        leader_aggregate_share,
                    });

                tx.update_collection_job::<0, TimeInterval, dummy::Vdaf>(&collection_job)
                    .await
                    .unwrap();
                Ok(())
            })
        })
        .await
        .unwrap();

    let mut test_conn = test_case.get_collection_job(&collection_job_id).await;

    assert_eq!(test_conn.status(), Some(Status::Ok));
    assert_headers!(
        &test_conn,
        "content-type" => (CollectionJobResp::<TimeInterval>::MEDIA_TYPE)
    );
    let collect_resp: CollectionJobResp<TimeInterval> = decode_response_body(&mut test_conn).await;
    let (
        report_count,
        interval,
        leader_encrypted_aggregate_share,
        helper_encrypted_aggregate_share,
    ) = assert_matches!(
        collect_resp,
        CollectionJobResp::Finished{
            report_count,
            interval,
            leader_encrypted_agg_share,
            helper_encrypted_agg_share,
            ..
        } => (report_count, interval, leader_encrypted_agg_share, helper_encrypted_agg_share)
    );

    assert_eq!(report_count, 12);
    assert_eq!(interval, batch_interval);

    let decrypted_leader_aggregate_share = hpke::open(
        test_case.task.collector_hpke_keypair(),
        &HpkeApplicationInfo::new(&Label::AggregateShare, &Role::Leader, &Role::Collector),
        &leader_encrypted_aggregate_share,
        &AggregateShareAad::new(
            *test_case.task.id(),
            aggregation_param.get_encoded().unwrap(),
            BatchSelector::new_time_interval(batch_interval),
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
        &helper_encrypted_aggregate_share,
        &AggregateShareAad::new(
            *test_case.task.id(),
            aggregation_param.get_encoded().unwrap(),
            BatchSelector::new_time_interval(batch_interval),
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

#[tokio::test]
async fn collection_job_get_request_no_such_collection_job() {
    let test_case = setup_collection_job_test_case(Role::Leader, BatchMode::TimeInterval).await;
    test_case
        .setup_time_interval_batch(Time::from_seconds_since_epoch(0))
        .await;

    let no_such_collection_job_id: CollectionJobId = random();

    let (header, value) = test_case
        .task
        .collector_auth_token()
        .request_authentication();
    let test_conn = get(format!(
        "/tasks/{}/collection_jobs/{no_such_collection_job_id}",
        test_case.task.id()
    ))
    .with_request_header(header, value)
    .run_async(&test_case.handler)
    .await;
    assert_eq!(test_conn.status(), Some(Status::NotFound));
}

#[tokio::test]
async fn collection_job_put_request_batch_queried_multiple_times() {
    let test_case = setup_collection_job_test_case(Role::Leader, BatchMode::TimeInterval).await;
    let interval = test_case
        .setup_time_interval_batch(Time::from_seconds_since_epoch(0))
        .await;

    // Sending this request will consume a query for [0, time_precision).
    let request = CollectionJobReq::new(
        Query::new_time_interval(interval),
        dummy::AggregationParam(0).get_encoded().unwrap(),
    );

    let test_conn = test_case.put_collection_job(&random(), &request).await;

    assert_eq!(test_conn.status(), Some(Status::Created));

    // This request will not be allowed due to the query count already being consumed.
    let invalid_request = CollectionJobReq::new(
        Query::new_time_interval(interval),
        dummy::AggregationParam(1).get_encoded().unwrap(),
    );

    let mut test_conn = test_case
        .put_collection_job(&random(), &invalid_request)
        .await;
    assert_eq!(test_conn.status(), Some(Status::BadRequest));
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": Status::BadRequest as u16,
            "type": "urn:ietf:params:ppm:dap:error:batchQueriedMultipleTimes",
            "title": "The batch described by the query has been queried already.",
            "taskid": format!("{}", test_case.task.id()),
        })
    );
}

#[tokio::test]
async fn collection_job_put_request_batch_overlap() {
    let test_case = setup_collection_job_test_case(Role::Leader, BatchMode::TimeInterval).await;
    let interval = test_case
        .setup_time_interval_batch(Time::from_seconds_since_epoch(0))
        .await;

    // Sending this request will consume a query for [0, 2 * time_precision).
    let request = CollectionJobReq::new(
        Query::new_time_interval(
            Interval::new(
                Time::from_seconds_since_epoch(0),
                Duration::from_seconds(2 * test_case.task.time_precision().as_seconds()),
            )
            .unwrap(),
        ),
        dummy::AggregationParam(0).get_encoded().unwrap(),
    );

    let test_conn = test_case.put_collection_job(&random(), &request).await;

    assert_eq!(test_conn.status(), Some(Status::Created));

    // This request will not be allowed due to overlapping with the previous request.
    let invalid_request = CollectionJobReq::new(
        Query::new_time_interval(interval),
        dummy::AggregationParam(1).get_encoded().unwrap(),
    );

    let mut test_conn = test_case
        .put_collection_job(&random(), &invalid_request)
        .await;
    assert_eq!(test_conn.status(), Some(Status::BadRequest));
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": Status::BadRequest as u16,
            "type": "urn:ietf:params:ppm:dap:error:batchOverlap",
            "title": "The queried batch overlaps with a previously queried batch.",
            "taskid": format!("{}", test_case.task.id()),
        })
    );
}

#[tokio::test]
async fn delete_collection_job() {
    let test_case = setup_collection_job_test_case(Role::Leader, BatchMode::TimeInterval).await;
    let batch_interval = test_case
        .setup_time_interval_batch(Time::from_seconds_since_epoch(0))
        .await;

    let collection_job_id: CollectionJobId = random();

    let (header, value) = test_case
        .task
        .collector_auth_token()
        .request_authentication();

    // Try to delete a collection job that doesn't exist
    let test_conn = delete(
        test_case
            .task
            .collection_job_uri(&collection_job_id)
            .unwrap()
            .path(),
    )
    .with_request_header(header, value.clone())
    .run_async(&test_case.handler)
    .await;
    assert_eq!(test_conn.status(), Some(Status::NotFound));

    // Create a collection job
    let request = CollectionJobReq::new(
        Query::new_time_interval(batch_interval),
        dummy::AggregationParam::default().get_encoded().unwrap(),
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
    .with_request_header(header, value)
    .run_async(&test_case.handler)
    .await;
    assert_eq!(test_conn.status(), Some(Status::NoContent));

    // Get the job again
    let test_conn = test_case.get_collection_job(&collection_job_id).await;
    assert_eq!(test_conn.status(), Some(Status::NoContent));
}
