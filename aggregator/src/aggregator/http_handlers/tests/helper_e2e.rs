use assert_matches::assert_matches;
use janus_aggregator_core::task::{test_util::TaskBuilder, AggregationMode, BatchMode};
use janus_core::{report_id::ReportIdChecksumExt, vdaf::VdafInstance};
use janus_messages::{
    batch_mode::LeaderSelected, AggregateShareReq, AggregationJobInitializeReq, AggregationJobResp,
    BatchSelector, PartialBatchSelector, PrepareStepResult, ReportError, ReportIdChecksum,
};
use prio::{
    codec::{Decode, Encode},
    vdaf::dummy,
};
use rand::random;
use trillium::Status;
use trillium_testing::assert_status;

use crate::aggregator::{
    aggregation_job_init::test_util::{put_aggregation_job, PrepareInitGenerator},
    http_handlers::{
        test_util::{take_response_body, HttpHandlerTest},
        tests::aggregate_share::post_aggregate_share_request,
    },
};

/// Send multiple aggregation job requests and aggregate share requests for a negative test that
/// reports cannot be aggregated with the same aggregation parameter into multiple batches.
#[tokio::test]
async fn helper_aggregation_report_share_replay() {
    let HttpHandlerTest {
        clock,
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        hpke_keypair,
        ..
    } = HttpHandlerTest::new().await;

    let task = TaskBuilder::new(
        BatchMode::LeaderSelected {
            batch_time_window_size: None,
        },
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .with_min_batch_size(1)
    .build();
    let vdaf = dummy::Vdaf::new(1);
    let agg_param = dummy::AggregationParam(0);

    let helper_task = task.helper_view().unwrap();
    datastore.put_aggregator_task(&helper_task).await.unwrap();

    let prep_init_generator = PrepareInitGenerator::new(
        clock.clone(),
        helper_task.clone(),
        hpke_keypair.config().clone(),
        vdaf.clone(),
        agg_param,
    );
    let (replayed_report, _replayed_report_transcript) = prep_init_generator.next(&7);
    let (other_report_1, _other_report_1_transcript) = prep_init_generator.next(&11);
    let (other_report_2, _other_report_2_transcript) = prep_init_generator.next(&23);

    let batch_id_1 = random();
    let batch_id_2 = random();
    let aggregation_job_id_1 = random();
    let aggregation_job_id_2 = random();

    let agg_init_req_1 = AggregationJobInitializeReq::new(
        agg_param.get_encoded().unwrap(),
        PartialBatchSelector::new_leader_selected(batch_id_1),
        Vec::from([replayed_report.clone(), other_report_1.clone()]),
    );
    let agg_init_req_2 = AggregationJobInitializeReq::new(
        agg_param.get_encoded().unwrap(),
        PartialBatchSelector::new_leader_selected(batch_id_2),
        Vec::from([replayed_report.clone(), other_report_2.clone()]),
    );

    let checksum_1 =
        ReportIdChecksum::for_report_id(replayed_report.report_share().metadata().id())
            .updated_with(other_report_1.report_share().metadata().id());
    let agg_share_req_1 = AggregateShareReq::<LeaderSelected>::new(
        BatchSelector::new(batch_id_1),
        agg_param.get_encoded().unwrap(),
        2,
        checksum_1,
    );
    let checksum_2 = ReportIdChecksum::for_report_id(other_report_2.report_share().metadata().id());
    let agg_share_req_2 = AggregateShareReq::<LeaderSelected>::new(
        BatchSelector::new(batch_id_2),
        agg_param.get_encoded().unwrap(),
        1,
        checksum_2,
    );

    // Make aggregation job initialization requests, and check the prepare step results.
    let mut test_conn =
        put_aggregation_job(&task, &aggregation_job_id_1, &agg_init_req_1, &handler).await;
    assert_status!(test_conn, Status::Created);
    let agg_init_resp_1 =
        AggregationJobResp::get_decoded(take_response_body(&mut test_conn).await.as_ref()).unwrap();
    let prepare_resps_1 = assert_matches!(
        agg_init_resp_1,
        AggregationJobResp::Finished { prepare_resps } => prepare_resps
    );
    assert_matches!(
        prepare_resps_1[0].result(),
        PrepareStepResult::Continue { .. }
    );
    assert_matches!(
        prepare_resps_1[1].result(),
        PrepareStepResult::Continue { .. }
    );

    let mut test_conn =
        put_aggregation_job(&task, &aggregation_job_id_2, &agg_init_req_2, &handler).await;
    assert_status!(test_conn, Status::Created);
    let agg_init_resp_2 =
        AggregationJobResp::get_decoded(take_response_body(&mut test_conn).await.as_ref()).unwrap();
    let prepare_resps_2 = assert_matches!(
        agg_init_resp_2,
        AggregationJobResp::Finished { prepare_resps } => prepare_resps
    );
    assert_matches!(
        prepare_resps_2[0].result(),
        PrepareStepResult::Reject(ReportError::ReportReplayed)
    );
    assert_matches!(
        prepare_resps_2[1].result(),
        PrepareStepResult::Continue { .. }
    );

    // Make aggregate share requests. If these succeed, then the helper's report_count and checksum
    // match those in the requests.
    let test_conn = post_aggregate_share_request(&task, &agg_share_req_1, &handler).await;
    assert_status!(test_conn, 200);

    let test_conn = post_aggregate_share_request(&task, &agg_share_req_2, &handler).await;
    assert_status!(test_conn, 200);
}
