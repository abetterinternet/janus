use crate::aggregator::{
    http_handlers::aggregator_handler, tests::generate_helper_report_share, Config,
};
use janus_aggregator_core::{
    datastore::{
        test_util::{ephemeral_datastore, EphemeralDatastore},
        Datastore,
    },
    task::{test_util::TaskBuilder, QueryType, Task},
    test_util::noop_meter,
};
use janus_core::{
    task::{AuthenticationToken, VdafInstance, DAP_AUTH_HEADER},
    test_util::{dummy_vdaf, install_test_trace_subscriber, run_vdaf, VdafTranscript},
    time::{Clock, MockClock, TimeExt as _},
};
use janus_messages::{
    query_type::FixedSize, AggregationJobId, AggregationJobInitializeReq, BatchId,
    PartialBatchSelector, ReportMetadata, ReportShare, Role,
};
use prio::codec::Encode;
use rand::random;
use std::sync::Arc;
use trillium::{Handler, KnownHeaderName, Status};
use trillium_testing::{prelude::put, TestConn};

pub(super) struct ReportShareGenerator {
    clock: MockClock,
    task: Task,
    aggregation_param: dummy_vdaf::AggregationParam,
    vdaf: dummy_vdaf::Vdaf,
}

impl ReportShareGenerator {
    pub(super) fn new(
        clock: MockClock,
        task: Task,
        aggregation_param: dummy_vdaf::AggregationParam,
    ) -> Self {
        Self {
            clock,
            task,
            aggregation_param,
            vdaf: dummy_vdaf::Vdaf::new(),
        }
    }

    fn with_vdaf(mut self, vdaf: dummy_vdaf::Vdaf) -> Self {
        self.vdaf = vdaf;
        self
    }

    pub(super) fn next(&self) -> (ReportShare, VdafTranscript<0, dummy_vdaf::Vdaf>) {
        self.next_with_metadata(ReportMetadata::new(
            random(),
            self.clock
                .now()
                .to_batch_interval_start(self.task.time_precision())
                .unwrap(),
        ))
    }

    pub(super) fn next_with_metadata(
        &self,
        report_metadata: ReportMetadata,
    ) -> (ReportShare, VdafTranscript<0, dummy_vdaf::Vdaf>) {
        let transcript = run_vdaf(
            &self.vdaf,
            self.task.primary_vdaf_verify_key().unwrap().as_bytes(),
            &self.aggregation_param,
            report_metadata.id(),
            &(),
        );
        let report_share = generate_helper_report_share::<dummy_vdaf::Vdaf>(
            *self.task.id(),
            report_metadata,
            self.task.current_hpke_key().config(),
            &transcript.public_share,
            Vec::new(),
            &transcript.input_shares[1],
        );

        (report_share, transcript)
    }
}

pub(super) struct AggregationJobInitTestCase {
    pub(super) clock: MockClock,
    pub(super) task: Task,
    pub(super) report_share_generator: ReportShareGenerator,
    pub(super) report_shares: Vec<ReportShare>,
    pub(super) aggregation_job_id: AggregationJobId,
    aggregation_job_init_req: AggregationJobInitializeReq<FixedSize>,
    pub(super) batch_id: BatchId,
    pub(super) aggregation_param: dummy_vdaf::AggregationParam,
    pub(super) handler: Box<dyn Handler>,
    pub(super) datastore: Arc<Datastore<MockClock>>,
    _ephemeral_datastore: EphemeralDatastore,
}

pub(super) async fn setup_aggregate_init_test() -> AggregationJobInitTestCase {
    let test_case = setup_aggregate_init_test_without_sending_request().await;

    let response = put_aggregation_job(
        &test_case.task,
        &test_case.aggregation_job_id,
        &test_case.aggregation_job_init_req,
        &test_case.handler,
    )
    .await;
    assert_eq!(response.status(), Some(Status::Ok));

    test_case
}

async fn setup_aggregate_init_test_without_sending_request() -> AggregationJobInitTestCase {
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

    datastore.put_task(&task).await.unwrap();

    let handler = aggregator_handler(
        Arc::clone(&datastore),
        clock.clone(),
        &noop_meter(),
        Config::default(),
    )
    .unwrap();

    let aggregation_param = dummy_vdaf::AggregationParam(0);

    let report_share_generator =
        ReportShareGenerator::new(clock.clone(), task.clone(), aggregation_param);

    let report_shares = Vec::from([
        report_share_generator.next().0,
        report_share_generator.next().0,
    ]);

    let aggregation_job_id = random();
    let batch_id = random();
    let aggregation_job_init_req = AggregationJobInitializeReq::new(
        aggregation_param.get_encoded(),
        PartialBatchSelector::new_fixed_size(batch_id),
        report_shares.clone(),
    );

    AggregationJobInitTestCase {
        clock,
        task,
        report_shares,
        report_share_generator,
        aggregation_job_id,
        aggregation_job_init_req,
        batch_id,
        aggregation_param,
        handler: Box::new(handler),
        datastore,
        _ephemeral_datastore: ephemeral_datastore,
    }
}

pub(crate) async fn put_aggregation_job(
    task: &Task,
    aggregation_job_id: &AggregationJobId,
    aggregation_job: &AggregationJobInitializeReq<FixedSize>,
    handler: &impl Handler,
) -> TestConn {
    put(task.aggregation_job_uri(aggregation_job_id).unwrap().path())
        .with_request_header(
            DAP_AUTH_HEADER,
            task.primary_aggregator_auth_token().as_ref().to_owned(),
        )
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE,
        )
        .with_request_body(aggregation_job.get_encoded())
        .run_async(handler)
        .await
}

#[tokio::test]
async fn aggregation_job_init_authorization_dap_auth_token() {
    let test_case = setup_aggregate_init_test_without_sending_request().await;
    // Find a DapAuthToken among the task's aggregator auth tokens
    let (auth_header, auth_value) = test_case
        .task
        .aggregator_auth_tokens()
        .iter()
        .find(|auth| matches!(auth, AuthenticationToken::DapAuth(_)))
        .unwrap()
        .request_authentication();

    let response = put(test_case
        .task
        .aggregation_job_uri(&test_case.aggregation_job_id)
        .unwrap()
        .path())
    .with_request_header(auth_header, auth_value)
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE,
    )
    .with_request_body(test_case.aggregation_job_init_req.get_encoded())
    .run_async(&test_case.handler)
    .await;

    assert_eq!(response.status(), Some(Status::Ok));
}

#[rstest::rstest]
#[case::not_bearer_token("wrong kind of token")]
#[case::not_base64("Bearer: ")]
#[tokio::test]
async fn aggregation_job_init_malformed_authorization_header(#[case] header_value: &'static str) {
    let test_case = setup_aggregate_init_test_without_sending_request().await;

    let response = put(test_case
        .task
        .aggregation_job_uri(&test_case.aggregation_job_id)
        .unwrap()
        .path())
    // Authenticate using a malformed "Authorization: Bearer <token>" header and a `DAP-Auth-Token`
    // header. The presence of the former should cause an error despite the latter being present and
    // well formed.
    .with_request_header(KnownHeaderName::Authorization, header_value.to_string())
    .with_request_header(
        DAP_AUTH_HEADER,
        test_case
            .task
            .primary_aggregator_auth_token()
            .as_ref()
            .to_owned(),
    )
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE,
    )
    .with_request_body(test_case.aggregation_job_init_req.get_encoded())
    .run_async(&test_case.handler)
    .await;

    assert_eq!(response.status(), Some(Status::BadRequest));
}

#[tokio::test]
async fn aggregation_job_mutation_aggregation_job() {
    let test_case = setup_aggregate_init_test().await;

    // Put the aggregation job again, but with a different aggregation parameter.
    let mutated_aggregation_job_init_req = AggregationJobInitializeReq::new(
        dummy_vdaf::AggregationParam(1).get_encoded(),
        PartialBatchSelector::new_fixed_size(test_case.batch_id),
        test_case.report_shares,
    );

    let response = put_aggregation_job(
        &test_case.task,
        &test_case.aggregation_job_id,
        &mutated_aggregation_job_init_req,
        &test_case.handler,
    )
    .await;
    assert_eq!(response.status(), Some(Status::Conflict));
}

#[tokio::test]
async fn aggregation_job_mutation_report_shares() {
    let test_case = setup_aggregate_init_test().await;

    // Put the aggregation job again, mutating the associated report shares' metadata such that
    // uniqueness constraints on client_reports are violated
    for mutated_report_shares in [
        // Omit a report share that was included previously
        Vec::from(&test_case.report_shares[0..test_case.report_shares.len() - 1]),
        // Include a different report share than was included previously
        [
            &test_case.report_shares[0..test_case.report_shares.len() - 1],
            &[test_case.report_share_generator.next().0],
        ]
        .concat(),
        // Include an extra report share than was included previously
        [
            test_case.report_shares.as_slice(),
            &[test_case.report_share_generator.next().0],
        ]
        .concat(),
        // Reverse the order of the reports
        test_case.report_shares.into_iter().rev().collect(),
    ] {
        let mutated_aggregation_job_init_req = AggregationJobInitializeReq::new(
            test_case.aggregation_param.get_encoded(),
            PartialBatchSelector::new_fixed_size(test_case.batch_id),
            mutated_report_shares,
        );
        let response = put_aggregation_job(
            &test_case.task,
            &test_case.aggregation_job_id,
            &mutated_aggregation_job_init_req,
            &test_case.handler,
        )
        .await;
        assert_eq!(response.status(), Some(Status::Conflict));
    }
}

#[tokio::test]
async fn aggregation_job_mutation_report_aggregations() {
    let test_case = setup_aggregate_init_test().await;

    // Generate some new reports using the existing reports' metadata, but varying the input shares
    // such that the prepare state computed during aggregation initializaton won't match the first
    // aggregation job.
    let mutated_report_shares_generator = test_case
        .report_share_generator
        .with_vdaf(dummy_vdaf::Vdaf::new().with_input_share(dummy_vdaf::InputShare(1)));
    let mutated_report_shares = test_case
        .report_shares
        .iter()
        .map(|s| {
            mutated_report_shares_generator
                .next_with_metadata(s.metadata().clone())
                .0
        })
        .collect();

    let mutated_aggregation_job_init_req = AggregationJobInitializeReq::new(
        test_case.aggregation_param.get_encoded(),
        PartialBatchSelector::new_fixed_size(test_case.batch_id),
        mutated_report_shares,
    );
    let response = put_aggregation_job(
        &test_case.task,
        &test_case.aggregation_job_id,
        &mutated_aggregation_job_init_req,
        &test_case.handler,
    )
    .await;
    assert_eq!(response.status(), Some(Status::Conflict));
}
