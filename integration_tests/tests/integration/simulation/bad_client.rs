use assert_matches::assert_matches;
use http::header::CONTENT_TYPE;
use janus_aggregator::aggregator::http_handlers::AggregatorHandlerBuilder;
use janus_aggregator_core::{
    datastore::{models::HpkeKeyState, test_util::ephemeral_datastore},
    task::{
        test_util::{Task, TaskBuilder},
        AggregationMode,
    },
    test_util::noop_meter,
};
use janus_core::{
    hpke::{self, HpkeApplicationInfo, HpkeKeypair, Label},
    http::HttpErrorResponse,
    retries::retry_http_request,
    test_util::{install_test_trace_subscriber, runtime::TestRuntime},
    time::{Clock, RealClock, TimeExt},
    vdaf::{vdaf_application_context, vdaf_dp_strategies, VdafInstance, VERIFY_KEY_LENGTH_PRIO3},
};
use janus_messages::{
    HpkeConfig, HpkeConfigList, InputShareAad, PlaintextInputShare, Report, ReportId,
    ReportMetadata, Role, TaskId, Time,
};
use prio::{
    codec::{Decode, Encode, ParameterizedDecode},
    field::{random_vector, Field128, FieldElement},
    flp::{gadgets::ParallelSum, types::Histogram, Flp},
    vdaf::{
        prio3::{optimal_chunk_length, Prio3, Prio3Histogram, Prio3InputShare, Prio3PublicShare},
        xof::{Seed, Xof, XofTurboShake128},
        AggregateShare, Aggregator, Client as _, Collector, PrepareTransition, Vdaf,
    },
};
use rand::{distributions::Standard, random, thread_rng, Rng};
use std::{net::Ipv4Addr, sync::Arc};
use tokio::net::TcpListener;
use trillium_tokio::Stopper;
use url::Url;

use crate::simulation::{http_request_exponential_backoff, run::MAX_REPORTS};

/// Shard and upload a report, but with a fixed ReportId.
pub(super) async fn upload_replay_report(
    measurement: usize,
    task: &Task,
    vdaf: &Prio3Histogram,
    report_time: &Time,
    http_client: &reqwest::Client,
) -> Result<(), janus_client::Error> {
    // This encodes to "replayreplayreplayrepl".
    let report_id = ReportId::from([
        173, 234, 101, 107, 42, 222, 166, 86, 178, 173, 234, 101, 107, 42, 222, 166,
    ]);
    let (public_share, input_shares) = vdaf.shard(
        &vdaf_application_context(task.id()),
        &measurement,
        report_id.as_ref(),
    )?;
    let rounded_time = report_time
        .to_batch_interval_start(task.time_precision())
        .unwrap();

    let report = prepare_report(
        http_client,
        task,
        public_share,
        input_shares,
        report_id,
        rounded_time,
    )
    .await?;
    upload_report(http_client, task, report).await
}

/// Shard and upload a report, but don't round the timestamp properly.
pub(super) async fn upload_report_not_rounded(
    measurement: usize,
    task: &Task,
    vdaf: &Prio3Histogram,
    report_time: &Time,
    http_client: &reqwest::Client,
) -> Result<(), janus_client::Error> {
    let report_id: ReportId = random();
    let (public_share, input_shares) = vdaf.shard(
        &vdaf_application_context(task.id()),
        &measurement,
        report_id.as_ref(),
    )?;

    let report = prepare_report(
        http_client,
        task,
        public_share,
        input_shares,
        report_id,
        *report_time,
    )
    .await?;
    upload_report(http_client, task, report).await
}

/// Shard and upload a report, using a measurement that does not pass the validity circuit.
pub(super) async fn upload_report_invalid_measurement(
    task: &Task,
    vdaf: &Prio3Histogram,
    report_time: &Time,
    http_client: &reqwest::Client,
) -> Result<(), janus_client::Error> {
    let mut encoded_measurement = Vec::from([Field128::zero(); MAX_REPORTS]);
    encoded_measurement[0] = Field128::one();
    encoded_measurement[1] = Field128::one();
    let task_id = random();
    let report_id = random();
    let (public_share, input_shares) =
        shard_encoded_measurement(vdaf, &task_id, encoded_measurement, report_id);

    let report = prepare_report(
        http_client,
        task,
        public_share,
        input_shares,
        report_id,
        *report_time,
    )
    .await?;
    upload_report(http_client, task, report).await
}

/// Take an already-encoded measurement as a vector of field elements, and run the Prio3 sharding
/// algorithm on it to produce a public share and a set of input shares.
fn shard_encoded_measurement(
    vdaf: &Prio3Histogram,
    task_id: &TaskId,
    encoded_measurement: Vec<Field128>,
    report_id: ReportId,
) -> (
    Prio3PublicShare<VERIFY_KEY_LENGTH_PRIO3>,
    Vec<Prio3InputShare<Field128, VERIFY_KEY_LENGTH_PRIO3>>,
) {
    const DST_MEASUREMENT_SHARE: u16 = 1;
    const DST_PROOF_SHARE: u16 = 2;
    const DST_JOINT_RANDOMNESS: u16 = 3;
    const DST_JOINT_RAND_SEED: u16 = 6;
    const DST_JOINT_RAND_PART: u16 = 7;

    const LEADER_AGGREGATOR_ID: u8 = 0;
    const HELPER_AGGREGATOR_ID: u8 = 1;

    const NUM_PROOFS: u8 = 1;

    let mut rng = thread_rng();

    let ctx = vdaf_application_context(task_id);

    assert_eq!(encoded_measurement.len(), MAX_REPORTS);
    let chunk_length = optimal_chunk_length(MAX_REPORTS);
    let circuit: Histogram<Field128, ParallelSum<_, _>> =
        Histogram::new(MAX_REPORTS, chunk_length).unwrap();

    // Share measurement.
    let helper_share_seed = rng.gen::<Seed<VERIFY_KEY_LENGTH_PRIO3>>();
    let mut helper_measurement_share_rng = XofTurboShake128::seed_stream(
        helper_share_seed.as_ref(),
        &[&domain_separation_tag(vdaf, DST_MEASUREMENT_SHARE), &ctx],
        &[&[HELPER_AGGREGATOR_ID]],
    );
    let mut expanded_helper_measurement_share: Vec<Field128> =
        Vec::with_capacity(circuit.input_len());
    let mut leader_measurement_share = encoded_measurement.clone();
    for leader_elem in leader_measurement_share.iter_mut() {
        let helper_elem = helper_measurement_share_rng.sample(Standard);
        *leader_elem -= helper_elem;
        expanded_helper_measurement_share.push(helper_elem);
    }

    // Derive joint randomness.
    let helper_joint_rand_blind = rng.gen::<Seed<VERIFY_KEY_LENGTH_PRIO3>>();
    let mut helper_joint_rand_part_xof = XofTurboShake128::init(
        helper_joint_rand_blind.as_ref(),
        &[&domain_separation_tag(vdaf, DST_JOINT_RAND_PART), &ctx],
    );
    helper_joint_rand_part_xof.update(&[HELPER_AGGREGATOR_ID]);
    helper_joint_rand_part_xof.update(report_id.as_ref());
    for helper_elem in expanded_helper_measurement_share.iter() {
        helper_joint_rand_part_xof.update(&helper_elem.get_encoded().unwrap());
    }
    let helper_joint_rand_seed_part = helper_joint_rand_part_xof.into_seed();

    let leader_joint_rand_blind = rng.gen::<Seed<VERIFY_KEY_LENGTH_PRIO3>>();
    let mut leader_joint_rand_part_xof = XofTurboShake128::init(
        leader_joint_rand_blind.as_ref(),
        &[&domain_separation_tag(vdaf, DST_JOINT_RAND_PART), &ctx],
    );
    leader_joint_rand_part_xof.update(&[LEADER_AGGREGATOR_ID]);
    leader_joint_rand_part_xof.update(report_id.as_ref());
    for leader_elem in leader_measurement_share.iter() {
        leader_joint_rand_part_xof.update(&leader_elem.get_encoded().unwrap());
    }
    let leader_joint_rand_seed_part = leader_joint_rand_part_xof.into_seed();

    let mut joint_rand_seed_xof = XofTurboShake128::init(
        &[0; VERIFY_KEY_LENGTH_PRIO3],
        &[&domain_separation_tag(vdaf, DST_JOINT_RAND_SEED), &ctx],
    );
    joint_rand_seed_xof.update(leader_joint_rand_seed_part.as_ref());
    joint_rand_seed_xof.update(helper_joint_rand_seed_part.as_ref());
    let joint_rand_seed = joint_rand_seed_xof.into_seed();
    let mut joint_rand: Vec<Field128> = Vec::with_capacity(circuit.joint_rand_len());
    let mut joint_rand_xof = XofTurboShake128::seed_stream(
        joint_rand_seed.as_ref(),
        &[&domain_separation_tag(vdaf, DST_JOINT_RANDOMNESS), &ctx],
        &[&[NUM_PROOFS]],
    );
    for _ in 0..circuit.joint_rand_len() {
        joint_rand.push(joint_rand_xof.sample(Standard));
    }

    // Construct and share FLP proof.
    let mut prove_rand: Vec<Field128> = Vec::new();
    for _ in 0..circuit.prove_rand_len() {
        prove_rand.push(random());
    }
    let mut leader_proof_share = circuit
        .prove(&encoded_measurement, &prove_rand, &joint_rand)
        .unwrap();
    let mut helper_proofs_share_xof = XofTurboShake128::seed_stream(
        helper_share_seed.as_ref(),
        &[&domain_separation_tag(vdaf, DST_PROOF_SHARE), &ctx],
        &[&[NUM_PROOFS, HELPER_AGGREGATOR_ID]],
    );
    for leader_elem in leader_proof_share.iter_mut() {
        let helper_elem = helper_proofs_share_xof.sample(Standard);
        *leader_elem -= helper_elem;
    }

    // Turn these fields into input shares via encoding and decoding.
    let mut encoded_public_share = Vec::new();
    leader_joint_rand_seed_part
        .encode(&mut encoded_public_share)
        .unwrap();
    helper_joint_rand_seed_part
        .encode(&mut encoded_public_share)
        .unwrap();
    let public_share =
        Prio3PublicShare::get_decoded_with_param(vdaf, &encoded_public_share).unwrap();

    let mut encoded_leader_input_share = Vec::new();
    for x in leader_measurement_share.iter() {
        x.encode(&mut encoded_leader_input_share).unwrap();
    }
    for x in leader_proof_share.iter() {
        x.encode(&mut encoded_leader_input_share).unwrap();
    }
    leader_joint_rand_blind
        .encode(&mut encoded_leader_input_share)
        .unwrap();
    let leader_input_share =
        Prio3InputShare::get_decoded_with_param(&(vdaf, 0), &encoded_leader_input_share).unwrap();
    let mut encoded_helper_input_share = Vec::new();
    helper_share_seed
        .encode(&mut encoded_helper_input_share)
        .unwrap();
    helper_joint_rand_blind
        .encode(&mut encoded_helper_input_share)
        .unwrap();
    let helper_input_share =
        Prio3InputShare::get_decoded_with_param(&(vdaf, 1), &encoded_helper_input_share).unwrap();

    (
        public_share,
        Vec::from([leader_input_share, helper_input_share]),
    )
}

fn domain_separation_tag(vdaf: &impl Vdaf, usage: u16) -> [u8; 8] {
    let mut dst = [0; 8];
    dst[0] = 12; // version
    dst[1] = 0; // algorithm class
    dst[2..6].copy_from_slice(vdaf.algorithm_id().to_be_bytes().as_slice());
    dst[6..8].copy_from_slice(usage.to_be_bytes().as_slice());
    dst
}

async fn prepare_report(
    http_client: &reqwest::Client,
    task: &Task,
    public_share: Prio3PublicShare<VERIFY_KEY_LENGTH_PRIO3>,
    input_shares: Vec<Prio3InputShare<Field128, VERIFY_KEY_LENGTH_PRIO3>>,
    report_id: ReportId,
    report_time: Time,
) -> Result<Report, janus_client::Error> {
    let task_id = *task.id();
    let report_metadata = ReportMetadata::new(report_id, report_time, Vec::new());
    let encoded_public_share = public_share.get_encoded().unwrap();

    let leader_hpke_config =
        aggregator_hpke_config(task.leader_aggregator_endpoint(), http_client).await?;
    let helper_hpke_config =
        aggregator_hpke_config(task.helper_aggregator_endpoint(), http_client).await?;

    let aad = InputShareAad::new(
        task_id,
        report_metadata.clone(),
        encoded_public_share.clone(),
    )
    .get_encoded()?;
    let leader_encrypted_input_share = hpke::seal(
        &leader_hpke_config,
        &HpkeApplicationInfo::new(&Label::InputShare, &Role::Client, &Role::Leader),
        &PlaintextInputShare::new(Vec::new(), input_shares[0].get_encoded()?).get_encoded()?,
        &aad,
    )?;
    let helper_encrypted_input_share = hpke::seal(
        &helper_hpke_config,
        &HpkeApplicationInfo::new(&Label::InputShare, &Role::Client, &Role::Helper),
        &PlaintextInputShare::new(Vec::new(), input_shares[1].get_encoded()?).get_encoded()?,
        &aad,
    )?;

    let report = Report::new(
        report_metadata,
        encoded_public_share,
        leader_encrypted_input_share,
        helper_encrypted_input_share,
    );
    Ok(report)
}

async fn upload_report(
    http_client: &reqwest::Client,
    task: &Task,
    report: Report,
) -> Result<(), janus_client::Error> {
    let task_id = task.id();
    let url = task
        .leader_aggregator_endpoint()
        .join(&format!("tasks/{task_id}/reports"))
        .unwrap();
    retry_http_request(http_request_exponential_backoff(), || async {
        http_client
            .post(url.clone())
            .header(CONTENT_TYPE, Report::MEDIA_TYPE)
            .body(report.get_encoded().unwrap())
            .send()
            .await
    })
    .await?;
    Ok(())
}

async fn aggregator_hpke_config(
    endpoint: &Url,
    http_client: &reqwest::Client,
) -> Result<HpkeConfig, janus_client::Error> {
    let response = retry_http_request(http_request_exponential_backoff(), || async {
        http_client
            .get(endpoint.join("hpke_config").unwrap())
            .send()
            .await
    })
    .await?;
    let status = response.status();
    if !status.is_success() {
        return Err(janus_client::Error::Http(Box::new(
            HttpErrorResponse::from(status),
        )));
    }

    let list = HpkeConfigList::get_decoded(response.body())?;

    Ok(list.hpke_configs()[0].clone())
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_client_report_validity() {
    install_test_trace_subscriber();

    let clock = RealClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let datastore = Arc::new(ephemeral_datastore.datastore(clock).await);
    let http_client = reqwest::Client::new();
    let keypair = HpkeKeypair::test();

    datastore
        .run_unnamed_tx(|tx| {
            let keypair = keypair.clone();
            Box::pin(async move {
                tx.put_hpke_keypair(&keypair).await?;
                tx.set_hpke_keypair_state(keypair.config().id(), &HpkeKeyState::Active)
                    .await
            })
        })
        .await
        .unwrap();

    let server = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let socket_address = server.local_addr().unwrap();

    let chunk_length = optimal_chunk_length(MAX_REPORTS);
    let vdaf = Prio3::new_histogram(2, MAX_REPORTS, chunk_length).unwrap();
    let vdaf_instance = VdafInstance::Prio3Histogram {
        length: MAX_REPORTS,
        chunk_length,
        dp_strategy: vdaf_dp_strategies::Prio3Histogram::NoDifferentialPrivacy,
    };
    let task = TaskBuilder::new(
        janus_aggregator_core::task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        vdaf_instance,
    )
    .with_leader_aggregator_endpoint(format!("http://{socket_address}/").parse().unwrap())
    .with_helper_aggregator_endpoint(format!("http://{socket_address}/").parse().unwrap())
    .build();
    datastore
        .put_aggregator_task(&task.leader_view().unwrap())
        .await
        .unwrap();

    let handler = AggregatorHandlerBuilder::new(
        Arc::clone(&datastore),
        clock,
        TestRuntime::default(),
        &noop_meter(),
        Default::default(),
    )
    .await
    .unwrap()
    .build()
    .unwrap();
    let stopper = Stopper::new();
    let server_handle = trillium_tokio::config()
        .with_stopper(stopper.clone())
        .without_signals()
        .with_prebound_server(server)
        .spawn(handler);

    let report_time = clock.now();
    upload_replay_report(0, &task, &vdaf, &report_time, &http_client)
        .await
        .unwrap();
    upload_report_not_rounded(0, &task, &vdaf, &report_time, &http_client)
        .await
        .unwrap();
    upload_report_invalid_measurement(&task, &vdaf, &report_time, &http_client)
        .await
        .unwrap();

    let task_id = *task.id();
    let counters = datastore
        .run_unnamed_tx(|tx| Box::pin(async move { tx.get_task_upload_counter(&task_id).await }))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(counters.report_success(), 3);

    stopper.stop();
    server_handle.await;
}

/// This checks that [`shard_encoded_measurement`] is correct by sharding a correctly-encoded
/// measurement, and confirming that it prepares successfully.
#[test]
fn shard_encoded_measurement_correct() {
    let chunk_length = optimal_chunk_length(MAX_REPORTS);
    let vdaf = Prio3::new_histogram(2, MAX_REPORTS, chunk_length).unwrap();

    let mut encoded_measurement = Vec::from([Field128::zero(); MAX_REPORTS]);
    encoded_measurement[0] = Field128::one();

    // Check the circuit output first.
    let histogram =
        Histogram::<Field128, ParallelSum<_, _>>::new(MAX_REPORTS, chunk_length).unwrap();
    let joint_rand = random_vector(histogram.joint_rand_len());
    let circuit_output = histogram
        .valid(
            &mut histogram.gadget(),
            &encoded_measurement,
            &joint_rand,
            1,
        )
        .unwrap();
    assert_eq!(circuit_output, [Field128::zero(); 2]);

    let task_id = random();
    let report_id = random();
    let (public_share, input_shares) =
        shard_encoded_measurement(&vdaf, &task_id, encoded_measurement, report_id);

    let verify_key: [u8; VERIFY_KEY_LENGTH_PRIO3] = random();
    let ctx = vdaf_application_context(&task_id);
    let (leader_prepare_state, leader_prepare_share) = vdaf
        .prepare_init(
            &verify_key,
            &ctx,
            0,
            &(),
            report_id.as_ref(),
            &public_share,
            &input_shares[0],
        )
        .unwrap();
    let (helper_prepare_state, helper_prepare_share) = vdaf
        .prepare_init(
            &verify_key,
            &ctx,
            1,
            &(),
            report_id.as_ref(),
            &public_share,
            &input_shares[1],
        )
        .unwrap();
    let prepare_message = vdaf
        .prepare_shares_to_prepare_message(&ctx, &(), [leader_prepare_share, helper_prepare_share])
        .unwrap();
    let leader_transition = vdaf
        .prepare_next(&ctx, leader_prepare_state, prepare_message.clone())
        .unwrap();
    let helper_transition = vdaf
        .prepare_next(&ctx, helper_prepare_state, prepare_message)
        .unwrap();
    let leader_output_share =
        assert_matches!(leader_transition, PrepareTransition::Finish(output_share) => output_share);
    let helper_output_share =
        assert_matches!(helper_transition, PrepareTransition::Finish(output_share) => output_share);
    let aggregate_result = vdaf
        .unshard(
            &(),
            [
                AggregateShare::from(leader_output_share),
                AggregateShare::from(helper_output_share),
            ],
            1,
        )
        .unwrap();
    assert_eq!(aggregate_result[0], 1);
    assert_eq!(aggregate_result[1..], [0; MAX_REPORTS - 1]);
}
