use backoff::ExponentialBackoffBuilder;
use itertools::Itertools;
use janus_aggregator::task::{test_util::TaskBuilder, QueryType, Task};
use janus_collector::{test_util::collect_with_rewritten_url, Collector, CollectorParameters};
use janus_core::{
    hpke::{test_util::generate_test_hpke_config_and_private_key, HpkePrivateKey},
    retries::test_http_request_exponential_backoff,
    task::VdafInstance,
    time::{Clock, RealClock, TimeExt},
};
use janus_integration_tests::client::{ClientBackend, ClientImplementation, InteropClientEncoding};
use janus_messages::{Duration, Interval, Role};
use prio::vdaf::{self, prio3::Prio3};
use rand::{random, thread_rng, Rng};
use reqwest::Url;
use std::iter;
use tokio::time;

// Returns (collector_private_key, leader_task, helper_task).
pub fn test_task_builders(vdaf: VdafInstance) -> (HpkePrivateKey, TaskBuilder, TaskBuilder) {
    let endpoint_random_value = hex::encode(random::<[u8; 4]>());
    let (collector_hpke_config, collector_private_key) =
        generate_test_hpke_config_and_private_key();
    let leader_task = TaskBuilder::new(QueryType::TimeInterval, vdaf, Role::Leader)
        .with_aggregator_endpoints(Vec::from([
            Url::parse(&format!("http://leader-{endpoint_random_value}:8080/")).unwrap(),
            Url::parse(&format!("http://helper-{endpoint_random_value}:8080/")).unwrap(),
        ]))
        .with_min_batch_size(46)
        .with_collector_hpke_config(collector_hpke_config);
    let helper_task = leader_task
        .clone()
        .with_role(Role::Helper)
        .with_collector_auth_tokens(Vec::new());

    (collector_private_key, leader_task, helper_task)
}

pub fn translate_url_for_external_access(url: &Url, external_port: u16) -> Url {
    let mut translated = url.clone();
    translated.set_host(Some("127.0.0.1")).unwrap();
    translated.set_port(Some(external_port)).unwrap();
    translated
}

/// A set of inputs and an expected output for a VDAF's aggregation.
pub struct AggregationTestCase<V>
where
    V: vdaf::Client + vdaf::Collector,
    Vec<u8>: for<'a> From<&'a V::AggregateShare>,
{
    measurements: Vec<V::Measurement>,
    aggregation_parameter: V::AggregationParam,
    aggregate_result: V::AggregateResult,
}

pub async fn submit_measurements_and_verify_aggregate_generic<V>(
    vdaf: V,
    aggregator_endpoints: Vec<Url>,
    leader_task: &Task,
    collector_private_key: &HpkePrivateKey,
    test_case: &AggregationTestCase<V>,
    client_implementation: &ClientImplementation<'_, V>,
) where
    V: vdaf::Client + vdaf::Collector + InteropClientEncoding,
    Vec<u8>: for<'a> From<&'a V::AggregateShare>,
    V::AggregateResult: PartialEq,
{
    // Submit some measurements, recording a timestamp before measurement upload to allow us to
    // determine the correct collect interval.
    let before_timestamp = RealClock::default().now();
    for measurement in test_case.measurements.iter() {
        client_implementation.upload(measurement).await.unwrap();
    }

    // Send a collect request.
    let batch_interval = Interval::new(
        before_timestamp
            .to_batch_interval_start(leader_task.time_precision())
            .unwrap(),
        // Use two time precisions as the interval duration in order to avoid a race condition if
        // this test happens to run very close to the end of a batch window.
        Duration::from_seconds(2 * leader_task.time_precision().as_seconds()),
    )
    .unwrap();
    let collector_params = CollectorParameters::new(
        *leader_task.id(),
        aggregator_endpoints[Role::Leader.index().unwrap()].clone(),
        leader_task.primary_collector_auth_token().clone(),
        leader_task.collector_hpke_config().clone(),
        collector_private_key.clone(),
    )
    .with_http_request_backoff(test_http_request_exponential_backoff())
    .with_collect_poll_backoff(
        ExponentialBackoffBuilder::new()
            .with_initial_interval(time::Duration::from_millis(500))
            .with_max_interval(time::Duration::from_millis(500))
            .with_max_elapsed_time(Some(time::Duration::from_secs(60)))
            .build(),
    );
    let collector = Collector::new(
        collector_params,
        vdaf,
        janus_collector::default_http_client().unwrap(),
    );
    let collection = collect_with_rewritten_url(
        &collector,
        batch_interval,
        &test_case.aggregation_parameter,
        "127.0.0.1",
        aggregator_endpoints[Role::Leader.index().unwrap()]
            .port()
            .unwrap(),
    )
    .await
    .unwrap();

    // Verify that we got the correct result.
    assert_eq!(
        collection.report_count(),
        u64::try_from(test_case.measurements.len()).unwrap()
    );
    assert_eq!(collection.aggregate_result(), &test_case.aggregate_result);
}

pub async fn submit_measurements_and_verify_aggregate(
    (leader_port, helper_port): (u16, u16),
    leader_task: &Task,
    collector_private_key: &HpkePrivateKey,
    client_backend: &ClientBackend<'_>,
) {
    // Translate aggregator endpoints for our perspective outside the container network.
    let aggregator_endpoints: Vec<_> = leader_task
        .aggregator_endpoints()
        .iter()
        .zip([leader_port, helper_port])
        .map(|(url, port)| translate_url_for_external_access(url, port))
        .collect();

    // We generate exactly one batch's worth of measurement uploads to work around an issue in
    // Daphne at time of writing.
    let total_measurements: usize = leader_task.min_batch_size().try_into().unwrap();

    match leader_task.vdaf() {
        VdafInstance::Prio3Aes128Count => {
            let vdaf = Prio3::new_aes128_count(2).unwrap();

            let num_nonzero_measurements = total_measurements / 2;
            let num_zero_measurements = total_measurements - num_nonzero_measurements;
            assert!(num_nonzero_measurements > 0 && num_zero_measurements > 0);
            let measurements = iter::repeat(1)
                .take(num_nonzero_measurements)
                .interleave(iter::repeat(0).take(num_zero_measurements))
                .collect::<Vec<_>>();
            let test_case = AggregationTestCase {
                measurements,
                aggregation_parameter: (),
                aggregate_result: num_nonzero_measurements.try_into().unwrap(),
            };

            let client_implementation = client_backend
                .build(leader_task, aggregator_endpoints.clone(), vdaf.clone())
                .await
                .unwrap();

            submit_measurements_and_verify_aggregate_generic(
                vdaf,
                aggregator_endpoints,
                leader_task,
                collector_private_key,
                &test_case,
                &client_implementation,
            )
            .await;
        }
        VdafInstance::Prio3Aes128Sum { bits } => {
            let vdaf = Prio3::new_aes128_sum(2, *bits).unwrap();

            let measurements = iter::repeat_with(|| (random::<u128>() as u128) >> (128 - bits))
                .take(total_measurements)
                .collect::<Vec<_>>();
            let aggregate_result = measurements.iter().sum();
            let test_case = AggregationTestCase {
                measurements,
                aggregation_parameter: (),
                aggregate_result,
            };

            let client_implementation = client_backend
                .build(leader_task, aggregator_endpoints.clone(), vdaf.clone())
                .await
                .unwrap();

            submit_measurements_and_verify_aggregate_generic(
                vdaf,
                aggregator_endpoints,
                leader_task,
                collector_private_key,
                &test_case,
                &client_implementation,
            )
            .await;
        }
        VdafInstance::Prio3Aes128Histogram { buckets } => {
            let vdaf = Prio3::new_aes128_histogram(2, buckets).unwrap();

            let mut aggregate_result = vec![0; buckets.len() + 1];
            aggregate_result.resize(buckets.len() + 1, 0);
            let measurements = iter::repeat_with(|| {
                let choice = thread_rng().gen_range(0..=buckets.len());
                aggregate_result[choice] += 1;
                let measurement = if choice == buckets.len() {
                    // This goes into the counter covering the range that extends to positive infinity.
                    buckets[buckets.len() - 1] + 1
                } else {
                    buckets[choice]
                };
                measurement as u128
            })
            .take(total_measurements)
            .collect::<Vec<_>>();
            let test_case = AggregationTestCase {
                measurements,
                aggregation_parameter: (),
                aggregate_result,
            };

            let client_implementation = client_backend
                .build(leader_task, aggregator_endpoints.clone(), vdaf.clone())
                .await
                .unwrap();

            submit_measurements_and_verify_aggregate_generic(
                vdaf,
                aggregator_endpoints,
                leader_task,
                collector_private_key,
                &test_case,
                &client_implementation,
            )
            .await;
        }
        VdafInstance::Prio3Aes128CountVec { length } => {
            let vdaf = Prio3::new_aes128_count_vec(2, *length).unwrap();

            let measurements = iter::repeat_with(|| {
                iter::repeat_with(|| random::<bool>() as u128)
                    .take(*length)
                    .collect::<Vec<_>>()
            })
            .take(total_measurements)
            .collect::<Vec<_>>();
            let aggregate_result =
                measurements
                    .iter()
                    .fold(vec![0u128; *length], |mut accumulator, measurement| {
                        for (sum, elem) in accumulator.iter_mut().zip(measurement.iter()) {
                            *sum += *elem;
                        }
                        accumulator
                    });
            let test_case = AggregationTestCase {
                measurements,
                aggregation_parameter: (),
                aggregate_result,
            };

            let client_implementation = client_backend
                .build(leader_task, aggregator_endpoints.clone(), vdaf.clone())
                .await
                .unwrap();

            submit_measurements_and_verify_aggregate_generic(
                vdaf,
                aggregator_endpoints,
                leader_task,
                collector_private_key,
                &test_case,
                &client_implementation,
            )
            .await;
        }
        _ => panic!("Unsupported VdafInstance: {:?}", leader_task.vdaf()),
    }
}
