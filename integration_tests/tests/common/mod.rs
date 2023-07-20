use backoff::{future::retry, ExponentialBackoffBuilder};
use itertools::Itertools;
use janus_aggregator_core::task::{test_util::TaskBuilder, QueryType};
use janus_collector::{
    test_util::collect_with_rewritten_url, Collection, Collector, CollectorParameters,
};
use janus_core::{
    hpke::test_util::generate_test_hpke_config_and_private_key,
    retries::test_http_request_exponential_backoff, task::VdafInstance,
};
use janus_integration_tests::{
    client::{ClientBackend, ClientImplementation, InteropClientEncoding},
    EndpointFragments, TaskParameters,
};
use janus_messages::{
    problem_type::DapProblemType,
    query_type::{self, FixedSize},
    FixedSizeQuery, Query, Role,
};
use prio::vdaf::{self, prio3::Prio3};
use rand::{random, thread_rng, Rng};
use std::{iter, time::Duration as StdDuration};
use tokio::time::{self, sleep};

/// Returns a tuple of [`TaskParameters`], a task builder for the leader, and a task builder for the
/// helper.
pub fn test_task_builders(
    vdaf: VdafInstance,
    query_type: QueryType,
) -> (TaskParameters, TaskBuilder, TaskBuilder) {
    let endpoint_random_value = hex::encode(random::<[u8; 4]>());
    let endpoint_fragments = EndpointFragments {
        leader_endpoint_host: format!("leader-{endpoint_random_value}"),
        leader_endpoint_path: "/".to_string(),
        helper_endpoint_host: format!("helper-{endpoint_random_value}"),
        helper_endpoint_path: "/".to_string(),
    };
    let collector_keypair = generate_test_hpke_config_and_private_key();
    let leader_task = TaskBuilder::new(query_type, vdaf.clone(), Role::Leader)
        .with_aggregator_endpoints(endpoint_fragments.container_network_endpoints())
        .with_min_batch_size(46)
        .with_collector_hpke_config(collector_keypair.config().clone());
    let helper_task = leader_task
        .clone()
        .with_role(Role::Helper)
        .with_collector_auth_tokens(Vec::new());
    let temporary_task = leader_task.clone().build();
    let task_parameters = TaskParameters {
        task_id: *temporary_task.id(),
        endpoint_fragments,
        query_type,
        vdaf,
        min_batch_size: temporary_task.min_batch_size(),
        time_precision: *temporary_task.time_precision(),
        collector_hpke_config: collector_keypair.config().clone(),
        collector_private_key: collector_keypair.private_key().clone(),
        collector_auth_token: temporary_task.primary_collector_auth_token().clone(),
    };

    (task_parameters, leader_task, helper_task)
}

/// A set of inputs and an expected output for a VDAF's aggregation.
pub struct AggregationTestCase<V>
where
    V: vdaf::Client<16> + vdaf::Collector,
{
    measurements: Vec<V::Measurement>,
    aggregation_parameter: V::AggregationParam,
    aggregate_result: V::AggregateResult,
}

pub async fn collect_generic<'a, V, Q>(
    collector: &Collector<V>,
    query: Query<Q>,
    aggregation_parameter: &V::AggregationParam,
    host: &str,
    port: u16,
) -> Result<Collection<V::AggregateResult, Q>, janus_collector::Error>
where
    V: vdaf::Client<16> + vdaf::Collector + InteropClientEncoding,
    Q: query_type::QueryType,
{
    // An extra retry loop is needed here because our collect request may race against the
    // aggregation job creator, which is responsible for assigning reports to batches in fixed-
    // size tasks.
    let backoff = ExponentialBackoffBuilder::new()
        .with_initial_interval(time::Duration::from_millis(500))
        .with_max_interval(time::Duration::from_millis(500))
        .with_max_elapsed_time(Some(time::Duration::from_secs(5)))
        .build();
    retry(backoff, || {
        let query = query.clone();
        async move {
            match collect_with_rewritten_url(collector, query, aggregation_parameter, host, port)
                .await
            {
                Ok(collection) => Ok(collection),
                Err(
                    error @ janus_collector::Error::Http {
                        dap_problem_type: Some(DapProblemType::InvalidBatchSize),
                        ..
                    },
                ) => Err(backoff::Error::transient(error)),
                Err(error) => Err(backoff::Error::permanent(error)),
            }
        }
    })
    .await
}

pub async fn submit_measurements_and_verify_aggregate_generic<V>(
    task_parameters: &TaskParameters,
    leader_port: u16,
    vdaf: V,
    test_case: &AggregationTestCase<V>,
    client_implementation: &ClientImplementation<'_, V>,
) where
    V: vdaf::Client<16> + vdaf::Collector + InteropClientEncoding,
    V::AggregateResult: PartialEq,
{
    // Submit some measurements.
    for measurement in test_case.measurements.iter() {
        client_implementation.upload(measurement).await.unwrap();
    }

    let leader_endpoint = task_parameters
        .endpoint_fragments
        .port_forwarded_leader_endpoint(leader_port);
    let collector_params = CollectorParameters::new(
        task_parameters.task_id,
        leader_endpoint,
        task_parameters.collector_auth_token.clone(),
        task_parameters.collector_hpke_config.clone(),
        task_parameters.collector_private_key.clone(),
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

    // Send a collect request and verify that we got the correct result.
    match &task_parameters.query_type {
        QueryType::FixedSize { .. } => {
            let mut requests = 0;
            let collection = loop {
                requests += 1;
                let collection_res = collect_generic::<_, FixedSize>(
                    &collector,
                    Query::new_fixed_size(FixedSizeQuery::CurrentBatch),
                    &test_case.aggregation_parameter,
                    "127.0.0.1",
                    leader_port,
                )
                .await;
                match collection_res {
                    Ok(collection) => break collection,
                    Err(e) => {
                        if requests >= 15 {
                            panic!(
                                "timed out waiting for a current batch query to succeed, error: \
                                 {e}"
                            );
                        }
                        sleep(StdDuration::from_secs(1)).await;
                        continue;
                    }
                }
            };

            assert_eq!(
                collection.report_count(),
                u64::try_from(test_case.measurements.len()).unwrap()
            );
            assert_eq!(collection.aggregate_result(), &test_case.aggregate_result);
        }
    };
}

pub async fn submit_measurements_and_verify_aggregate(
    task_parameters: &TaskParameters,
    (leader_port, helper_port): (u16, u16),
    client_backend: &ClientBackend<'_>,
) {
    // We generate exactly one batch's worth of measurement uploads to work around an issue in
    // Daphne at time of writing.
    let total_measurements: usize = task_parameters.min_batch_size.try_into().unwrap();

    match &task_parameters.vdaf {
        VdafInstance::Prio3Count => {
            let vdaf = Prio3::new_count(2).unwrap();

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
                .build(task_parameters, (leader_port, helper_port), vdaf.clone())
                .await
                .unwrap();

            submit_measurements_and_verify_aggregate_generic(
                task_parameters,
                leader_port,
                vdaf,
                &test_case,
                &client_implementation,
            )
            .await;
        }
        VdafInstance::Prio3Sum { bits } => {
            let vdaf = Prio3::new_sum(2, *bits).unwrap();

            let measurements = iter::repeat_with(|| (random::<u128>()) >> (128 - bits))
                .take(total_measurements)
                .collect::<Vec<_>>();
            let aggregate_result = measurements.iter().sum();
            let test_case = AggregationTestCase {
                measurements,
                aggregation_parameter: (),
                aggregate_result,
            };

            let client_implementation = client_backend
                .build(task_parameters, (leader_port, helper_port), vdaf.clone())
                .await
                .unwrap();

            submit_measurements_and_verify_aggregate_generic(
                task_parameters,
                leader_port,
                vdaf,
                &test_case,
                &client_implementation,
            )
            .await;
        }
        VdafInstance::Prio3SumVec { bits, length } => {
            let vdaf = Prio3::new_sum_vec_multithreaded(2, *bits, *length).unwrap();

            let measurements = iter::repeat_with(|| {
                iter::repeat_with(|| (random::<u128>()) >> (128 - bits))
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
                .build(task_parameters, (leader_port, helper_port), vdaf.clone())
                .await
                .unwrap();

            submit_measurements_and_verify_aggregate_generic(
                task_parameters,
                leader_port,
                vdaf,
                &test_case,
                &client_implementation,
            )
            .await;
        }
        VdafInstance::Prio3Histogram { buckets } => {
            let vdaf = Prio3::new_histogram(2, buckets).unwrap();

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
                .build(task_parameters, (leader_port, helper_port), vdaf.clone())
                .await
                .unwrap();

            submit_measurements_and_verify_aggregate_generic(
                task_parameters,
                leader_port,
                vdaf,
                &test_case,
                &client_implementation,
            )
            .await;
        }
        VdafInstance::Prio3CountVec { length } => {
            let vdaf = Prio3::new_sum_vec_multithreaded(2, 1, *length).unwrap();

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
                .build(task_parameters, (leader_port, helper_port), vdaf.clone())
                .await
                .unwrap();

            submit_measurements_and_verify_aggregate_generic(
                task_parameters,
                leader_port,
                vdaf,
                &test_case,
                &client_implementation,
            )
            .await;
        }
        _ => panic!("Unsupported VdafInstance: {:?}", task_parameters.vdaf),
    }
}
