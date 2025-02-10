//! In-memory data structure to incrementally build leader-selected batches.

use crate::aggregator::aggregation_job_writer::{AggregationJobWriter, InitialWrite};
use futures::future::try_join_all;
use janus_aggregator_core::datastore::{
    models::{
        AggregationJob, AggregationJobState, OutstandingBatch, ReportAggregationMetadata,
        ReportAggregationMetadataState, UnaggregatedReport,
    },
    Error, Transaction,
};
use janus_core::time::{Clock, DurationExt, TimeExt};
use janus_messages::{
    batch_mode::LeaderSelected, AggregationJobStep, BatchId, Duration, Interval, ReportId, TaskId,
    Time,
};
use prio::{codec::Encode, vdaf::Aggregator};
use rand::random;
use std::{
    cmp::{max, min, Ordering},
    collections::{binary_heap::PeekMut, hash_map, BinaryHeap, HashMap, HashSet, VecDeque},
    ops::RangeInclusive,
    sync::Arc,
};
use tokio::try_join;
use tracing::debug;

/// This data structure loads existing outstanding batches, incrementally assigns new reports to
/// outstanding batches and aggregation jobs, and provides unused reports at the end.
///
/// If time bucketing is enabled, reports will be separated by timestamp into different sets of
/// outstanding reports.
pub struct BatchCreator<'a, const SEED_SIZE: usize, A>
where
    A: Aggregator<SEED_SIZE, 16>,
{
    properties: Properties,
    aggregation_job_writer: &'a mut AggregationJobWriter<
        SEED_SIZE,
        LeaderSelected,
        A,
        InitialWrite,
        ReportAggregationMetadata,
    >,
    buckets: HashMap<Option<Time>, Bucket>,
    new_batches: Vec<(BatchId, Option<Time>)>,
    newly_filled_batches: HashSet<BatchId>,
    report_ids_to_scrub: HashSet<ReportId>,
}

/// Common properties used by [`BatchCreator`]. This is broken out into a separate structure to make
/// them easier to pass to helper associated functions that also take mutable references to map
/// entries.
struct Properties {
    min_aggregation_job_size: usize,
    max_aggregation_job_size: usize,
    task_id: TaskId,
    task_min_batch_size: usize,
    task_batch_time_window_size: Option<Duration>,
}

impl<'a, const SEED_SIZE: usize, A> BatchCreator<'a, SEED_SIZE, A>
where
    A: Aggregator<SEED_SIZE, 16, AggregationParam = ()> + Send + Sync + 'a,
    A::PrepareState: Encode,
{
    pub fn new(
        min_aggregation_job_size: usize,
        max_aggregation_job_size: usize,
        task_id: TaskId,
        task_min_batch_size: usize,
        task_batch_time_window_size: Option<Duration>,
        aggregation_job_writer: &'a mut AggregationJobWriter<
            SEED_SIZE,
            LeaderSelected,
            A,
            InitialWrite,
            ReportAggregationMetadata,
        >,
    ) -> Self {
        Self {
            properties: Properties {
                min_aggregation_job_size,
                max_aggregation_job_size,
                task_id,
                task_min_batch_size,
                task_batch_time_window_size,
            },
            aggregation_job_writer,
            buckets: HashMap::new(),
            new_batches: Vec::new(),
            newly_filled_batches: HashSet::new(),
            report_ids_to_scrub: HashSet::new(),
        }
    }

    pub async fn add_report<C>(
        &mut self,
        tx: &Transaction<'_, C>,
        report: UnaggregatedReport,
    ) -> Result<(), Error>
    where
        C: Clock,
    {
        let time_bucket_start_opt = self
            .properties
            .task_batch_time_window_size
            .map(|batch_time_window_size| {
                report
                    .client_timestamp()
                    .to_batch_interval_start(&batch_time_window_size)
            })
            .transpose()?;
        let mut map_entry = self.buckets.entry(time_bucket_start_opt);
        let bucket = match &mut map_entry {
            hash_map::Entry::Occupied(occupied) => occupied.get_mut(),
            hash_map::Entry::Vacant(_) => {
                // Lazily find existing unfilled batches.
                let outstanding_batches = tx
                    .get_unfilled_outstanding_batches(
                        &self.properties.task_id,
                        &time_bucket_start_opt,
                    )
                    .await?
                    .into_iter()
                    .filter(|outstanding_batch| {
                        if *outstanding_batch.size().start() < self.properties.task_min_batch_size {
                            true
                        } else {
                            // This outstanding batch has completed enough aggregations to meet the
                            // minimum batch size. Prepare to mark it filled, and ignore it for
                            // purposes of assigning reports to batches.
                            self.newly_filled_batches.insert(*outstanding_batch.id());
                            false
                        }
                    })
                    .collect();

                self.buckets
                    .entry(time_bucket_start_opt)
                    .or_insert_with(|| Bucket::new(outstanding_batches))
            }
        };

        // Add to the list of unaggregated reports for this combination of task and time bucket.
        bucket.unaggregated_reports.push_back(report);

        Self::process_batches(
            &self.properties,
            self.aggregation_job_writer,
            &mut self.report_ids_to_scrub,
            &mut self.new_batches,
            &time_bucket_start_opt,
            bucket,
            false,
        )?;

        Ok(())
    }

    /// Helper function to extract common batch creation and aggregate job creation logic within the
    /// scope of one set of outstanding batches.
    ///
    /// If `greedy` is false, aggregation jobs will be created whenever there are enough reports to
    /// meet the maximum aggregation job size. If `greedy` is true, aggregation jobs will be created
    /// when there are enough reports to meet the minimum aggregation job size. In either case,
    /// aggregation jobs may also be created when there are enough reports to make up the difference
    /// between the task's min_batch_size parameter and the upper limit of an outstanding batch's
    /// size range.
    fn process_batches(
        properties: &Properties,
        aggregation_job_writer: &mut AggregationJobWriter<
            SEED_SIZE,
            LeaderSelected,
            A,
            InitialWrite,
            ReportAggregationMetadata,
        >,
        report_ids_to_scrub: &mut HashSet<ReportId>,
        new_batches: &mut Vec<(BatchId, Option<Time>)>,
        time_bucket_start: &Option<Time>,
        bucket: &mut Bucket,
        greedy: bool,
    ) -> Result<(), Error> {
        loop {
            // Consider creating aggregation jobs inside existing batches.
            while let Some(mut largest_outstanding_batch) = bucket.outstanding_batches.peek_mut() {
                // Short-circuit if the reports are exhausted.
                if bucket.unaggregated_reports.is_empty() {
                    return Ok(());
                }

                // Discard any outstanding batches that do not currently have room for more reports.
                if largest_outstanding_batch.max_size() >= properties.task_min_batch_size {
                    PeekMut::pop(largest_outstanding_batch);
                    continue;
                }

                if greedy {
                    let desired_aggregation_job_size = min(
                        min(
                            bucket.unaggregated_reports.len(),
                            properties.max_aggregation_job_size,
                        ),
                        properties.task_min_batch_size - largest_outstanding_batch.max_size(),
                    );
                    if (desired_aggregation_job_size >= properties.min_aggregation_job_size)
                        || (largest_outstanding_batch.max_size() < properties.task_min_batch_size
                            && largest_outstanding_batch.max_size() + desired_aggregation_job_size
                                >= properties.task_min_batch_size)
                    {
                        // First condition: Create an aggregation job with between
                        // min_aggregation_job_size and max_aggregation_job_size reports.
                        //
                        // Second condition: This outstanding batch doesn't meet the minimum batch
                        // size, and an aggregation job with the currently available reports (less
                        // than min_aggregation_job_size) would be sufficient to meet the minimum
                        // batch size (assuming the reports are successfully aggregated). Create
                        // such an aggregation job.

                        Self::create_aggregation_job(
                            properties.task_id,
                            *largest_outstanding_batch.id(),
                            desired_aggregation_job_size,
                            &mut bucket.unaggregated_reports,
                            aggregation_job_writer,
                            report_ids_to_scrub,
                        )?;
                        largest_outstanding_batch.add_reports(desired_aggregation_job_size);
                    } else {
                        // There are not enough reports available to finish this outstanding batch.
                        // Since `greedy` is true, we won't see any more reports in this run of the
                        // aggregation job creator.
                        //
                        // All other outstanding batches will have lesser upper bounds on the number
                        // of reports they contain, so they will not meet the criteria to create
                        // less-than-min_aggregation_job_size jobs in order to finish out batches.
                        // Exit now, to skip looping over them.
                        return Ok(());
                    }
                } else {
                    // Create an aggregation job if there are enough reports that we couldn't use
                    // any more.
                    let desired_aggregation_job_size = min(
                        properties.max_aggregation_job_size,
                        properties.task_min_batch_size - largest_outstanding_batch.max_size(),
                    );
                    if bucket.unaggregated_reports.len() >= desired_aggregation_job_size {
                        Self::create_aggregation_job(
                            properties.task_id,
                            *largest_outstanding_batch.id(),
                            desired_aggregation_job_size,
                            &mut bucket.unaggregated_reports,
                            aggregation_job_writer,
                            report_ids_to_scrub,
                        )?;
                        largest_outstanding_batch.add_reports(desired_aggregation_job_size);
                    } else {
                        // Cannot yet fill an aggregation job for the most-full outstanding batch.
                        // Exit now, as we do not need to create any more batches.
                        return Ok(());
                    }
                }
            }

            // If there are enough reports, create a new batch and a new full aggregation job.
            let new_batch_threshold = if greedy {
                properties.min_aggregation_job_size
            } else {
                properties.max_aggregation_job_size
            };
            let desired_aggregation_job_size = min(
                min(
                    bucket.unaggregated_reports.len(),
                    properties.max_aggregation_job_size,
                ),
                properties.task_min_batch_size,
            );
            if desired_aggregation_job_size >= new_batch_threshold {
                let batch_id = random();
                new_batches.push((batch_id, *time_bucket_start));
                let outstanding_batch = OutstandingBatch::new(
                    properties.task_id,
                    batch_id,
                    RangeInclusive::new(0, desired_aggregation_job_size),
                );
                bucket
                    .outstanding_batches
                    .push(UpdatedOutstandingBatch::new(outstanding_batch));
                Self::create_aggregation_job(
                    properties.task_id,
                    batch_id,
                    desired_aggregation_job_size,
                    &mut bucket.unaggregated_reports,
                    aggregation_job_writer,
                    report_ids_to_scrub,
                )?;

                // Loop to the top of this method to create more aggregation jobs in this newly
                // outstanding batch.
                continue;
            } else {
                // Done adding reports to existing batches, and do not need to create a new batch
                // yet. Exit.
                return Ok(());
            }
        }
    }

    fn create_aggregation_job(
        task_id: TaskId,
        batch_id: BatchId,
        aggregation_job_size: usize,
        unaggregated_reports: &mut VecDeque<UnaggregatedReport>,
        aggregation_job_writer: &mut AggregationJobWriter<
            SEED_SIZE,
            LeaderSelected,
            A,
            InitialWrite,
            ReportAggregationMetadata,
        >,
        report_ids_to_scrub: &mut HashSet<ReportId>,
    ) -> Result<(), Error> {
        let aggregation_job_id = random();
        debug!(
            task_id = %task_id,
            %batch_id,
            %aggregation_job_id,
            report_count = aggregation_job_size,
            "Creating aggregation job"
        );
        let mut min_client_timestamp = None;
        let mut max_client_timestamp = None;

        let report_aggregations: Vec<_> = (0u64..)
            .zip(unaggregated_reports.drain(..aggregation_job_size))
            .map(|(ord, report)| {
                let client_timestamp = *report.client_timestamp();
                min_client_timestamp = Some(
                    min_client_timestamp.map_or(client_timestamp, |ts| min(ts, client_timestamp)),
                );
                max_client_timestamp = Some(
                    max_client_timestamp.map_or(client_timestamp, |ts| max(ts, client_timestamp)),
                );

                ReportAggregationMetadata::new(
                    task_id,
                    aggregation_job_id,
                    *report.report_id(),
                    client_timestamp,
                    ord,
                    ReportAggregationMetadataState::Init,
                )
            })
            .collect();
        report_ids_to_scrub.extend(report_aggregations.iter().map(|ra| *ra.report_id()));

        let min_client_timestamp = min_client_timestamp.unwrap(); // unwrap safety: aggregation_job_size > 0
        let max_client_timestamp = max_client_timestamp.unwrap(); // unwrap safety: aggregation_job_size > 0
        let client_timestamp_interval = Interval::new(
            min_client_timestamp,
            max_client_timestamp
                .difference(&min_client_timestamp)?
                .add(&Duration::from_seconds(1))?,
        )?;
        let aggregation_job = AggregationJob::<SEED_SIZE, LeaderSelected, A>::new(
            task_id,
            aggregation_job_id,
            (),
            batch_id,
            client_timestamp_interval,
            AggregationJobState::Active,
            AggregationJobStep::from(0),
        );
        aggregation_job_writer.put(aggregation_job, report_aggregations)?;

        Ok(())
    }

    /// Finish creating aggregation jobs with the remaining reports where possible. Marks remaining
    /// unused reports as unaggregated.
    pub async fn finish<C>(mut self, tx: &Transaction<'_, C>, vdaf: Arc<A>) -> Result<(), Error>
    where
        C: Clock,
    {
        let mut unaggregated_report_ids = Vec::new();

        // Create additional aggregation jobs with the remaining reports where possible. These will
        // be smaller than max_aggregation_job_size. We will only create jobs smaller than
        // min_aggregation_job_size if the remaining headroom in a batch requires it, otherwise
        // remaining reports will be added to unaggregated_report_ids, to be marked as unaggregated.
        for (time_bucket_start, mut bucket) in self.buckets.into_iter() {
            Self::process_batches(
                &self.properties,
                self.aggregation_job_writer,
                &mut self.report_ids_to_scrub,
                &mut self.new_batches,
                &time_bucket_start,
                &mut bucket,
                true,
            )?;
            unaggregated_report_ids.extend(
                bucket
                    .unaggregated_reports
                    .into_iter()
                    .map(|report| *report.report_id()),
            );
        }

        self.aggregation_job_writer.write(tx, vdaf).await?;

        // Report scrubbing must wait until after report aggregations have been created,
        // because they have a write-after-read antidependency on the report shares.
        try_join!(
            try_join_all(self.newly_filled_batches.iter().map(|batch_id| {
                tx.mark_outstanding_batch_filled(&self.properties.task_id, batch_id)
            })),
            try_join_all(
                self.report_ids_to_scrub
                    .iter()
                    .map(|report_id| tx.scrub_client_report(&self.properties.task_id, report_id))
            ),
            try_join_all(
                self.new_batches
                    .iter()
                    .map(|(batch_id, time_bucket_start)| tx.put_outstanding_batch(
                        &self.properties.task_id,
                        batch_id,
                        time_bucket_start,
                    ))
            ),
            try_join_all(
                unaggregated_report_ids
                    .iter()
                    .map(|report_id| tx
                        .mark_report_unaggregated(&self.properties.task_id, report_id))
            ),
        )?;
        Ok(())
    }
}

/// Tracks reports and batches for one partition of a task.
struct Bucket {
    outstanding_batches: BinaryHeap<UpdatedOutstandingBatch>,
    unaggregated_reports: VecDeque<UnaggregatedReport>,
}

impl Bucket {
    fn new(outstanding_batches: Vec<OutstandingBatch>) -> Self {
        Self {
            outstanding_batches: outstanding_batches
                .into_iter()
                .map(UpdatedOutstandingBatch::new)
                .collect(),
            unaggregated_reports: VecDeque::new(),
        }
    }
}

/// This serves as both a wrapper type for sorting outstanding batches by their current maximum
/// size, and as in-memory storage for pending changes to a batch's maximum size.
struct UpdatedOutstandingBatch {
    inner: OutstandingBatch,
    new_max_size: usize,
}

impl UpdatedOutstandingBatch {
    fn new(outstanding_batch: OutstandingBatch) -> Self {
        Self {
            new_max_size: *outstanding_batch.size().end(),
            inner: outstanding_batch,
        }
    }

    fn add_reports(&mut self, count: usize) {
        self.new_max_size += count;
    }

    fn max_size(&self) -> usize {
        self.new_max_size
    }

    fn id(&self) -> &BatchId {
        self.inner.id()
    }
}

impl PartialEq for UpdatedOutstandingBatch {
    fn eq(&self, other: &Self) -> bool {
        self.new_max_size == other.new_max_size
    }
}

impl Eq for UpdatedOutstandingBatch {}

impl PartialOrd for UpdatedOutstandingBatch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.new_max_size.cmp(&other.new_max_size))
    }
}

impl Ord for UpdatedOutstandingBatch {
    fn cmp(&self, other: &Self) -> Ordering {
        self.new_max_size.cmp(&other.new_max_size)
    }
}
