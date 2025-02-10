//! Various in-memory caches that can be used by an aggregator.

use crate::aggregator::{report_writer::ReportWriteBatcher, Error, TaskAggregator};
use janus_aggregator_core::{
    datastore::{
        models::{HpkeKeyState, HpkeKeypair},
        Datastore,
    },
    taskprov::PeerAggregator,
};
use janus_core::{hpke, time::Clock};
use janus_messages::{HpkeConfig, HpkeConfigId, Role, TaskId};
use moka::{
    future::{Cache, CacheBuilder},
    ops::compute::Op,
    Entry,
};
use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, Mutex as StdMutex},
    time::{Duration, Instant},
};
use tokio::{spawn, task::JoinHandle, time::sleep};
use tracing::{debug, error};
use url::Url;

type HpkeConfigs = Arc<Vec<HpkeConfig>>;
type HpkeKeypairs = HashMap<HpkeConfigId, Arc<hpke::HpkeKeypair>>;

#[derive(Debug)]
pub struct HpkeKeypairCache {
    // We use a std::sync::Mutex in this cache because we won't hold locks across `.await`
    // boundaries. StdMutex is lighter weight than `tokio::sync::Mutex`.
    /// The cache state: HPKE configs for advertisement, and keypairs for decryption.
    state: Arc<StdMutex<HpkeKeypairCacheState>>,

    /// Handle for task responsible for periodically refreshing the cache.
    refresh_handle: JoinHandle<()>,
}

#[derive(Debug, Default)]
struct HpkeKeypairCacheState {
    /// HPKE configs for advertisement.
    configs: HpkeConfigs,

    /// HPKE keypairs for report decryption.
    keypairs: HpkeKeypairs,
}

impl HpkeKeypairCache {
    pub const DEFAULT_REFRESH_INTERVAL: Duration =
        Duration::from_secs(60 * 30 /* 30 minutes */);

    const WAIT_RETRY_INTERVAL: Duration = Duration::from_secs(1);
    const WAIT_MAX_RETRIES: u32 = 10;

    pub async fn new<C: Clock>(
        datastore: Arc<Datastore<C>>,
        refresh_interval: Duration,
    ) -> Result<Self, Error> {
        let state = Arc::new(Default::default());

        // Initial cache load.
        Self::refresh_inner(&datastore, &state).await?;

        // Start refresh task.
        let refresh_handle = spawn({
            let datastore = Arc::clone(&datastore);
            let state = Arc::clone(&state);

            async move {
                loop {
                    sleep(refresh_interval).await;

                    let now = Instant::now();
                    let result = Self::refresh_inner(&datastore, &state).await;
                    let elapsed = now.elapsed();

                    match result {
                        Ok(_) => debug!(?elapsed, "successfully refreshed HPKE keypair cache"),
                        Err(err) => error!(?err, ?elapsed, "failed to refresh HPKE keypair cache"),
                    }
                }
            }
        });

        Ok(Self {
            state,
            refresh_handle,
        })
    }

    async fn get_hpke_keypairs<C: Clock>(
        datastore: &Datastore<C>,
    ) -> Result<Vec<HpkeKeypair>, Error> {
        // We need to ensure that there's at least one active keypair in the database before
        // proceeding.
        for _ in 0..Self::WAIT_MAX_RETRIES {
            let keypairs = datastore
                .run_tx("refresh_hpke_keypairs_cache", |tx| {
                    Box::pin(async move { tx.get_hpke_keypairs().await })
                })
                .await?;

            if keypairs.iter().any(|keypair| keypair.is_active()) {
                return Ok(keypairs);
            }

            // We sleep and retry to ensure that a separate concurrently running key rotator has
            // the chance to do its work, before failing the process.
            debug!("no active HPKE keys present in database, retrying");
            sleep(Self::WAIT_RETRY_INTERVAL).await;
        }
        Err(Error::Internal(
            "no active HPKE keys present in database".to_string(),
        ))
    }

    #[tracing::instrument(skip_all, err)]
    async fn refresh_inner<C: Clock>(
        datastore: &Datastore<C>,
        state: &StdMutex<HpkeKeypairCacheState>,
    ) -> Result<(), Error> {
        let hpke_keypairs = Self::get_hpke_keypairs(datastore).await?;

        let configs = Arc::new(
            hpke_keypairs
                .iter()
                .filter_map(|keypair| match keypair.state() {
                    HpkeKeyState::Active => Some(keypair.hpke_keypair().config().clone()),
                    _ => None,
                })
                .collect(),
        );

        let keypairs = hpke_keypairs
            .iter()
            .map(|keypair| {
                let keypair = keypair.hpke_keypair().clone();
                (*keypair.config().id(), Arc::new(keypair))
            })
            .collect();

        let mut state = state.lock().unwrap();
        *state = HpkeKeypairCacheState { configs, keypairs };
        Ok(())
    }

    #[cfg(feature = "test-util")]
    pub async fn refresh<C: Clock>(&self, datastore: &Datastore<C>) -> Result<(), Error> {
        Self::refresh_inner(datastore, &self.state).await
    }

    /// Retrieve active configs for config advertisement. This only returns configs for keypairs
    /// that are in the `[HpkeKeyState::Active]` state.
    pub fn configs(&self) -> HpkeConfigs {
        let state = self.state.lock().unwrap();
        Arc::clone(&state.configs)
    }

    /// Retrieve a keypair by ID for report decryption. This retrieves keypairs that are in any
    /// state.
    pub fn keypair(&self, id: &HpkeConfigId) -> Option<Arc<hpke::HpkeKeypair>> {
        let state = self.state.lock().unwrap();
        state.keypairs.get(id).cloned()
    }
}

impl Drop for HpkeKeypairCache {
    fn drop(&mut self) {
        self.refresh_handle.abort()
    }
}

/// Caches taskprov [`PeerAggregator`]'s. This cache is never invalidated, so the process needs to
/// be restarted if there are any changes to peer aggregators.
#[derive(Debug)]
pub struct PeerAggregatorCache {
    peers: Vec<PeerAggregator>,
}

impl PeerAggregatorCache {
    pub async fn new<C: Clock>(datastore: &Datastore<C>) -> Result<Self, Error> {
        Ok(Self {
            peers: datastore
                .run_tx("refresh_peer_aggregators_cache", |tx| {
                    Box::pin(async move { tx.get_taskprov_peer_aggregators().await })
                })
                .await?
                .into_iter()
                .collect(),
        })
    }

    pub fn get(&self, endpoint: &Url, role: &Role) -> Option<&PeerAggregator> {
        // The peer aggregator table is unlikely to be more than a few entries long (1-2 entries),
        // so a linear search should be fine.
        self.peers
            .iter()
            .find(|peer| peer.endpoint() == endpoint && peer.peer_role() == role)
    }
}

#[derive(Debug)]
pub struct TaskAggregatorCache<C: Clock> {
    datastore: Arc<Datastore<C>>,
    report_writer: Arc<ReportWriteBatcher<C>>,
    cache: Cache<TaskId, TaskAggregatorRef<C>>,
    cache_none: bool,
}

/// An Arc reference to a TaskAggregator. None indicates that there is no such task aggregator in
/// the database.
type TaskAggregatorRef<C> = Option<Arc<TaskAggregator<C>>>;

pub const TASK_AGGREGATOR_CACHE_DEFAULT_TTL: Duration = Duration::from_secs(600);
pub const TASK_AGGREGATOR_CACHE_DEFAULT_CAPACITY: u64 = 10_000;

impl<C: Clock> TaskAggregatorCache<C> {
    pub fn new(
        datastore: Arc<Datastore<C>>,
        report_writer: ReportWriteBatcher<C>,
        cache_none: bool,
        capacity: u64,
        ttl: Duration,
    ) -> Self {
        Self {
            datastore,
            report_writer: Arc::new(report_writer),
            cache: CacheBuilder::new(capacity).time_to_live(ttl).build(),
            cache_none,
        }
    }

    pub async fn get(&self, task_id: &TaskId) -> Result<TaskAggregatorRef<C>, Error> {
        Ok(self
            .cache
            .entry(*task_id)
            .and_try_compute_with(|entry| async move {
                match entry {
                    Some(_) => Ok::<_, Error>(Op::Nop),
                    None => {
                        let task = self
                            .datastore
                            .run_tx("task_aggregator_get_task", |tx| {
                                let task_id = *task_id;
                                Box::pin(async move { tx.get_aggregator_task(&task_id).await })
                            })
                            .await?
                            .map(|task| TaskAggregator::new(task, Arc::clone(&self.report_writer)))
                            .transpose()?
                            .map(Arc::new);
                        match task {
                            Some(task) => Ok(Op::Put(Some(task))),
                            None => {
                                if self.cache_none {
                                    Ok(Op::Put(None))
                                } else {
                                    Ok(Op::Nop)
                                }
                            }
                        }
                    }
                }
            })
            .await?
            .into_entry()
            .map_or_else(|| None, Entry::into_value))
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use janus_aggregator_core::{
        datastore::{models::HpkeKeyState, test_util::ephemeral_datastore},
        task::{test_util::TaskBuilder, AggregationMode, BatchMode},
    };
    use janus_core::{
        hpke::HpkeKeypair,
        test_util::{install_test_trace_subscriber, runtime::TestRuntime},
        time::MockClock,
        vdaf::VdafInstance,
    };
    use janus_messages::Time;
    use tokio::time::sleep;

    use crate::{
        aggregator::report_writer::ReportWriteBatcher,
        cache::{HpkeKeypairCache, TaskAggregatorCache},
    };

    #[tokio::test]
    async fn hpke_keypair_cache() {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        // On empty DB, this should block waiting for an HPKE keypair to be placed, so spawn it to
        // let it poll in the background.
        let cache = tokio::spawn({
            let datastore = datastore.clone();
            async move {
                HpkeKeypairCache::new(datastore, HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL).await
            }
        });

        // Insert a new active key in the foreground, after a short wait.
        sleep(Duration::from_secs(1)).await;
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

        let cache = cache.await.unwrap().unwrap();
        assert_eq!(
            cache.keypair(keypair.config().id()).unwrap(),
            Arc::new(keypair),
        );
    }

    #[tokio::test]
    async fn task_aggregator_cache() {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        let ttl = Duration::from_millis(500);
        let task_aggregators = TaskAggregatorCache::new(
            Arc::clone(&datastore),
            ReportWriteBatcher::new(
                Arc::clone(&datastore),
                TestRuntime::default(),
                100,                      // doesn't matter
                100,                      // doesn't matter
                Duration::from_secs(100), // doesn't matter
            ),
            false,
            10000,
            ttl,
        );

        let task = TaskBuilder::new(
            BatchMode::TimeInterval,
            AggregationMode::Synchronous,
            VdafInstance::Prio3Count,
        )
        .build()
        .leader_view()
        .unwrap();

        assert!(task_aggregators.get(task.id()).await.unwrap().is_none());
        // We shouldn't have cached that last call.
        assert_eq!(task_aggregators.cache.entry_count(), 0);

        // A wild task appears!
        datastore.put_aggregator_task(&task).await.unwrap();
        let task_aggregator = task_aggregators.get(task.id()).await.unwrap().unwrap();
        assert_eq!(task_aggregator.task.id(), task.id());

        // Modify the task.
        let new_end = Time::from_seconds_since_epoch(100);
        datastore
            .run_unnamed_tx(|tx| {
                let task_id = *task.id();
                Box::pin(async move {
                    tx.update_task_end(&task_id, Some(&new_end)).await.unwrap();
                    Ok(())
                })
            })
            .await
            .unwrap();

        // At this point, the above change may or may not be reflected yet, because we've cached the
        // previous task.
        let task_aggregator = task_aggregators.get(task.id()).await.unwrap().unwrap();
        assert!(
            (task_aggregator.task.task_end() == task.task_end())
                || (task_aggregator.task.task_end() == Some(&new_end))
        );

        // Unfortunately, because moka doesn't provide any facility for a fake clock, we have to resort
        // to sleeps to test TTL functionality.
        sleep(Duration::from_secs(1)).await;

        let task_aggregator = task_aggregators.get(task.id()).await.unwrap().unwrap();
        assert_eq!(task_aggregator.task.task_end(), Some(&new_end));
    }

    #[tokio::test]
    async fn task_aggregator_cache_none() {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        let ttl = Duration::from_millis(500);
        let task_aggregators = TaskAggregatorCache::new(
            Arc::clone(&datastore),
            ReportWriteBatcher::new(
                Arc::clone(&datastore),
                TestRuntime::default(),
                100,                      // doesn't matter
                100,                      // doesn't matter
                Duration::from_secs(100), // doesn't matter
            ),
            true,
            10000,
            ttl,
        );

        let task = TaskBuilder::new(
            BatchMode::TimeInterval,
            AggregationMode::Synchronous,
            VdafInstance::Prio3Count,
        )
        .build()
        .leader_view()
        .unwrap();

        assert!(task_aggregators.get(task.id()).await.unwrap().is_none());

        // A wild task appears!
        datastore.put_aggregator_task(&task).await.unwrap();

        // Unfortunately, because moka doesn't provide any facility for a fake clock, we have to resort
        // to sleeps to test TTL functionality.
        sleep(Duration::from_secs(1)).await;

        // Now we should see it.
        let task_aggregator = task_aggregators.get(task.id()).await.unwrap().unwrap();
        assert_eq!(task_aggregator.task.id(), task.id());

        // Modify the task.
        let new_end = Time::from_seconds_since_epoch(100);
        datastore
            .run_unnamed_tx(|tx| {
                let task_id = *task.id();
                Box::pin(async move {
                    tx.update_task_end(&task_id, Some(&new_end)).await.unwrap();
                    Ok(())
                })
            })
            .await
            .unwrap();

        // At this point, the above change may or may not be reflected yet because we've cached the
        // previous value.
        let task_aggregator = task_aggregators.get(task.id()).await.unwrap().unwrap();
        assert!(
            (task_aggregator.task.task_end() == task.task_end())
                || (task_aggregator.task.task_end() == Some(&new_end))
        );

        sleep(Duration::from_secs(1)).await;

        let task_aggregator = task_aggregators.get(task.id()).await.unwrap().unwrap();
        assert_eq!(task_aggregator.task.task_end(), Some(&new_end));
    }
}
