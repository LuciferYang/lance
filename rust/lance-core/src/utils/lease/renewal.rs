// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinHandle;

use super::{LeaseOptions, LeaseRegistry, VersionLease};

/// RAII handle that keeps a lease alive via background renewal.
///
/// **Prefer `release().await`** for deterministic cleanup. `Drop` is best-effort:
/// - If a Tokio runtime is current, it spawns an async release task.
/// - Otherwise, TTL expiry is the backstop.
pub struct LeaseGuard {
    inner: Option<Inner>,
    #[cfg_attr(not(any(test, feature = "testing")), allow(dead_code))]
    renewal_count: Arc<AtomicU64>,
    /// Set to `true` by the renewer if it gives up (e.g., `renew()` failed).
    /// Callers can poll this via `is_renewer_healthy()` — a `false` return
    /// means the guard no longer guarantees freshness and the caller should
    /// stop trusting `current()` and re-acquire a new lease.
    renewer_healthy: Arc<std::sync::atomic::AtomicBool>,
}

struct Inner {
    lease: Arc<Mutex<VersionLease>>,
    registry: Arc<dyn LeaseRegistry>,
    renewal: JoinHandle<()>,
    release_signal: Option<oneshot::Sender<()>>,
}

impl LeaseGuard {
    pub async fn acquire(
        registry: Arc<dyn LeaseRegistry>,
        version: u64,
        opts: LeaseOptions,
        renew_every: Duration,
    ) -> crate::Result<Self> {
        let lease = registry.acquire(version, &opts).await?;
        let lease = Arc::new(Mutex::new(lease));
        let renewal_count = Arc::new(AtomicU64::new(0));
        let renewer_healthy = Arc::new(std::sync::atomic::AtomicBool::new(true));

        let registry_clone = registry.clone();
        let lease_clone = lease.clone();
        let opts_clone = opts.clone();
        let count_clone = renewal_count.clone();
        let healthy_clone = renewer_healthy.clone();

        let renewal = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(renew_every);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let current = { lease_clone.lock().await.clone() };
                match registry_clone.renew(&current, &opts_clone).await {
                    Ok(renewed) => {
                        *lease_clone.lock().await = renewed;
                        count_clone.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "lease renewal failed; aborting renewer");
                        healthy_clone.store(false, std::sync::atomic::Ordering::SeqCst);
                        break;
                    }
                }
            }
        });

        Ok(Self {
            inner: Some(Inner {
                lease,
                registry,
                renewal,
                release_signal: None,
            }),
            renewal_count,
            renewer_healthy,
        })
    }

    /// Returns `false` if the background renewer has given up (e.g., `renew()`
    /// repeatedly failed). Callers holding the guard should treat this as a
    /// signal that the on-disk lease is no longer being extended: release and
    /// re-acquire, or stop using the version.
    pub fn is_renewer_healthy(&self) -> bool {
        self.renewer_healthy
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    pub async fn current(&self) -> VersionLease {
        // Reachable only between construction and `release(self)` / `drop`.
        self.inner
            .as_ref()
            .expect("LeaseGuard after release")
            .lease
            .lock()
            .await
            .clone()
    }

    /// Explicit, deterministic release. Prefer over relying on `Drop`.
    ///
    /// Order of operations matters to avoid zombie-lease resurrection:
    /// 1. `abort()` signals the renewer.
    /// 2. **Await the JoinHandle**: this returns immediately with
    ///    `Err(JoinError::Cancelled)` if the renewer was at an await point,
    ///    and waits until it reaches one otherwise. Guarantees no in-flight
    ///    `renew()` can write the lease file after the subsequent `release()`.
    /// 3. Then delete the lease file.
    pub async fn release(mut self) -> crate::Result<()> {
        if let Some(inner) = self.inner.take() {
            inner.renewal.abort();
            // Ignore JoinError (Cancelled is expected; Panicked is surfaced via
            // tracing so release always proceeds to delete the lease).
            if let Err(e) = inner.renewal.await
                && e.is_panic()
            {
                tracing::warn!(error = %e, "renewer task panicked; releasing anyway");
            }
            let snapshot = inner.lease.lock().await.clone();
            inner.registry.release(&snapshot).await?;
            if let Some(tx) = inner.release_signal {
                let _ = tx.send(());
            }
        }
        Ok(())
    }

    /// Test-only: observable counter incremented on each successful renew.
    /// Lets tests avoid wall-clock assertions on `expires_at`.
    #[cfg(any(test, feature = "testing"))]
    pub fn renewal_count_for_test(&self) -> Arc<AtomicU64> {
        self.renewal_count.clone()
    }

    /// Testing-only helper; not covered by semver. Installs a oneshot that
    /// fires after Drop's spawned release task completes.
    ///
    /// Consumes `self`. Drop runs at the end of this method call — it
    /// aborts the renewer and *schedules* an async release task on the
    /// current Tokio runtime. The scheduled task itself completes
    /// asynchronously; the signal fires after that task's `release` call
    /// returns. Callers should wait on the oneshot with a generous timeout.
    #[cfg(any(test, feature = "testing"))]
    pub fn set_release_signal_for_test(mut self, tx: oneshot::Sender<()>) {
        if let Some(inner) = self.inner.as_mut() {
            inner.release_signal = Some(tx);
        }
        // `self` drops at end-of-fn → `Drop::drop` runs → schedules async release.
    }
}

impl std::fmt::Debug for LeaseGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Manual impl — JoinHandle/Mutex/oneshot fields aren't all Debug.
        // Print only the caller-visible state.
        f.debug_struct("LeaseGuard")
            .field("released", &self.inner.is_none())
            .field(
                "renewer_healthy",
                &self
                    .renewer_healthy
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "renewal_count",
                &self.renewal_count.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl Drop for LeaseGuard {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        inner.renewal.abort();
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::debug!("LeaseGuard dropped without tokio runtime; relying on TTL");
            return;
        };
        let registry = inner.registry.clone();
        let lease = inner.lease.clone();
        let release_signal = inner.release_signal;
        let renewal = inner.renewal;
        handle.spawn(async move {
            // Await the renewer's JoinHandle BEFORE deleting the lease file
            // so any in-flight `put_opts` completes (or is cancelled) first.
            // Without this, a renew mid-flight can land after the delete and
            // resurrect the lease with a fresh TTL — same ordering hazard as
            // `release().await` handles for the explicit path.
            let _ = renewal.await;
            let snapshot = lease.lock().await.clone();
            let _ = registry.release(&snapshot).await;
            if let Some(tx) = release_signal {
                let _ = tx.send(());
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::lease::{LeaseOptions, LeaseRegistry, ObjectStoreLeaseRegistry};
    use object_store::{memory::InMemory, path::Path};
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    fn registry() -> Arc<ObjectStoreLeaseRegistry> {
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        Arc::new(ObjectStoreLeaseRegistry::new(
            store,
            Path::from("d/_versions/.leases"),
            Duration::from_secs(5),
        ))
    }

    /// Registry that always fails `renew` — used to drive `renewer_healthy`
    /// to `false` in the health test below.
    #[derive(Debug)]
    struct FailingRenewRegistry {
        inner: Arc<ObjectStoreLeaseRegistry>,
    }

    #[async_trait::async_trait]
    impl LeaseRegistry for FailingRenewRegistry {
        async fn acquire(
            &self,
            version: u64,
            opts: &LeaseOptions,
        ) -> crate::Result<VersionLease> {
            self.inner.acquire(version, opts).await
        }
        async fn renew(
            &self,
            _lease: &VersionLease,
            _opts: &LeaseOptions,
        ) -> crate::Result<VersionLease> {
            Err(crate::Error::io("injected renew failure"))
        }
        async fn release(&self, lease: &VersionLease) -> crate::Result<()> {
            self.inner.release(lease).await
        }
        async fn list_active(&self) -> crate::Result<Vec<VersionLease>> {
            self.inner.list_active().await
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn renewer_healthy_flips_to_false_on_renew_failure() {
        // Drives the `renewer_healthy` flag to `false` via a registry whose
        // `renew` always errors. Without this test the `false`-store branch in
        // the renewer loop is dead from a coverage standpoint and could be
        // silently inverted by a refactor.
        let reg: Arc<dyn LeaseRegistry> = Arc::new(FailingRenewRegistry {
            inner: registry(),
        });
        let guard = LeaseGuard::acquire(
            reg,
            1,
            LeaseOptions::default(),
            Duration::from_millis(10),
        )
        .await
        .unwrap();

        assert!(guard.is_renewer_healthy(), "starts healthy");

        // Wait for the renewer to tick once, fail, and flip the flag.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while guard.is_renewer_healthy() {
            if std::time::Instant::now() > deadline {
                panic!("renewer_healthy never flipped to false");
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(!guard.is_renewer_healthy());

        guard.release().await.unwrap();
    }

    // Real (non-paused) time is fine because we observe `renewal_count`, which is
    // purely a counter incremented inside the renewer. No wall-clock assertions.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn guard_renews_at_least_once_before_release() {
        let reg = registry();
        let opts = LeaseOptions::try_new(Duration::from_secs(60))
            .unwrap()
            .with_holder("h1");
        // 50 ms renew interval — with a multi-thread runtime, the renewer
        // will fire at least once within 300 ms.
        let guard = LeaseGuard::acquire(reg.clone(), 1, opts, Duration::from_millis(50))
            .await
            .unwrap();
        let counter = guard.renewal_count_for_test();
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        // Deterministic wait: poll the counter with a bounded timeout.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while counter.load(Ordering::SeqCst) < 1 {
            if std::time::Instant::now() > deadline {
                panic!("renewer did not fire within 2s");
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        guard.release().await.unwrap();
        assert_eq!(reg.list_active().await.unwrap().len(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_release_awaits_renewer_and_no_resurrection() {
        // Short renew_every so a renew might be in-flight when release hits.
        // `release()` contract: abort → join → delete. After release, the
        // lease file must stay deleted (no resurrection from a pending
        // renew that had already crossed its put_opts boundary).
        let reg = registry();
        let guard = LeaseGuard::acquire(
            reg.clone(),
            1,
            LeaseOptions::default(),
            Duration::from_millis(10),
        )
        .await
        .unwrap();
        assert_eq!(reg.list_active().await.unwrap().len(), 1);

        let counter = guard.renewal_count_for_test();
        guard.release().await.unwrap();
        assert_eq!(reg.list_active().await.unwrap().len(), 0);

        // Give any stragglers time to misbehave. InMemory's put is synchronous,
        // so if the renewer could resurrect the lease, 200ms is plenty for it
        // to do so.
        let count_after_release = counter.load(Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            reg.list_active().await.unwrap().len(),
            0,
            "lease must remain deleted after release — no zombie resurrection"
        );
        // The renewer may have ticked once between acquire and abort; we
        // only require that NO FURTHER ticks happen after release has
        // awaited the join handle.
        let count_final = counter.load(Ordering::SeqCst);
        assert_eq!(
            count_final, count_after_release,
            "renewer must stop ticking once release() returns"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn drop_releases_via_test_signal() {
        // Short renew_every so a renew is likely mid-flight when Drop fires —
        // exercises the Drop-path abort→join→delete ordering (no zombie
        // resurrection). Wait past the signal, then sleep and re-check.
        let reg = registry();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        {
            let guard = LeaseGuard::acquire(
                reg.clone(),
                1,
                LeaseOptions::default(),
                Duration::from_millis(10),
            )
            .await
            .unwrap();
            assert_eq!(reg.list_active().await.unwrap().len(), 1);
            guard.set_release_signal_for_test(tx); // consumes guard → Drop fires here
        }

        // 5s generous bound — heavy CI can delay multi-thread scheduling.
        tokio::time::timeout(Duration::from_secs(5), rx)
            .await
            .expect("release signal timed out")
            .expect("sender dropped");
        assert_eq!(reg.list_active().await.unwrap().len(), 0);

        // Wait for any stragglers; InMemory put is synchronous so 200ms is
        // plenty for a racing renew to misbehave. The lease must stay gone.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            reg.list_active().await.unwrap().len(),
            0,
            "Drop must not resurrect a lease after delete"
        );
    }
}
