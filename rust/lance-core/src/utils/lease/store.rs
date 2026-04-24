// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use futures::StreamExt;
use object_store::{Error as OsError, ObjectStore, PutMode, PutOptions, path::Path};

use super::{LEASE_EXTENSION, LeaseOptions, LeaseRegistry, VersionLease};

/// Object-store-backed `LeaseRegistry`. One file per lease under
/// `lease_dir`. Multi-process safe.
#[derive(Debug)]
pub struct ObjectStoreLeaseRegistry {
    store: Arc<dyn ObjectStore>,
    lease_dir: Path,
    skew_grace: Duration,
}

impl ObjectStoreLeaseRegistry {
    /// Constructor is `pub` so `Dataset::lease_registry()` in the `lance`
    /// crate can wire it up. Direct users bypass `Dataset` path conventions
    /// at their own risk.
    pub fn new(store: Arc<dyn ObjectStore>, lease_dir: Path, skew_grace: Duration) -> Self {
        Self {
            store,
            lease_dir,
            skew_grace,
        }
    }

    fn lease_path(&self, lease_id: uuid::Uuid) -> Path {
        self.lease_dir
            .child(format!("{lease_id}.{LEASE_EXTENSION}"))
    }

    async fn write_lease(&self, lease: &VersionLease, mode: PutMode) -> crate::Result<()> {
        let path = self.lease_path(lease.lease_id);
        let body = lease.to_json_bytes()?;
        let opts = PutOptions {
            mode,
            ..Default::default()
        };
        self.store
            .put_opts(&path, body.into(), opts)
            .await
            .map_err(|e| crate::Error::io(format!("write lease {path}: {e}")))?;
        Ok(())
    }

    /// Testing-only helper; not covered by semver.
    #[cfg(any(test, feature = "testing"))]
    pub async fn write_lease_for_test(&self, lease: &VersionLease) -> crate::Result<()> {
        self.write_lease(lease, PutMode::Overwrite).await
    }

    #[cfg(any(test, feature = "testing"))]
    pub async fn write_raw_for_test(&self, path: &Path, bytes: Vec<u8>) -> crate::Result<()> {
        self.store
            .put(path, bytes.into())
            .await
            .map_err(|e| crate::Error::io(format!("write_raw: {e}")))?;
        Ok(())
    }

    #[cfg(any(test, feature = "testing"))]
    pub fn lease_dir_for_test(&self) -> &Path {
        &self.lease_dir
    }
}

#[async_trait]
impl LeaseRegistry for ObjectStoreLeaseRegistry {
    async fn acquire(&self, version: u64, opts: &LeaseOptions) -> crate::Result<VersionLease> {
        let lease = VersionLease {
            lease_id: uuid::Uuid::new_v4(),
            version,
            expires_at: Utc::now()
                + chrono::Duration::from_std(opts.ttl())
                    .map_err(|e| crate::Error::invalid_input(format!("invalid ttl: {e}")))?,
            holder: opts.holder().map(|s| s.to_string()),
        };
        // PutMode::Create would be ideal on backends that support conditional
        // writes, but LocalFileSystem does not. A fresh v4 UUID makes
        // collision probability ≈ 2^-122 so Overwrite is safe.
        self.write_lease(&lease, PutMode::Overwrite).await?;
        Ok(lease)
    }

    async fn renew(
        &self,
        lease: &VersionLease,
        opts: &LeaseOptions,
    ) -> crate::Result<VersionLease> {
        let path = self.lease_path(lease.lease_id);
        self.store.head(&path).await.map_err(|e| match e {
            OsError::NotFound { .. } => crate::Error::io(format!(
                "renew: lease {} not found (released or reaped)",
                lease.lease_id
            )),
            other => crate::Error::io(format!("renew head {path}: {other}")),
        })?;

        let computed_expiry = Utc::now()
            + chrono::Duration::from_std(opts.ttl())
                .map_err(|e| crate::Error::invalid_input(format!("invalid ttl: {e}")))?;
        // Monotonicity: never let renew REDUCE expires_at. A backward NTP
        // step on the writer would otherwise shrink protection on a
        // just-renewed lease. Take the max of previous and newly computed.
        let expires_at = std::cmp::max(computed_expiry, lease.expires_at);
        let renewed = VersionLease {
            lease_id: lease.lease_id,
            version: lease.version,
            expires_at,
            holder: opts
                .holder()
                .map(|s| s.to_string())
                .or_else(|| lease.holder.clone()),
        };
        self.write_lease(&renewed, PutMode::Overwrite).await?;
        Ok(renewed)
    }

    async fn release(&self, lease: &VersionLease) -> crate::Result<()> {
        match self.store.delete(&self.lease_path(lease.lease_id)).await {
            Ok(()) | Err(OsError::NotFound { .. }) => Ok(()),
            Err(e) => Err(crate::Error::io(format!("release lease: {e}"))),
        }
    }

    async fn list_active(&self) -> crate::Result<Vec<VersionLease>> {
        // Snapshot `now` up front so a long list doesn't expire earlier leases mid-walk.
        let now = Utc::now();
        let skew_grace = self.skew_grace;

        // Collect paths first so a listing error fails fast (cleanup fails closed).
        let mut paths: Vec<Path> = Vec::new();
        let mut stream = self.store.list(Some(&self.lease_dir));
        while let Some(item) = stream.next().await {
            match item {
                Ok(meta) => paths.push(meta.location),
                Err(OsError::NotFound { .. }) => return Ok(Vec::new()), // dir absent
                Err(e) => return Err(crate::Error::io(format!("list leases: {e}"))),
            }
        }

        let mut out = Vec::with_capacity(paths.len());
        for path in paths {
            // Fail-closed for transient I/O errors: a `get` that times out
            // or fails mid-stream could hide a real conflicting lease from
            // cleanup's phase-2 probe. `NotFound` is the one legitimate skip
            // — it just means the lease was released between our `list` and
            // our `get`, which is harmless. Parse errors are also skipped
            // (one malformed file cannot block cleanup).
            let bytes = match self.store.get(&path).await {
                Ok(g) => g.bytes().await.map_err(|e| {
                    crate::Error::io(format!("read lease body {path}: {e}"))
                })?,
                Err(OsError::NotFound { .. }) => continue,
                Err(e) => return Err(crate::Error::io(format!("get lease {path}: {e}"))),
            };
            match VersionLease::from_json_bytes(&bytes) {
                Ok(lease) if !lease.is_expired(now, skew_grace) => out.push(lease),
                Ok(_) => {}
                Err(e) => tracing::warn!(path = %path, error = %e, "parse lease"),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::{memory::InMemory, path::Path};
    use std::sync::Arc;
    use std::time::Duration;

    fn registry() -> ObjectStoreLeaseRegistry {
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        ObjectStoreLeaseRegistry::new(
            store,
            Path::from("d/_versions/.leases"),
            Duration::from_secs(10),
        )
    }

    #[tokio::test]
    async fn acquire_then_list_returns_lease_with_correct_ttl() {
        let reg = registry();
        let opts = LeaseOptions::try_new(Duration::from_secs(60))
            .unwrap()
            .with_holder("r1");
        let before = chrono::Utc::now();
        let lease = reg.acquire(7, &opts).await.unwrap();
        let after = chrono::Utc::now();

        assert_eq!(lease.version(), 7);
        assert_eq!(lease.holder(), Some("r1"));
        let min = before + chrono::Duration::seconds(60);
        let max = after + chrono::Duration::seconds(60) + chrono::Duration::milliseconds(50);
        assert!(
            lease.expires_at() >= min && lease.expires_at() <= max,
            "expires_at={} outside [{min}, {max}]",
            lease.expires_at()
        );

        let active = reg.list_active().await.unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].lease_id(), lease.lease_id());
    }

    #[tokio::test]
    async fn expired_lease_is_filtered_out() {
        let reg = registry();
        let expired = VersionLease::new_for_test(
            uuid::Uuid::new_v4(),
            1,
            chrono::Utc::now() - chrono::Duration::seconds(30),
            None,
        );
        reg.write_lease_for_test(&expired).await.unwrap();
        assert!(reg.list_active().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn skew_grace_keeps_recently_expired_lease_alive() {
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let reg = ObjectStoreLeaseRegistry::new(
            store,
            Path::from("d/_versions/.leases"),
            Duration::from_secs(30),
        );
        let stale = VersionLease::new_for_test(
            uuid::Uuid::new_v4(),
            1,
            chrono::Utc::now() - chrono::Duration::seconds(10),
            None,
        );
        reg.write_lease_for_test(&stale).await.unwrap();
        assert_eq!(reg.list_active().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn renew_extends_expiry_by_ttl() {
        // NOTE: Not `start_paused` — `ObjectStoreLeaseRegistry::renew` uses
        // `chrono::Utc::now()` which is NOT affected by tokio's virtual clock.
        // The assertion tolerates the wall-clock delta between acquire and renew
        // (microseconds on a warm CPU); we verify the DELTA BETWEEN TTLs,
        // independent of when the calls happen.
        let reg = registry();
        let short = LeaseOptions::try_new(Duration::from_secs(5)).unwrap();
        let long = LeaseOptions::try_new(Duration::from_secs(60)).unwrap();
        let lease = reg.acquire(1, &short).await.unwrap();
        let renewed = reg.renew(&lease, &long).await.unwrap();
        let gap = renewed.expires_at() - lease.expires_at();
        // gap = (renewed_time + 60s) - (acquire_time + 5s) ≈ 55s + elapsed
        assert!(
            gap.num_seconds() >= 55 && gap.num_seconds() <= 58,
            "gap was {gap} seconds; expected ≈55s (with small wall-clock jitter)"
        );
    }

    #[tokio::test]
    async fn renew_of_released_lease_fails() {
        let reg = registry();
        let opts = LeaseOptions::default();
        let lease = reg.acquire(1, &opts).await.unwrap();
        reg.release(&lease).await.unwrap();
        let err = reg.renew(&lease, &opts).await.unwrap_err();
        // Match the unique plan-guaranteed phrase, not just "not found" which
        // may flake on object_store Display format changes.
        assert!(
            err.to_string().contains("released or reaped"),
            "renew of released lease should mention 'released or reaped': {err}"
        );
    }

    #[tokio::test]
    async fn renew_is_monotonic_under_backward_clock_step() {
        // Simulate a backward NTP step: existing lease has `expires_at` far
        // in the future; renew with a SHORT ttl whose `now + ttl` is
        // earlier than the existing expiry. The `max(prev, computed)` guard
        // must preserve the longer expiry — a swap to `min`, or dropping the
        // guard, would shrink protection silently.
        let reg = registry();
        let far_future = chrono::Utc::now() + chrono::Duration::hours(1);
        let existing = VersionLease::new_for_test(uuid::Uuid::new_v4(), 1, far_future, None);
        reg.write_lease_for_test(&existing).await.unwrap();

        let short = LeaseOptions::try_new(Duration::from_secs(5)).unwrap();
        let renewed = reg.renew(&existing, &short).await.unwrap();
        assert!(
            renewed.expires_at() >= existing.expires_at(),
            "renew must not reduce expires_at (got {}, previously {})",
            renewed.expires_at(),
            existing.expires_at()
        );
    }

    #[tokio::test]
    async fn release_is_idempotent() {
        let reg = registry();
        let lease = reg.acquire(1, &LeaseOptions::default()).await.unwrap();
        reg.release(&lease).await.unwrap();
        reg.release(&lease).await.unwrap();
        assert!(reg.list_active().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn list_active_on_missing_dir_is_empty() {
        let reg = registry();
        assert!(reg.list_active().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn list_active_fails_closed_on_get_error() {
        // A transient `get` failure (not `NotFound`) must propagate — otherwise
        // a mid-flight object-store blip could hide a conflicting lease from
        // cleanup's phase-2 probe. NotFound is still tolerated (release race).
        use crate::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
        let inner: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let policy = Arc::new(std::sync::Mutex::new(ProxyObjectStorePolicy::new()));
        let proxy: Arc<dyn object_store::ObjectStore> =
            Arc::new(ProxyObjectStore::new(inner, policy.clone()));
        let reg = ObjectStoreLeaseRegistry::new(
            proxy,
            Path::from("d/_versions/.leases"),
            Duration::from_secs(10),
        );
        reg.acquire(1, &LeaseOptions::default()).await.unwrap();

        // Inject a transient `get` failure for lease files. The policy
        // hook needs to return the error for `get_opts` (what `get` uses).
        policy.lock().unwrap().set_before_policy(
            "flaky_get",
            Arc::new(|op, path| -> crate::Result<()> {
                if (op == "get_opts" || op == "get_range")
                    && path.as_ref().contains(".leases")
                {
                    Err(crate::Error::io("simulated transient get failure"))
                } else {
                    Ok(())
                }
            }),
        );

        let err = reg.list_active().await.expect_err("must fail closed");
        assert!(
            err.to_string().contains("simulated") || err.to_string().contains("get lease"),
            "error should surface get failure: {err}"
        );
    }

    #[tokio::test]
    async fn list_active_skips_malformed_files() {
        let reg = registry();
        reg.write_raw_for_test(
            &reg.lease_dir_for_test()
                .child(format!("{}.lease", uuid::Uuid::new_v4())),
            b"this is not JSON".to_vec(),
        )
        .await
        .unwrap();
        let lease = reg.acquire(1, &LeaseOptions::default()).await.unwrap();
        let active = reg.list_active().await.unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].lease_id(), lease.lease_id());
    }

    #[tokio::test]
    async fn two_registries_sharing_store_see_each_others_leases() {
        // Cross-process contract: two registries rooted at the same object
        // store + lease_dir must observe each other's leases.
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let dir = Path::from("d/_versions/.leases");
        let reg_a =
            ObjectStoreLeaseRegistry::new(store.clone(), dir.clone(), Duration::from_secs(10));
        let reg_b = ObjectStoreLeaseRegistry::new(store, dir, Duration::from_secs(10));

        let lease = reg_a.acquire(42, &LeaseOptions::default()).await.unwrap();
        let seen_by_b = reg_b.list_active().await.unwrap();
        assert_eq!(seen_by_b.len(), 1);
        assert_eq!(seen_by_b[0].lease_id(), lease.lease_id());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_acquires_produce_distinct_leases() {
        // Smoke test: barrier-synchronized parallel acquires must each persist
        // their lease (list_active sees TWO distinct ids). Fresh v4 UUIDs are
        // always distinct regardless of concurrency, so the stronger assertion
        // is `len == 2` on the registry — a broken impl that overwrites by
        // path collision would fail it.
        let reg = Arc::new(registry());
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let reg_a = reg.clone();
        let barrier_a = barrier.clone();
        let a = tokio::spawn(async move {
            barrier_a.wait().await;
            reg_a.acquire(5, &LeaseOptions::default()).await.unwrap()
        });
        let reg_b = reg.clone();
        let barrier_b = barrier.clone();
        let b = tokio::spawn(async move {
            barrier_b.wait().await;
            reg_b.acquire(5, &LeaseOptions::default()).await.unwrap()
        });
        let (la, lb) = tokio::join!(a, b);
        let (la, lb) = (la.unwrap(), lb.unwrap());
        assert_ne!(la.lease_id(), lb.lease_id());
        assert_eq!(reg.list_active().await.unwrap().len(), 2);
    }
}
