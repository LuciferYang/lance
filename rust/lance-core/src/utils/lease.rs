// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Version leases that pin dataset versions against cleanup.
//!
//! See `docs/design/active-metadata-lease.md` for the full design.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::Error;

mod renewal;
mod store;
pub use renewal::LeaseGuard;
pub use store::ObjectStoreLeaseRegistry;

pub(crate) const LEASE_EXTENSION: &str = "lease";
pub(crate) const DEFAULT_LEASE_TTL: std::time::Duration = std::time::Duration::from_secs(60);
pub(crate) const MAX_LEASE_TTL: std::time::Duration = std::time::Duration::from_secs(600);
/// Recommended default renewal interval for `LeaseGuard::acquire` — one-third
/// of `DEFAULT_LEASE_TTL`, giving three renewal chances per TTL window.
pub const DEFAULT_LEASE_RENEW_EVERY: std::time::Duration = std::time::Duration::from_secs(20);

/// A lease claimed by an active task on a specific dataset version.
///
/// Cleanup treats any version with a live (non-expired) lease as if it were
/// tagged — its manifest, data, index, and deletion files are retained.
///
/// Fields are `pub(crate)`; callers use the accessors. The on-disk layout is
/// JSON and is part of the cross-process contract. Unknown fields are
/// intentionally accepted on deserialization for forward compatibility.
///
/// Wire-format codec note: this type relies on `serde_json`'s lenient
/// handling of missing `Option<T>` keys (treated as `None`). If the wire
/// codec is ever switched to one that requires every key to be present
/// (`bincode`, `messagepack`, etc.), annotate all optional fields with
/// `#[serde(default)]` before the codec change.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionLease {
    pub(crate) lease_id: uuid::Uuid,
    pub(crate) version: u64,
    pub(crate) expires_at: DateTime<Utc>,
    /// Diagnostic label for the holder (hostname, PID, job ID). Not used for
    /// retention decisions — identity is by `lease_id` alone.
    pub(crate) holder: Option<String>,
}

impl VersionLease {
    pub fn lease_id(&self) -> uuid::Uuid {
        self.lease_id
    }
    pub fn version(&self) -> u64 {
        self.version
    }
    pub fn expires_at(&self) -> DateTime<Utc> {
        self.expires_at
    }
    pub fn holder(&self) -> Option<&str> {
        self.holder.as_deref()
    }

    /// A lease is expired at `now` iff `expires_at + skew_grace <= now`.
    /// Grace overflow (should never happen with reasonable durations) is
    /// treated conservatively: grace saturates to `max_value`, keeping the
    /// lease alive rather than silently removing protection.
    pub fn is_expired(&self, now: DateTime<Utc>, skew_grace: std::time::Duration) -> bool {
        let grace = chrono::Duration::from_std(skew_grace).unwrap_or(chrono::Duration::MAX);
        self.expires_at + grace <= now
    }

    pub fn to_json_bytes(&self) -> crate::Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| Error::io(format!("serialize lease: {e}")))
    }

    pub fn from_json_bytes(bytes: &[u8]) -> crate::Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| Error::io(format!("parse lease: {e}")))
    }

    /// Testing-only helper; not covered by semver. Enable via
    /// `features = ["testing"]` in dev-dependencies only. Used by cross-crate
    /// integration tests; `#[cfg(test)]` alone would not be visible.
    #[cfg(any(test, feature = "testing"))]
    pub fn new_for_test(
        lease_id: uuid::Uuid,
        version: u64,
        expires_at: DateTime<Utc>,
        holder: Option<String>,
    ) -> Self {
        Self {
            lease_id,
            version,
            expires_at,
            holder,
        }
    }
}

/// Options controlling a single lease acquisition. Construct via `try_new`
/// for TTL validation, then chain `with_holder` if a diagnostic label is desired.
///
/// Deliberately does NOT use `impl Into<Option<String>>` on `try_new` — that
/// trait bound does not accept `&str` literals because `Option<String>` has no
/// `From<&str>` impl. The builder pattern below keeps the ergonomics while
/// compiling.
#[derive(Clone, Debug)]
pub struct LeaseOptions {
    ttl: std::time::Duration,
    holder: Option<String>,
}

impl LeaseOptions {
    pub fn try_new(ttl: std::time::Duration) -> crate::Result<Self> {
        if ttl.is_zero() {
            return Err(Error::invalid_input("lease ttl must be > 0"));
        }
        if ttl > MAX_LEASE_TTL {
            return Err(Error::invalid_input(format!(
                "lease ttl {ttl:?} exceeds MAX_LEASE_TTL ({MAX_LEASE_TTL:?})"
            )));
        }
        Ok(Self { ttl, holder: None })
    }

    pub fn with_holder(mut self, holder: impl Into<String>) -> Self {
        self.holder = Some(holder.into());
        self
    }

    pub fn ttl(&self) -> std::time::Duration {
        self.ttl
    }
    pub fn holder(&self) -> Option<&str> {
        self.holder.as_deref()
    }
}

impl Default for LeaseOptions {
    fn default() -> Self {
        Self {
            ttl: DEFAULT_LEASE_TTL,
            holder: None,
        }
    }
}

use async_trait::async_trait;

#[async_trait]
pub trait LeaseRegistry: Send + Sync + std::fmt::Debug {
    async fn acquire(&self, version: u64, opts: &LeaseOptions) -> crate::Result<VersionLease>;
    async fn renew(&self, lease: &VersionLease, opts: &LeaseOptions) -> crate::Result<VersionLease>;
    async fn release(&self, lease: &VersionLease) -> crate::Result<()>;
    async fn list_active(&self) -> crate::Result<Vec<VersionLease>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use std::time::Duration;

    fn lease_at(expires_at: chrono::DateTime<chrono::Utc>) -> VersionLease {
        VersionLease {
            lease_id: uuid::Uuid::nil(),
            version: 42,
            expires_at,
            holder: Some("reader-1".into()),
        }
    }

    #[test]
    fn round_trips_through_json() {
        let lease = lease_at(chrono::Utc.with_ymd_and_hms(2030, 1, 1, 0, 0, 0).unwrap());
        let bytes = lease.to_json_bytes().unwrap();
        let back = VersionLease::from_json_bytes(&bytes).unwrap();
        assert_eq!(lease, back);
    }

    #[test]
    fn is_expired_boundary_inclusive_at_equal_now() {
        let t = chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        assert!(lease_at(t).is_expired(t, Duration::ZERO));
    }

    #[test]
    fn is_expired_respects_skew_grace() {
        let now = chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 5).unwrap();
        let lease = lease_at(chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap());
        assert!(!lease.is_expired(now, Duration::from_secs(10)));
        assert!(lease.is_expired(now, Duration::from_secs(4)));
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(VersionLease::from_json_bytes(b"").is_err());
        assert!(VersionLease::from_json_bytes(b"{}").is_err());
        assert!(VersionLease::from_json_bytes(b"{\"version\": \"nope\"}").is_err());
    }

    #[test]
    fn forward_compat_accepts_unknown_fields() {
        let json = br#"{
            "lease_id": "00000000-0000-0000-0000-000000000000",
            "version": 1,
            "expires_at": "2030-01-01T00:00:00Z",
            "holder": null,
            "future_field_we_dont_know_about": 123
        }"#;
        let lease = VersionLease::from_json_bytes(json).unwrap();
        assert_eq!(lease.version(), 1);
    }

    #[test]
    fn lease_options_rejects_zero_ttl_and_over_cap() {
        assert!(LeaseOptions::try_new(Duration::ZERO).is_err());
        assert!(LeaseOptions::try_new(Duration::from_secs(10_000)).is_err());
        assert!(LeaseOptions::try_new(Duration::from_secs(60)).is_ok());
    }

    #[test]
    fn lease_options_builder_accepts_holder() {
        let a = LeaseOptions::try_new(Duration::from_secs(10))
            .unwrap()
            .with_holder("r1");
        assert_eq!(a.holder(), Some("r1"));
        let b = LeaseOptions::try_new(Duration::from_secs(10))
            .unwrap()
            .with_holder(String::from("r2"));
        assert_eq!(b.holder(), Some("r2"));
        let c = LeaseOptions::try_new(Duration::from_secs(10)).unwrap();
        assert_eq!(c.holder(), None);
    }

    #[test]
    fn lease_registry_is_object_safe_and_has_expected_methods() {
        use async_trait::async_trait;

        #[derive(Debug, Default)]
        struct Noop;

        #[async_trait]
        impl LeaseRegistry for Noop {
            async fn acquire(&self, version: u64, _opts: &LeaseOptions) -> crate::Result<VersionLease> {
                Ok(VersionLease {
                    lease_id: uuid::Uuid::nil(),
                    version,
                    expires_at: chrono::Utc::now(),
                    holder: None,
                })
            }
            async fn renew(&self, lease: &VersionLease, _opts: &LeaseOptions) -> crate::Result<VersionLease> {
                Ok(lease.clone())
            }
            async fn release(&self, _lease: &VersionLease) -> crate::Result<()> { Ok(()) }
            async fn list_active(&self) -> crate::Result<Vec<VersionLease>> { Ok(vec![]) }
        }

        let reg: std::sync::Arc<dyn LeaseRegistry> = std::sync::Arc::new(Noop);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let lease = rt.block_on(reg.acquire(7, &LeaseOptions::default())).unwrap();
        assert_eq!(lease.version(), 7);
        assert!(rt.block_on(reg.list_active()).unwrap().is_empty());
    }
}
