// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Testing utilities

use crate::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use object_store::path::Path;
use object_store::{
    Error as OSError, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as OSResult,
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

// A policy function takes in the name of the operation (e.g. "put") and the location
// that is being accessed / modified and returns an optional error.
pub trait PolicyFnT: Fn(&str, &Path) -> Result<()> + Send + Sync {}
impl<F> PolicyFnT for F where F: Fn(&str, &Path) -> Result<()> + Send + Sync {}
impl Debug for dyn PolicyFnT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PolicyFn")
    }
}
type PolicyFn = Arc<dyn PolicyFnT>;

// These policy functions receive (and optionally transform) an ObjectMeta
// They apply to functions that list file info
pub trait ObjectMetaPolicyFnT: Fn(&str, ObjectMeta) -> Result<ObjectMeta> + Send + Sync {}
impl<F> ObjectMetaPolicyFnT for F where F: Fn(&str, ObjectMeta) -> Result<ObjectMeta> + Send + Sync {}
impl Debug for dyn ObjectMetaPolicyFnT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PolicyFn")
    }
}
type ObjectMetaPolicyFn = Arc<dyn ObjectMetaPolicyFnT>;

/// A policy container, meant to be shared between test code and the proxy object store.
///
/// This container allows you to configure policies that should apply to the proxied calls.
///
/// Typically, you would use this to simulate I/O errors or mock out data.
///
/// Currently, for simplicity, we only proxy calls that involve some kind of path.  Calls
/// to copy functions, which have a src and dst, will provide the source to the policy
#[derive(Debug, Default)]
pub struct ProxyObjectStorePolicy {
    /// Policies which run before a method is invoked.  If the policy returns
    /// an error then the target method will not be invoked and the error will
    /// be returned instead.
    before_policies: HashMap<String, PolicyFn>,
    /// Policies which run after calls that return ObjectMeta.  The policy can
    /// transform the returned ObjectMeta to mock out file listing results.
    object_meta_policies: HashMap<String, ObjectMetaPolicyFn>,
}

impl ProxyObjectStorePolicy {
    pub fn new() -> Self {
        Default::default()
    }

    /// Set a new policy with the given name
    ///
    /// The name can be used to later remove this policy
    pub fn set_before_policy(&mut self, name: &str, policy: PolicyFn) {
        self.before_policies.insert(name.to_string(), policy);
    }

    pub fn clear_before_policy(&mut self, name: &str) {
        self.before_policies.remove(name);
    }

    pub fn set_obj_meta_policy(&mut self, name: &str, policy: ObjectMetaPolicyFn) {
        self.object_meta_policies.insert(name.to_string(), policy);
    }
}

/// A proxy object store
///
/// This store wraps another object store and applies the given policy to all calls
/// made to the underlying store.  This can be used to simulate failures or, perhaps
/// in the future, to mock out results or provide other fine-grained control.
#[derive(Debug)]
pub struct ProxyObjectStore {
    target: Arc<dyn ObjectStore>,
    policy: Arc<Mutex<ProxyObjectStorePolicy>>,
}

impl ProxyObjectStore {
    pub fn new(target: Arc<dyn ObjectStore>, policy: Arc<Mutex<ProxyObjectStorePolicy>>) -> Self {
        Self { target, policy }
    }

    fn before_method(&self, method: &str, location: &Path) -> OSResult<()> {
        // Snapshot policies under the lock, then drop the guard before invoking
        // them so a policy that re-enters the proxy won't deadlock/panic.
        let snapshot: Vec<PolicyFn> = {
            let policy = self.policy.lock().unwrap();
            policy.before_policies.values().cloned().collect()
        };
        for p in snapshot {
            // `PolicyFn` returns `crate::Result<()>`; map into `object_store::Error`.
            p(method, location).map_err(OSError::from)?;
        }
        Ok(())
    }

    fn transform_meta(&self, method: &str, meta: ObjectMeta) -> OSResult<ObjectMeta> {
        let policy = self.policy.lock().unwrap();
        let mut meta = meta;
        for policy in policy.object_meta_policies.values() {
            meta = policy(method, meta).map_err(OSError::from)?;
        }
        Ok(meta)
    }
}

impl std::fmt::Display for ProxyObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ProxyObjectStore({})", self.target)
    }
}

/// An object store wrapper that counts listing operations.
///
/// This increments the shared counter for both `list` and `list_with_delimiter`
/// so tests can observe all listing-based directory and version discovery calls.
#[derive(Debug)]
pub struct CountingObjectStore {
    target: Arc<dyn ObjectStore>,
    listing_count: Arc<AtomicUsize>,
}

impl CountingObjectStore {
    pub fn new(target: Arc<dyn ObjectStore>, listing_count: Arc<AtomicUsize>) -> Self {
        Self {
            target,
            listing_count,
        }
    }

    fn record_listing(&self) {
        self.listing_count.fetch_add(1, Ordering::SeqCst);
    }

    fn delegate_list(
        &self,
        prefix: Option<&Path>,
    ) -> Pin<Box<dyn futures::Stream<Item = OSResult<ObjectMeta>> + Send>> {
        self.target.list(prefix)
    }
}

impl std::fmt::Display for CountingObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CountingObjectStore({})", self.target)
    }
}

#[async_trait]
impl ObjectStore for ProxyObjectStore {
    async fn put_opts(
        &self,
        location: &Path,
        bytes: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.before_method("put", location)?;
        self.target.put_opts(location, bytes, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        self.before_method("put_multipart", location)?;
        self.target.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.before_method("get_opts", location)?;
        self.target.get_opts(location, options).await
    }

    async fn get_range(&self, location: &Path, range: Range<u64>) -> OSResult<Bytes> {
        self.before_method("get_range", location)?;
        self.target.get_range(location, range).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
        self.before_method("get_ranges", location)?;
        self.target.get_ranges(location, ranges).await
    }

    async fn head(&self, location: &Path) -> OSResult<ObjectMeta> {
        self.before_method("head", location)?;
        let meta = self.target.head(location).await?;
        self.transform_meta("head", meta)
    }

    async fn delete(&self, location: &Path) -> OSResult<()> {
        self.before_method("delete", location)?;
        self.target.delete(location).await
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        let target = self.target.clone();
        let policy = Arc::clone(&self.policy);

        let prefix_path: Path = prefix.cloned().unwrap_or_else(|| Path::from(""));
        if let Err(e) = self.before_method("list", &prefix_path) {
            return futures::stream::once(async move { Err(e) }).boxed();
        }

        target
            .list(prefix)
            .and_then(move |meta| {
                let object_meta_policies: Vec<ObjectMetaPolicyFn> = {
                    let policy = policy.lock().unwrap();
                    policy.object_meta_policies.values().cloned().collect()
                };
                let mut meta = meta;
                // object_meta_policies are infallible transformations; panic
                // on policy error. Only `before_method` policies above can
                // fail, and those are propagated via `?`.
                for p in object_meta_policies {
                    meta = p("list", meta).map_err(OSError::from).unwrap();
                }
                future::ready(Ok(meta))
            })
            .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.target.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.before_method("copy", from)?;
        self.target.copy(from, to).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.before_method("rename", from)?;
        self.target.rename(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.before_method("copy_if_not_exists", from)?;
        self.target.copy_if_not_exists(from, to).await
    }
}

#[async_trait]
impl ObjectStore for CountingObjectStore {
    async fn put_opts(
        &self,
        location: &Path,
        bytes: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.target.put_opts(location, bytes, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        self.target.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.target.get_opts(location, options).await
    }

    async fn get_range(&self, location: &Path, range: Range<u64>) -> OSResult<Bytes> {
        self.target.get_range(location, range).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
        self.target.get_ranges(location, ranges).await
    }

    async fn head(&self, location: &Path) -> OSResult<ObjectMeta> {
        self.target.head(location).await
    }

    async fn delete(&self, location: &Path) -> OSResult<()> {
        self.target.delete(location).await
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.record_listing();
        self.delegate_list(prefix).boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.record_listing();
        self.target.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.target.copy(from, to).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.target.rename(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.target.copy_if_not_exists(from, to).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn list_fires_before_method() {
        use futures::StreamExt;
        let store: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        let policy = Arc::new(std::sync::Mutex::new(ProxyObjectStorePolicy::new()));
        let proxy = ProxyObjectStore::new(store, policy.clone());
        policy.lock().unwrap().set_before_policy(
            "fail_list",
            Arc::new(|op, _| -> crate::Result<()> {
                if op == "list" {
                    Err(crate::Error::io("boom"))
                } else {
                    Ok(())
                }
            }),
        );
        let results: Vec<_> = proxy.list(None).collect().await;
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err(), "list should surface the injected error");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reentrant_policy_does_not_deadlock() {
        // Regression for the snapshot-then-release refactor: a policy that calls
        // back into the proxy (triggering another `before_method`) must not
        // deadlock/panic on a non-reentrant `std::sync::Mutex`.
        //
        // Requires `multi_thread` flavor because the re-entry uses `block_in_place`,
        // which panics on a `current_thread` runtime.
        use object_store::{path::Path, ObjectStore, PutPayload};
        let inner: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let policy = Arc::new(std::sync::Mutex::new(ProxyObjectStorePolicy::new()));
        let proxy = Arc::new(ProxyObjectStore::new(inner, policy.clone()));

        let proxy_for_policy = proxy.clone();
        policy.lock().unwrap().set_before_policy(
            "reenter_on_head",
            Arc::new(move |op, _| -> crate::Result<()> {
                if op == "head" {
                    let proxy = proxy_for_policy.clone();
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async move {
                            let _ = proxy
                                .put(&Path::from("nested/blob"), PutPayload::from_static(b"x"))
                                .await;
                        });
                    });
                }
                Ok(())
            }),
        );

        let _ = proxy.head(&Path::from("does_not_exist")).await;
        // If we got here without panic, the re-entry fix holds.
    }
}
