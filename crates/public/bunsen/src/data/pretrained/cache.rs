//! # Weights cache
//!
//! Digest-pinned weights under the disk cache's cache directory.
//! [`BunsenDiskCache`] decides *where*; this decides *what is trusted there*.
//! A pinned descriptor's file lives at
//!
//! ```text
//! <cache>/weights/<kit>/<provider>/<sha256>/<file>
//! ```
//!
//! The digest in the path is the pin: a file at that path was verified when
//! it was written, so it is trusted on later runs without re-hashing 3 GB,
//! and a re-pinned model cannot collide with a stale one. An unpinned
//! descriptor keeps the older `<cache>/weights/<key>/<file>` layout, keyed
//! by its first URL, so entries fetched before pinning existed stay valid.
//!
//! [`WeightsCache::resolve`] consults the cache first, then each source in
//! the descriptor's order: a bundled file is used in place, a file in
//! another tool's directory is checked and linked in, and a URL is fetched,
//! checked and written in. Every transfer reports to the disk cache's
//! observer stack.
//!
//! A [`Resource`] of a [`ResourceMap`] lives under a root of its own,
//!
//! ```text
//! <cache>/pretrained/<kit>/<namespace>/<sha256>/<file>
//! ```
//!
//! and is resolved by [`WeightsCache::resolve_resource`]: the cache first,
//! then each source in order, a file in another tool's directory used in
//! place (a "trust me" source, hashed only when the options ask), a URL
//! fetched, checked and written in. Nothing is linked or copied into the
//! cache. [`WeightsCache::load`] does it for a whole map, fetching what is
//! remote together under the options' policy. The descriptor path above is
//! the older one, and keeps its behavior until it retires.

use std::{
    collections::BTreeMap,
    fmt,
    fs,
    path::{
        Path,
        PathBuf,
    },
};

use super::{
    LoadedResources,
    PretrainedWeightsDescriptor,
    Resource,
    ResourceMap,
    WeightsSource,
    pretrained_weights_resource_key,
};
#[cfg(feature = "fetch")]
use crate::data::cache::{
    FetchJob,
    FetchOutcome,
    FetchPolicy,
};
use crate::{
    data::cache::{
        BunsenDiskCache,
        BunsenDiskCacheOptions,
        link_or_copy,
        verify_sha256,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// The directory under the cache dir that holds weights.
pub const WEIGHTS_DIR: &str = "weights";

/// The directory under the cache dir that holds a pretrained's resources:
/// `<cache>/pretrained/<kit>/<namespace>/<sha256>/<file>`.
pub const PRETRAINED_DIR: &str = "pretrained";

/// Options for [`WeightsCache`].
#[derive(Clone, Debug, Default)]
pub struct WeightsCacheOptions {
    /// The disk cache underneath: where the cache directory is, and who
    /// watches transfers.
    pub disk: BunsenDiskCacheOptions,

    /// Never reach the network: weights not already local are an error.
    pub offline: bool,

    /// Overrides for [`WeightsSource::LocalDir`] sources, by the source's
    /// name: where another tool's directory is on this machine.
    pub local_dirs: BTreeMap<String, PathBuf>,

    /// Hash a pinned file found in a local-dir source before it is used in
    /// place. Off by default: a local dir is a "trust me" source, and
    /// hashing a 3 GB checkpoint on every run is not trust. A caller that
    /// is not sure has `verify_sha256`.
    pub verify_local_dirs: bool,

    /// How the remote resources of a map are fetched together.
    #[cfg(feature = "fetch")]
    pub fetch: FetchPolicy,
}

impl WeightsCacheOptions {
    /// Sets the disk cache options.
    pub fn with_disk(
        mut self,
        disk: BunsenDiskCacheOptions,
    ) -> Self {
        self.disk = disk;
        self
    }

    /// Sets whether the network is off limits.
    pub fn with_offline(
        mut self,
        offline: bool,
    ) -> Self {
        self.offline = offline;
        self
    }

    /// Overrides the directory of the local-dir source called `name`.
    pub fn with_local_dir(
        mut self,
        name: impl Into<String>,
        dir: impl Into<PathBuf>,
    ) -> Self {
        self.local_dirs.insert(name.into(), dir.into());
        self
    }

    /// Sets whether a pinned local-dir file is hashed before use.
    pub fn with_verify_local_dirs(
        mut self,
        verify: bool,
    ) -> Self {
        self.verify_local_dirs = verify;
        self
    }

    /// Sets how a map's remote resources are fetched together.
    #[cfg(feature = "fetch")]
    pub fn with_fetch_policy(
        mut self,
        fetch: FetchPolicy,
    ) -> Self {
        self.fetch = fetch;
        self
    }
}

/// Where a resolved file came from, this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// Already in the cache.
    Cached,
    /// A bundled file, used in place.
    File,
    /// Found in another tool's directory, checked, and linked into the cache.
    LocalDir,
    /// Fetched from a URL, checked, and written to the cache.
    Downloaded,
    /// A path the caller gave; not a cache entry.
    Given,
}

impl fmt::Display for Provenance {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(match self {
            Self::Cached => "cached",
            Self::File => "bundled file",
            Self::LocalDir => "local dir",
            Self::Downloaded => "downloaded",
            Self::Given => "given",
        })
    }
}

/// A local file holding a descriptor's weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedWeights {
    /// The file.
    pub path: PathBuf,

    /// Where it came from.
    pub provenance: Provenance,
}

/// Where a descriptor's weights stand before anything is fetched, for a
/// listing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStatus {
    /// In the cache.
    Cached,
    /// Not in the cache, but a bundled file is on disk.
    File,
    /// Not in the cache, but another tool's directory has a file of that
    /// name, unchecked.
    LocalDir,
    /// Only a URL.
    Remote,
}

impl fmt::Display for CacheStatus {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(match self {
            Self::Cached => "cached",
            Self::File => "bundled file",
            Self::LocalDir => "local dir",
            Self::Remote => "remote",
        })
    }
}

/// Digest-pinned weights under the disk cache's cache directory.
pub struct WeightsCache {
    disk: BunsenDiskCache,
    offline: bool,
    local_dirs: BTreeMap<String, PathBuf>,
    verify_local_dirs: bool,
    #[cfg(feature = "fetch")]
    fetch: FetchPolicy,
}

impl WeightsCache {
    /// Opens the cache.
    ///
    /// # Errors
    /// As [`BunsenDiskCache::new`].
    pub fn new(options: WeightsCacheOptions) -> BunsenResult<Self> {
        Ok(Self {
            disk: BunsenDiskCache::new(options.disk)?,
            offline: options.offline,
            local_dirs: options.local_dirs,
            verify_local_dirs: options.verify_local_dirs,
            #[cfg(feature = "fetch")]
            fetch: options.fetch,
        })
    }

    /// The disk cache underneath.
    pub fn disk(&self) -> &BunsenDiskCache {
        &self.disk
    }

    /// The cache directory.
    pub fn cache_dir(&self) -> &Path {
        self.disk.cache_dir()
    }

    /// Whether the network is off limits.
    pub fn offline(&self) -> bool {
        self.offline
    }

    /// Whether a pinned local-dir file is hashed before use.
    pub fn verify_local_dirs(&self) -> bool {
        self.verify_local_dirs
    }

    /// The local-dir overrides, by source name.
    pub fn local_dirs(&self) -> &BTreeMap<String, PathBuf> {
        &self.local_dirs
    }

    /// The directory a local-dir source resolves to: the override for its
    /// name, else the directory the source carries.
    pub fn local_dir(
        &self,
        source: &WeightsSource,
    ) -> Option<PathBuf> {
        match source {
            WeightsSource::LocalDir { name, dir } => {
                self.local_dirs.get(name).cloned().or_else(|| dir.clone())
            }
            _ => None,
        }
    }

    /// Where a descriptor's file lives in the cache, present or not.
    pub fn cached_path(
        &self,
        kit: &str,
        provider: &str,
        desc: &PretrainedWeightsDescriptor,
    ) -> PathBuf {
        match &desc.sha256 {
            Some(sha256) => self
                .disk
                .cache_path(&[WEIGHTS_DIR, kit, provider, sha256], &desc.file),
            None => {
                let key = pretrained_weights_resource_key(&desc.cache_key());
                self.disk.cache_path(&key, &desc.file)
            }
        }
    }

    /// Where a descriptor's weights stand, without fetching or hashing
    /// anything.
    pub fn status(
        &self,
        kit: &str,
        provider: &str,
        desc: &PretrainedWeightsDescriptor,
    ) -> CacheStatus {
        if self.cached_path(kit, provider, desc).is_file() {
            return CacheStatus::Cached;
        }
        for source in &desc.sources {
            match source {
                WeightsSource::File(path) if path.is_file() => return CacheStatus::File,
                WeightsSource::LocalDir { .. } => {
                    if let Some(dir) = self.local_dir(source)
                        && dir.join(&desc.file).is_file()
                    {
                        return CacheStatus::LocalDir;
                    }
                }
                _ => {}
            }
        }
        CacheStatus::Remote
    }

    /// Brings a descriptor's weights local, and says where they came from.
    ///
    /// The cache is consulted first; then each source in the descriptor's
    /// order: a bundled file is used in place, a file in another tool's
    /// directory is checked against the digest (when pinned) and linked in,
    /// and the URLs are fetched in order, checked, and written in. A local
    /// file that fails its digest is passed over rather than fatal: it is
    /// another tool's directory, and may hold a partial download.
    ///
    /// # Errors
    /// [`BunsenError::ResourceNotFound`] if nothing local matches and the
    /// cache is offline or the descriptor has no URL;
    /// [`BunsenError::Invalid`] if a download's digest does not match;
    /// [`BunsenError::External`] for a transfer or file-system failure.
    pub fn resolve(
        &self,
        kit: &str,
        provider: &str,
        desc: &PretrainedWeightsDescriptor,
    ) -> BunsenResult<ResolvedWeights> {
        let id = format!("{provider}/{}", desc.name);
        let dest = self.cached_path(kit, provider, desc);

        if dest.is_file() {
            return Ok(ResolvedWeights {
                path: dest,
                provenance: Provenance::Cached,
            });
        }
        // A dangling link, from a source that has since gone.
        if dest.symlink_metadata().is_ok() {
            fs::remove_file(&dest).map_err(BunsenError::external)?;
        }

        let mut urls = Vec::new();
        for source in &desc.sources {
            match source {
                WeightsSource::File(path) => {
                    if path.is_file() {
                        return Ok(ResolvedWeights {
                            path: path.clone(),
                            provenance: Provenance::File,
                        });
                    }
                }
                WeightsSource::LocalDir { .. } => {
                    let Some(dir) = self.local_dir(source) else {
                        continue;
                    };
                    let path = dir.join(&desc.file);
                    if !path.is_file() {
                        continue;
                    }
                    let checks_out = match &desc.sha256 {
                        Some(sha256) => verify_sha256(&path, sha256).is_ok(),
                        None => true,
                    };
                    if checks_out {
                        link_or_copy(&path, &dest)?;
                        return Ok(ResolvedWeights {
                            path: dest,
                            provenance: Provenance::LocalDir,
                        });
                    }
                }
                WeightsSource::Url(url) => urls.push(url.as_str()),
            }
        }

        if urls.is_empty() {
            return Err(BunsenError::ResourceNotFound(format!(
                "{id}: not local, and no URL to fetch it from"
            )));
        }
        if self.offline {
            return Err(BunsenError::ResourceNotFound(format!(
                "{id}: not in the cache ({}) and the cache is offline",
                dest.display()
            )));
        }
        self.download(dest, &urls, desc.sha256.as_deref())
    }

    /// The URLs, in order, through the disk cache's fetch.
    #[cfg(feature = "fetch")]
    fn download(
        &self,
        dest: PathBuf,
        urls: &[&str],
        sha256: Option<&str>,
    ) -> BunsenResult<ResolvedWeights> {
        self.disk.fetch_from_urls(urls, &dest, sha256)?;
        Ok(ResolvedWeights {
            path: dest,
            provenance: Provenance::Downloaded,
        })
    }

    /// Without the `fetch` feature there is no network: not local is not
    /// found.
    #[cfg(not(feature = "fetch"))]
    fn download(
        &self,
        dest: PathBuf,
        _urls: &[&str],
        _sha256: Option<&str>,
    ) -> BunsenResult<ResolvedWeights> {
        Err(BunsenError::ResourceNotFound(format!(
            "{}: not local, and fetching needs the `fetch` feature",
            dest.display()
        )))
    }
}

/// What a resource's local lookup found.
enum Local {
    /// A file, with where it came from.
    Found(ResolvedWeights),

    /// Nothing local: the URLs to try, in order, and where the file lands.
    Remote { dest: PathBuf, urls: Vec<String> },
}

/// A resource a map load has to fetch.
#[cfg_attr(not(feature = "fetch"), allow(dead_code))]
struct RemotePart {
    key: String,
    dest: PathBuf,
    urls: Vec<String>,
    sha256: Option<String>,
}

fn remote_keys(remote: &[RemotePart]) -> String {
    remote
        .iter()
        .map(|r| r.key.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The resource side: a [`Resource`] under `pretrained/`, one at a time or
/// a whole [`ResourceMap`] together.
impl WeightsCache {
    /// Where a resource's file lives in the cache, present or not:
    /// `pretrained/<kit>/<namespace>/<sha256>/<file>` when it is pinned,
    /// and `pretrained/<kit>/<namespace>/<cache key>/<file>` when not.
    pub fn resource_path(
        &self,
        kit: &str,
        res: &Resource,
    ) -> PathBuf {
        let pin = match &res.sha256 {
            Some(sha256) => sha256.clone(),
            None => res.cache_key(),
        };
        self.disk.cache_path(
            &[PRETRAINED_DIR, kit, res.namespace.as_str(), pin.as_str()],
            &res.file,
        )
    }

    /// Where a resource stands, without fetching or hashing anything.
    pub fn resource_status(
        &self,
        kit: &str,
        res: &Resource,
    ) -> CacheStatus {
        if self.resource_path(kit, res).is_file() {
            return CacheStatus::Cached;
        }
        for source in &res.sources {
            match source {
                WeightsSource::File(path) if path.is_file() => return CacheStatus::File,
                WeightsSource::LocalDir { .. } => {
                    if let Some(dir) = self.local_dir(source)
                        && dir.join(&res.file).is_file()
                    {
                        return CacheStatus::LocalDir;
                    }
                }
                _ => {}
            }
        }
        CacheStatus::Remote
    }

    /// Where every resource of `map` stands, by key.
    pub fn map_status(
        &self,
        kit: &str,
        map: &ResourceMap,
    ) -> BTreeMap<String, CacheStatus> {
        map.resources
            .iter()
            .map(|(key, res)| (key.clone(), self.resource_status(kit, res)))
            .collect()
    }

    /// Brings a resource local, and says where it came from.
    ///
    /// The cache is consulted first; then each source in the resource's
    /// order: a file in another tool's directory is used in place, hashed
    /// first only when the options ask; the URLs are fetched in order,
    /// checked, and written into the cache. Nothing is linked or copied
    /// in.
    ///
    /// # Errors
    /// [`BunsenError::ResourceNotFound`] if nothing local matches and the
    /// cache is offline or the resource has no URL;
    /// [`BunsenError::Invalid`] if a download's digest does not match, or a
    /// local-dir file's when the options verify them;
    /// [`BunsenError::External`] for a transfer or file-system failure.
    pub fn resolve_resource(
        &self,
        kit: &str,
        res: &Resource,
    ) -> BunsenResult<ResolvedWeights> {
        match self.resolve_local(kit, res)? {
            Local::Found(found) => Ok(found),
            Local::Remote { dest, urls } => {
                if self.offline {
                    return Err(BunsenError::ResourceNotFound(format!(
                        "{}: not in the cache ({}) and the cache is offline",
                        res.key,
                        dest.display()
                    )));
                }
                let urls: Vec<&str> = urls.iter().map(String::as_str).collect();
                self.download(dest, &urls, res.sha256.as_deref())
            }
        }
    }

    /// Every resource of `map` local, under `kit`, all or nothing.
    ///
    /// Each resource is looked for locally as
    /// [`resolve_resource`](Self::resolve_resource) does; the ones only a
    /// URL can supply are then fetched together, under the options' fetch
    /// policy, into the cache. A resource that cannot be had fails the
    /// load with its key named; the files that did land stay.
    ///
    /// # Errors
    /// [`BunsenError::ResourceNotFound`] naming the remote keys when the
    /// cache is offline, or a key with no URL; [`BunsenError::External`]
    /// naming each key that did not land; [`BunsenError::Invalid`] as
    /// [`resolve_resource`](Self::resolve_resource).
    pub fn load(
        &self,
        kit: &str,
        map: &ResourceMap,
    ) -> BunsenResult<LoadedResources> {
        let mut parts = BTreeMap::new();
        let mut remote = Vec::new();
        for (key, res) in &map.resources {
            match self.resolve_local(kit, res)? {
                Local::Found(found) => {
                    parts.insert(key.clone(), found);
                }
                Local::Remote { dest, urls } => remote.push(RemotePart {
                    key: key.clone(),
                    dest,
                    urls,
                    sha256: res.sha256.clone(),
                }),
            }
        }
        if !remote.is_empty() {
            if self.offline {
                return Err(BunsenError::ResourceNotFound(format!(
                    "{}: not in the cache, and the cache is offline: {}",
                    map.name,
                    remote_keys(&remote)
                )));
            }
            self.fetch_remote(&map.name, remote, &mut parts)?;
        }
        Ok(LoadedResources {
            map: map.clone(),
            parts,
        })
    }

    /// The cache, then the local sources in order; the URLs otherwise.
    fn resolve_local(
        &self,
        kit: &str,
        res: &Resource,
    ) -> BunsenResult<Local> {
        let dest = self.resource_path(kit, res);
        if dest.is_file() {
            return Ok(Local::Found(ResolvedWeights {
                path: dest,
                provenance: Provenance::Cached,
            }));
        }

        let mut urls = Vec::new();
        for source in &res.sources {
            match source {
                WeightsSource::File(path) => {
                    if path.is_file() {
                        return Ok(Local::Found(ResolvedWeights {
                            path: path.clone(),
                            provenance: Provenance::File,
                        }));
                    }
                }
                WeightsSource::LocalDir { .. } => {
                    let Some(dir) = self.local_dir(source) else {
                        continue;
                    };
                    let path = dir.join(&res.file);
                    if !path.is_file() {
                        continue;
                    }
                    if self.verify_local_dirs
                        && let Some(sha256) = &res.sha256
                    {
                        verify_sha256(&path, sha256)?;
                    }
                    return Ok(Local::Found(ResolvedWeights {
                        path,
                        provenance: Provenance::LocalDir,
                    }));
                }
                WeightsSource::Url(url) => urls.push(url.clone()),
            }
        }

        if urls.is_empty() {
            return Err(BunsenError::ResourceNotFound(format!(
                "{}: not local, and no URL to fetch it from",
                res.key
            )));
        }
        Ok(Local::Remote { dest, urls })
    }

    /// The remote parts, together, under the fetch policy.
    #[cfg(feature = "fetch")]
    fn fetch_remote(
        &self,
        name: &str,
        remote: Vec<RemotePart>,
        parts: &mut BTreeMap<String, ResolvedWeights>,
    ) -> BunsenResult<()> {
        let jobs: Vec<FetchJob> = remote
            .iter()
            .map(|r| {
                let mut job = FetchJob::new(r.urls.clone(), r.dest.clone());
                job.sha256 = r.sha256.clone();
                job
            })
            .collect();
        let report = self.disk.fetch_many(&jobs, &self.fetch);

        let mut missing = Vec::new();
        for (outcome, part) in report.outcomes.iter().zip(remote) {
            let provenance = match outcome {
                FetchOutcome::Cached(_) => Provenance::Cached,
                FetchOutcome::Fetched(_) => Provenance::Downloaded,
                FetchOutcome::Failed { error, .. } => {
                    missing.push(format!("{}: {error}", part.key));
                    continue;
                }
                FetchOutcome::Skipped { .. } => {
                    missing.push(format!("{}: skipped", part.key));
                    continue;
                }
            };
            parts.insert(
                part.key,
                ResolvedWeights {
                    path: part.dest,
                    provenance,
                },
            );
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(BunsenError::External(format!(
                "{name}: {} of {} resources did not land: {}",
                missing.len(),
                jobs.len(),
                missing.join("; ")
            )))
        }
    }

    /// Without the `fetch` feature there is no network: not local is not
    /// found.
    #[cfg(not(feature = "fetch"))]
    fn fetch_remote(
        &self,
        name: &str,
        remote: Vec<RemotePart>,
        _parts: &mut BTreeMap<String, ResolvedWeights>,
    ) -> BunsenResult<()> {
        Err(BunsenError::ResourceNotFound(format!(
            "{name}: not local, and fetching needs the `fetch` feature: {}",
            remote_keys(&remote)
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "fetch")]
    use crate::data::cache::{
        OnFailure,
        testing::{
            refused_url,
            serve_once,
        },
    };
    use crate::data::{
        cache::testing::ABC_SHA256,
        pretrained::WeightsFormat,
    };

    fn cache_in(
        dir: &Path,
        offline: bool,
    ) -> WeightsCache {
        WeightsCache::new(
            WeightsCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(offline),
        )
        .unwrap()
    }

    /// `abc.pt`, pinned to the digest of `abc`, from `sources`.
    fn abc(
        sha256: Option<&str>,
        sources: Vec<WeightsSource>,
    ) -> PretrainedWeightsDescriptor {
        PretrainedWeightsDescriptor {
            name: "abc".to_string(),
            description: "three bytes".to_string(),
            license: None,
            origin: None,
            prefab: "abc".to_string(),
            aliases: vec![],
            file: "abc.pt".to_string(),
            sha256: sha256.map(str::to_string),
            format: WeightsFormat::PYTORCH_F16,
            sources,
        }
    }

    fn url(u: &str) -> WeightsSource {
        WeightsSource::Url(u.to_string())
    }

    #[test]
    fn test_cached_paths_by_pin() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path(), true);
        let root = dir.path().join("cache").join("weights");

        let pinned = abc(Some(ABC_SHA256), vec![url("https://a.example/abc.pt")]);
        assert_eq!(
            cache.cached_path("kit", "prov", &pinned),
            root.join("kit")
                .join("prov")
                .join(ABC_SHA256)
                .join("abc.pt")
        );

        let unpinned = abc(None, vec![url("https://a.example/abc.pt")]);
        assert_eq!(
            cache.cached_path("kit", "prov", &unpinned),
            root.join(unpinned.cache_key()).join("abc.pt"),
            "unpinned keeps the URL-keyed layout"
        );
        assert_eq!(cache.cache_dir(), dir.path().join("cache"));
        assert!(cache.offline());
    }

    #[test]
    fn test_offline_without_a_local_source_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path(), true);
        let desc = abc(Some(ABC_SHA256), vec![url("https://a.example/abc.pt")]);

        assert_eq!(cache.status("kit", "prov", &desc), CacheStatus::Remote);
        assert!(matches!(
            cache.resolve("kit", "prov", &desc),
            Err(BunsenError::ResourceNotFound(_))
        ));

        let no_url = abc(Some(ABC_SHA256), vec![]);
        assert!(matches!(
            cache_in(dir.path(), false).resolve("kit", "prov", &no_url),
            Err(BunsenError::ResourceNotFound(_))
        ));
    }

    /// Whatever is at the pinned path is taken as verified-when-written; the
    /// digest is in the path, not re-checked per run.
    #[test]
    fn test_a_cached_file_is_trusted_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path(), true);
        let desc = abc(Some(ABC_SHA256), vec![url("https://a.example/abc.pt")]);

        let dest = cache.cached_path("kit", "prov", &desc);
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(&dest, b"not really abc").unwrap();

        assert_eq!(cache.status("kit", "prov", &desc), CacheStatus::Cached);
        assert_eq!(
            cache.resolve("kit", "prov", &desc).unwrap(),
            ResolvedWeights {
                path: dest,
                provenance: Provenance::Cached,
            }
        );
    }

    /// A file of the right name in a local dir is checked before it is
    /// adopted: the wrong bytes are passed over, the right ones linked in.
    /// The dir comes from the source, or from an override by name.
    #[test]
    fn test_local_dir_is_checked_before_adoption() {
        let dir = tempfile::tempdir().unwrap();
        let upstream = dir.path().join("upstream");
        fs::create_dir_all(&upstream).unwrap();
        let desc = abc(
            Some(ABC_SHA256),
            vec![
                WeightsSource::LocalDir {
                    name: "up".to_string(),
                    dir: Some(upstream.clone()),
                },
                url("https://a.example/abc.pt"),
            ],
        );
        let cache = cache_in(dir.path(), true);

        fs::write(upstream.join("abc.pt"), b"a partial download").unwrap();
        assert_eq!(cache.status("kit", "prov", &desc), CacheStatus::LocalDir);
        assert!(matches!(
            cache.resolve("kit", "prov", &desc),
            Err(BunsenError::ResourceNotFound(_))
        ));
        assert!(!cache.cached_path("kit", "prov", &desc).exists());

        fs::write(upstream.join("abc.pt"), b"abc").unwrap();
        let resolved = cache.resolve("kit", "prov", &desc).unwrap();
        assert_eq!(resolved.provenance, Provenance::LocalDir);
        assert_eq!(resolved.path, cache.cached_path("kit", "prov", &desc));
        assert_eq!(fs::read(&resolved.path).unwrap(), b"abc");
        assert_eq!(cache.status("kit", "prov", &desc), CacheStatus::Cached);

        // An override by name wins over the directory the source carries.
        let elsewhere = dir.path().join("elsewhere");
        fs::create_dir_all(&elsewhere).unwrap();
        fs::write(elsewhere.join("abc.pt"), b"abc").unwrap();
        let cache = WeightsCache::new(
            WeightsCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache2")))
                        .without_transfer_observers(),
                )
                .with_offline(true)
                .with_local_dir("up", elsewhere.clone()),
        )
        .unwrap();
        assert_eq!(cache.local_dir(&desc.sources[0]), Some(elsewhere));
        assert_eq!(cache.local_dirs().len(), 1);
        assert_eq!(
            cache.resolve("kit", "prov", &desc).unwrap().provenance,
            Provenance::LocalDir
        );
    }

    /// A bundled file is used where it is; nothing is written to the cache.
    #[test]
    fn test_a_bundled_file_is_used_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let bundled = dir.path().join("bundled").join("abc.pt");
        fs::create_dir_all(bundled.parent().unwrap()).unwrap();
        fs::write(&bundled, b"abc").unwrap();
        let desc = abc(
            Some(ABC_SHA256),
            vec![
                WeightsSource::File(bundled.clone()),
                url("https://a.example/abc.pt"),
            ],
        );
        let cache = cache_in(dir.path(), true);

        assert_eq!(cache.status("kit", "prov", &desc), CacheStatus::File);
        assert_eq!(
            cache.resolve("kit", "prov", &desc).unwrap(),
            ResolvedWeights {
                path: bundled,
                provenance: Provenance::File,
            }
        );
        assert!(!dir.path().join("cache").join("weights").exists());
    }

    /// A URL is fetched, checked and written in; the next resolve is cached.
    #[cfg(feature = "fetch")]
    #[test]
    fn test_a_url_is_fetched_then_cached() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path(), false);
        let live = serve_once("abc.pt", b"abc");
        let desc = abc(Some(ABC_SHA256), vec![url(&live)]);

        let resolved = cache.resolve("kit", "prov", &desc).unwrap();
        assert_eq!(resolved.provenance, Provenance::Downloaded);
        assert_eq!(resolved.path, cache.cached_path("kit", "prov", &desc));
        assert_eq!(fs::read(&resolved.path).unwrap(), b"abc");
        assert_eq!(
            cache.resolve("kit", "prov", &desc).unwrap().provenance,
            Provenance::Cached
        );

        // The same pin under another provider is another path, so the file
        // just cached does not answer for it.
        let wrong = serve_once("abc.pt", b"abd");
        let desc = abc(Some(ABC_SHA256), vec![url(&wrong)]);
        assert!(matches!(
            cache.resolve("kit", "other", &desc),
            Err(BunsenError::Invalid(_))
        ));
        assert!(!cache.cached_path("kit", "other", &desc).exists());
    }

    #[test]
    fn test_display() {
        assert_eq!(Provenance::LocalDir.to_string(), "local dir");
        assert_eq!(Provenance::Given.to_string(), "given");
        assert_eq!(CacheStatus::File.to_string(), "bundled file");
        assert_eq!(CacheStatus::Remote.to_string(), "remote");
    }

    /// `<key>.bin`, three bytes when pinned to `abc`, from `sources`.
    fn res(
        key: &str,
        sha256: Option<&str>,
        sources: Vec<WeightsSource>,
    ) -> Resource {
        Resource {
            key: key.to_string(),
            file: format!("{key}.bin"),
            sha256: sha256.map(str::to_string),
            kind: None,
            namespace: "prov".to_string(),
            sources,
        }
    }

    fn local_dir(dir: &Path) -> WeightsSource {
        WeightsSource::LocalDir {
            name: "up".to_string(),
            dir: Some(dir.to_path_buf()),
        }
    }

    #[test]
    fn test_resource_paths_by_pin() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path(), true);
        let root = dir.path().join("cache").join("pretrained");

        let pinned = res(
            "abc",
            Some(ABC_SHA256),
            vec![url("https://a.example/abc.bin")],
        );
        assert_eq!(
            cache.resource_path("kit", &pinned),
            root.join("kit")
                .join("prov")
                .join(ABC_SHA256)
                .join("abc.bin")
        );

        let unpinned = res("abc", None, vec![url("https://a.example/abc.bin")]);
        assert_eq!(
            cache.resource_path("kit", &unpinned),
            root.join("kit")
                .join("prov")
                .join(unpinned.cache_key())
                .join("abc.bin"),
            "unpinned is keyed by its first URL"
        );
        assert!(!cache.verify_local_dirs());
    }

    /// Whatever is at the pinned path is taken as verified-when-written.
    #[test]
    fn test_a_cached_resource_is_trusted_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path(), true);
        let r = res(
            "abc",
            Some(ABC_SHA256),
            vec![url("https://a.example/abc.bin")],
        );

        assert_eq!(cache.resource_status("kit", &r), CacheStatus::Remote);
        let dest = cache.resource_path("kit", &r);
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(&dest, b"not really abc").unwrap();

        assert_eq!(cache.resource_status("kit", &r), CacheStatus::Cached);
        assert_eq!(
            cache.resolve_resource("kit", &r).unwrap(),
            ResolvedWeights {
                path: dest,
                provenance: Provenance::Cached,
            }
        );
    }

    /// A file in a local dir is used where it is, and the cache stays
    /// empty. It is not hashed unless the options ask, and then a wrong
    /// file is an error rather than passed over. An override by name wins
    /// over the directory the source carries.
    #[test]
    fn test_local_dir_is_used_in_place_and_trusted_unless_asked() {
        let dir = tempfile::tempdir().unwrap();
        let upstream = dir.path().join("upstream");
        fs::create_dir_all(&upstream).unwrap();
        let r = res(
            "abc",
            Some(ABC_SHA256),
            vec![local_dir(&upstream), url("https://a.example/abc.bin")],
        );
        fs::write(upstream.join("abc.bin"), b"a partial download").unwrap();

        let trusting = cache_in(dir.path(), true);
        assert_eq!(trusting.resource_status("kit", &r), CacheStatus::LocalDir);
        let resolved = trusting.resolve_resource("kit", &r).unwrap();
        assert_eq!(resolved.provenance, Provenance::LocalDir);
        assert_eq!(resolved.path, upstream.join("abc.bin"));
        assert!(!dir.path().join("cache").join("pretrained").exists());

        let verifying = WeightsCache::new(
            WeightsCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true)
                .with_verify_local_dirs(true),
        )
        .unwrap();
        assert!(verifying.verify_local_dirs());
        assert!(matches!(
            verifying.resolve_resource("kit", &r),
            Err(BunsenError::Invalid(_))
        ));
        fs::write(upstream.join("abc.bin"), b"abc").unwrap();
        assert_eq!(
            verifying.resolve_resource("kit", &r).unwrap().provenance,
            Provenance::LocalDir
        );
        assert!(!dir.path().join("cache").join("pretrained").exists());

        let elsewhere = dir.path().join("elsewhere");
        fs::create_dir_all(&elsewhere).unwrap();
        fs::write(elsewhere.join("abc.bin"), b"abc").unwrap();
        let overridden = WeightsCache::new(
            WeightsCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true)
                .with_local_dir("up", elsewhere.clone()),
        )
        .unwrap();
        assert_eq!(
            overridden.resolve_resource("kit", &r).unwrap().path,
            elsewhere.join("abc.bin")
        );
    }

    /// A path on the command line resolves to itself.
    #[test]
    fn test_a_given_resource_resolves_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("mine.pt");
        fs::write(&file, b"x").unwrap();
        let cache = cache_in(dir.path(), true);
        let r = Resource::given("checkpoint", &file);

        assert_eq!(cache.resource_status("kit", &r), CacheStatus::LocalDir);
        assert_eq!(
            cache.resolve_resource("kit", &r).unwrap(),
            ResolvedWeights {
                path: file,
                provenance: Provenance::LocalDir,
            }
        );
    }

    #[test]
    fn test_resource_offline_or_without_a_url_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let offline = cache_in(dir.path(), true);
        let remote = res(
            "abc",
            Some(ABC_SHA256),
            vec![url("https://a.example/abc.bin")],
        );
        assert!(matches!(
            offline.resolve_resource("kit", &remote),
            Err(BunsenError::ResourceNotFound(m)) if m.contains("offline")
        ));

        let no_url = res("abc", Some(ABC_SHA256), vec![]);
        assert!(matches!(
            cache_in(dir.path(), false).resolve_resource("kit", &no_url),
            Err(BunsenError::ResourceNotFound(m)) if m.contains("no URL")
        ));
    }

    /// A URL is fetched, checked and written under `pretrained/`; the next
    /// resolve is cached.
    #[cfg(feature = "fetch")]
    #[test]
    fn test_a_resource_url_is_fetched_then_cached() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path(), false);
        let live = serve_once("abc.bin", b"abc");
        let r = res("abc", Some(ABC_SHA256), vec![url(&live)]);

        let resolved = cache.resolve_resource("kit", &r).unwrap();
        assert_eq!(resolved.provenance, Provenance::Downloaded);
        assert_eq!(resolved.path, cache.resource_path("kit", &r));
        assert_eq!(fs::read(&resolved.path).unwrap(), b"abc");
        assert_eq!(
            cache.resolve_resource("kit", &r).unwrap().provenance,
            Provenance::Cached
        );
    }

    /// A map loads with one provenance per resource: the cached one is
    /// trusted, the local-dir one used in place, the remote one fetched.
    #[cfg(feature = "fetch")]
    #[test]
    fn test_load_mixes_provenances() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path(), false);
        let upstream = dir.path().join("upstream");
        fs::create_dir_all(&upstream).unwrap();
        fs::write(upstream.join("local.bin"), b"abc").unwrap();
        let live = serve_once("remote.bin", b"abc");

        let cached = res(
            "cached",
            Some(ABC_SHA256),
            vec![url("https://a.example/cached.bin")],
        );
        let dest = cache.resource_path("kit", &cached);
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(&dest, b"abc").unwrap();
        let local = res(
            "local",
            Some(ABC_SHA256),
            vec![local_dir(&upstream), url("https://a.example/local.bin")],
        );
        let remote = res("remote", Some(ABC_SHA256), vec![url(&live)]);
        let map = ResourceMap::new("m")
            .with_resource(cached)
            .with_resource(local.clone())
            .with_resource(remote.clone());

        let status = cache.map_status("kit", &map);
        assert_eq!(status["cached"], CacheStatus::Cached);
        assert_eq!(status["local"], CacheStatus::LocalDir);
        assert_eq!(status["remote"], CacheStatus::Remote);

        let loaded = cache.load("kit", &map).unwrap();
        assert_eq!(loaded.map, map);
        assert_eq!(loaded.keys(), ["cached", "local", "remote"]);
        assert_eq!(loaded.get("cached").unwrap().provenance, Provenance::Cached);
        assert_eq!(
            loaded.get("local").unwrap().provenance,
            Provenance::LocalDir
        );
        assert_eq!(
            loaded.get("local").unwrap().path,
            upstream.join("local.bin")
        );
        assert_eq!(
            loaded.get("remote").unwrap().provenance,
            Provenance::Downloaded
        );
        assert_eq!(
            loaded.expect("remote").unwrap(),
            cache.resource_path("kit", &remote)
        );
        assert!(!cache.resource_path("kit", &local).exists());
        assert_eq!(cache.map_status("kit", &map)["remote"], CacheStatus::Cached);
    }

    /// One remote that cannot be had fails the load and is named by key;
    /// the one that could be had is on disk.
    #[cfg(feature = "fetch")]
    #[test]
    fn test_load_names_the_key_that_did_not_land() {
        let dir = tempfile::tempdir().unwrap();
        let cache = WeightsCache::new(
            WeightsCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_fetch_policy(FetchPolicy::default().with_on_failure(OnFailure::Continue)),
        )
        .unwrap();
        let live = res(
            "live",
            Some(ABC_SHA256),
            vec![url(&serve_once("live.bin", b"abc"))],
        );
        let dead = res(
            "dead",
            Some(ABC_SHA256),
            vec![url(&refused_url("dead.bin"))],
        );
        let map = ResourceMap::new("m")
            .with_resource(live.clone())
            .with_resource(dead);

        let err = cache.load("kit", &map).unwrap_err();
        assert!(
            matches!(&err, BunsenError::External(m) if m.contains("dead:")),
            "{err}"
        );
        assert!(!err.to_string().contains("live:"), "{err}");
        assert!(cache.resource_path("kit", &live).is_file());
    }

    /// Offline, a map with a remote resource is not found, and the error
    /// names the remote keys; the local ones are not touched first.
    #[test]
    fn test_load_offline_names_the_remote_keys() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path(), true);
        let cached = res(
            "cached",
            Some(ABC_SHA256),
            vec![url("https://a.example/cached.bin")],
        );
        let dest = cache.resource_path("kit", &cached);
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(&dest, b"abc").unwrap();
        let remote = res(
            "remote",
            Some(ABC_SHA256),
            vec![url("https://a.example/remote.bin")],
        );
        let map = ResourceMap::new("m")
            .with_resource(cached)
            .with_resource(remote);

        let err = cache.load("kit", &map).unwrap_err();
        assert!(
            matches!(&err, BunsenError::ResourceNotFound(m) if m.ends_with("offline: remote")),
            "{err}"
        );

        let local_only = ResourceMap::new("m").with_resource(res(
            "cached",
            Some(ABC_SHA256),
            vec![url("https://a.example/cached.bin")],
        ));
        let loaded = cache.load("kit", &local_only).unwrap();
        assert_eq!(loaded.get("cached").unwrap().provenance, Provenance::Cached);
    }
}
