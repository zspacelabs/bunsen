//! # Pretrained cache
//!
//! Digest-pinned resources under the disk cache's cache directory.
//! [`BunsenDiskCache`] decides *where*; this decides *what is trusted
//! there*. A pinned resource's file lives at
//!
//! ```text
//! <cache>/pretrained/<kit>/<namespace>/<sha256>/<file>
//! ```
//!
//! The digest in the path is the pin: a file at that path was verified when
//! it was written, so it is trusted on later runs without re-hashing 3 GB,
//! and a re-pinned model cannot collide with a stale one. An unpinned
//! resource lives under its URL-derived cache key instead.
//!
//! [`PretrainedCache::resolve`] consults the cache first, then each source
//! in the resource's order: a file in a directory another tool keeps is
//! used in place, a "trust me" source hashed only when the options ask, and
//! a URL is fetched, checked and written in. Nothing is linked or copied
//! into the cache; only a download writes, and every transfer reports to
//! the disk cache's observer stack. [`PretrainedCache::load`] does it for a
//! whole [`ResourceMap`], fetching what is remote together under the
//! options' policy.
//!
//! Bundling is not modeled. A deployment that wants to hit populates this
//! directory ahead of time, and the cache hits with no knowledge of it;
//! `bunsen-bundled-whisper` lays its build output out this way.

use std::{
    collections::BTreeMap,
    fmt,
    path::{
        Path,
        PathBuf,
    },
};

use super::{
    LoadedResources,
    Resource,
    ResourceMap,
    Source,
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
        verify_sha256,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// The directory under the cache dir that holds a pretrained's resources:
/// `<cache>/pretrained/<kit>/<namespace>/<sha256>/<file>`.
pub const PRETRAINED_DIR: &str = "pretrained";

/// Options for [`PretrainedCache`].
#[derive(Clone, Debug, Default)]
pub struct PretrainedCacheOptions {
    /// The disk cache underneath: where the cache directory is, and who
    /// watches transfers.
    pub disk: BunsenDiskCacheOptions,

    /// Never reach the network: a resource not already local is an error.
    pub offline: bool,

    /// Overrides for [`Source::LocalDir`] sources, by the source's name:
    /// where another tool's directory is on this machine.
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

impl PretrainedCacheOptions {
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
    /// Found in another tool's directory, or given as a path, and used in
    /// place.
    LocalDir,
    /// Fetched from a URL, checked, and written to the cache.
    Downloaded,
}

impl fmt::Display for Provenance {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(match self {
            Self::Cached => "cached",
            Self::LocalDir => "local dir",
            Self::Downloaded => "downloaded",
        })
    }
}

/// A local file holding a resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedResource {
    /// The file.
    pub path: PathBuf,

    /// Where it came from.
    pub provenance: Provenance,
}

/// Where a resource stands before anything is fetched, for a listing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStatus {
    /// In the cache.
    Cached,
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
            Self::LocalDir => "local dir",
            Self::Remote => "remote",
        })
    }
}

/// What a resource's local lookup found.
enum Local {
    /// A file, with where it came from.
    Found(ResolvedResource),

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

/// Digest-pinned resources under the disk cache's cache directory.
pub struct PretrainedCache {
    disk: BunsenDiskCache,
    offline: bool,
    local_dirs: BTreeMap<String, PathBuf>,
    verify_local_dirs: bool,
    #[cfg(feature = "fetch")]
    fetch: FetchPolicy,
}

impl PretrainedCache {
    /// Opens the cache.
    ///
    /// # Errors
    /// As [`BunsenDiskCache::new`].
    pub fn new(options: PretrainedCacheOptions) -> BunsenResult<Self> {
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
        source: &Source,
    ) -> Option<PathBuf> {
        match source {
            Source::LocalDir { name, dir } => {
                self.local_dirs.get(name).cloned().or_else(|| dir.clone())
            }
            Source::Url(_) => None,
        }
    }

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
    pub fn status(
        &self,
        kit: &str,
        res: &Resource,
    ) -> CacheStatus {
        if self.resource_path(kit, res).is_file() {
            return CacheStatus::Cached;
        }
        for source in &res.sources {
            if let Source::LocalDir { .. } = source
                && let Some(dir) = self.local_dir(source)
                && dir.join(&res.file).is_file()
            {
                return CacheStatus::LocalDir;
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
            .map(|(key, res)| (key.clone(), self.status(kit, res)))
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
    pub fn resolve(
        &self,
        kit: &str,
        res: &Resource,
    ) -> BunsenResult<ResolvedResource> {
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
    /// Each resource is looked for locally as [`resolve`](Self::resolve)
    /// does; the ones only a URL can supply are then fetched together,
    /// under the options' fetch policy, into the cache. A resource that
    /// cannot be had fails the load with its key named; the files that did
    /// land stay.
    ///
    /// # Errors
    /// [`BunsenError::ResourceNotFound`] naming the remote keys when the
    /// cache is offline, or a key with no URL; [`BunsenError::External`]
    /// naming each key that did not land; [`BunsenError::Invalid`] as
    /// [`resolve`](Self::resolve).
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
            return Ok(Local::Found(ResolvedResource {
                path: dest,
                provenance: Provenance::Cached,
            }));
        }

        let mut urls = Vec::new();
        for source in &res.sources {
            match source {
                Source::LocalDir { .. } => {
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
                    return Ok(Local::Found(ResolvedResource {
                        path,
                        provenance: Provenance::LocalDir,
                    }));
                }
                Source::Url(url) => urls.push(url.clone()),
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

    /// The URLs, in order, through the disk cache's fetch.
    #[cfg(feature = "fetch")]
    fn download(
        &self,
        dest: PathBuf,
        urls: &[&str],
        sha256: Option<&str>,
    ) -> BunsenResult<ResolvedResource> {
        self.disk.fetch_from_urls(urls, &dest, sha256)?;
        Ok(ResolvedResource {
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
    ) -> BunsenResult<ResolvedResource> {
        Err(BunsenError::ResourceNotFound(format!(
            "{}: not local, and fetching needs the `fetch` feature",
            dest.display()
        )))
    }

    /// The remote parts, together, under the fetch policy.
    #[cfg(feature = "fetch")]
    fn fetch_remote(
        &self,
        name: &str,
        remote: Vec<RemotePart>,
        parts: &mut BTreeMap<String, ResolvedResource>,
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
                ResolvedResource {
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
        _parts: &mut BTreeMap<String, ResolvedResource>,
    ) -> BunsenResult<()> {
        Err(BunsenError::ResourceNotFound(format!(
            "{name}: not local, and fetching needs the `fetch` feature: {}",
            remote_keys(&remote)
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::data::cache::testing::ABC_SHA256;
    #[cfg(feature = "fetch")]
    use crate::data::cache::{
        OnFailure,
        testing::{
            refused_url,
            serve_once,
        },
    };

    fn cache_in(
        dir: &Path,
        offline: bool,
    ) -> PretrainedCache {
        PretrainedCache::new(
            PretrainedCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(offline),
        )
        .unwrap()
    }

    fn url(u: &str) -> Source {
        Source::Url(u.to_string())
    }

    /// `<key>.bin`, three bytes when pinned to `abc`, from `sources`.
    fn res(
        key: &str,
        sha256: Option<&str>,
        sources: Vec<Source>,
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

    fn local_dir(dir: &Path) -> Source {
        Source::LocalDir {
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
        assert_eq!(cache.cache_dir(), dir.path().join("cache"));
        assert!(cache.offline());
        assert!(!cache.verify_local_dirs());
        assert!(cache.local_dirs().is_empty());
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

        assert_eq!(cache.status("kit", &r), CacheStatus::Remote);
        let dest = cache.resource_path("kit", &r);
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(&dest, b"not really abc").unwrap();

        assert_eq!(cache.status("kit", &r), CacheStatus::Cached);
        assert_eq!(
            cache.resolve("kit", &r).unwrap(),
            ResolvedResource {
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
        assert_eq!(trusting.status("kit", &r), CacheStatus::LocalDir);
        let resolved = trusting.resolve("kit", &r).unwrap();
        assert_eq!(resolved.provenance, Provenance::LocalDir);
        assert_eq!(resolved.path, upstream.join("abc.bin"));
        assert!(!dir.path().join("cache").join("pretrained").exists());

        let verifying = PretrainedCache::new(
            PretrainedCacheOptions::default()
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
            verifying.resolve("kit", &r),
            Err(BunsenError::Invalid(_))
        ));
        fs::write(upstream.join("abc.bin"), b"abc").unwrap();
        assert_eq!(
            verifying.resolve("kit", &r).unwrap().provenance,
            Provenance::LocalDir
        );
        assert!(!dir.path().join("cache").join("pretrained").exists());

        let elsewhere = dir.path().join("elsewhere");
        fs::create_dir_all(&elsewhere).unwrap();
        fs::write(elsewhere.join("abc.bin"), b"abc").unwrap();
        let overridden = PretrainedCache::new(
            PretrainedCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true)
                .with_local_dir("up", elsewhere.clone()),
        )
        .unwrap();
        assert_eq!(overridden.local_dir(&r.sources[0]), Some(elsewhere.clone()));
        assert_eq!(overridden.local_dir(&r.sources[1]), None);
        assert_eq!(
            overridden.resolve("kit", &r).unwrap().path,
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

        assert_eq!(cache.status("kit", &r), CacheStatus::LocalDir);
        assert_eq!(
            cache.resolve("kit", &r).unwrap(),
            ResolvedResource {
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
            offline.resolve("kit", &remote),
            Err(BunsenError::ResourceNotFound(m)) if m.contains("offline")
        ));

        let no_url = res("abc", Some(ABC_SHA256), vec![]);
        assert!(matches!(
            cache_in(dir.path(), false).resolve("kit", &no_url),
            Err(BunsenError::ResourceNotFound(m)) if m.contains("no URL")
        ));
    }

    /// A URL is fetched, checked and written under `pretrained/`; the next
    /// resolve is cached. The wrong bytes under another pin are refused
    /// and leave nothing.
    #[cfg(feature = "fetch")]
    #[test]
    fn test_a_resource_url_is_fetched_then_cached() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path(), false);
        let live = serve_once("abc.bin", b"abc");
        let r = res("abc", Some(ABC_SHA256), vec![url(&live)]);

        let resolved = cache.resolve("kit", &r).unwrap();
        assert_eq!(resolved.provenance, Provenance::Downloaded);
        assert_eq!(resolved.path, cache.resource_path("kit", &r));
        assert_eq!(fs::read(&resolved.path).unwrap(), b"abc");
        assert_eq!(
            cache.resolve("kit", &r).unwrap().provenance,
            Provenance::Cached
        );

        let wrong = serve_once("abd.bin", b"abd");
        let r = res("abd", Some(ABC_SHA256), vec![url(&wrong)]);
        assert!(matches!(
            cache.resolve("kit", &r),
            Err(BunsenError::Invalid(_))
        ));
        assert!(!cache.resource_path("kit", &r).exists());
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
        let cache = PretrainedCache::new(
            PretrainedCacheOptions::default()
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
    /// names the remote keys; a map of local resources loads.
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

    #[test]
    fn test_display() {
        assert_eq!(Provenance::Cached.to_string(), "cached");
        assert_eq!(Provenance::LocalDir.to_string(), "local dir");
        assert_eq!(Provenance::Downloaded.to_string(), "downloaded");
        assert_eq!(CacheStatus::Cached.to_string(), "cached");
        assert_eq!(CacheStatus::LocalDir.to_string(), "local dir");
        assert_eq!(CacheStatus::Remote.to_string(), "remote");
    }
}
