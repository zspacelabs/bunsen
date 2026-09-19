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
    PretrainedWeightsDescriptor,
    WeightsSource,
    pretrained_weights_resource_key,
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "fetch")]
    use crate::data::cache::testing::serve_once;
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
}
