//! The local weights cache: digest-pinned files under bunsen's cache
//! directory.
//!
//! [`BunsenDiskCache`] decides *where*: `--cache-dir`, then
//! `$BUNSEN_CACHE_DIR`, then the platform's cache directory. This module
//! decides *what is trusted there*. A pretrained's file lives at
//!
//! ```text
//! <cache>/whisper/<provider>/<sha256>/<file>
//! ```
//!
//! The digest in the path is the pin, as it is in upstream's URLs: a file at
//! that path was verified when it was written, so it is trusted on later
//! runs without re-hashing 3 GB, and a re-pinned model cannot collide with
//! a stale one.
//!
//! [`BunsenDiskCache::load_cached_path`] is not used for the transfer. It has
//! no digest (`// TODO: hash`), and it drops the per-file result of the
//! download, so a 404 comes back `Ok` with nothing on disk. The transfer here
//! streams to a `.partial` beside the destination, hashes as it goes, and
//! renames only on a match, as `bunsen-bundled-whisper`'s build script does.

use std::{
    fmt,
    fs,
    io::{
        self,
        Read,
    },
    path::{
        Path,
        PathBuf,
    },
};

use bunsen::{
    data::cache::{
        BunsenDiskCache,
        BunsenDiskCacheOptions,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};
use indicatif::{
    ProgressBar,
    ProgressStyle,
};
use sha2::{
    Digest,
    Sha256,
};

use crate::models::pretrained::{
    WeightsSource,
    WhisperPretrained,
};

/// Options for [`WeightsCache`].
#[derive(Debug, Clone, Default)]
pub struct WeightsCacheOptions {
    /// The cache directory; resolved by [`BunsenDiskCache`] when omitted.
    pub cache_dir: Option<PathBuf>,

    /// Never reach the network: a model not already local is an error.
    pub offline: bool,

    /// `openai-whisper`'s download cache, read as a local source;
    /// `$XDG_CACHE_HOME/whisper` or `~/.cache/whisper` when omitted, as
    /// upstream resolves it.
    pub upstream_cache_dir: Option<PathBuf>,
}

/// Where a resolved file came from, this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// Already in the cache.
    Cached,
    /// The file `bunsen-bundled-whisper` fetched at build time.
    Bundled,
    /// Found in upstream's cache, verified, and linked into ours.
    UpstreamCache,
    /// Fetched from a URL, verified, and written to the cache.
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
            Self::Bundled => "bundled",
            Self::UpstreamCache => "upstream cache",
            Self::Downloaded => "downloaded",
            Self::Given => "given",
        })
    }
}

/// A local file holding a pretrained's weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedWeights {
    /// The file.
    pub path: PathBuf,

    /// Where it came from.
    pub provenance: Provenance,
}

/// Where a pretrained stands before anything is fetched, for a listing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStatus {
    /// In the cache.
    Cached,
    /// Not in the cache, but the bundled file is on disk.
    Bundled,
    /// Not in the cache, but upstream's cache has a file of that name,
    /// unverified.
    UpstreamCache,
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
            Self::Bundled => "bundled",
            Self::UpstreamCache => "upstream cache",
            Self::Remote => "remote",
        })
    }
}

/// Digest-pinned weights under bunsen's cache directory.
pub struct WeightsCache {
    disk: BunsenDiskCache,
    offline: bool,
    upstream_dir: Option<PathBuf>,
}

impl WeightsCache {
    /// Opens the cache.
    ///
    /// # Errors
    /// [`BunsenError`] if no cache directory can be resolved.
    pub fn new(options: WeightsCacheOptions) -> BunsenResult<Self> {
        let disk = BunsenDiskCache::new(
            BunsenDiskCacheOptions::default().with_cache_dir(options.cache_dir),
        )?;
        let upstream_dir = options
            .upstream_cache_dir
            .or_else(default_upstream_cache_dir);
        Ok(Self {
            disk,
            offline: options.offline,
            upstream_dir,
        })
    }

    /// The cache directory.
    pub fn cache_dir(&self) -> &Path {
        self.disk.cache_dir()
    }

    /// Whether the network is off limits.
    pub fn offline(&self) -> bool {
        self.offline
    }

    /// Upstream's cache directory, if one could be resolved.
    pub fn upstream_cache_dir(&self) -> Option<&Path> {
        self.upstream_dir.as_deref()
    }

    /// Where a pretrained's file lives in the cache, present or not.
    pub fn cached_path(
        &self,
        provider: &str,
        pretrained: &WhisperPretrained,
    ) -> PathBuf {
        self.disk
            .cache_path(&["whisper", provider, pretrained.sha256], pretrained.file)
    }

    /// Where a pretrained stands, without fetching or hashing anything.
    pub fn status(
        &self,
        provider: &str,
        pretrained: &WhisperPretrained,
    ) -> CacheStatus {
        if self.cached_path(provider, pretrained).is_file() {
            return CacheStatus::Cached;
        }
        for source in pretrained.sources {
            match source {
                WeightsSource::Bundled(path) if path().is_file() => return CacheStatus::Bundled,
                WeightsSource::UpstreamCache
                    if self.upstream_path(pretrained).is_some_and(|p| p.is_file()) =>
                {
                    return CacheStatus::UpstreamCache;
                }
                _ => {}
            }
        }
        CacheStatus::Remote
    }

    /// Brings a pretrained's file local, and says where it came from.
    ///
    /// The cache is consulted first; then each source in the pretrained's
    /// order: a bundled file is used in place, a file in upstream's cache is
    /// hashed and linked in, and a URL is fetched, hashed and written in.
    /// A local file that fails its digest is skipped with a warning rather
    /// than fatal: it is upstream's cache, and may hold a partial download.
    ///
    /// # Errors
    /// [`BunsenError::ResourceNotFound`] if nothing local matches and the
    /// cache is offline or the pretrained has no URL;
    /// [`BunsenError::Invalid`] if a download's digest does not match;
    /// [`BunsenError::External`] for a transfer or file-system failure.
    pub fn resolve(
        &mut self,
        provider: &str,
        pretrained: &WhisperPretrained,
    ) -> BunsenResult<ResolvedWeights> {
        let id = format!("{provider}/{}", pretrained.name);
        let dest = self.cached_path(provider, pretrained);

        if dest.is_file() {
            return Ok(ResolvedWeights {
                path: dest,
                provenance: Provenance::Cached,
            });
        }
        // A dangling link, from a source that has since gone.
        if dest.symlink_metadata().is_ok() {
            log::warn!("{id}: removing a dangling cache entry {}", dest.display());
            fs::remove_file(&dest).map_err(BunsenError::external)?;
        }

        let mut urls = Vec::new();
        for source in pretrained.sources {
            match source {
                WeightsSource::Bundled(path) => {
                    let path = path();
                    if path.is_file() {
                        return Ok(ResolvedWeights {
                            path: path.to_path_buf(),
                            provenance: Provenance::Bundled,
                        });
                    }
                    log::debug!("{id}: bundled file {} is gone", path.display());
                }
                WeightsSource::UpstreamCache => {
                    let Some(path) = self.upstream_path(pretrained) else {
                        continue;
                    };
                    if !path.is_file() {
                        continue;
                    }
                    log::info!("{id}: verifying {}", path.display());
                    match Self::verify(&path, pretrained.sha256) {
                        Ok(()) => {
                            link_or_copy(&path, &dest)?;
                            return Ok(ResolvedWeights {
                                path: dest,
                                provenance: Provenance::UpstreamCache,
                            });
                        }
                        Err(e) => log::warn!("{id}: skipping upstream's copy: {e}"),
                    }
                }
                WeightsSource::Url(url) => urls.push(*url),
            }
        }

        if urls.is_empty() {
            return Err(BunsenError::ResourceNotFound(format!(
                "{id}: not local, and no URL to fetch it from"
            )));
        }
        if self.offline {
            return Err(BunsenError::ResourceNotFound(format!(
                "{id}: not in the cache ({}) and --offline is set",
                dest.display()
            )));
        }

        let mut last = None;
        for url in urls {
            log::info!("{id}: fetching {url}");
            match fetch_verified(url, &dest, pretrained.sha256) {
                Ok(()) => {
                    return Ok(ResolvedWeights {
                        path: dest,
                        provenance: Provenance::Downloaded,
                    });
                }
                Err(e) => {
                    log::warn!("{id}: {url}: {e}");
                    last = Some(e);
                }
            }
        }
        Err(last.expect("at least one URL was tried"))
    }

    /// Checks a file against a digest.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] on a mismatch, [`BunsenError::External`] if
    /// the file cannot be read.
    pub fn verify(
        path: &Path,
        sha256: &str,
    ) -> BunsenResult<()> {
        let found = sha256_of(path).map_err(BunsenError::external)?;
        if found == sha256 {
            Ok(())
        } else {
            Err(BunsenError::Invalid(format!(
                "{}: sha256 is {found}, expected {sha256}",
                path.display()
            )))
        }
    }

    /// Where upstream's cache would keep this file.
    fn upstream_path(
        &self,
        pretrained: &WhisperPretrained,
    ) -> Option<PathBuf> {
        self.upstream_dir.as_ref().map(|d| d.join(pretrained.file))
    }
}

/// `openai-whisper`'s default `download_root`: `$XDG_CACHE_HOME/whisper`, or
/// `~/.cache/whisper`. Upstream uses `~/.cache` on every platform, so this
/// does not ask the platform for its cache directory.
fn default_upstream_cache_dir() -> Option<PathBuf> {
    let cache_home = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::home_dir().map(|h| h.join(".cache")))?;
    Some(cache_home.join("whisper"))
}

/// The lowercase hex SHA-256 of a file.
pub fn sha256_of(path: &Path) -> io::Result<String> {
    let file = fs::File::open(path)?;
    let mut reader = HashingReader::new(file);
    io::copy(&mut reader, &mut io::sink())?;
    Ok(reader.hex_digest())
}

/// A reader that digests what passes through it, so a download is hashed
/// as it lands rather than read back afterwards.
struct HashingReader<R> {
    inner: R,
    hasher: Sha256,
}

impl<R: Read> HashingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    fn hex_digest(self) -> String {
        self.hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

impl<R: Read> Read for HashingReader<R> {
    fn read(
        &mut self,
        buf: &mut [u8],
    ) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
}

/// Streams `url` to `dest`, verified.
///
/// The bytes land in `<dest>.partial` and are hashed on the way; only a
/// matching file is renamed into place, so an interrupted or corrupt
/// transfer never leaves a trusted-looking file at `dest`.
fn fetch_verified(
    url: &str,
    dest: &Path,
    sha256: &str,
) -> BunsenResult<()> {
    let parent = dest.parent().expect("a cache path has a parent");
    fs::create_dir_all(parent).map_err(BunsenError::external)?;
    let partial = partial_path(dest);

    let mut response = ureq::get(url).call().map_err(BunsenError::external)?;
    let len = response.body().content_length();
    let bar = progress_bar(len, dest);

    let mut reader = HashingReader::new(bar.wrap_read(response.body_mut().as_reader()));
    let copied = (|| -> io::Result<()> {
        let mut file = fs::File::create(&partial)?;
        io::copy(&mut reader, &mut file)?;
        file.sync_all()
    })();
    bar.finish_and_clear();
    if let Err(e) = copied {
        let _ = fs::remove_file(&partial);
        return Err(BunsenError::external(e));
    }

    let found = reader.hex_digest();
    if found != sha256 {
        let _ = fs::remove_file(&partial);
        return Err(BunsenError::Invalid(format!(
            "{url}: sha256 is {found}, expected {sha256}; the transfer was corrupt or the asset changed"
        )));
    }

    fs::rename(&partial, dest).map_err(BunsenError::external)
}

/// `<dest>.partial`, keeping every dot of the name (`tiny.en.pt.partial`).
fn partial_path(dest: &Path) -> PathBuf {
    let mut name = dest
        .file_name()
        .expect("a cache path has a file name")
        .to_os_string();
    name.push(".partial");
    dest.with_file_name(name)
}

/// A byte progress bar for a transfer, hidden when stderr is not a terminal.
fn progress_bar(
    len: Option<u64>,
    dest: &Path,
) -> ProgressBar {
    let name = dest.file_name().unwrap_or_default().display().to_string();
    let bar = match len {
        Some(len) => ProgressBar::new(len).with_style(
            ProgressStyle::with_template(
                "{msg} {bar:32} {bytes}/{total_bytes} {bytes_per_sec} eta {eta}",
            )
            .expect("a literal template"),
        ),
        None => ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template("{msg} {spinner} {bytes} {bytes_per_sec}")
                .expect("a literal template"),
        ),
    };
    bar.set_message(name);
    bar
}

/// Puts a verified file at `dest` without copying 3 GB where a link will
/// do: a symlink where the platform has them, then a hard link, then a copy.
fn link_or_copy(
    src: &Path,
    dest: &Path,
) -> BunsenResult<()> {
    let parent = dest.parent().expect("a cache path has a parent");
    fs::create_dir_all(parent).map_err(BunsenError::external)?;

    #[cfg(unix)]
    if std::os::unix::fs::symlink(src, dest).is_ok() {
        return Ok(());
    }
    if fs::hard_link(src, dest).is_ok() {
        return Ok(());
    }
    fs::copy(src, dest)
        .map(|_| ())
        .map_err(BunsenError::external)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::pretrained::OPENAI;

    /// SHA-256 of the three bytes `abc`, from FIPS 180-4's examples.
    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn cache_in(dir: &Path) -> WeightsCache {
        WeightsCache::new(WeightsCacheOptions {
            cache_dir: Some(dir.to_path_buf()),
            offline: true,
            upstream_cache_dir: Some(dir.join("upstream")),
        })
        .unwrap()
    }

    #[test]
    fn test_sha256_of_and_verify() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("abc");
        fs::write(&path, b"abc").unwrap();

        assert_eq!(sha256_of(&path).unwrap(), ABC_SHA256);
        WeightsCache::verify(&path, ABC_SHA256).unwrap();
        assert!(matches!(
            WeightsCache::verify(&path, &"0".repeat(64)),
            Err(BunsenError::Invalid(_))
        ));
    }

    #[test]
    fn test_cached_path_is_digest_addressed() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        let tiny = OPENAI.lookup("tiny.en").unwrap();

        assert_eq!(
            cache.cached_path("openai", tiny),
            dir.path()
                .join("whisper")
                .join("openai")
                .join(tiny.sha256)
                .join("tiny.en.pt"),
        );
        assert_eq!(
            partial_path(Path::new("/c/tiny.en.pt")),
            PathBuf::from("/c/tiny.en.pt.partial")
        );
    }

    #[test]
    fn test_offline_resolution_without_a_local_source_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_in(dir.path());
        let tiny = OPENAI.lookup("tiny.en").unwrap();

        assert_eq!(cache.status("openai", tiny), CacheStatus::Remote);
        assert!(matches!(
            cache.resolve("openai", tiny),
            Err(BunsenError::ResourceNotFound(_))
        ));
    }

    #[test]
    fn test_a_cached_file_is_trusted_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_in(dir.path());
        let tiny = OPENAI.lookup("tiny.en").unwrap();

        // Whatever is at the pinned path is taken as verified-when-written;
        // the digest is in the path, not re-checked per run.
        let dest = cache.cached_path("openai", tiny);
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(&dest, b"not really tiny.en").unwrap();

        assert_eq!(cache.status("openai", tiny), CacheStatus::Cached);
        assert_eq!(
            cache.resolve("openai", tiny).unwrap(),
            ResolvedWeights {
                path: dest,
                provenance: Provenance::Cached
            }
        );
    }

    #[test]
    fn test_upstream_cache_is_verified_before_adoption() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_in(dir.path());
        let tiny = OPENAI.lookup("tiny.en").unwrap();

        // A file of the right name with the wrong bytes: seen, but not taken.
        let upstream = dir.path().join("upstream");
        fs::create_dir_all(&upstream).unwrap();
        fs::write(upstream.join("tiny.en.pt"), b"a partial download").unwrap();

        assert_eq!(cache.status("openai", tiny), CacheStatus::UpstreamCache);
        assert!(matches!(
            cache.resolve("openai", tiny),
            Err(BunsenError::ResourceNotFound(_))
        ));
        assert!(!cache.cached_path("openai", tiny).exists());
    }

    #[test]
    fn test_the_bundled_base_resolves_offline() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_in(dir.path());
        let base = OPENAI.lookup("base").unwrap();

        assert_eq!(cache.status("openai", base), CacheStatus::Bundled);
        let resolved = cache.resolve("openai", base).unwrap();
        assert_eq!(resolved.provenance, Provenance::Bundled);
        assert!(resolved.path.is_file());
        // Nothing was written: the bundled file is used where it is.
        assert!(!dir.path().join("whisper").exists());
    }
}
