//! # Disk Cache Management

use std::{
    path::{
        Path,
        PathBuf,
    },
    sync::Arc,
};
#[cfg(feature = "fetch")]
use std::{
    thread,
    time::Duration,
};

use crate::{
    data::cache::{
        BUNSEN_CACHE_CONFIG,
        TransferObserver,
        TransferObservers,
        path_utils,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};
#[cfg(feature = "fetch")]
use crate::{
    data::cache::{
        FetchFailure,
        fetch_file,
        fetch_verified,
        file_name_from_url,
    },
    errors::{
        BunsenErrorKind,
        ConstraintError,
        Multiple,
        ResultContext,
    },
};

/// Environment variable key to override the default cache directory.
pub const BUNSEN_CACHE_DIR: &str = "BUNSEN_CACHE_DIR";
/// Environment variable key to override the default data directory.
pub const BUNSEN_DATA_DIR: &str = "BUNSEN_DATA_DIR";

/// Options for [`BunsenDiskCache`].
#[derive(Clone, Debug)]
pub struct BunsenDiskCacheOptions {
    /// Optional path to the cache directory.
    pub cache_dir: Option<PathBuf>,

    /// Optional path to the data directory.
    pub data_dir: Option<PathBuf>,

    /// Told about every transfer, in this order; see [`TransferObserver`].
    ///
    /// [`Default`] seeds it from [`default_transfer_observers`]: one
    /// `IndicatifObserver` with the `indicatif` feature, none without.
    pub transfer_observers: TransferObservers,
}

impl Default for BunsenDiskCacheOptions {
    fn default() -> Self {
        Self {
            cache_dir: None,
            data_dir: None,
            transfer_observers: default_transfer_observers(),
        }
    }
}

/// The observers [`BunsenDiskCacheOptions::default()`] carries: one
/// `IndicatifObserver` drawing on stderr with the `indicatif` feature, none
/// without.
pub fn default_transfer_observers() -> TransferObservers {
    // Only the feature-on build mutates it, and the workspace denies
    // `unused_mut`; the other build is told the `mut` is expected.
    #[cfg_attr(not(feature = "indicatif"), allow(unused_mut))]
    let mut observers: TransferObservers = Vec::new();

    #[cfg(feature = "indicatif")]
    {
        use crate::data::cache::IndicatifObserver;
        observers.push(Arc::new(IndicatifObserver::default()));
    }

    observers
}

impl BunsenDiskCacheOptions {
    /// Sets the cache directory.
    pub fn with_cache_dir<P: AsRef<Path>>(
        mut self,
        cache_dir: Option<P>,
    ) -> Self {
        self.cache_dir = cache_dir.map(|p| p.as_ref().to_path_buf());
        self
    }

    /// Sets the data directory.
    pub fn with_data_dir<P: AsRef<Path>>(
        mut self,
        data_dir: Option<P>,
    ) -> Self {
        self.data_dir = data_dir.map(|p| p.as_ref().to_path_buf());
        self
    }

    /// Adds an observer after the ones already there.
    pub fn with_transfer_observer(
        mut self,
        observer: Arc<dyn TransferObserver>,
    ) -> Self {
        self.transfer_observers.push(observer);
        self
    }

    /// Replaces the observers.
    pub fn with_transfer_observers(
        mut self,
        observers: TransferObservers,
    ) -> Self {
        self.transfer_observers = observers;
        self
    }

    /// Drops every observer, the default `indicatif` one included.
    pub fn without_transfer_observers(mut self) -> Self {
        self.transfer_observers.clear();
        self
    }
}

/// Disk cache for downloaded files: a cache directory, a data directory,
/// and the observers every transfer reports to.
///
/// [`new`](Self::new) resolves both directories from the
/// [`BunsenDiskCacheOptions`], then the environment, then the platform,
/// through [`BUNSEN_CACHE_CONFIG`](super::BUNSEN_CACHE_CONFIG), a
/// [`PathResolver`](super::PathResolver). Every transfer is reported to
/// the [`TransferObserver`] stack the options carried.
///
/// The layers above keep their files here. A
/// [`PretrainedCache`](crate::data::pretrained::PretrainedCache) wraps one,
/// opened from its options' `disk`, and a
/// [`ShardSet`](crate::data::shards::ShardSet) borrows one.
#[cfg_attr(
    feature = "fetch",
    doc = "",
    doc = "With the `fetch` feature, [`fetch_file`] brings a missing file in: streamed to",
    doc = "a `.partial`, digest-checked when a digest is given, and renamed into place."
)]
pub struct BunsenDiskCache {
    /// Cache directory.
    cache_dir: PathBuf,

    /// Data directory.
    data_dir: PathBuf,

    /// Told about every transfer, in this order.
    transfer_observers: TransferObservers,
}

impl Default for BunsenDiskCache {
    fn default() -> Self {
        Self::new(BunsenDiskCacheOptions::default()).unwrap()
    }
}

impl BunsenDiskCache {
    /// Constructs a new [`BunsenDiskCache`].
    ///
    /// # Errors
    /// [`Policy`](crate::errors::BunsenErrorKind::Policy) when a directory
    /// cannot be resolved: no option, no environment variable, and no
    /// platform default. Setting either of the first two fixes it.
    pub fn new(options: BunsenDiskCacheOptions) -> BunsenResult<Self> {
        let cache_dir = BUNSEN_CACHE_CONFIG
            .resolve_cache_dir(options.cache_dir)
            .ok_or_else(|| {
                BunsenError::policy(format!(
                    "no cache directory: none was given, {BUNSEN_CACHE_DIR} is not set, and the platform has no default"
                ))
            })?;

        let data_dir = BUNSEN_CACHE_CONFIG
            .resolve_data_dir(options.data_dir)
            .ok_or_else(|| {
                BunsenError::policy(format!(
                    "no data directory: none was given, {BUNSEN_DATA_DIR} is not set, and the platform has no default"
                ))
            })?;

        Ok(Self {
            cache_dir,
            data_dir,
            transfer_observers: options.transfer_observers,
        })
    }

    /// Returns the cache directory.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Returns the data directory.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// The transfer observers, in the order each transfer reaches them.
    pub fn transfer_observers(&self) -> &[Arc<dyn TransferObserver>] {
        &self.transfer_observers
    }

    /// Returns the cache path for the given key.
    ///
    /// * Does not check that the path exists.
    /// * Does not initialize the containing directories.
    ///
    /// # Arguments
    /// * `context` - prefix dirs, inserted between `self.cache_dir` and `file`.
    /// * `file` - the final file name.
    pub fn cache_path<C, F>(
        &self,
        context: &[C],
        file: F,
    ) -> PathBuf
    where
        C: AsRef<Path>,
        F: AsRef<Path>,
    {
        path_utils::extend_path(&self.cache_dir, context, file)
    }

    /// Returns the data path for the given key.
    ///
    /// * Does not check that the path exists.
    /// * Does not initialize the containing directories.
    ///
    /// # Arguments
    /// * `context` - prefix dirs, inserted between `self.cache_dir` and `file`.
    /// * `file` - the final file name.
    pub fn data_path<C, F>(
        &self,
        context: &[C],
        file: F,
    ) -> PathBuf
    where
        C: AsRef<Path>,
        F: AsRef<Path>,
    {
        path_utils::extend_path(&self.data_dir, context, file)
    }
}

/// The network side: every method here needs the `fetch` feature.
#[cfg(feature = "fetch")]
impl BunsenDiskCache {
    /// Streams `url` to `dest`, verified against `sha256`, reporting to the
    /// observer stack; see [`fetch_verified`](super::fetch_verified).
    ///
    /// # Errors
    /// As [`fetch_verified`](super::fetch_verified).
    pub fn fetch_verified(
        &self,
        url: &str,
        dest: &Path,
        sha256: &str,
    ) -> BunsenResult<()> {
        fetch_verified(url, dest, sha256, &self.transfer_observers)
    }

    /// Streams `url` to `dest`, checked against `sha256` when one is given,
    /// reporting to the observer stack; see [`fetch_file`](super::fetch_file).
    ///
    /// # Errors
    /// As [`fetch_file`](super::fetch_file).
    pub fn fetch_file(
        &self,
        url: &str,
        dest: &Path,
        sha256: Option<&str>,
    ) -> BunsenResult<()> {
        fetch_file(url, dest, sha256, &self.transfer_observers)
    }

    /// Brings `dest` in from `urls`, mirrors tried in order, checked against
    /// `sha256` when one is given; the first URL whose file checks out wins.
    /// Every attempt reports to the observer stack.
    ///
    /// A mirror is passed over when it fails in a way another mirror may
    /// not: it is down ([`Unavailable`](BunsenErrorKind::Unavailable)), it
    /// does not have the file or will not serve it (a
    /// [`Lookup`](BunsenErrorKind::Lookup) of the URL), or it served bytes
    /// that do not match the digest ([`FetchFailure::Digest`]). Any other
    /// failure, such as the local disk failing, stops the fetch: no other
    /// mirror would help. Nothing is retried;
    /// [`fetch_from_urls_retrying`](Self::fetch_from_urls_retrying) retries
    /// a mirror that is down.
    ///
    /// # Errors
    /// - [`Illegal`](BunsenErrorKind::Illegal), with a [`ConstraintError`]
    ///   cause, with no URL at all.
    /// - The error that stopped the fetch, as [`fetch_file`](super::fetch_file)
    ///   returns it.
    /// - When every mirror failed: with one URL, its error; with several, a
    ///   [`Multiple`] holding each mirror's error, labelled by its URL, whose
    ///   kind comes from theirs (see [`Multiple::kind`]).
    pub fn fetch_from_urls(
        &self,
        urls: &[&str],
        dest: &Path,
        sha256: Option<&str>,
    ) -> BunsenResult<()> {
        self.fetch_mirrors(urls, dest, sha256, 0).0
    }

    /// [`fetch_from_urls`](Self::fetch_from_urls), retrying a mirror that is
    /// down up to `retries` more times before passing it over: the retry
    /// count of a [`FetchPolicy`](super::FetchPolicy).
    ///
    /// Only a failure whose kind
    /// [is retryable](BunsenErrorKind::is_retryable) is retried. Each retry
    /// first waits as long as the server's `Retry-After` asked, or else a
    /// backoff that doubles from 100 ms; either is capped at a minute.
    ///
    /// # Errors
    /// As [`fetch_from_urls`](Self::fetch_from_urls).
    pub fn fetch_from_urls_retrying(
        &self,
        urls: &[&str],
        dest: &Path,
        sha256: Option<&str>,
        retries: u32,
    ) -> BunsenResult<()> {
        self.fetch_mirrors(urls, dest, sha256, retries).0
    }

    /// The mirror loop behind [`fetch_from_urls_retrying`]: the outcome, and
    /// the number of fetches tried.
    ///
    /// [`fetch_from_urls_retrying`]: Self::fetch_from_urls_retrying
    pub(super) fn fetch_mirrors(
        &self,
        urls: &[&str],
        dest: &Path,
        sha256: Option<&str>,
        retries: u32,
    ) -> (BunsenResult<()>, u32) {
        if urls.is_empty() {
            let error: BunsenResult<()> =
                Err(ConstraintError::zero_or_empty("BunsenDiskCache", "urls").into());
            return (error.with_context(|| dest.display().to_string()), 0);
        }
        let mut attempts = 0;
        let mut failures: Vec<(String, BunsenError)> = Vec::with_capacity(urls.len());
        for url in urls {
            let mut retried = 0;
            let error = loop {
                attempts += 1;
                let error = match fetch_file(url, dest, sha256, &self.transfer_observers) {
                    Ok(()) => return (Ok(()), attempts),
                    Err(error) => error,
                };
                if error.kind().is_retryable() && retried < retries {
                    retried += 1;
                    thread::sleep(retry_delay(&error, retried));
                    continue;
                }
                break error;
            };
            let failure = error.find::<FetchFailure>();
            let next_mirror = error.kind().is_retryable()
                || matches!(failure, Some(FetchFailure::Digest(_)))
                || (error.kind() == BunsenErrorKind::Lookup && failure.is_some());
            if !next_mirror {
                return (Err(error), attempts);
            }
            failures.push((url.to_string(), error));
        }
        let error = if failures.len() == 1 {
            failures.pop().expect("one failure").1
        } else {
            let summary = format!(
                "{}: none of {} mirrors could be fetched",
                dest.display(),
                failures.len()
            );
            Multiple::new(summary, failures).into()
        };
        (Err(error), attempts)
    }

    /// Finds `context/<file>` under `root`, fetching it from `urls` when it
    /// is not there. The file is named by the last path segment of the first
    /// URL.
    ///
    /// The fetch is [`fetch_from_urls`](Self::fetch_from_urls).
    fn _load_resource<P, C, S>(
        &self,
        root: &P,
        context: &[C],
        urls: &[S],
        download: bool,
        sha256: Option<&str>,
    ) -> BunsenResult<PathBuf>
    where
        P: AsRef<Path>,
        C: AsRef<Path>,
        S: AsRef<str>,
    {
        let urls: Vec<&str> = urls.iter().map(AsRef::as_ref).collect();
        let Some(first) = urls.first() else {
            return Err(ConstraintError::zero_or_empty("BunsenDiskCache", "urls").into());
        };
        let Some(file_name) = file_name_from_url(first) else {
            return Err(BunsenError::illegal(format!(
                "{first}: the URL names no file"
            )));
        };
        let path = path_utils::extend_path(root, context, file_name);

        if path.exists() {
            return Ok(path);
        }
        if !download {
            return Err(BunsenError::policy(format!(
                "{}: not cached, and downloads are off",
                path.display()
            )));
        }

        self.fetch_from_urls(&urls, &path, sha256)?;
        Ok(path)
    }

    /// Loads a file under the cache directory, fetching it if it is not
    /// there.
    ///
    /// # Arguments
    /// * `context`: A slice of `C` containing path-related context used in
    ///   determining the cache location. These paths are combined to build the
    ///   cached file's location.
    /// * `urls`: The URLs to fetch the file from, tried in order, if it is not
    ///   already cached. The file is named by the first URL's last path
    ///   segment.
    /// * `download`: A boolean flag indicating whether to attempt downloading
    ///   the file from the provided URLs if it does not already exist in the
    ///   cache.
    /// * `sha256`: The lowercase hex SHA-256 a fetched file must have, when
    ///   pinned. A file already in the cache is trusted as it is.
    ///
    /// # Returns
    /// * Returns a [`PathBuf`] pointing to the cached file if it exists or is
    ///   successfully downloaded.
    ///
    /// # Errors
    /// * [`Policy`](BunsenErrorKind::Policy) if the file is not cached and
    ///   `download` is `false`.
    /// * [`Illegal`](BunsenErrorKind::Illegal) if there is no URL, or the first
    ///   URL names no file.
    /// * Otherwise as [`fetch_from_urls`](Self::fetch_from_urls): a digest
    ///   mismatch is [`InvalidResource`](BunsenErrorKind::InvalidResource), and
    ///   with several URLs that all failed, the error is a [`Multiple`] of each
    ///   URL's.
    pub fn load_cached_path<C, S>(
        &self,
        context: &[C],
        urls: &[S],
        download: bool,
        sha256: Option<&str>,
    ) -> BunsenResult<PathBuf>
    where
        C: AsRef<Path>,
        S: AsRef<str>,
    {
        self._load_resource(&self.cache_dir, context, urls, download, sha256)
    }

    /// Loads a file under the data directory, fetching it if it is not there.
    ///
    /// # Arguments
    /// * `context`: A slice of `C` containing path-related context used in
    ///   determining the cache location. These paths are combined to build the
    ///   data file's location.
    /// * `urls`: The URLs to fetch the file from, tried in order, if it is not
    ///   already there. The file is named by the first URL's last path segment.
    /// * `download`: A boolean flag indicating whether to attempt downloading
    ///   the file from the provided URLs if it does not already exist in the
    ///   cache.
    /// * `sha256`: The lowercase hex SHA-256 a fetched file must have, when
    ///   pinned. A file already there is trusted as it is.
    ///
    /// # Returns
    /// * Returns a [`PathBuf`] pointing to the data file if it exists or is
    ///   successfully downloaded.
    ///
    /// # Errors
    /// As [`load_cached_path`](Self::load_cached_path).
    pub fn load_data_path<C, S>(
        &self,
        context: &[C],
        urls: &[S],
        download: bool,
        sha256: Option<&str>,
    ) -> BunsenResult<PathBuf>
    where
        C: AsRef<Path>,
        S: AsRef<str>,
    {
        self._load_resource(&self.data_dir, context, urls, download, sha256)
    }
}

/// The longest a retry of a mirror waits.
#[cfg(feature = "fetch")]
const MAX_RETRY_DELAY: Duration = Duration::from_secs(60);

/// The first backoff between retries of a mirror; it doubles with each.
#[cfg(feature = "fetch")]
const RETRY_BASE_DELAY: Duration = Duration::from_millis(100);

/// How long to wait before retry number `retry` (from 1) after `error`: the
/// server's `Retry-After` when it sent one, or else the backoff; capped at
/// [`MAX_RETRY_DELAY`].
#[cfg(feature = "fetch")]
fn retry_delay(
    error: &BunsenError,
    retry: u32,
) -> Duration {
    if let Some(FetchFailure::Status {
        retry_after: Some(after),
        ..
    }) = error.find::<FetchFailure>()
    {
        return (*after).min(MAX_RETRY_DELAY);
    }
    let doublings = retry.saturating_sub(1).min(16);
    (RETRY_BASE_DELAY * 2u32.pow(doublings)).min(MAX_RETRY_DELAY)
}

#[cfg(test)]
mod tests {
    use std::env;

    use serial_test::serial;

    use super::*;
    use crate::data::cache::testing::RecordingObserver;

    #[test]
    #[serial]
    fn test_resolve_dirs() {
        let orig_cache_dir = env::var(BUNSEN_CACHE_DIR);
        let orig_data_dir = env::var(BUNSEN_DATA_DIR);

        let pds = BUNSEN_CACHE_CONFIG
            .project_dirs()
            .expect("failed to get project dirs");

        let user_cache_dir = PathBuf::from("/tmp/bunsen/cache");
        let user_data_dir = PathBuf::from("/tmp/bunsen/data");

        let env_cache_dir = PathBuf::from("/tmp/bunsen/env_cache");
        let env_data_dir = PathBuf::from("/tmp/bunsen/env_data");

        // No env vars
        unsafe {
            env::remove_var(BUNSEN_CACHE_DIR);
            env::remove_var(BUNSEN_DATA_DIR);
        }

        let cache = BunsenDiskCache::new(
            BunsenDiskCacheOptions::default()
                .with_cache_dir(Some(user_cache_dir.clone()))
                .with_data_dir(Some(user_data_dir.clone())),
        )
        .unwrap();
        assert_eq!(&cache.cache_dir(), &user_cache_dir);
        assert_eq!(&cache.data_dir(), &user_data_dir);

        let cache = BunsenDiskCache::new(BunsenDiskCacheOptions::default()).unwrap();
        assert_eq!(&cache.cache_dir(), &pds.cache_dir().to_path_buf());
        assert_eq!(&cache.data_dir(), &pds.data_dir().to_path_buf());

        // With env var.
        unsafe {
            env::set_var(BUNSEN_CACHE_DIR, env_cache_dir.to_str().unwrap());
            env::set_var(BUNSEN_DATA_DIR, env_data_dir.to_str().unwrap());
        }

        let cache = BunsenDiskCache::new(
            BunsenDiskCacheOptions::default()
                .with_cache_dir(Some(user_cache_dir.clone()))
                .with_data_dir(Some(user_data_dir.clone())),
        )
        .unwrap();
        assert_eq!(&cache.cache_dir(), &user_cache_dir);
        assert_eq!(&cache.data_dir(), &user_data_dir);

        let cache = BunsenDiskCache::new(BunsenDiskCacheOptions::default()).unwrap();
        assert_eq!(&cache.cache_dir(), &env_cache_dir);
        assert_eq!(&cache.data_dir(), &env_data_dir);

        // restore original env var.
        match orig_cache_dir {
            Ok(original) => unsafe { env::set_var(BUNSEN_CACHE_DIR, original) },
            Err(_) => unsafe { env::remove_var(BUNSEN_CACHE_DIR) },
        }
        match orig_data_dir {
            Ok(original) => unsafe { env::set_var(BUNSEN_DATA_DIR, original) },
            Err(_) => unsafe { env::remove_var(BUNSEN_DATA_DIR) },
        }
    }

    #[test]
    fn test_data_path() {
        let cache = BunsenDiskCache::new(BunsenDiskCacheOptions::default()).unwrap();
        let path = cache.data_path(&["prefix"], "file.txt");
        assert_eq!(path, cache.data_dir.join("prefix").join("file.txt"));
    }

    #[test]
    fn test_cache_path() {
        let cache = BunsenDiskCache::new(BunsenDiskCacheOptions::default()).unwrap();
        let path = cache.cache_path(&["prefix"], "file.txt");
        assert_eq!(path, cache.cache_dir.join("prefix").join("file.txt"));
    }

    /// `Default` seeds the stack from the `indicatif` feature: one observer
    /// with it, none without.
    #[test]
    fn test_default_transfer_observers_follow_the_feature() {
        let expected = if cfg!(feature = "indicatif") { 1 } else { 0 };
        assert_eq!(default_transfer_observers().len(), expected);
        assert_eq!(
            BunsenDiskCacheOptions::default().transfer_observers.len(),
            expected
        );
    }

    /// The builders: one appends after the defaults, one replaces, one
    /// clears; and the cache carries the result through, in order.
    #[test]
    fn test_transfer_observer_builders() {
        let a: Arc<dyn TransferObserver> = Arc::new(RecordingObserver::default());
        let b: Arc<dyn TransferObserver> = Arc::new(RecordingObserver::default());

        let options = BunsenDiskCacheOptions::default().with_transfer_observer(a.clone());
        assert_eq!(
            options.transfer_observers.len(),
            default_transfer_observers().len() + 1
        );
        assert!(Arc::ptr_eq(options.transfer_observers.last().unwrap(), &a));

        let options = options.without_transfer_observers();
        assert!(options.transfer_observers.is_empty());

        let options =
            BunsenDiskCacheOptions::default().with_transfer_observers(vec![a.clone(), b.clone()]);
        assert_eq!(options.transfer_observers.len(), 2);
        assert!(Arc::ptr_eq(&options.transfer_observers[0], &a));
        assert!(Arc::ptr_eq(&options.transfer_observers[1], &b));

        let cache = BunsenDiskCache::new(options).unwrap();
        assert_eq!(cache.transfer_observers().len(), 2);
        assert!(Arc::ptr_eq(&cache.transfer_observers()[0], &a));
        assert!(Arc::ptr_eq(&cache.transfer_observers()[1], &b));
    }
}

#[cfg(all(test, feature = "fetch"))]
mod fetch_tests {
    use std::fs;

    use super::*;
    use crate::{
        data::cache::{
            partial_path,
            testing::{
                ABC_SHA256,
                CacheProgressEvent,
                RecordingObserver,
                refused_url,
                serve_flaky,
                serve_once,
                serve_status,
            },
        },
        errors::testing::{
            ErrorMatcher,
            predicate,
        },
    };

    /// A download opens one transfer on every observer: the URL and the
    /// cache path in `begin`, the byte count as it lands, `Complete` at the
    /// end.
    #[test]
    fn test_load_cached_path_reports_the_transfer() {
        let dir = tempfile::tempdir().unwrap();
        let observer = Arc::new(RecordingObserver::default());
        let cache = BunsenDiskCache::new(
            BunsenDiskCacheOptions::default()
                .with_cache_dir(Some(dir.path().join("cache")))
                .without_transfer_observers()
                .with_transfer_observer(observer.clone()),
        )
        .unwrap();

        let body: &'static [u8] = b"hello, observers";
        let url = serve_once("hello.txt", body);
        let path = cache
            .load_cached_path(&["t"], &[url.as_str()], true, None)
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), body);

        let events = observer.events();
        assert_eq!(
            events.first(),
            Some(&CacheProgressEvent::Begin {
                source: url.clone(),
                dest: path.clone(),
                total: Some(body.len() as u64),
            })
        );
        assert_eq!(events.last(), Some(&CacheProgressEvent::Finish(Ok(()))));
        let last_position = events.iter().rev().find_map(|e| match e {
            CacheProgressEvent::Position(n) => Some(*n),
            _ => None,
        });
        assert_eq!(last_position, Some(body.len() as u64));
    }

    /// The cache's `fetch_verified` reports through the observers it
    /// carries.
    #[test]
    fn test_fetch_verified_reports_through_the_cache_observers() {
        let dir = tempfile::tempdir().unwrap();
        let observer = Arc::new(RecordingObserver::default());
        let cache = BunsenDiskCache::new(
            BunsenDiskCacheOptions::default()
                .with_cache_dir(Some(dir.path().join("cache")))
                .without_transfer_observers()
                .with_transfer_observer(observer.clone()),
        )
        .unwrap();

        let url = serve_once("abc.bin", b"abc");
        let dest = cache.cache_path(&["t"], "abc.bin");
        cache.fetch_verified(&url, &dest, ABC_SHA256).unwrap();

        assert_eq!(fs::read(&dest).unwrap(), b"abc");
        let events = observer.events();
        assert!(matches!(
            events.first(),
            Some(CacheProgressEvent::Begin { .. })
        ));
        assert_eq!(events.last(), Some(&CacheProgressEvent::Finish(Ok(()))));
    }

    /// `load_data_path` looks under the data directory, `load_cached_path`
    /// under the cache directory, and neither sees the other's files.
    #[test]
    fn test_load_paths_root_at_their_own_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let cache = BunsenDiskCache::new(
            BunsenDiskCacheOptions::default()
                .with_cache_dir(Some(dir.path().join("cache")))
                .with_data_dir(Some(dir.path().join("data"))),
        )
        .unwrap();

        // `download = false` throughout: the file is found, or it is not.
        let context = ["prefix"];
        let urls = ["https://example.invalid/mirror/file.txt"];

        let data_file = cache.data_path(&context, "file.txt");
        let cache_file = cache.cache_path(&context, "file.txt");
        assert_ne!(data_file, cache_file);

        // Nothing on disk: neither finds anything, and downloads are off.
        let not_cached = ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_contains("not cached, and downloads are off");
        not_cached.assert_err(&cache.load_data_path(&context, &urls, false, None));
        not_cached.assert_err(&cache.load_cached_path(&context, &urls, false, None));

        // A data file is found by `load_data_path` only.
        fs::create_dir_all(data_file.parent().unwrap()).unwrap();
        fs::write(&data_file, b"data").unwrap();
        assert_eq!(
            cache.load_data_path(&context, &urls, false, None).unwrap(),
            data_file
        );
        not_cached.assert_err(&cache.load_cached_path(&context, &urls, false, None));

        // A cache file is found by `load_cached_path` only.
        fs::remove_file(&data_file).unwrap();
        fs::create_dir_all(cache_file.parent().unwrap()).unwrap();
        fs::write(&cache_file, b"cache").unwrap();
        assert_eq!(
            cache
                .load_cached_path(&context, &urls, false, None)
                .unwrap(),
            cache_file
        );
        not_cached.assert_err(&cache.load_data_path(&context, &urls, false, None));
    }

    fn observing_cache(
        dir: &Path,
        observer: &Arc<RecordingObserver>,
    ) -> BunsenDiskCache {
        BunsenDiskCache::new(
            BunsenDiskCacheOptions::default()
                .with_cache_dir(Some(dir.join("cache")))
                .without_transfer_observers()
                .with_transfer_observer(observer.clone()),
        )
        .unwrap()
    }

    /// URLs are tried in order: one with nothing listening is passed over,
    /// the next serves the file, and only that transfer is reported.
    #[test]
    fn test_load_cached_path_tries_urls_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let observer = Arc::new(RecordingObserver::default());
        let cache = observing_cache(dir.path(), &observer);

        let dead = refused_url("abc.bin");
        let live = serve_once("abc.bin", b"abc");
        let path = cache
            .load_cached_path(
                &["t"],
                &[dead.as_str(), live.as_str()],
                true,
                Some(ABC_SHA256),
            )
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"abc");

        let begins: Vec<_> = observer
            .events()
            .into_iter()
            .filter(|e| matches!(e, CacheProgressEvent::Begin { .. }))
            .collect();
        assert_eq!(
            begins,
            vec![CacheProgressEvent::Begin {
                source: live.clone(),
                dest: path.clone(),
                total: Some(3),
            }]
        );
    }

    /// A pinned file whose bytes do not match is refused, leaves nothing on
    /// disk, and comes back `InvalidResource` when it was the only URL.
    #[test]
    fn test_load_cached_path_refuses_a_bad_digest() {
        let dir = tempfile::tempdir().unwrap();
        let observer = Arc::new(RecordingObserver::default());
        let cache = observing_cache(dir.path(), &observer);

        let url = serve_once("abc.bin", b"abd");
        let result = cache.load_cached_path(&["t"], &[url.as_str()], true, Some(ABC_SHA256));

        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .has_cause::<FetchFailure>()
            .assert_err(&result);
        let path = cache.cache_path(&["t"], "abc.bin");
        assert!(!path.exists());
        assert!(!partial_path(&path).exists());
    }

    /// When every URL fails, the error holds each one's, labelled by its URL,
    /// and takes its kind from theirs: two mirrors that are down are
    /// `Unavailable`.
    #[test]
    fn test_load_cached_path_names_every_failed_url() {
        let dir = tempfile::tempdir().unwrap();
        let observer = Arc::new(RecordingObserver::default());
        let cache = observing_cache(dir.path(), &observer);

        let a = refused_url("abc.bin");
        let b = refused_url("abc.bin");
        let result = cache.load_cached_path(&["t"], &[a.as_str(), b.as_str()], true, None);
        let down = || ErrorMatcher::kind(BunsenErrorKind::Unavailable).has_cause::<FetchFailure>();
        ErrorMatcher::kind(BunsenErrorKind::Unavailable)
            .message_contains("none of 2 mirrors could be fetched")
            .details_contains(&format!("{a}: [Unavailable]"))
            .details_contains(&format!("{b}: [Unavailable]"))
            .member(0, down())
            .member(1, down())
            .assert_err(&result);
        assert!(observer.events().is_empty());
    }

    /// A mirror without the file (`404`) and one with the wrong bytes are both
    /// passed over for the next; the aggregate holds both, and is a `Lookup`
    /// after its first member.
    #[test]
    fn test_fetch_from_urls_passes_over_missing_and_corrupt_mirrors() {
        let dir = tempfile::tempdir().unwrap();
        let observer = Arc::new(RecordingObserver::default());
        let cache = observing_cache(dir.path(), &observer);
        let dest = cache.cache_path(&["t"], "abc.bin");

        let missing = serve_status("abc.bin", 404);
        let corrupt = serve_once("abc.bin", b"abd");
        let live = serve_once("abc.bin", b"abc");
        cache
            .fetch_from_urls(
                &[missing.as_str(), corrupt.as_str(), live.as_str()],
                &dest,
                Some(ABC_SHA256),
            )
            .unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"abc");

        let dest = cache.cache_path(&["t"], "abd.bin");
        let missing = serve_status("abc.bin", 404);
        let corrupt = serve_once("abc.bin", b"abd");
        let result = cache.fetch_from_urls(
            &[missing.as_str(), corrupt.as_str()],
            &dest,
            Some(ABC_SHA256),
        );
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .member(0, ErrorMatcher::kind(BunsenErrorKind::Lookup))
            .member(
                1,
                ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
                    .cause(predicate("a digest failure", |f: &FetchFailure| {
                        matches!(f, FetchFailure::Digest(_))
                    })),
            )
            .assert_err(&result);
    }

    /// A failure no other mirror could help stops the loop: a destination
    /// that is not a file path is `Illegal`, and the second mirror is not
    /// tried.
    #[test]
    fn test_fetch_from_urls_stops_on_a_failure_no_mirror_helps() {
        let dir = tempfile::tempdir().unwrap();
        let observer = Arc::new(RecordingObserver::default());
        let cache = observing_cache(dir.path(), &observer);
        let a = serve_once("abc.bin", b"abc");
        let b = serve_once("abc.bin", b"abc");
        let result = cache.fetch_mirrors(&[a.as_str(), b.as_str()], Path::new("/"), None, 0);
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("not a file path")
            .assert_err(&result.0);
        assert_eq!(result.1, 1, "one fetch was tried");
    }

    /// A mirror that is down is retried within the budget before it is
    /// passed over; without a budget it is not retried.
    #[test]
    fn test_fetch_from_urls_retrying_retries_a_mirror_that_is_down() {
        let dir = tempfile::tempdir().unwrap();
        let observer = Arc::new(RecordingObserver::default());
        let cache = observing_cache(dir.path(), &observer);

        let flaky = serve_flaky("abc.bin", b"abc", 1);
        let dest = cache.cache_path(&["t"], "abc.bin");
        let (result, attempts) = cache.fetch_mirrors(&[flaky.as_str()], &dest, None, 2);
        result.unwrap();
        assert_eq!(attempts, 2);

        let flaky = serve_flaky("abc.bin", b"abc", 1);
        let dest = cache.cache_path(&["t"], "abd.bin");
        ErrorMatcher::kind(BunsenErrorKind::Unavailable).assert_err(&cache.fetch_from_urls(
            &[flaky.as_str()],
            &dest,
            None,
        ));

        // A 404 is not retried, whatever the budget.
        let missing = serve_status("abc.bin", 404);
        let (result, attempts) = cache.fetch_mirrors(&[missing.as_str()], &dest, None, 3);
        ErrorMatcher::kind(BunsenErrorKind::Lookup).assert_err(&result);
        assert_eq!(attempts, 1);
    }

    /// A `Retry-After` is honored, and capped.
    #[test]
    fn test_retry_delay() {
        let status = |retry_after: Option<Duration>| {
            BunsenError::from(FetchFailure::Status {
                url: "u".to_string(),
                status: 503,
                retry_after,
            })
        };
        assert_eq!(
            retry_delay(&status(Some(Duration::from_secs(2))), 1),
            Duration::from_secs(2)
        );
        assert_eq!(
            retry_delay(&status(Some(Duration::from_secs(3600))), 1),
            MAX_RETRY_DELAY
        );
        assert_eq!(retry_delay(&status(None), 1), RETRY_BASE_DELAY);
        assert_eq!(retry_delay(&status(None), 3), RETRY_BASE_DELAY * 4);
        assert_eq!(retry_delay(&status(None), 40), MAX_RETRY_DELAY);
    }

    /// A URL that names no file is refused before anything is fetched.
    #[test]
    fn test_load_cached_path_needs_a_file_name() {
        let dir = tempfile::tempdir().unwrap();
        let observer = Arc::new(RecordingObserver::default());
        let cache = observing_cache(dir.path(), &observer);

        let result = cache.load_cached_path(&["t"], &["https://example.invalid/dir/"], true, None);
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("names no file")
            .assert_err(&result);
        let no_url = ErrorMatcher::kind(BunsenErrorKind::Illegal).has_cause::<ConstraintError>();
        no_url.assert_err(&cache.load_cached_path::<&str, &str>(&["t"], &[], true, None));
        no_url.assert_err(&cache.fetch_from_urls(&[], Path::new("/d/f.bin"), None));
    }
}
