//! # Disk Cache Management

use std::{
    path::{
        Path,
        PathBuf,
    },
    sync::Arc,
};

use crate::{
    data::cache::{
        BUNSEN_CACHE_CONFIG,
        TransferObserver,
        TransferObservers,
        fetch_file,
        fetch_verified,
        file_name_from_url,
        path_utils,
    },
    errors::{
        BunsenError,
        BunsenResult,
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

/// Disk cache for downloaded files.
///
/// [`PathResolver`](`super::PathResolver`) decides where the cache and data
/// directories are for a user/system combo, with environment overrides.
/// [`fetch_file`] brings a missing file in: streamed to a `.partial`,
/// digest-checked when a digest is given, and renamed into place. Every
/// transfer is reported to the [`TransferObserver`] stack the options
/// carried.
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
    pub fn new(options: BunsenDiskCacheOptions) -> BunsenResult<Self> {
        let cache_dir = BUNSEN_CACHE_CONFIG
            .resolve_cache_dir(options.cache_dir)
            .ok_or(BunsenError::ResourceNotFound(
                "failed to resolve cache directory".to_string(),
            ))?;

        let data_dir = BUNSEN_CACHE_CONFIG
            .resolve_data_dir(options.data_dir)
            .ok_or(BunsenError::ResourceNotFound(
                "failed to resolve data directory".to_string(),
            ))?;

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

    /// Brings `dest` in from `urls`, tried in order, checked against
    /// `sha256` when one is given; the first URL whose file checks out wins.
    /// Every attempt reports to the observer stack.
    ///
    /// # Errors
    /// With one URL its error comes back as is; with several,
    /// [`BunsenError::External`] naming every failure.
    /// [`BunsenError::Invalid`] with no URL at all.
    pub fn fetch_from_urls(
        &self,
        urls: &[&str],
        dest: &Path,
        sha256: Option<&str>,
    ) -> BunsenResult<()> {
        if urls.is_empty() {
            return Err(BunsenError::Invalid(format!(
                "{}: no URL to fetch from",
                dest.display()
            )));
        }
        let mut failures = Vec::with_capacity(urls.len());
        for url in urls {
            match fetch_file(url, dest, sha256, &self.transfer_observers) {
                Ok(()) => return Ok(()),
                Err(e) => failures.push((*url, e)),
            }
        }
        if failures.len() == 1 {
            return Err(failures.pop().expect("one failure").1);
        }
        let failures: Vec<String> = failures
            .iter()
            .map(|(url, e)| format!("{url}: {e}"))
            .collect();
        Err(BunsenError::External(format!(
            "{}: no URL could be fetched: {}",
            dest.display(),
            failures.join("; ")
        )))
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
            return Err(BunsenError::Invalid("no URL to load from".to_string()));
        };
        let Some(file_name) = file_name_from_url(first) else {
            return Err(BunsenError::Invalid(format!(
                "{first}: the URL names no file"
            )));
        };
        let path = path_utils::extend_path(root, context, file_name);

        if path.exists() {
            return Ok(path);
        }
        if !download {
            return Err(BunsenError::ResourceNotFound(format!(
                "cached file not found: {}",
                path.display()
            )));
        }

        self.fetch_from_urls(&urls, &path, sha256)?;
        Ok(path)
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
    /// * [`BunsenError::ResourceNotFound`] if the file is not cached and
    ///   `download` is `false`.
    /// * [`BunsenError::Invalid`] if a fetched file does not match `sha256`, or
    ///   no URL names a file.
    /// * [`BunsenError::External`] if the transfer fails; with several URLs,
    ///   the message names every failure.
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

#[cfg(test)]
mod tests {
    use std::{
        env,
        fs,
    };

    use serial_test::serial;

    use super::*;
    use crate::data::cache::{
        partial_path,
        transfer::testing::{
            ABC_SHA256,
            CacheProgressEvent,
            RecordingObserver,
            refused_url,
            serve_once,
        },
    };

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

        // Nothing on disk: neither finds anything.
        assert!(matches!(
            cache.load_data_path(&context, &urls, false, None),
            Err(BunsenError::ResourceNotFound(_))
        ));
        assert!(matches!(
            cache.load_cached_path(&context, &urls, false, None),
            Err(BunsenError::ResourceNotFound(_))
        ));

        // A data file is found by `load_data_path` only.
        fs::create_dir_all(data_file.parent().unwrap()).unwrap();
        fs::write(&data_file, b"data").unwrap();
        assert_eq!(
            cache.load_data_path(&context, &urls, false, None).unwrap(),
            data_file
        );
        assert!(matches!(
            cache.load_cached_path(&context, &urls, false, None),
            Err(BunsenError::ResourceNotFound(_))
        ));

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
        assert!(matches!(
            cache.load_data_path(&context, &urls, false, None),
            Err(BunsenError::ResourceNotFound(_))
        ));
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
    /// disk, and comes back `Invalid` when it was the only URL.
    #[test]
    fn test_load_cached_path_refuses_a_bad_digest() {
        let dir = tempfile::tempdir().unwrap();
        let observer = Arc::new(RecordingObserver::default());
        let cache = observing_cache(dir.path(), &observer);

        let url = serve_once("abc.bin", b"abd");
        let result = cache.load_cached_path(&["t"], &[url.as_str()], true, Some(ABC_SHA256));

        assert!(matches!(result, Err(BunsenError::Invalid(_))), "{result:?}");
        let path = cache.cache_path(&["t"], "abc.bin");
        assert!(!path.exists());
        assert!(!partial_path(&path).exists());
    }

    /// When every URL fails, the error names each one.
    #[test]
    fn test_load_cached_path_names_every_failed_url() {
        let dir = tempfile::tempdir().unwrap();
        let observer = Arc::new(RecordingObserver::default());
        let cache = observing_cache(dir.path(), &observer);

        let a = refused_url("abc.bin");
        let b = refused_url("abc.bin");
        match cache.load_cached_path(&["t"], &[a.as_str(), b.as_str()], true, None) {
            Err(BunsenError::External(message)) => {
                assert!(message.contains(&a), "{message}");
                assert!(message.contains(&b), "{message}");
            }
            other => panic!("expected External, got {other:?}"),
        }
        assert!(observer.events().is_empty());
    }

    /// A URL that names no file is refused before anything is fetched.
    #[test]
    fn test_load_cached_path_needs_a_file_name() {
        let dir = tempfile::tempdir().unwrap();
        let observer = Arc::new(RecordingObserver::default());
        let cache = observing_cache(dir.path(), &observer);

        let result = cache.load_cached_path(&["t"], &["https://example.invalid/dir/"], true, None);
        assert!(matches!(result, Err(BunsenError::Invalid(_))), "{result:?}");
        let result = cache.load_cached_path::<&str, &str>(&["t"], &[], true, None);
        assert!(matches!(result, Err(BunsenError::Invalid(_))), "{result:?}");
    }
}
