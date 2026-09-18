//! # Disk Cache Management

use std::{
    fs,
    path::{
        Path,
        PathBuf,
    },
    sync::Arc,
};

use downloader::{
    Download,
    Downloader,
};

use crate::{
    data::cache::{
        BUNSEN_CACHE_CONFIG,
        TransferObserver,
        TransferObservers,
        downloader_progress::DownloaderReporter,
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

    /// Optional [`Downloader`] builder.
    pub downloader: Option<fn() -> Downloader>,

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
            downloader: None,
            transfer_observers: default_transfer_observers(),
        }
    }
}

/// The observers [`BunsenDiskCacheOptions::default()`] carries: one
/// `IndicatifObserver` drawing on stderr, because the `indicatif` feature
/// is on.
#[cfg(feature = "indicatif")]
pub fn default_transfer_observers() -> TransferObservers {
    vec![Arc::new(super::IndicatifObserver::default())]
}

/// The observers [`BunsenDiskCacheOptions::default()`] carries: none,
/// because the `indicatif` feature is off.
#[cfg(not(feature = "indicatif"))]
pub fn default_transfer_observers() -> TransferObservers {
    Vec::new()
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

    /// Sets the downloader builder.
    pub fn with_downloader(
        mut self,
        downloader: Option<fn() -> Downloader>,
    ) -> Self {
        self.downloader = downloader;
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
/// Leverages [`Downloader`] for downloading files,
/// and [`PathResolver`](`super::PathResolver`) for resolving cache and data
/// paths appropriate for a user/system combo, and any environment overrides.
/// Every transfer is reported to the [`TransferObserver`] stack the options
/// carried.
pub struct BunsenDiskCache {
    /// Cache directory.
    cache_dir: PathBuf,

    /// Data directory.
    data_dir: PathBuf,

    /// Connection pool for downloading files.
    downloader: Downloader,

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

        let downloader = match options.downloader {
            Some(builder) => builder(),
            None => Downloader::builder()
                .build()
                .map_err(BunsenError::external)?,
        };

        Ok(Self {
            cache_dir,
            data_dir,
            downloader,
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

    /// Returns the downloader.
    pub fn downloader(&self) -> &Downloader {
        &self.downloader
    }

    /// The transfer observers, in the order each transfer reaches them.
    pub fn transfer_observers(&self) -> &[Arc<dyn TransferObserver>] {
        &self.transfer_observers
    }

    /// Loads a file from a specified path or downloads it if it does not exist.
    ///
    /// # Arguments
    /// * `context`: A slice of `C` containing path-related context used in
    ///   determining the cache location. These paths are combined to build the
    ///   cached file's location.
    /// * `urls`: A slice of string references specifying the URLs to download
    ///   the file from if it is not already cached.
    /// * `download`: A boolean flag indicating whether to attempt downloading
    ///   the file from the provided URLs if it does not already exist in the
    ///   cache.
    ///
    /// # Returns
    /// * Returns a [`PathBuf`] pointing to the cached file if it exists or is
    ///   successfully downloaded.
    /// * Returns an error if the file is not found in the cache and downloading
    ///   is not allowed or fails.
    ///
    /// # Errors
    /// * Returns an error if the cached file does not exist and `download` is
    ///   `false`.
    /// * Returns an error if the downloading process fails.
    fn _load_resource<P, C, S>(
        &mut self,
        root: &P,
        context: &[C],
        urls: &[S],
        download: bool,
        // TODO: hash: Option<&str>,
    ) -> BunsenResult<PathBuf>
    where
        P: AsRef<Path>,
        C: AsRef<Path>,
        S: AsRef<str>,
    {
        let urls: Vec<_> = urls.iter().map(|s| s.as_ref()).collect();
        let mut dl = Download::new_mirrored(&urls);
        let file_name = dl.file_name.clone();
        let path = path_utils::extend_path(root, context, &file_name);
        dl.file_name = path.clone();

        if path.exists() {
            return Ok(path);
        }

        if !download {
            return Err(BunsenError::ResourceNotFound(format!(
                "cached file not found: {}",
                path.display()
            )));
        }

        fs::create_dir_all(path.parent().unwrap()).map_err(BunsenError::external)?;

        // One transfer on the observer stack per attempt. `downloader`
        // reports `done` only on success, so an attempt it gave up on is
        // failed here, once `download` returns.
        let reporter = Arc::new(DownloaderReporter::new(
            &self.transfer_observers,
            urls.join(", "),
            path.clone(),
        ));
        let dl = dl.progress(reporter.clone());
        let result = self.downloader.download(&[dl]);
        reporter.fail_open("the download did not complete");
        result.map_err(BunsenError::external)?;

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

    /// Loads a cached file from a specified path or downloads it if it does not
    /// exist.
    ///
    /// # Arguments
    /// * `context`: A slice of `C` containing path-related context used in
    ///   determining the cache location. These paths are combined to build the
    ///   cached file's location.
    /// * `urls`: A slice of string references specifying the URLs to download
    ///   the file from if it is not already cached.
    /// * `download`: A boolean flag indicating whether to attempt downloading
    ///   the file from the provided URLs if it does not already exist in the
    ///   cache.
    ///
    /// # Returns
    /// * Returns a [`PathBuf`] pointing to the cached file if it exists or is
    ///   successfully downloaded.
    /// * Returns an error if the file is not found in the cache and downloading
    ///   is not allowed or fails.
    ///
    /// # Errors
    /// * Returns an error if the cached file does not exist and `download` is
    ///   `false`.
    /// * Returns an error if the downloading process fails.
    pub fn load_cached_path<C, S>(
        &mut self,
        context: &[C],
        urls: &[S],
        download: bool,
        // TODO: hash: Option<&str>,
    ) -> BunsenResult<PathBuf>
    where
        C: AsRef<Path>,
        S: AsRef<str>,
    {
        let root = self.cache_dir.clone();
        self._load_resource(&root, context, urls, download)
    }

    /// Loads a data file from a specified path or downloads it if it does not
    /// exist.
    ///
    /// # Arguments
    /// * `context`: A slice of `C` containing path-related context used in
    ///   determining the cache location. These paths are combined to build the
    ///   data file's location.
    /// * `urls`: A slice of string references specifying the URLs to download
    ///   the file from if it is not already data.
    /// * `download`: A boolean flag indicating whether to attempt downloading
    ///   the file from the provided URLs if it does not already exist in the
    ///   cache.
    ///
    /// # Returns
    /// * Returns a [`PathBuf`] pointing to the data file if it exists or is
    ///   successfully downloaded.
    /// * Returns an error if the file is not found in the cache and downloading
    ///   is not allowed or fails.
    ///
    /// # Errors
    /// * Returns an error if the data file does not exist and `download` is
    ///   `false`.
    /// * Returns an error if the downloading process fails.
    pub fn load_data_path<C, S>(
        &mut self,
        context: &[C],
        urls: &[S],
        download: bool,
        // TODO: hash: Option<&str>,
    ) -> BunsenResult<PathBuf>
    where
        C: AsRef<Path>,
        S: AsRef<str>,
    {
        let root = self.data_dir.clone();
        self._load_resource(&root, context, urls, download)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        env,
        fs,
        io::{
            Read,
            Write,
        },
        net::TcpListener,
        path::PathBuf,
        sync::Arc,
        thread,
    };

    use serial_test::serial;

    use crate::{
        data::cache::{
            BUNSEN_CACHE_CONFIG,
            BUNSEN_CACHE_DIR,
            BUNSEN_DATA_DIR,
            BunsenDiskCache,
            BunsenDiskCacheOptions,
            TransferObserver,
            default_transfer_observers,
            transfer::testing::{
                Event,
                RecordingObserver,
            },
        },
        errors::BunsenError,
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

    /// Serves one HTTP/1.1 `200` with `body` on a loopback port, once, and
    /// returns the URL for `name`.
    fn serve_once(
        name: &str,
        body: &'static [u8],
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/{name}", listener.local_addr().unwrap());
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            // Read the request head; a GET carries no body.
            let mut head = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = stream.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                head.extend_from_slice(&buf[..n]);
                if head.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
            stream.flush().unwrap();
        });
        url
    }

    /// A download opens one transfer on every observer: the URL and the
    /// cache path in `begin`, the byte count as it lands, `Complete` at the
    /// end.
    #[test]
    fn test_load_cached_path_reports_the_transfer() {
        let dir = tempfile::tempdir().unwrap();
        let observer = Arc::new(RecordingObserver::default());
        let mut cache = BunsenDiskCache::new(
            BunsenDiskCacheOptions::default()
                .with_cache_dir(Some(dir.path().join("cache")))
                .without_transfer_observers()
                .with_transfer_observer(observer.clone()),
        )
        .unwrap();

        let body: &'static [u8] = b"hello, observers";
        let url = serve_once("hello.txt", body);
        let path = cache
            .load_cached_path(&["t"], &[url.as_str()], true)
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), body);

        let events = observer.events();
        assert_eq!(
            events.first(),
            Some(&Event::Begin {
                source: url.clone(),
                dest: path.clone(),
                total: Some(body.len() as u64),
            })
        );
        assert_eq!(events.last(), Some(&Event::Finish(Ok(()))));
        let last_position = events.iter().rev().find_map(|e| match e {
            Event::Position(n) => Some(*n),
            _ => None,
        });
        assert_eq!(last_position, Some(body.len() as u64));
    }

    /// `load_data_path` looks under the data directory, `load_cached_path`
    /// under the cache directory, and neither sees the other's files.
    #[test]
    fn test_load_paths_root_at_their_own_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = BunsenDiskCache::new(
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
            cache.load_data_path(&context, &urls, false),
            Err(BunsenError::ResourceNotFound(_))
        ));
        assert!(matches!(
            cache.load_cached_path(&context, &urls, false),
            Err(BunsenError::ResourceNotFound(_))
        ));

        // A data file is found by `load_data_path` only.
        fs::create_dir_all(data_file.parent().unwrap()).unwrap();
        fs::write(&data_file, b"data").unwrap();
        assert_eq!(
            cache.load_data_path(&context, &urls, false).unwrap(),
            data_file
        );
        assert!(matches!(
            cache.load_cached_path(&context, &urls, false),
            Err(BunsenError::ResourceNotFound(_))
        ));

        // A cache file is found by `load_cached_path` only.
        fs::remove_file(&data_file).unwrap();
        fs::create_dir_all(cache_file.parent().unwrap()).unwrap();
        fs::write(&cache_file, b"cache").unwrap();
        assert_eq!(
            cache.load_cached_path(&context, &urls, false).unwrap(),
            cache_file
        );
        assert!(matches!(
            cache.load_data_path(&context, &urls, false),
            Err(BunsenError::ResourceNotFound(_))
        ));
    }
}
