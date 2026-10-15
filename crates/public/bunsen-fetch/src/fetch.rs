//! Fetch: a URL streamed to disk, hashed as it lands.
use std::{
    fmt,
    fs,
    io::{
        self,
        Read,
        Write,
    },
    path::{
        Path,
        PathBuf,
    },
    sync::Arc,
    time::Duration,
};

use crate::{
    HashingReader,
    IoAt,
    TransferDesc,
    TransferObserver,
    TransferOutcome,
    TransferProgress,
    TransferProgressStack,
    partial_path,
};

/// Why a fetch failed on the network side.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum FetchFailure {
    /// The server answered with an error status.
    Status {
        /// The URL requested.
        url: String,
        /// The HTTP status.
        status: u16,
        /// How long the server asked the client to wait before retrying
        /// (`Retry-After`, in seconds), when it said.
        retry_after: Option<Duration>,
    },

    /// No usable response, or the body stream broke: a refused or dropped
    /// connection, a DNS or TLS failure, a timeout.
    Transport {
        /// The URL requested.
        url: String,
        /// The transport's error.
        source: Arc<ureq::Error>,
    },

    /// The body ended short of its `Content-Length`.
    ShortTransfer {
        /// The URL requested.
        url: String,
        /// Bytes received.
        received: u64,
        /// Bytes the server declared.
        expected: u64,
    },

    /// The bytes do not match the pinned digest.
    Digest {
        /// The URL requested.
        url: String,
        /// Where the file was to land.
        dest: PathBuf,
        /// The pinned digest.
        expected: String,
        /// The digest of what arrived.
        found: String,
    },
}

impl FetchFailure {
    /// The URL the failed fetch requested.
    pub fn url(&self) -> &str {
        match self {
            Self::Status { url, .. }
            | Self::Transport { url, .. }
            | Self::ShortTransfer { url, .. }
            | Self::Digest { url, .. } => url,
        }
    }
}

impl fmt::Display for FetchFailure {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Status {
                url,
                status,
                retry_after,
            } => {
                write!(f, "{url}: HTTP status {status}")?;
                if let Some(after) = retry_after {
                    write!(f, " (retry after {}s)", after.as_secs())?;
                }
                Ok(())
            }
            Self::Transport { url, source } => write!(f, "{url}: {source}"),
            Self::ShortTransfer {
                url,
                received,
                expected,
            } => write!(f, "{url}: short transfer, {received} of {expected} bytes"),
            Self::Digest {
                url,
                dest,
                expected,
                found,
            } => write!(
                f,
                "{}: sha256 is {found}, not the pinned {expected} (downloaded from {url}: \
                 the transfer was corrupt, or the source changed)",
                dest.display()
            ),
        }
    }
}

impl std::error::Error for FetchFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

/// Why [`fetch_file`] failed.
#[non_exhaustive]
#[derive(Debug)]
pub enum FetchError {
    /// The network side: the server, the transport, or the bytes.
    Failure(FetchFailure),

    /// The local disk under the transfer: the `.partial`, its directory,
    /// the rename.
    Io(IoAt),

    /// `dest` is not a file path.
    NotAFilePath(PathBuf),

    /// `url` is not a URL.
    BadUrl {
        /// The URL given.
        url: String,
        /// The client's error.
        source: ureq::Error,
    },
}

impl fmt::Display for FetchError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Failure(failure) => write!(f, "{failure}"),
            Self::Io(io) => write!(f, "{io}"),
            Self::NotAFilePath(dest) => write!(f, "{}: not a file path", dest.display()),
            Self::BadUrl { url, .. } => write!(f, "{url}: not a URL"),
        }
    }
}

impl std::error::Error for FetchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Failure(failure) => Some(failure),
            Self::Io(io) => Some(io),
            Self::NotAFilePath(_) => None,
            Self::BadUrl { source, .. } => Some(source),
        }
    }
}

impl From<FetchFailure> for FetchError {
    fn from(failure: FetchFailure) -> Self {
        Self::Failure(failure)
    }
}

impl From<IoAt> for FetchError {
    fn from(io: IoAt) -> Self {
        Self::Io(io)
    }
}

/// Streams `url` to `dest`, checked against `sha256` when one is given,
/// reporting to `observers`.
///
/// The bytes are hashed as they land, in `<dest>.partial`
/// ([`partial_path`]), and renamed into place only once they check out. A
/// pinned file (`Some`) must match its digest; every file must match the
/// `Content-Length` when the server sent one. Nothing is left at `dest` or
/// beside it when either check fails, so an interrupted or corrupt transfer
/// never leaves a trusted-looking file at `dest`.
///
/// The transfer is reported to every observer: `begin` once the response
/// headers are in (which is when the length is known), `position` as bytes
/// land, and `finish` with the outcome. A request that fails before a
/// response body, unanswered or with an error status, is an error with no
/// transfer reported.
///
/// # Errors
/// [`FetchError::Failure`] for the network side, [`FetchError::Io`] for the
/// local disk, [`FetchError::NotAFilePath`] and [`FetchError::BadUrl`] for
/// bad arguments.
pub fn fetch_file(
    url: &str,
    dest: &Path,
    sha256: Option<&str>,
    observers: &[Arc<dyn TransferObserver>],
) -> Result<(), FetchError> {
    let (Some(parent), Some(_)) = (dest.parent(), dest.file_name()) else {
        return Err(FetchError::NotAFilePath(dest.to_path_buf()));
    };
    // The request first, the slot second: a URL that answers with an error
    // status (a repo that is not there) leaves no directory behind.
    let mut response = match ureq::get(url)
        .config()
        .http_status_as_error(false)
        .build()
        .call()
    {
        Ok(response) => response,
        Err(source @ ureq::Error::BadUri(_)) => {
            return Err(FetchError::BadUrl {
                url: url.to_string(),
                source,
            });
        }
        Err(e) => {
            return Err(FetchFailure::Transport {
                url: url.to_string(),
                source: Arc::new(e),
            }
            .into());
        }
    };
    let status = response.status().as_u16();
    if status >= 400 {
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(Duration::from_secs);
        return Err(FetchFailure::Status {
            url: url.to_string(),
            status,
            retry_after,
        }
        .into());
    }
    let total = response.body().content_length();
    fs::create_dir_all(parent).map_err(IoAt::at("create directory", parent))?;
    let partial = partial_path(dest);
    let progress = TransferProgressStack::begin(
        observers,
        &TransferDesc {
            source: url,
            dest,
            total,
        },
    );
    let fail = |e: FetchError| {
        let _ = fs::remove_file(&partial);
        progress.finish(TransferOutcome::Failed(&e.to_string()));
        e
    };

    let mut reader = HashingReader::new(ProgressReader::new(
        response.body_mut().as_reader(),
        &progress,
    ));
    let received = copy_body(&mut reader, &partial, url).map_err(fail)?;
    if let Some(expected) = total
        && expected != received
    {
        return Err(fail(
            FetchFailure::ShortTransfer {
                url: url.to_string(),
                received,
                expected,
            }
            .into(),
        ));
    }

    let found = reader.hex_digest();
    if let Some(expected) = sha256
        && expected != found
    {
        return Err(fail(
            FetchFailure::Digest {
                url: url.to_string(),
                dest: dest.to_path_buf(),
                expected: expected.to_string(),
                found,
            }
            .into(),
        ));
    }

    if let Err(e) = fs::rename(&partial, dest) {
        return Err(fail(IoAt::at("rename", &partial)(e).into()));
    }
    progress.finish(TransferOutcome::Complete);
    Ok(())
}

/// [`fetch_file`] with a required digest.
///
/// # Errors
/// As [`fetch_file`].
pub fn fetch_verified(
    url: &str,
    dest: &Path,
    sha256: &str,
    observers: &[Arc<dyn TransferObserver>],
) -> Result<(), FetchError> {
    fetch_file(url, dest, Some(sha256), observers)
}

/// Copies the response body from `reader` into a new file at `partial`, and
/// syncs it; the byte count. A read failure is the network side, a create,
/// write or sync the local disk.
fn copy_body(
    reader: &mut impl Read,
    partial: &Path,
    url: &str,
) -> Result<u64, FetchError> {
    let mut file = fs::File::create(partial).map_err(IoAt::at("create", partial))?;
    let mut buf = vec![0u8; 64 * 1024];
    let mut received = 0u64;
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => {
                return Err(FetchFailure::Transport {
                    url: url.to_string(),
                    source: Arc::new(ureq::Error::from(e)),
                }
                .into());
            }
        };
        file.write_all(&buf[..n])
            .map_err(IoAt::at("write", partial))?;
        received += n as u64;
    }
    file.sync_all().map_err(IoAt::at("sync", partial))?;
    Ok(received)
}

/// A reader that reports its running byte count to a transfer's sink.
struct ProgressReader<'a, R> {
    inner: R,
    position: u64,
    progress: &'a dyn TransferProgress,
}

impl<'a, R: Read> ProgressReader<'a, R> {
    fn new(
        inner: R,
        progress: &'a dyn TransferProgress,
    ) -> Self {
        Self {
            inner,
            position: 0,
            progress,
        }
    }
}

impl<R: Read> Read for ProgressReader<'_, R> {
    fn read(
        &mut self,
        buf: &mut [u8],
    ) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.position += n as u64;
        self.progress.position(self.position);
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{
        ABC_SHA256,
        CacheProgressEvent,
        RecordingObserver,
        refused_url,
        serve_once,
        serve_once_declaring,
        serve_status,
    };

    fn observers(observer: &Arc<RecordingObserver>) -> Vec<Arc<dyn TransferObserver>> {
        vec![observer.clone()]
    }

    /// A pinned file lands, reported from begin to a complete finish.
    #[test]
    fn test_fetch_verified_lands_and_reports() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("sub").join("abc.bin");
        let url = serve_once("abc.bin", b"abc");
        let observer = Arc::new(RecordingObserver::default());
        fetch_verified(&url, &dest, ABC_SHA256, &observers(&observer)).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"abc");
        let events = observer.events();
        assert!(matches!(
            events.first(),
            Some(CacheProgressEvent::Begin { total: Some(3), .. })
        ));
        assert_eq!(events.last(), Some(&CacheProgressEvent::Finish(Ok(()))));
    }

    /// Wrong bytes, a short body, an error status and no server each fail
    /// as their own kind, and leave nothing at `dest`.
    #[test]
    fn test_failures() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.bin");
        let pinned = "0".repeat(64);

        let observer = Arc::new(RecordingObserver::default());
        let err = fetch_verified(
            &serve_once("x.bin", b"abc"),
            &dest,
            &pinned,
            &observers(&observer),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            FetchError::Failure(FetchFailure::Digest { .. })
        ));
        match observer.events().last() {
            Some(CacheProgressEvent::Finish(Err(message))) => {
                assert!(message.contains("sha256 is"), "{message}");
                assert!(message.contains(&pinned), "{message}");
            }
            other => panic!("expected a failed finish, got {other:?}"),
        }

        let err =
            fetch_file(&serve_once_declaring("x.bin", b"ab", 3), &dest, None, &[]).unwrap_err();
        assert!(
            matches!(
                err,
                FetchError::Failure(FetchFailure::ShortTransfer { .. })
                    | FetchError::Failure(FetchFailure::Transport { .. })
            ),
            "{err}"
        );

        let err = fetch_file(&serve_status("x.bin", 404), &dest, None, &[]).unwrap_err();
        assert!(matches!(
            err,
            FetchError::Failure(FetchFailure::Status { status: 404, .. })
        ));

        let err = fetch_file(&refused_url("x.bin"), &dest, None, &[]).unwrap_err();
        assert!(matches!(
            err,
            FetchError::Failure(FetchFailure::Transport { .. })
        ));

        assert!(!dest.exists());
        assert!(!partial_path(&dest).exists());
        assert!(matches!(
            fetch_file("http://127.0.0.1:1/x", Path::new("/"), None, &[]).unwrap_err(),
            FetchError::NotAFilePath(_)
        ));
    }
}
