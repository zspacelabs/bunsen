//! Fetch: a URL streamed into the cache, hashed as it lands.

use std::{
    fmt,
    fs,
    io::{
        self,
        Read,
        Write,
    },
    path::Path,
    sync::Arc,
    time::Duration,
};

use super::{
    HashingReader,
    TransferDesc,
    TransferObserver,
    TransferOutcome,
    TransferProgress,
    TransferProgressStack,
    partial_path,
};
use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    DigestMismatch,
    DigestOrigin,
    LookupError,
    LookupProblem,
    sys_at,
};

/// Why a fetch from a URL failed: the cause of the errors [`fetch_file`]
/// returns for the network side of a transfer.
///
/// A generic consumer reads only the error's kind. A handler with its own
/// recovery, such as the mirror loop of
/// [`BunsenDiskCache::fetch_from_urls`](super::BunsenDiskCache::fetch_from_urls),
/// reads this cause with [`BunsenError::find`] to tell a mirror that is down
/// from one that served the wrong bytes.
///
/// The kind comes from the variant, through `From`:
///
/// - [`Status`](Self::Status) 401, 403, 404 and 410 are a
///   [`Lookup`](BunsenErrorKind::Lookup) whose [`LookupError`] (`Unauthorized`,
///   `Denied`, `Missing`) wraps this failure; `find::<FetchFailure>()` still
///   reaches it.
/// - [`Status`](Self::Status) 408, 429 and 5xx, [`Transport`](Self::Transport)
///   and [`ShortTransfer`](Self::ShortTransfer) are
///   [`Unavailable`](BunsenErrorKind::Unavailable): a retry may succeed.
/// - Any other status, and [`Digest`](Self::Digest), are
///   [`InvalidResource`](BunsenErrorKind::InvalidResource).
///
/// A failure of the local disk under a transfer (the `.partial`, its
/// directory, the rename) is not a fetch failure: it is sorted by its
/// `io::Error`, with the local path, through [`sys_at`].
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

    /// The bytes do not match the pinned digest; its origin is a
    /// [`DigestOrigin::Download`] naming the URL.
    Digest(DigestMismatch),
}

impl FetchFailure {
    /// The URL the failed fetch requested.
    pub fn url(&self) -> &str {
        match self {
            Self::Status { url, .. }
            | Self::Transport { url, .. }
            | Self::ShortTransfer { url, .. } => url,
            Self::Digest(mismatch) => match &mismatch.origin {
                DigestOrigin::Download { url } => url,
                DigestOrigin::AtRest | DigestOrigin::Bundled => &mismatch.subject,
            },
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
            Self::Digest(mismatch) => write!(f, "{mismatch}"),
        }
    }
}

impl core::error::Error for FetchFailure {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Transport { source, .. } => Some(source.as_ref()),
            Self::Digest(mismatch) => Some(mismatch),
            Self::Status { .. } | Self::ShortTransfer { .. } => None,
        }
    }
}

impl From<FetchFailure> for BunsenError {
    #[track_caller]
    fn from(failure: FetchFailure) -> Self {
        let (kind, lookup) = match &failure {
            FetchFailure::Status { status, .. } => match status {
                401 => (BunsenErrorKind::Lookup, Some(LookupProblem::Unauthorized)),
                403 => (BunsenErrorKind::Lookup, Some(LookupProblem::Denied)),
                404 | 410 => (BunsenErrorKind::Lookup, Some(LookupProblem::Missing)),
                408 | 429 | 500..=599 => (BunsenErrorKind::Unavailable, None),
                _ => (BunsenErrorKind::InvalidResource, None),
            },
            FetchFailure::Transport { .. } | FetchFailure::ShortTransfer { .. } => {
                (BunsenErrorKind::Unavailable, None)
            }
            FetchFailure::Digest(_) => (BunsenErrorKind::InvalidResource, None),
        };
        match lookup {
            Some(problem) => {
                let url = failure.url().to_string();
                BunsenError::lookup(LookupError::url(url, problem).with_source(failure))
            }
            None => BunsenError::from_cause(kind, failure),
        }
    }
}

/// Streams `url` to `dest`, checked against `sha256` when one is given,
/// reporting to `observers`.
///
/// The bytes are hashed as they land, in `<dest>.partial`
/// ([`partial_path`]), and renamed into place only once they check out. A
/// pinned file (`Some`) must match its digest; every file must match the
/// `Content-Length` when the server sent one. Nothing is left at `dest` or
/// beside it when either check fails, so an interrupted or corrupt
/// transfer never leaves a trusted-looking file at `dest`.
///
/// The transfer is reported to every observer: `begin` once the response
/// headers are in (which is when the length is known), `position` as bytes
/// land, and `finish` with the outcome. A request that fails before a
/// response body, unanswered or with an error status, is an error with no
/// transfer reported.
///
/// This is the one place bunsen reaches the network at run time; it is
/// behind the `fetch` feature. Most callers reach it through
/// [`BunsenDiskCache`](super::BunsenDiskCache), which brings its own
/// observers.
///
/// # Errors
/// - A failure of the network side has a [`FetchFailure`] cause, which picks
///   the kind: [`Lookup`](BunsenErrorKind::Lookup) for an HTTP 401, 403, 404 or
///   410; [`Unavailable`](BunsenErrorKind::Unavailable) for a transport
///   failure, a short transfer, or an HTTP 408, 429 or 5xx;
///   [`InvalidResource`](BunsenErrorKind::InvalidResource) for a digest
///   mismatch ([`FetchFailure::Digest`]; nothing is left at `dest`) or any
///   other status.
/// - A failure of the local disk is sorted by its `io::Error` and names the
///   local path (see [`sys_at`]): usually [`Sys`](BunsenErrorKind::Sys).
/// - [`Illegal`](BunsenErrorKind::Illegal) if `dest` is not a file path, or
///   `url` is not a URL.
pub fn fetch_file(
    url: &str,
    dest: &Path,
    sha256: Option<&str>,
    observers: &[Arc<dyn TransferObserver>],
) -> BunsenResult<()> {
    let (Some(parent), Some(_)) = (dest.parent(), dest.file_name()) else {
        return Err(BunsenError::illegal(format!(
            "{}: not a file path",
            dest.display()
        )));
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
        Err(e @ ureq::Error::BadUri(_)) => {
            return Err(BunsenError::illegal(format!("{url}: not a URL")).with_cause(e));
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
    fs::create_dir_all(parent).map_err(sys_at("create directory", parent))?;
    let partial = partial_path(dest);
    let progress = TransferProgressStack::begin(
        observers,
        &TransferDesc {
            source: url,
            dest,
            total,
        },
    );

    let mut reader = HashingReader::new(ProgressReader::new(
        response.body_mut().as_reader(),
        &progress,
    ));
    let received = match copy_body(&mut reader, &partial, url) {
        Ok(received) => received,
        Err(e) => {
            let _ = fs::remove_file(&partial);
            return Err(fail(&progress, e));
        }
    };
    if let Some(expected) = total
        && expected != received
    {
        let _ = fs::remove_file(&partial);
        return Err(fail(
            &progress,
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
        let _ = fs::remove_file(&partial);
        let mismatch = DigestMismatch::new(
            dest.display().to_string(),
            DigestOrigin::Download {
                url: url.to_string(),
            },
            expected,
            found,
        );
        return Err(fail(&progress, FetchFailure::Digest(mismatch).into()));
    }

    if let Err(e) = fs::rename(&partial, dest) {
        let _ = fs::remove_file(&partial);
        return Err(fail(&progress, sys_at("rename", &partial)(e)));
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
) -> BunsenResult<()> {
    fetch_file(url, dest, Some(sha256), observers)
}

/// Copies the response body from `reader` into a new file at `partial`, and
/// syncs it; the byte count.
///
/// A read is the network side, a [`FetchFailure::Transport`] for `url`; a
/// create, write or sync is the local disk, sorted by [`sys_at`] on
/// `partial`.
fn copy_body(
    reader: &mut impl Read,
    partial: &Path,
    url: &str,
) -> BunsenResult<u64> {
    let mut file = fs::File::create(partial).map_err(sys_at("create", partial))?;
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
            .map_err(sys_at("write", partial))?;
        received += n as u64;
    }
    file.sync_all().map_err(sys_at("sync", partial))?;
    Ok(received)
}

/// Finishes `progress` as failed with `error`'s message, and returns the
/// error.
fn fail(
    progress: &TransferProgressStack,
    error: BunsenError,
) -> BunsenError {
    progress.finish(TransferOutcome::Failed(&error.to_string()));
    error
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
    use std::path::Path;

    use super::*;
    use crate::{
        data::cache::testing::{
            ABC_SHA256,
            CacheProgressEvent,
            RecordingObserver,
            refused_url,
            serve_once,
            serve_once_declaring,
            serve_status,
        },
        errors::testing::{
            ErrorMatcher,
            predicate,
        },
    };

    fn observers(observer: &Arc<RecordingObserver>) -> Vec<Arc<dyn TransferObserver>> {
        vec![observer.clone()]
    }

    /// Each status picks its kind; a lookup status keeps the failure under
    /// its `LookupError`, where `find` still reaches it.
    #[test]
    fn test_fetch_failure_kinds() {
        use BunsenErrorKind::*;
        let status = |status: u16| {
            BunsenError::from(FetchFailure::Status {
                url: "https://h.example/f".to_string(),
                status,
                retry_after: None,
            })
        };
        for (code, kind) in [
            (401, Lookup),
            (403, Lookup),
            (404, Lookup),
            (410, Lookup),
            (408, Unavailable),
            (429, Unavailable),
            (500, Unavailable),
            (503, Unavailable),
            (400, InvalidResource),
            (418, InvalidResource),
        ] {
            assert_eq!(status(code).kind(), kind, "{code}");
        }
        for (code, problem) in [
            (401, LookupProblem::Unauthorized),
            (403, LookupProblem::Denied),
            (410, LookupProblem::Missing),
        ] {
            ErrorMatcher::kind(Lookup)
                .cause(predicate("the problem", move |l: &LookupError| {
                    l.problem == problem
                }))
                .has_cause::<FetchFailure>()
                .assert(&status(code));
        }

        let short = BunsenError::from(FetchFailure::ShortTransfer {
            url: "u".to_string(),
            received: 3,
            expected: 10,
        });
        assert_eq!(short.kind(), Unavailable);
        assert_eq!(short.to_string(), "u: short transfer, 3 of 10 bytes");

        let digest = BunsenError::from(FetchFailure::Digest(DigestMismatch::new(
            "f",
            DigestOrigin::Download {
                url: "u".to_string(),
            },
            "aa",
            "bb",
        )));
        assert_eq!(digest.kind(), InvalidResource);
        assert_eq!(digest.find::<FetchFailure>().unwrap().url(), "u");
    }

    /// The bytes land at `dest`, the `.partial` is gone, and the observer saw
    /// the whole transfer: the URL, the destination and the length in
    /// `begin`, the count as it landed, `Complete` at the end.
    #[test]
    fn test_fetch_verified_streams_hashes_and_reports() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("weights").join("abc.bin");
        let url = serve_once("abc.bin", b"abc");
        let observer = Arc::new(RecordingObserver::default());

        fetch_verified(&url, &dest, ABC_SHA256, &observers(&observer)).unwrap();

        assert_eq!(fs::read(&dest).unwrap(), b"abc");
        assert!(!partial_path(&dest).exists());
        let events = observer.events();
        assert_eq!(
            events.first(),
            Some(&CacheProgressEvent::Begin {
                source: url.clone(),
                dest: dest.clone(),
                total: Some(3),
            })
        );
        assert!(events.contains(&CacheProgressEvent::Position(3)));
        assert_eq!(events.last(), Some(&CacheProgressEvent::Finish(Ok(()))));
    }

    /// A digest mismatch is `InvalidResource` with a download-origin digest
    /// cause, leaves nothing at `dest` or beside it, and the observer sees the
    /// failure with the reason.
    #[test]
    fn test_fetch_verified_rejects_a_mismatch_and_leaves_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("abc.bin");
        let url = serve_once("abc.bin", b"abd");
        let observer = Arc::new(RecordingObserver::default());

        let result = fetch_verified(&url, &dest, ABC_SHA256, &observers(&observer));

        let origin = DigestOrigin::Download { url: url.clone() };
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .cause(predicate(
                "a digest failure from the URL",
                move |f: &FetchFailure| matches!(f, FetchFailure::Digest(m) if m.origin == origin),
            ))
            .assert_err(&result);
        assert!(!dest.exists());
        assert!(!partial_path(&dest).exists());
        match observer.events().last() {
            Some(CacheProgressEvent::Finish(Err(message))) => {
                assert!(message.contains("sha256 is"), "{message}");
                assert!(message.contains(ABC_SHA256), "{message}");
            }
            other => panic!("expected a failed finish, got {other:?}"),
        }
    }

    /// No response at all is `Unavailable`, a transport failure, and no
    /// transfer is reported: there was nothing to begin.
    #[test]
    fn test_fetch_verified_without_a_response_reports_no_transfer() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("gone.bin");
        let url = refused_url("gone.bin");
        let observer = Arc::new(RecordingObserver::default());

        let result = fetch_verified(&url, &dest, ABC_SHA256, &observers(&observer));

        ErrorMatcher::kind(BunsenErrorKind::Unavailable)
            .cause(predicate("a transport failure", |f: &FetchFailure| {
                matches!(f, FetchFailure::Transport { .. })
            }))
            .assert_err(&result);
        assert!(observer.events().is_empty());
        assert!(!dest.exists());
    }

    #[test]
    fn test_fetch_verified_rejects_a_non_file_dest() {
        let observer = Arc::new(RecordingObserver::default());
        let result = fetch_verified(
            "http://127.0.0.1:1/never",
            Path::new("/"),
            ABC_SHA256,
            &observers(&observer),
        );
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("not a file path")
            .assert_err(&result);
        assert!(observer.events().is_empty());
    }

    /// An unpinned file lands on the strength of its length alone, and the
    /// observer sees the same transfer a pinned one gets.
    #[test]
    fn test_fetch_file_unpinned_lands_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("abc.bin");
        let url = serve_once("abc.bin", b"abc");
        let observer = Arc::new(RecordingObserver::default());

        fetch_file(&url, &dest, None, &observers(&observer)).unwrap();

        assert_eq!(fs::read(&dest).unwrap(), b"abc");
        assert!(!partial_path(&dest).exists());
        assert_eq!(
            observer.events().last(),
            Some(&CacheProgressEvent::Finish(Ok(())))
        );
    }

    /// A 404 is a `Lookup` of the URL, with the status underneath, and the
    /// slot the file would have landed in is not created for it.
    #[test]
    fn test_fetch_file_names_the_url_on_an_error_status() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("slot").join("missing.bin");
        let url = serve_status("missing.bin", 404);

        let result = fetch_file(&url, &dest, None, &[]);

        let key = url.clone();
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_eq(&format!("{url}: not found"))
            .cause(predicate("a missing URL", move |l: &LookupError| {
                l.problem == LookupProblem::Missing && l.key == key
            }))
            .cause(predicate("status 404", |f: &FetchFailure| {
                matches!(f, FetchFailure::Status { status: 404, .. })
            }))
            .assert_err(&result);
        assert!(!dest.parent().unwrap().exists(), "no empty slot is left");
    }

    /// A transfer cut off short of its `Content-Length` is `Unavailable`,
    /// leaves nothing behind, and the observer sees it fail.
    #[test]
    fn test_fetch_file_rejects_a_short_transfer() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("abc.bin");
        let url = serve_once_declaring("abc.bin", b"abc", 10);
        let observer = Arc::new(RecordingObserver::default());

        let result = fetch_file(&url, &dest, None, &observers(&observer));

        ErrorMatcher::kind(BunsenErrorKind::Unavailable)
            .has_cause::<FetchFailure>()
            .assert_err(&result);
        assert!(!dest.exists());
        assert!(!partial_path(&dest).exists());
        assert!(matches!(
            observer.events().last(),
            Some(CacheProgressEvent::Finish(Err(_)))
        ));
    }
}
