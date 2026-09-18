//! # Fetch
//!
//! Streams a URL into the cache, hashed as it lands. The bytes go to
//! `<dest>.partial` and are renamed into place only once they check out: the
//! digest matches when the caller pinned one, and the byte count matches the
//! `Content-Length` when the server sent one. An interrupted or corrupt
//! transfer never leaves a trusted-looking file at `dest`.
//!
//! Every transfer is reported to the [`TransferObserver`] stack it is given:
//! `begin` once the response headers are in (that is when the length is
//! known), `position` as bytes land, and `finish` with the outcome. A
//! request that never gets a response is an error with no transfer
//! reported.

use std::{
    fs,
    io::{
        self,
        Read,
    },
    path::{
        Path,
        PathBuf,
    },
    sync::Arc,
};

use super::{
    HashingReader,
    TransferDesc,
    TransferObserver,
    TransferOutcome,
    TransferProgress,
    TransferProgressStack,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// Streams `url` to `dest`, checked against `sha256` when one is given,
/// reporting to `observers`.
///
/// A pinned file (`Some`) must match its digest; every file must match the
/// `Content-Length` when the server sent one. Nothing is left at `dest` or
/// beside it when either check fails.
///
/// # Errors
/// [`BunsenError::Invalid`] if the digest does not match (nothing is left at
/// `dest`), or if `dest` is not a file path; [`BunsenError::External`] for a
/// transport or file-system failure, an HTTP error status or a short
/// transfer included.
pub fn fetch_file(
    url: &str,
    dest: &Path,
    sha256: Option<&str>,
    observers: &[Arc<dyn TransferObserver>],
) -> BunsenResult<()> {
    let (Some(parent), Some(_)) = (dest.parent(), dest.file_name()) else {
        return Err(BunsenError::Invalid(format!(
            "{}: not a file path",
            dest.display()
        )));
    };
    fs::create_dir_all(parent).map_err(BunsenError::external)?;
    let partial = partial_path(dest);

    let mut response = ureq::get(url).call().map_err(BunsenError::external)?;
    let total = response.body().content_length();
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
    let copied = (|| -> io::Result<u64> {
        let mut file = fs::File::create(&partial)?;
        let received = io::copy(&mut reader, &mut file)?;
        file.sync_all()?;
        Ok(received)
    })();
    let received = match copied {
        Ok(received) => received,
        Err(e) => {
            let _ = fs::remove_file(&partial);
            return Err(fail(&progress, format!("{url}: {e}")));
        }
    };
    if total.is_some_and(|expected| expected != received) {
        let _ = fs::remove_file(&partial);
        let expected = total.unwrap_or_default();
        return Err(fail(
            &progress,
            format!("{url}: short transfer, {received} of {expected} bytes"),
        ));
    }

    let found = reader.hex_digest();
    if sha256.is_some_and(|expected| expected != found) {
        let _ = fs::remove_file(&partial);
        let expected = sha256.unwrap_or_default();
        let message = format!(
            "{url}: sha256 is {found}, expected {expected}; the transfer was corrupt or the asset changed"
        );
        progress.finish(TransferOutcome::Failed(&message));
        return Err(BunsenError::Invalid(message));
    }

    if let Err(e) = fs::rename(&partial, dest) {
        let _ = fs::remove_file(&partial);
        return Err(fail(&progress, format!("{}: {e}", dest.display())));
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

/// The file a URL names: its last path segment, before any query or
/// fragment. `None` when the URL has no path or its path ends in `/`.
pub fn file_name_from_url(url: &str) -> Option<&str> {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let path = path.split_once("://").map_or(path, |(_, rest)| rest);
    let (_, name) = path.rsplit_once('/')?;
    (!name.is_empty()).then_some(name)
}

/// Finishes `progress` as failed with `message`, and makes the error.
fn fail(
    progress: &TransferProgressStack,
    message: String,
) -> BunsenError {
    progress.finish(TransferOutcome::Failed(&message));
    BunsenError::External(message)
}

/// `<dest>.partial`, keeping every dot of the name (`tiny.en.pt.partial`).
///
/// # Panics
/// If `dest` has no file name.
pub fn partial_path(dest: &Path) -> PathBuf {
    let mut name = dest
        .file_name()
        .expect("a cache path has a file name")
        .to_os_string();
    name.push(".partial");
    dest.with_file_name(name)
}

/// Puts a verified file at `dest` without copying 3 GB where a link will
/// do: a symlink where the platform has them, then a hard link, then a copy.
///
/// # Errors
/// [`BunsenError::Invalid`] if `dest` has no parent directory;
/// [`BunsenError::External`] if none of the three could be made.
pub fn link_or_copy(
    src: &Path,
    dest: &Path,
) -> BunsenResult<()> {
    let Some(parent) = dest.parent() else {
        return Err(BunsenError::Invalid(format!(
            "{}: not a file path",
            dest.display()
        )));
    };
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
    use std::{
        fs,
        path::{
            Path,
            PathBuf,
        },
        sync::Arc,
    };

    use super::*;
    use crate::{
        data::cache::{
            TransferObserver,
            transfer::testing::{
                ABC_SHA256,
                CacheProgressEvent,
                RecordingObserver,
                refused_url,
                serve_once,
                serve_once_declaring,
            },
        },
        errors::BunsenError,
    };

    fn observers(observer: &Arc<RecordingObserver>) -> Vec<Arc<dyn TransferObserver>> {
        vec![observer.clone()]
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

    /// A digest mismatch is `Invalid`, leaves nothing at `dest` or beside
    /// it, and the observer sees the failure with the reason.
    #[test]
    fn test_fetch_verified_rejects_a_mismatch_and_leaves_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("abc.bin");
        let url = serve_once("abc.bin", b"abd");
        let observer = Arc::new(RecordingObserver::default());

        let result = fetch_verified(&url, &dest, ABC_SHA256, &observers(&observer));

        assert!(matches!(result, Err(BunsenError::Invalid(_))));
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

    /// No response at all is `External`, and no transfer is reported: there
    /// was nothing to begin.
    #[test]
    fn test_fetch_verified_without_a_response_reports_no_transfer() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("gone.bin");
        let url = refused_url("gone.bin");
        let observer = Arc::new(RecordingObserver::default());

        let result = fetch_verified(&url, &dest, ABC_SHA256, &observers(&observer));

        assert!(
            matches!(result, Err(BunsenError::External(_))),
            "{result:?}"
        );
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
        assert!(matches!(result, Err(BunsenError::Invalid(_))), "{result:?}");
        assert!(observer.events().is_empty());
    }

    #[test]
    fn test_partial_path_keeps_every_dot() {
        assert_eq!(
            partial_path(Path::new("/c/tiny.en.pt")),
            PathBuf::from("/c/tiny.en.pt.partial")
        );
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

    /// A transfer cut off short of its `Content-Length` is `External`,
    /// leaves nothing behind, pinned or not, and the observer sees it fail.
    #[test]
    fn test_fetch_file_rejects_a_short_transfer() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("abc.bin");
        let url = serve_once_declaring("abc.bin", b"abc", 10);
        let observer = Arc::new(RecordingObserver::default());

        let result = fetch_file(&url, &dest, None, &observers(&observer));

        assert!(
            matches!(result, Err(BunsenError::External(_))),
            "{result:?}"
        );
        assert!(!dest.exists());
        assert!(!partial_path(&dest).exists());
        assert!(matches!(
            observer.events().last(),
            Some(CacheProgressEvent::Finish(Err(_)))
        ));
    }

    #[test]
    fn test_file_name_from_url() {
        assert_eq!(
            file_name_from_url("https://h.example/a/b/tiny.en.pt?x=1#frag"),
            Some("tiny.en.pt")
        );
        assert_eq!(file_name_from_url("https://h.example/f"), Some("f"));
        assert_eq!(file_name_from_url("https://h.example/dir/"), None);
        assert_eq!(file_name_from_url("https://h.example/"), None);
        assert_eq!(file_name_from_url("https://h.example"), None);
        assert_eq!(file_name_from_url("relative/path/f.bin"), Some("f.bin"));
    }

    /// The file is reachable at `dest`, in a directory made on the way; on
    /// unix it is a symlink to `src`.
    #[test]
    fn test_link_or_copy_places_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src.bin");
        fs::write(&src, b"bytes").unwrap();
        let dest = dir.path().join("sub").join("dest.bin");

        link_or_copy(&src, &dest).unwrap();

        assert_eq!(fs::read(&dest).unwrap(), b"bytes");
        #[cfg(unix)]
        assert!(
            fs::symlink_metadata(&dest)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(matches!(
            link_or_copy(&src, Path::new("/")),
            Err(BunsenError::Invalid(_))
        ));
    }
}
