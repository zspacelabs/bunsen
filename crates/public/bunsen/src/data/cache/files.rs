//! Files: naming and placing files in the cache.

use std::path::Path;

pub use bunsen_fetch::{
    file_name_from_url,
    partial_path,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
    ResultContext,
    sys_at,
};

/// Puts a verified file at `dest` without copying 3 GB where a link will
/// do: a symlink where the platform has them, then a hard link, then a copy.
///
/// # Errors
/// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if `dest` has no
/// parent directory. If `dest`'s directory cannot be made, or none of the
/// three could be made, the `io::Error` sorted with its path (see
/// [`sys_at`]): [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a
/// missing `src`, usually [`Sys`](crate::errors::BunsenErrorKind::Sys)
/// otherwise.
pub fn link_or_copy(
    src: &Path,
    dest: &Path,
) -> BunsenResult<()> {
    if dest.parent().is_none() {
        return Err(BunsenError::illegal(format!(
            "{}: not a file path",
            dest.display()
        )));
    }
    bunsen_fetch::link_or_copy(src, dest)
        .map_err(io_at)
        .with_context(|| format!("placing {}", dest.display()))
}

/// A local-disk failure from `bunsen_fetch`, sorted by its `io::Error` and
/// naming its path (see [`sys_at`]).
pub(crate) fn io_at(e: bunsen_fetch::IoAt) -> BunsenError {
    sys_at(e.op, &e.path)(e.source)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
    };

    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    #[test]
    fn test_partial_path_keeps_every_dot() {
        assert_eq!(
            partial_path(Path::new("/c/tiny.en.pt")),
            PathBuf::from("/c/tiny.en.pt.partial")
        );
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
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("not a file path")
            .assert_err(&link_or_copy(&src, Path::new("/")));
    }
}
