//! Files: naming and placing files in the cache.

use std::{
    fs,
    path::{
        Path,
        PathBuf,
    },
};

use crate::errors::{
    BunsenError,
    BunsenResult,
    ResultContext,
    sys_at,
};

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

/// The file a URL names: its last path segment, before any query or
/// fragment. `None` when the URL has no path or its path ends in `/`.
pub fn file_name_from_url(url: &str) -> Option<&str> {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let path = path.split_once("://").map_or(path, |(_, rest)| rest);
    let (_, name) = path.rsplit_once('/')?;
    (!name.is_empty()).then_some(name)
}

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
    let Some(parent) = dest.parent() else {
        return Err(BunsenError::illegal(format!(
            "{}: not a file path",
            dest.display()
        )));
    };
    fs::create_dir_all(parent).map_err(sys_at("create directory", parent))?;

    #[cfg(unix)]
    if std::os::unix::fs::symlink(src, dest).is_ok() {
        return Ok(());
    }
    if fs::hard_link(src, dest).is_ok() {
        return Ok(());
    }
    fs::copy(src, dest)
        .map(|_| ())
        .map_err(sys_at("copy", src))
        .with_context(|| format!("placing {}", dest.display()))
}

#[cfg(test)]
mod tests {
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
