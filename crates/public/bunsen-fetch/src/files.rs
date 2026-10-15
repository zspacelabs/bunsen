//! Files: naming and placing fetched files.
use std::{
    fmt,
    fs,
    io,
    path::{
        Path,
        PathBuf,
    },
};

/// A local-disk failure: the operation, the path it was on, and the
/// `io::Error`.
///
/// The disk side of a transfer, kept apart from the network side so a caller
/// can sort it by its `io::Error` and name the path.
#[derive(Debug)]
pub struct IoAt {
    /// What was being done: `"create directory"`, `"write"`, `"rename"`.
    pub op: &'static str,

    /// The path it was done to.
    pub path: PathBuf,

    /// The error.
    pub source: io::Error,
}

impl IoAt {
    /// A closure for `map_err` that attaches `op` and `path`.
    pub fn at<'a>(
        op: &'static str,
        path: &'a Path,
    ) -> impl FnOnce(io::Error) -> Self + 'a {
        move |source| Self {
            op,
            path: path.to_path_buf(),
            source,
        }
    }
}

impl fmt::Display for IoAt {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{} {}: {}", self.op, self.path.display(), self.source)
    }
}

impl std::error::Error for IoAt {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
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
/// `dest`'s directory is made on the way.
///
/// # Errors
/// When `dest`'s directory cannot be made (`"create directory"`), or none of
/// the three could be made (`"copy"`, on `src`). A `dest` without a parent
/// directory is an `InvalidInput` error on `dest`.
pub fn link_or_copy(
    src: &Path,
    dest: &Path,
) -> Result<(), IoAt> {
    let Some(parent) = dest.parent() else {
        return Err(IoAt::at("place", dest)(io::Error::new(
            io::ErrorKind::InvalidInput,
            "not a file path",
        )));
    };
    fs::create_dir_all(parent).map_err(IoAt::at("create directory", parent))?;

    #[cfg(unix)]
    if std::os::unix::fs::symlink(src, dest).is_ok() {
        return Ok(());
    }
    if fs::hard_link(src, dest).is_ok() {
        return Ok(());
    }
    fs::copy(src, dest)
        .map(|_| ())
        .map_err(IoAt::at("copy", src))
}

/// Makes `dest` a real file with `src`'s contents: a hard link where the two
/// share a filesystem, else a copy. A no-op when `dest` already is `src`.
///
/// For a reader that refuses symlinks: `onnx-ir` canonicalizes an ONNX
/// graph's external-data path and refuses one that resolves outside the
/// graph's directory.
///
/// # Errors
/// When `src` cannot be resolved, `dest`'s directory made, or the file
/// placed.
pub fn place_file(
    src: &Path,
    dest: &Path,
) -> Result<(), IoAt> {
    if src == dest && !dest.is_symlink() {
        return Ok(());
    }
    let src = fs::canonicalize(src).map_err(IoAt::at("resolve", src))?;
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(IoAt::at("create directory", parent))?;
    }
    let tmp = dest.with_extension("placing");
    let _ = fs::remove_file(&tmp);
    if fs::hard_link(&src, &tmp).is_err() {
        fs::copy(&src, &tmp).map_err(IoAt::at("copy", &src))?;
    }
    fs::rename(&tmp, dest).map_err(IoAt::at("rename", &tmp))
}

/// Replaces whatever is at `dest` with `src`: a symlink where the platform
/// has them, else a copy.
///
/// # Errors
/// When `dest`'s directory cannot be made, what is there removed, or the
/// file placed.
pub fn replace_with_link(
    src: &Path,
    dest: &Path,
) -> Result<(), IoAt> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(IoAt::at("create directory", parent))?;
    }
    if dest.symlink_metadata().is_ok() {
        fs::remove_file(dest).map_err(IoAt::at("remove", dest))?;
    }
    #[cfg(unix)]
    if std::os::unix::fs::symlink(src, dest).is_ok() {
        return Ok(());
    }
    fs::copy(src, dest)
        .map(|_| ())
        .map_err(IoAt::at("copy", src))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Each placement makes `dest` hold `src`'s bytes, in a directory made
    /// on the way; `place_file`'s is a real file, and the others on unix a
    /// symlink.
    #[test]
    fn test_placements() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src.bin");
        fs::write(&src, b"bytes").unwrap();

        let linked = dir.path().join("a").join("linked.bin");
        link_or_copy(&src, &linked).unwrap();
        assert_eq!(fs::read(&linked).unwrap(), b"bytes");
        #[cfg(unix)]
        assert!(linked.is_symlink());

        let real = dir.path().join("b").join("real.bin");
        place_file(&src, &real).unwrap();
        assert_eq!(fs::read(&real).unwrap(), b"bytes");
        assert!(!real.is_symlink());

        replace_with_link(&src, &real).unwrap();
        assert_eq!(fs::read(&real).unwrap(), b"bytes");
        #[cfg(unix)]
        assert!(real.is_symlink());

        let err = link_or_copy(&src, Path::new("/")).unwrap_err();
        assert_eq!(err.source.kind(), io::ErrorKind::InvalidInput);
    }
}
