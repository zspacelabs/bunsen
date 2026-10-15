//! Build-script support for bunsen's bundled-asset crates.
//!
//! A bundled-asset crate fetches large files at build time rather than
//! committing them. A build that silently downloads is a bad neighbour, so
//! every asset goes through the same rules:
//!
//! * A fetch only happens under the feature that needs the asset; the caller
//!   gates the call.
//! * Assets land in `OUT_DIR`, the one directory a build script may write to.
//! * Every asset is pinned to a SHA-256 and re-verified on each build. A cached
//!   copy that fails is deleted and re-fetched once.
//! * An environment variable may name a local copy instead. A copy that is the
//!   pinned file is laid out where the download would have gone (a symlink on
//!   Unix, a copy elsewhere); any other file is used where it is, with a cargo
//!   warning.
//!
//! ```no_run
//! use bunsen_asset_fetch::Asset;
//!
//! let model = Asset {
//!     override_var: "MY_MODEL",
//!     url: "https://example.com/model.bin",
//!     sha256: "0000000000000000000000000000000000000000000000000000000000000000",
//! };
//! let path = model.resolve(&bunsen_asset_fetch::out_dir().join("model.bin"));
//! println!("cargo:rustc-env=MY_MODEL_PATH={}", path.display());
//! ```

use std::{
    env,
    fs,
    io,
    io::Read,
    path::{
        Path,
        PathBuf,
    },
};

use sha2::{
    Digest,
    Sha256,
};

/// One pinned file.
#[derive(Clone, Copy, Debug)]
pub struct Asset<'a> {
    /// The environment variable naming a local file to use instead. The
    /// build reruns when it changes.
    pub override_var: &'a str,

    /// Where the file is fetched from.
    pub url: &'a str,

    /// The file's lowercase hex SHA-256.
    pub sha256: &'a str,
}

impl Asset<'_> {
    /// A path to the asset: its override, or `dest`, fetched and verified if
    /// it is not already there.
    ///
    /// A caller-supplied file is deliberately **not** required to be the
    /// pinned file, since the point of the override is to try something else.
    /// One that is the pinned file is laid out at `dest` too (a symlink where
    /// the platform has them); one that is not stays where it is.
    ///
    /// # Panics
    /// When the override names something that is not a file, the download
    /// fails, or the downloaded file does not have the pinned digest: a build
    /// script has no better way to fail.
    pub fn resolve(
        &self,
        dest: &Path,
    ) -> PathBuf {
        let var = self.override_var;
        println!("cargo:rerun-if-env-changed={var}");
        if let Some(local) = env::var_os(var) {
            let path = PathBuf::from(local);
            assert!(
                path.is_file(),
                "{var} is set but not a file: {}",
                path.display()
            );
            return match digest_of(&path) {
                Ok(found) if found == self.sha256 => {
                    place_link(&path, dest);
                    dest.to_path_buf()
                }
                Ok(found) => {
                    println!(
                        "cargo:warning={var}: {} has digest {found}, not {}; used in place",
                        path.display(),
                        self.sha256,
                    );
                    path
                }
                Err(e) => {
                    println!(
                        "cargo:warning={var}: cannot read {}: {e}; used in place",
                        path.display()
                    );
                    path
                }
            };
        }
        fetch_verified(self.url, self.sha256, dest)
    }
}

/// `OUT_DIR`.
///
/// # Panics
/// Outside a build script, where cargo does not set it.
pub fn out_dir() -> PathBuf {
    PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR for build scripts"))
}

/// `pretrained/<kit>/<namespace>/<sha256>/<file>` under `root`: where
/// bunsen's `PretrainedCache` keeps a pinned resource, so a cache rooted at
/// `root` finds the file without knowing it was bundled.
pub fn pretrained_cache_path(
    root: &Path,
    kit: &str,
    namespace: &str,
    sha256: &str,
    file: &str,
) -> PathBuf {
    root.join("pretrained")
        .join(kit)
        .join(namespace)
        .join(sha256)
        .join(file)
}

/// Puts `src` at `dest`: a symlink where the platform has them, else a copy.
///
/// # Panics
/// When the directory cannot be created or the file not placed.
pub fn place_link(
    src: &Path,
    dest: &Path,
) {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).expect("create the asset directory");
    }
    if dest.exists() || dest.symlink_metadata().is_ok() {
        fs::remove_file(dest).expect("replace the laid-out asset");
    }
    #[cfg(unix)]
    if std::os::unix::fs::symlink(src, dest).is_ok() {
        return;
    }
    fs::copy(src, dest)
        .unwrap_or_else(|e| panic!("placing {} at {}: {e}", src.display(), dest.display()));
}

/// Makes `dest` a real file with `src`'s contents: a hard link where the two
/// share a filesystem, else a copy. A no-op when `dest` already is `src`.
///
/// For a reader that refuses symlinks: `onnx-ir` canonicalizes an ONNX
/// graph's external-data path and refuses one that resolves outside the
/// graph's directory.
///
/// # Panics
/// When `src` cannot be resolved or the file not placed.
pub fn place_file(
    src: &Path,
    dest: &Path,
) {
    if src == dest && !dest.is_symlink() {
        return;
    }
    let src = fs::canonicalize(src).unwrap_or_else(|e| panic!("resolving {}: {e}", src.display()));
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).expect("create the asset directory");
    }
    let tmp = dest.with_extension("placing");
    let _ = fs::remove_file(&tmp);
    if fs::hard_link(&src, &tmp).is_err() {
        fs::copy(&src, &tmp)
            .unwrap_or_else(|e| panic!("copying {} to {}: {e}", src.display(), tmp.display()));
    }
    fs::rename(&tmp, dest)
        .unwrap_or_else(|e| panic!("placing {} at {}: {e}", src.display(), dest.display()));
}

/// Ensures `dest` holds the file at `url` with digest `sha256`.
///
/// A cached file whose digest does not match is treated as corrupt: it is
/// removed and fetched once more, and a second mismatch is fatal.
///
/// # Panics
/// When the download fails or the downloaded file has another digest.
pub fn fetch_verified(
    url: &str,
    sha256: &str,
    dest: &Path,
) -> PathBuf {
    if dest.is_file() {
        match digest_of(dest) {
            Ok(found) if found == sha256 => return dest.to_path_buf(),
            Ok(found) => {
                println!(
                    "cargo:warning=cached {} has digest {found}, refetching",
                    dest.display()
                );
                let _ = fs::remove_file(dest);
            }
            Err(e) => {
                println!(
                    "cargo:warning=cannot read cached {}: {e}, refetching",
                    dest.display()
                );
                let _ = fs::remove_file(dest);
            }
        }
    }

    println!("cargo:warning=fetching {url}");
    download(url, dest).unwrap_or_else(|e| panic!("fetching {url}: {e}"));

    let found = digest_of(dest).expect("digest the freshly fetched asset");
    assert_eq!(
        found, sha256,
        "{url} has digest {found}, expected {sha256}. The upstream asset \
         changed, or the transfer was corrupt.",
    );
    dest.to_path_buf()
}

/// Streams `url` to `dest`, through a temporary file so an interrupted
/// transfer cannot leave a truncated asset behind.
fn download(
    url: &str,
    dest: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = dest.with_extension("partial");
    let mut response = ureq::get(url).call()?;
    let mut reader = response.body_mut().as_reader();
    let mut file = fs::File::create(&tmp)?;
    io::copy(&mut reader, &mut file)?;
    drop(file);
    fs::rename(&tmp, dest)?;
    Ok(())
}

/// The lowercase hex SHA-256 of a file, read in chunks so a multi-gigabyte
/// asset stays off the heap.
///
/// # Errors
/// When the file cannot be read.
pub fn digest_of(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digest_and_layout() {
        let dir = env::temp_dir().join(format!("bunsen-asset-fetch-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("a.txt");
        fs::write(&file, b"abc").unwrap();
        assert_eq!(
            digest_of(&file).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        let copy = dir.join("nested/b.txt");
        place_file(&file, &copy);
        assert_eq!(fs::read(&copy).unwrap(), b"abc");
        assert!(!copy.is_symlink());

        assert_eq!(
            pretrained_cache_path(Path::new("/r"), "k", "ns", "ff", "f.bin"),
            Path::new("/r/pretrained/k/ns/ff/f.bin")
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
