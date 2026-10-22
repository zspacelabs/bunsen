//! Build-script assets.
use std::{
    env,
    fs,
    path::{
        Path,
        PathBuf,
    },
    thread,
    time::Duration,
};

use crate::{
    FetchError,
    FetchFailure,
    fetch_verified,
    place_file,
    replace_with_link,
    sha256_of,
};

/// The environment variable naming a directory of assets kept across builds,
/// each under its SHA-256: [`Asset::resolve`] takes a pinned file from there
/// before fetching it, and leaves one there after.
///
/// Unset, nothing outside `OUT_DIR` is read or written. For CI, which keeps
/// the directory in its own cache: a path in it never changes with the
/// toolchain or the lockfile, as an `OUT_DIR` path does, and a file in it is
/// verified against its pin before it is used, as every asset is.
pub const BUNSEN_ASSET_CACHE: &str = "BUNSEN_ASSET_CACHE";

/// How many times [`Asset::resolve`] tries a download that fails on the way.
const ATTEMPTS: u32 = 3;

/// Whether a fetch failure may pass on a retry: a dropped or refused
/// connection, a body cut short, or a server asking for time (`408`, `429`,
/// `5xx`). A wrong digest, a missing file or a local-disk failure will not.
fn is_transient(e: &FetchError) -> bool {
    match e {
        FetchError::Failure(
            FetchFailure::Transport { .. } | FetchFailure::ShortTransfer { .. },
        ) => true,
        FetchError::Failure(FetchFailure::Status { status, .. }) => {
            matches!(status, 408 | 429 | 500..=599)
        }
        _ => false,
    }
}

/// One pinned file a build script needs.
///
/// A build that silently downloads is a bad neighbour, so every asset goes
/// through the same rules:
///
/// * A fetch only happens under the feature that needs the asset; the caller
///   gates the call.
/// * Assets land in `OUT_DIR`, the one directory a build script may write to.
/// * Every asset is pinned to a SHA-256 and re-verified on each build. A cached
///   copy that fails is deleted and re-fetched once.
/// * A download that fails on the way (a dropped connection, a body cut short,
///   a server asking for time) is tried again, up to three times in all, with a
///   cargo warning each time.
/// * An environment variable may name a local copy instead. A copy that is the
///   pinned file is laid out where the download would have gone (a symlink on
///   Unix, a copy elsewhere); any other file is used where it is, with a cargo
///   warning.
/// * [`BUNSEN_ASSET_CACHE`] may name a directory of assets kept by digest: a
///   pinned file there is placed in `OUT_DIR` instead of fetched, and a file
///   fetched, or found in `OUT_DIR`, is kept there.
///
/// ```no_run
/// use bunsen_fetch::{
///     Asset,
///     out_dir,
/// };
///
/// let model = Asset {
///     override_var: "MY_MODEL",
///     url: "https://example.com/model.bin",
///     sha256: "0000000000000000000000000000000000000000000000000000000000000000",
/// };
/// let path = model.resolve(&out_dir().join("model.bin"));
/// println!("cargo:rustc-env=MY_MODEL_PATH={}", path.display());
/// ```
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
    /// One that is the pinned file is laid out at `dest` too; one that is not
    /// stays where it is.
    ///
    /// # Panics
    /// When the override names something that is not a file, the download
    /// fails, or the downloaded file does not have the pinned digest: a build
    /// script has no better way to fail.
    pub fn resolve(
        &self,
        dest: &Path,
    ) -> PathBuf {
        println!("cargo:rerun-if-env-changed={BUNSEN_ASSET_CACHE}");
        let shared = env::var_os(BUNSEN_ASSET_CACHE).map(PathBuf::from);
        self.resolve_with(dest, shared.as_deref())
    }

    /// [`resolve`](Self::resolve), with the asset cache, if any, given.
    fn resolve_with(
        &self,
        dest: &Path,
        shared: Option<&Path>,
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
            return match sha256_of(&path) {
                Ok(found) if found == self.sha256 => {
                    replace_with_link(&path, dest).unwrap_or_else(|e| panic!("{e}"));
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

        let kept = shared.map(|dir| dir.join(self.sha256));
        if dest.is_file() {
            match sha256_of(dest) {
                Ok(found) if found == self.sha256 => {
                    self.keep(dest, kept.as_deref());
                    return dest.to_path_buf();
                }
                Ok(found) => println!(
                    "cargo:warning=cached {} has digest {found}, refetching",
                    dest.display()
                ),
                Err(e) => println!(
                    "cargo:warning=cannot read cached {}: {e}, refetching",
                    dest.display()
                ),
            }
            let _ = fs::remove_file(dest);
        }
        if let Some(kept) = &kept
            && kept.is_file()
        {
            match sha256_of(kept) {
                Ok(found) if found == self.sha256 => {
                    // A real file, not a link: `onnx-ir` refuses a graph's
                    // external data that resolves outside its directory.
                    place_file(kept, dest).unwrap_or_else(|e| panic!("{e}"));
                    return dest.to_path_buf();
                }
                _ => {
                    println!(
                        "cargo:warning={BUNSEN_ASSET_CACHE}: {} is not the pinned file, refetching",
                        kept.display()
                    );
                    let _ = fs::remove_file(kept);
                }
            }
        }
        println!("cargo:warning=fetching {}", self.url);
        for attempt in 1..=ATTEMPTS {
            match fetch_verified(self.url, dest, self.sha256, &[]) {
                Ok(()) => break,
                Err(e) if attempt < ATTEMPTS && is_transient(&e) => {
                    println!(
                        "cargo:warning=fetching {} (attempt {attempt} of {ATTEMPTS}): {e}; retrying",
                        self.url
                    );
                    thread::sleep(Duration::from_secs(2 << attempt));
                }
                Err(e) => panic!("fetching {}: {e}", self.url),
            }
        }
        self.keep(dest, kept.as_deref());
        dest.to_path_buf()
    }

    /// Keeps the verified file at `dest` in the asset cache as `kept`, when
    /// there is one and it does not hold it yet. A failure costs only a fetch
    /// next time, so it warns rather than failing the build.
    fn keep(
        &self,
        dest: &Path,
        kept: Option<&Path>,
    ) {
        let Some(kept) = kept else {
            return;
        };
        if kept.is_file() {
            return;
        }
        if let Err(e) = place_file(dest, kept) {
            println!(
                "cargo:warning={BUNSEN_ASSET_CACHE}: cannot keep {}: {e}",
                dest.display()
            );
        }
    }
}

/// `OUT_DIR`.
///
/// # Panics
/// Outside a build script, where cargo does not set it.
pub fn out_dir() -> PathBuf {
    PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR for build scripts"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// A fetched asset is kept in the asset cache under its digest, and a
    /// later build with an empty `OUT_DIR` takes it from there, with no
    /// server to fetch from; a cached file that is not the pin is refetched.
    #[test]
    fn test_the_asset_cache_keeps_and_serves() {
        use crate::testing::{
            ABC_SHA256,
            serve_n,
        };

        let dir = tempfile::tempdir().unwrap();
        let shared = dir.path().join("assets");
        let url = serve_n("abc", b"abc", 2);
        let asset = Asset {
            override_var: "BUNSEN_FETCH_TEST_UNSET",
            url: &url,
            sha256: ABC_SHA256,
        };

        let first = dir.path().join("out1/abc");
        assert_eq!(asset.resolve_with(&first, Some(&shared)), first);
        let kept = shared.join(ABC_SHA256);
        assert_eq!(fs::read(&kept).unwrap(), b"abc");

        let unreachable = Asset {
            url: "http://127.0.0.1:9/abc",
            ..asset
        };
        let second = dir.path().join("out2/abc");
        assert_eq!(unreachable.resolve_with(&second, Some(&shared)), second);
        assert_eq!(fs::read(&second).unwrap(), b"abc");
        assert!(!second.is_symlink(), "a real file, for onnx-ir");

        fs::remove_file(&kept).unwrap();
        fs::write(&kept, b"xyz").unwrap();
        let third = dir.path().join("out3/abc");
        assert_eq!(asset.resolve_with(&third, Some(&shared)), third);
        assert_eq!(fs::read(&third).unwrap(), b"abc");
        assert_eq!(fs::read(&kept).unwrap(), b"abc", "kept again, verified");
    }

    /// A file already in `OUT_DIR` is kept too, so a warm local build fills an
    /// asset cache it is first pointed at.
    #[test]
    fn test_a_verified_out_dir_file_is_kept() {
        use crate::testing::ABC_SHA256;

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out/abc");
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(&dest, b"abc").unwrap();
        let shared = dir.path().join("assets");
        let asset = Asset {
            override_var: "BUNSEN_FETCH_TEST_UNSET",
            url: "http://127.0.0.1:9/abc",
            sha256: ABC_SHA256,
        };
        assert_eq!(asset.resolve_with(&dest, Some(&shared)), dest);
        assert_eq!(fs::read(shared.join(ABC_SHA256)).unwrap(), b"abc");
    }

    /// A transport failure, a short body and a busy server are retried; a
    /// wrong digest and a missing file are not.
    #[test]
    fn test_is_transient() {
        let url = "https://h.example/f".to_string();
        let status = |status| {
            FetchError::Failure(FetchFailure::Status {
                url: url.clone(),
                status,
                retry_after: None,
            })
        };
        assert!(is_transient(&status(503)));
        assert!(is_transient(&status(429)));
        assert!(!is_transient(&status(404)));
        assert!(is_transient(&FetchError::Failure(
            FetchFailure::ShortTransfer {
                url: url.clone(),
                received: 1,
                expected: 2,
            }
        )));
        assert!(is_transient(&FetchError::Failure(
            FetchFailure::Transport {
                url: url.clone(),
                source: Arc::new(ureq::Error::ConnectionFailed),
            }
        )));
        assert!(!is_transient(&FetchError::Failure(FetchFailure::Digest {
            url,
            dest: PathBuf::from("f"),
            expected: "a".into(),
            found: "b".into(),
        })));
    }
}
