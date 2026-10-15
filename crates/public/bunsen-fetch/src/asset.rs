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
    replace_with_link,
    sha256_of,
};

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

        if dest.is_file() {
            match sha256_of(dest) {
                Ok(found) if found == self.sha256 => return dest.to_path_buf(),
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
        dest.to_path_buf()
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
