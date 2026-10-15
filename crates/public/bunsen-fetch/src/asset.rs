//! Build-script assets.
use std::{
    env,
    fs,
    path::{
        Path,
        PathBuf,
    },
};

use crate::{
    fetch_verified,
    replace_with_link,
    sha256_of,
};

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
        fetch_verified(self.url, dest, self.sha256, &[])
            .unwrap_or_else(|e| panic!("fetching {}: {e}", self.url));
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
