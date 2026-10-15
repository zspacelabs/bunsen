//! Fetches the Parakeet TDT assets.
//!
//! One set, behind its feature: `onnx_gen` is the `istupakov` ONNX export of
//! NVIDIA's `parakeet-tdt-0.6b-v3` that `parakeet-model-validation` compares
//! bunsen against: its encoder graph and its prediction-and-joint graph. The
//! export's feature-extractor graph (`nemo128.onnx`) is not generated:
//! `burn-onnx` 0.22 cannot infer its types (`ArgType::Shape has no DType`).
//!
//! As in `bunsen-bundled-whisper`, a build that silently downloads is a bad
//! neighbour, so:
//!
//! * A fetch only happens under the feature that needs the asset.
//! * Assets land in `OUT_DIR`, the one directory a build script may write to.
//! * Every asset is pinned to a SHA-256 and re-verified on each build. A cached
//!   copy that fails is deleted and re-fetched once.
//! * `PARAKEET_ONNX_ENCODER`, `PARAKEET_ONNX_ENCODER_DATA` and
//!   `PARAKEET_ONNX_DECODER_JOINT` point the build at local files instead, for
//!   working offline or against a different export. An override that is the
//!   pinned file is laid out where the download would have gone; one that is
//!   not is used where it is.

// Each feature uses its own constants and helpers; the rest are dead in that
// build. Enumerating `cfg` on every item would be noisier than this.
#![allow(unused)]

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

/// The export's repository, pinned to one revision so every URL names one
/// file forever.
const ONNX_BASE_URL: &str = "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce";

/// One fetched asset: where it comes from, its pin, and its local name.
struct Asset {
    /// The environment variable naming a local file to use instead.
    override_var: &'static str,

    /// The file's name in the export's repository.
    remote: &'static str,

    /// SHA-256 of the file.
    sha256: &'static str,

    /// Local name under `OUT_DIR`. `ModelGen` derives the generated module's
    /// name from a graph's file name, so it is also the name `src/lib.rs`
    /// includes.
    local: &'static str,
}

/// The `FastConformer` encoder graph. Its weights are external, in
/// [`ENCODER_DATA`].
const ENCODER: Asset = Asset {
    override_var: "PARAKEET_ONNX_ENCODER",
    remote: "encoder-model.onnx",
    sha256: "98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1",
    local: "parakeet_encoder.onnx",
};

/// The encoder's external weights. The graph names this file by its remote
/// name, relative to the graph's directory, so it keeps that name locally.
const ENCODER_DATA: Asset = Asset {
    override_var: "PARAKEET_ONNX_ENCODER_DATA",
    remote: "encoder-model.onnx.data",
    sha256: "9a22d372c51455c34f13405da2520baefb7125bd16981397561423ed32d24f36",
    local: "encoder-model.onnx.data",
};

/// The TDT prediction network and joint, one step at a time.
const DECODER_JOINT: Asset = Asset {
    override_var: "PARAKEET_ONNX_DECODER_JOINT",
    remote: "decoder_joint-model.onnx",
    sha256: "e978ddf6688527182c10fde2eb4b83068421648985ef23f7a86be732be8706c1",
    local: "parakeet_decoder_joint.onnx",
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for asset in [&ENCODER, &ENCODER_DATA, &DECODER_JOINT] {
        println!("cargo:rerun-if-env-changed={}", asset.override_var);
    }

    // `cfg` rather than a `CARGO_FEATURE_*` lookup: `burn_onnx` is an
    // optional build-dependency, so without the feature it is not linked.
    #[cfg(feature = "onnx_gen")]
    generate_reference();
}

/// Fetches the ONNX export and generates Rust models from it.
#[cfg(feature = "onnx_gen")]
fn generate_reference() {
    // The generated weights are loaded from `OUT_DIR` at run time rather than
    // embedded: the encoder's alone are 2.4 GB.
    println!(
        "cargo:rustc-env=PARAKEET_ONNX_OUT_DIR={}",
        env::var("OUT_DIR").unwrap()
    );

    // The encoder graph reads its weights from beside itself, so they are
    // resolved first, into the same directory.
    // Both are real files there, never symlinks: `onnx-ir` canonicalizes the
    // data path and refuses one that resolves outside the graph's directory.
    let encoder_dir = cache_dir();
    let data = resolve_asset(&ENCODER_DATA);
    place_file(&data, &encoder_dir.join(ENCODER_DATA.local));

    for asset in [&ENCODER, &DECODER_JOINT] {
        let mut onnx = resolve_asset(asset);
        if asset.local == ENCODER.local {
            let dest = encoder_dir.join(ENCODER.local);
            place_file(&onnx, &dest);
            onnx = dest;
        }
        println!("cargo:rerun-if-changed={}", onnx.display());

        burn_onnx::ModelGen::new()
            .input(onnx.to_str().expect("asset path is UTF-8"))
            .out_dir("./")
            .run_from_script();
    }
}

/// Where fetched assets land: `OUT_DIR`.
fn cache_dir() -> PathBuf {
    PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR for build scripts"))
}

/// Returns a path to an asset, honouring its override or fetching it.
///
/// A caller-supplied file is deliberately **not** digest-pinned — the point of
/// the override is to try something else. One that happens to be the pinned
/// asset is laid out where the download would have gone; one that is not
/// stays where it is.
fn resolve_asset(asset: &Asset) -> PathBuf {
    let dest = cache_dir().join(asset.local);
    let override_var = asset.override_var;
    if let Some(local) = env::var_os(override_var) {
        let path = PathBuf::from(local);
        assert!(
            path.is_file(),
            "{override_var} is set but not a file: {}",
            path.display(),
        );
        return match digest_of(&path) {
            Ok(found) if found == asset.sha256 => {
                place(&path, &dest);
                dest
            }
            Ok(found) => {
                println!(
                    "cargo:warning={override_var}: {} has digest {found}, not {}; used in place",
                    path.display(),
                    asset.sha256,
                );
                path
            }
            Err(e) => {
                println!(
                    "cargo:warning={override_var}: cannot read {}: {e}; used in place",
                    path.display()
                );
                path
            }
        };
    }

    let url = format!("{ONNX_BASE_URL}/{}", asset.remote);
    fetch_verified(&url, asset.sha256, &dest)
}

/// Puts `src` at `dest`: a symlink where the platform has them, else a copy.
fn place(
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
fn place_file(
    src: &Path,
    dest: &Path,
) {
    if src == dest && !dest.is_symlink() {
        return;
    }
    let src = fs::canonicalize(src).unwrap_or_else(|e| panic!("resolving {}: {e}", src.display()));
    let tmp = dest.with_extension("placing");
    let _ = fs::remove_file(&tmp);
    if fs::hard_link(&src, &tmp).is_err() {
        fs::copy(&src, &tmp)
            .unwrap_or_else(|e| panic!("copying {} to {}: {e}", src.display(), tmp.display()));
    }
    fs::rename(&tmp, dest)
        .unwrap_or_else(|e| panic!("placing {} at {}: {e}", src.display(), dest.display()));
}

/// Ensures `dest` holds the asset at `url` with the given digest.
///
/// A cache entry whose digest does not match is treated as corrupt: it is
/// removed and fetched once more, and a second failure is fatal.
fn fetch_verified(
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

/// Streams `url` to `dest`, via a temporary file so an interrupted transfer
/// cannot leave a truncated asset in the cache.
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

/// The lowercase hex SHA-256 of a file, read in chunks so a 2.4 GB asset stays
/// off the heap.
fn digest_of(path: &Path) -> io::Result<String> {
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
