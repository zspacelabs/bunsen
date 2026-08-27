//! Fetches the reference assets and generates the ONNX model.
//!
//! Cargo places no restriction on network access in a build script, but a
//! build that silently downloads is a bad neighbour: it breaks `--offline`,
//! breaks air-gapped CI, and makes builds non-reproducible. So:
//!
//! * The fetch only happens under the `download` feature, which is off by
//!   default — `cargo build --workspace` never reaches the network.
//! * Assets land in a `.cache/` directory beside the manifest, not `OUT_DIR`,
//!   so `cargo clean` does not force an 82 MB re-download.
//! * Every asset is pinned to a SHA-256 and re-verified on each build. A cache
//!   entry that fails is deleted and re-fetched once.
//! * `WHISPER_ONNX_ENCODER` points the build at a local file instead, for
//!   working offline or against a different export.

use std::{
    env,
    fs,
    io,
    path::{
        Path,
        PathBuf,
    },
};

use sha2::{
    Digest,
    Sha256,
};

/// The encoder graph, exported by the `onnx-community` mirror of
/// `openai/whisper-base`.
const ENCODER_URL: &str =
    "https://huggingface.co/onnx-community/whisper-base/resolve/main/onnx/encoder_model.onnx";

/// SHA-256 of [`ENCODER_URL`]'s payload.
const ENCODER_SHA256: &str = "a9f3b752833b49e880dec91ee5b6d936112be7c3ea07c221024ba493439f46fe";

/// Local name for the fetched graph. `ModelGen` derives the generated module
/// name from this, so it is also the name `src/lib.rs` includes.
const ENCODER_FILE: &str = "whisper_base_encoder.onnx";

/// `OpenAI`'s multilingual `base.pt` — the checkpoint the ONNX export above was
/// converted from. Both are needed, and they must be the same model, or the
/// comparison is meaningless.
const CHECKPOINT_URL: &str = "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt";

/// SHA-256 of [`CHECKPOINT_URL`]'s payload. `OpenAI` embeds it in the URL.
const CHECKPOINT_SHA256: &str = "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e";

/// Local name for the fetched checkpoint.
const CHECKPOINT_FILE: &str = "whisper_base.pt";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=WHISPER_ONNX_ENCODER");
    println!("cargo:rerun-if-env-changed=WHISPER_BASE_PT");

    if env::var_os("CARGO_FEATURE_DOWNLOAD").is_none() {
        // Nothing to do: `src/lib.rs` gates the generated module on the same
        // feature, so the crate still compiles (and is trivially empty).
        return;
    }

    let cache = cache_dir();

    // Hand the checkpoint's location to the test as a compile-time constant,
    // so the comparison has no reason to skip itself.
    let checkpoint = resolve_asset(
        "WHISPER_BASE_PT",
        CHECKPOINT_URL,
        CHECKPOINT_SHA256,
        &cache.join(CHECKPOINT_FILE),
    );
    println!(
        "cargo:rustc-env=WHISPER_BASE_PT_PATH={}",
        checkpoint.display()
    );

    let onnx = resolve_asset(
        "WHISPER_ONNX_ENCODER",
        ENCODER_URL,
        ENCODER_SHA256,
        &cache.join(ENCODER_FILE),
    );
    println!("cargo:rerun-if-changed={}", onnx.display());

    burn_onnx::ModelGen::new()
        .input(onnx.to_str().expect("asset path is UTF-8"))
        .out_dir("./")
        .run_from_script();
}

/// The asset cache, beside the manifest rather than in `OUT_DIR` so a
/// `cargo clean` does not force a re-download.
fn cache_dir() -> PathBuf {
    let cache = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join(".cache");
    fs::create_dir_all(&cache).expect("create the asset cache");
    cache
}

/// Returns a path to an asset, honouring an override or fetching it.
///
/// # Arguments
/// * `override_var`: env var naming a local file to use instead. A
///   caller-supplied file is deliberately **not** digest-pinned — the point of
///   the override is to try something else.
fn resolve_asset(
    override_var: &str,
    url: &str,
    sha256: &str,
    dest: &Path,
) -> PathBuf {
    if let Some(local) = env::var_os(override_var) {
        let path = PathBuf::from(local);
        assert!(
            path.is_file(),
            "{override_var} is set but not a file: {}",
            path.display(),
        );
        return path;
    }

    fetch_verified(url, sha256, dest)
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
    let tmp = dest.with_extension("partial");

    let mut response = ureq::get(url).call()?;
    let mut reader = response.body_mut().as_reader();
    let mut file = fs::File::create(&tmp)?;
    io::copy(&mut reader, &mut file)?;
    drop(file);

    fs::rename(&tmp, dest)?;
    Ok(())
}

/// The lowercase hex SHA-256 of a file.
fn digest_of(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;

    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
