//! Fetches the pretrained Whisper assets.
//!
//! Three independent sets, each behind its own feature: `checkpoint` is the
//! `base.pt` bunsen loads, `vocab` is the pair of `.tiktoken` rank files its
//! tokenizer reads, and `onnx_gen` is the `onnx-community` export that
//! `whisper-model-validation` compares it against. None implies another.
//!
//! Cargo places no restriction on network access in a build script, but a
//! build that silently downloads is a bad neighbour: it breaks `--offline`,
//! breaks air-gapped CI, and makes builds non-reproducible. So:
//!
//! * A fetch only happens under the feature that needs the asset.
//! * Assets land in `OUT_DIR`, the one directory a build script may write to.
//!   `cargo publish` verifies that the build left the package source untouched,
//!   and a crate unpacked from crates.io must not write into the registry — so
//!   a `cache/` beside the manifest is not an option, and `cargo clean` does
//!   cost a re-download. The override variables below are the way around that.
//! * Every asset is pinned to a SHA-256 and re-verified on each build. A cached
//!   copy that fails is deleted and re-fetched once.
//! * `WHISPER_BASE_PT`, `WHISPER_MULTILINGUAL_TIKTOKEN`,
//!   `WHISPER_GPT2_TIKTOKEN`, `WHISPER_ONNX_ENCODER` and `WHISPER_ONNX_DECODER`
//!   point the build at local files instead, for working offline or against a
//!   different export.
//!
//! **`checkpoint` is on by default**, so building this crate — including as
//! part of `cargo build --workspace` — fetches 145 MB into a fresh `OUT_DIR`.
//! That is deliberate for a crate whose whole purpose is to bundle weights, but
//! it does mean a clean CI run pays for it. `--no-default-features` opts out,
//! and `WHISPER_BASE_PT` points at a local copy.

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

/// `OpenAI`'s multilingual `base.pt`.
///
/// `onnx-community/whisper-base` is a conversion of this checkpoint, which is
/// what lets `whisper-model-validation` compare the two.
const BASE_URL: &str = "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt";

/// SHA-256 of [`BASE_URL`]'s payload. `OpenAI` embeds it in the URL.
const BASE_SHA256: &str = "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e";

/// Local name for the fetched checkpoint.
const BASE_FILE: &str = "whisper_base.pt";

/// The rank file behind every multilingual checkpoint's tokenizer.
///
/// Pinned to the commit that last touched it ("Use tiktoken", openai/whisper
/// #1044) rather than `main`, so the URL names one file forever. Its last line
/// is `= 50256` — base64 of nothing — and that empty token is real; bunsen's
/// parser reads it as such.
const MULTILINGUAL_TIKTOKEN_URL: &str = "https://raw.githubusercontent.com/openai/whisper/839639a223b92ad61851baae9ad8a695ccb41ce5/whisper/assets/multilingual.tiktoken";

/// SHA-256 of [`MULTILINGUAL_TIKTOKEN_URL`]'s payload.
const MULTILINGUAL_TIKTOKEN_SHA256: &str =
    "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126";

/// Local name for the fetched multilingual rank file.
const MULTILINGUAL_TIKTOKEN_FILE: &str = "multilingual.tiktoken";

/// The rank file behind the English-only (`*.en`) checkpoints' tokenizer:
/// GPT-2's, one rank shorter than the multilingual file.
const GPT2_TIKTOKEN_URL: &str = "https://raw.githubusercontent.com/openai/whisper/839639a223b92ad61851baae9ad8a695ccb41ce5/whisper/assets/gpt2.tiktoken";

/// SHA-256 of [`GPT2_TIKTOKEN_URL`]'s payload.
const GPT2_TIKTOKEN_SHA256: &str =
    "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930";

/// Local name for the fetched English-only rank file.
const GPT2_TIKTOKEN_FILE: &str = "gpt2.tiktoken";

/// The encoder graph, exported by the `onnx-community` mirror of
/// `openai/whisper-base`.
const ENCODER_URL: &str =
    "https://huggingface.co/onnx-community/whisper-base/resolve/main/onnx/encoder_model.onnx";

/// SHA-256 of [`ENCODER_URL`]'s payload.
const ENCODER_SHA256: &str = "a9f3b752833b49e880dec91ee5b6d936112be7c3ea07c221024ba493439f46fe";

/// Local name for the fetched graph. `ModelGen` derives the generated module
/// name from this, so it is also the name `src/lib.rs` includes.
const ENCODER_FILE: &str = "whisper_base_encoder.onnx";

/// The decoder graph, without the KV-cache inputs — it takes a whole token
/// sequence at once, which is the shape `TextDecoder::forward` has.
const DECODER_URL: &str =
    "https://huggingface.co/onnx-community/whisper-base/resolve/main/onnx/decoder_model.onnx";

/// SHA-256 of [`DECODER_URL`]'s payload.
const DECODER_SHA256: &str = "70d26763610c0d6bb407373b7f30d415252ee470e62a0f816c8a46b2caca7326";

/// Local name for the fetched decoder graph.
const DECODER_FILE: &str = "whisper_base_decoder.onnx";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=WHISPER_BASE_PT");
    println!("cargo:rerun-if-env-changed=WHISPER_MULTILINGUAL_TIKTOKEN");
    println!("cargo:rerun-if-env-changed=WHISPER_GPT2_TIKTOKEN");
    println!("cargo:rerun-if-env-changed=WHISPER_ONNX_ENCODER");
    println!("cargo:rerun-if-env-changed=WHISPER_ONNX_DECODER");

    // Each feature gates its own asset. `src/lib.rs` gates the matching item
    // on the same feature, so with neither the crate is trivially empty.
    //
    // These are `cfg` rather than `CARGO_FEATURE_*` lookups because
    // `burn_onnx` is an optional build-dependency: without the feature it is
    // not linked, so a runtime check would still fail to compile.
    #[cfg(feature = "checkpoint")]
    fetch_checkpoint();

    #[cfg(feature = "vocab")]
    fetch_vocab();

    #[cfg(feature = "onnx_gen")]
    generate_reference();
}

/// Fetches `base.pt` and names it as a compile-time constant, so a caller
/// never has to discover the path or thread it through.
#[cfg(feature = "checkpoint")]
fn fetch_checkpoint() {
    let checkpoint = resolve_asset(
        "WHISPER_BASE_PT",
        BASE_URL,
        BASE_SHA256,
        &cache_dir().join(BASE_FILE),
    );
    println!(
        "cargo:rustc-env=WHISPER_BASE_PT_PATH={}",
        checkpoint.display()
    );
}

/// Fetches both `.tiktoken` rank files and names them as compile-time
/// constants.
#[cfg(feature = "vocab")]
fn fetch_vocab() {
    let cache = cache_dir();

    for (override_var, url, sha, file, path_var) in [
        (
            "WHISPER_MULTILINGUAL_TIKTOKEN",
            MULTILINGUAL_TIKTOKEN_URL,
            MULTILINGUAL_TIKTOKEN_SHA256,
            MULTILINGUAL_TIKTOKEN_FILE,
            "WHISPER_MULTILINGUAL_TIKTOKEN_PATH",
        ),
        (
            "WHISPER_GPT2_TIKTOKEN",
            GPT2_TIKTOKEN_URL,
            GPT2_TIKTOKEN_SHA256,
            GPT2_TIKTOKEN_FILE,
            "WHISPER_GPT2_TIKTOKEN_PATH",
        ),
    ] {
        let vocab = resolve_asset(override_var, url, sha, &cache.join(file));
        println!("cargo:rustc-env={path_var}={}", vocab.display());
    }
}

/// Fetches the ONNX export and generates Rust models from it.
#[cfg(feature = "onnx_gen")]
fn generate_reference() {
    let cache = cache_dir();

    // The generated weights are loaded from `OUT_DIR` at run time rather than
    // embedded: together these are ~290 MB, and `include_bytes!` of that would
    // dominate both compile time and binary size. That is the one way this
    // differs from the Silero weights, which are small enough to inline.
    println!(
        "cargo:rustc-env=WHISPER_ONNX_OUT_DIR={}",
        env::var("OUT_DIR").unwrap()
    );

    for (var, url, sha, file) in [
        (
            "WHISPER_ONNX_ENCODER",
            ENCODER_URL,
            ENCODER_SHA256,
            ENCODER_FILE,
        ),
        (
            "WHISPER_ONNX_DECODER",
            DECODER_URL,
            DECODER_SHA256,
            DECODER_FILE,
        ),
    ] {
        let onnx = resolve_asset(var, url, sha, &cache.join(file));
        println!("cargo:rerun-if-changed={}", onnx.display());

        burn_onnx::ModelGen::new()
            .input(onnx.to_str().expect("asset path is UTF-8"))
            .out_dir("./")
            .run_from_script();
    }
}

/// Where fetched assets land: `OUT_DIR`.
///
/// The only directory a build script may write to. `cargo publish` fails
/// verification if the build touched the package source, and a crate unpacked
/// from crates.io must not write into `~/.cargo/registry`. The cost is that
/// `cargo clean` discards the assets along with everything else.
fn cache_dir() -> PathBuf {
    PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR for build scripts"))
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
///
/// Read in chunks rather than with `io::copy`: `sha2` 0.11 dropped the
/// `io::Write` impl on its hashers, and `digest` 0.11 has no feature to bring
/// it back. Chunking also keeps a 145 MB asset off the heap.
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
