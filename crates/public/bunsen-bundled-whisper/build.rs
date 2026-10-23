//! Fetches the pretrained Whisper assets.
//!
//! Four independent sets, each behind its own feature: `checkpoint` is the
//! `base.pt` bunsen loads, `hf_checkpoint` the same weights as
//! `transformers` names them, `vocab` is the pair of `.tiktoken` rank files
//! its tokenizer reads, and `onnx_gen` is the `onnx-community` export that
//! `whisper-model-validation` compares it against. None implies another.
//!
//! Cargo places no restriction on network access in a build script, but a
//! build that silently downloads is a bad neighbour: it breaks `--offline`,
//! breaks air-gapped CI, and makes builds non-reproducible. So:
//!
//! * A fetch only happens under the feature that needs the asset.
//! * Assets land in `OUT_DIR`, the one directory a build script may write to,
//!   laid out as a pretrained cache (see [`cache_path`]) so that a cache
//!   pointed at `OUT_DIR` hits them without knowing they were bundled. `cargo
//!   publish` verifies that the build left the package source untouched, and a
//!   crate unpacked from crates.io must not write into the registry — so a
//!   `cache/` beside the manifest is not an option, and `cargo clean` does cost
//!   a re-download. The override variables below are the way around that.
//! * Every asset is pinned to a SHA-256 and re-verified on each build. A cached
//!   copy that fails is deleted and re-fetched once. `bunsen-fetch`'s `Asset`
//!   does the fetching, pinning and overriding, as it does for every bundled
//!   crate.
//! * `WHISPER_BASE_PT`, `WHISPER_BASE_SAFETENSORS`,
//!   `WHISPER_MULTILINGUAL_TIKTOKEN`, `WHISPER_GPT2_TIKTOKEN`,
//!   `WHISPER_ONNX_ENCODER` and `WHISPER_ONNX_DECODER` point the build at local
//!   files instead, for working offline or against a different export. A
//!   pretrained override that is the pinned file is laid out in the cache too;
//!   one that is not is used where it is, and only its accessor knows it.
//!
//! **No feature is on by default**, so building this crate as a workspace
//! member fetches nothing. `whisper-model-validation/weights` turns on
//! `checkpoint` and `vocab` (145 MB into a fresh `OUT_DIR`);
//! `whisper-model-validation/download` turns on `onnx_gen` too.
//! `WHISPER_BASE_PT` points at a local copy.

// Each feature uses its own constants and helpers; the rest are dead in that
// build. Enumerating `cfg` on every item would be noisier than this.
#![allow(unused)]

use std::path::PathBuf;

use bunsen_fetch::{
    Asset,
    out_dir,
    pretrained_cache_path,
};

/// `OpenAI`'s multilingual `base.pt`.
///
/// `onnx-community/whisper-base` is a conversion of this checkpoint, which is
/// what lets `whisper-model-validation` compare the two.
const BASE_URL: &str = "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt";

/// SHA-256 of [`BASE_URL`]'s payload. `OpenAI` embeds it in the URL.
const BASE_SHA256: &str = "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e";

/// Local name for the fetched checkpoint.
///
/// The name the Whisper kit's resource for `openai/base` uses, so the cache
/// layout below names the same file.
const BASE_FILE: &str = "base.pt";

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

/// `transformers`' conversion of `OpenAI`'s multilingual base checkpoint,
/// `openai/whisper-base`'s `model.safetensors`, pinned to a commit.
///
/// The same weights as [`BASE_URL`]'s under `transformers`' names, which is
/// what lets `whisper-model-validation` check the kit's `transformers` loader
/// against its `.pt` one.
const BASE_SAFETENSORS_URL: &str = "https://huggingface.co/openai/whisper-base/resolve/e37978b90ca9030d5170a5c07aadb050351a65bb/model.safetensors";

/// SHA-256 of [`BASE_SAFETENSORS_URL`]'s payload.
const BASE_SAFETENSORS_SHA256: &str =
    "07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b";

/// Local name for the fetched `transformers` checkpoint.
const BASE_SAFETENSORS_FILE: &str = "whisper-base.safetensors";

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

/// The Whisper kit's segment of the layout: `WHISPER_KIT` in bunsen.
const WHISPER_KIT: &str = "whisper";

/// The namespace every `openai` resource is cached under.
const OPENAI_NAMESPACE: &str = "openai";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=WHISPER_BASE_PT");
    println!("cargo:rerun-if-env-changed=WHISPER_BASE_SAFETENSORS");
    println!("cargo:rerun-if-env-changed=WHISPER_MULTILINGUAL_TIKTOKEN");
    println!("cargo:rerun-if-env-changed=WHISPER_GPT2_TIKTOKEN");
    println!("cargo:rerun-if-env-changed=WHISPER_ONNX_ENCODER");
    println!("cargo:rerun-if-env-changed=WHISPER_ONNX_DECODER");

    // The pretrained assets are laid out under this as a cache; the
    // accessor for it is gated on the features that fill it.
    println!(
        "cargo:rustc-env=WHISPER_CACHE_DIR={}",
        cache_dir().display()
    );

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

    #[cfg(feature = "hf_checkpoint")]
    fetch_hf_checkpoint();

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
        &cache_path(BASE_SHA256, BASE_FILE),
    );
    println!(
        "cargo:rustc-env=WHISPER_BASE_PT_PATH={}",
        checkpoint.display()
    );
}

/// Fetches `openai/whisper-base`'s `model.safetensors` and names it as a
/// compile-time constant. It is not laid out as a cache: the kit's `hf:`
/// rows are served by Hugging Face's cache, not bunsen's.
#[cfg(feature = "hf_checkpoint")]
fn fetch_hf_checkpoint() {
    let checkpoint = resolve_asset(
        "WHISPER_BASE_SAFETENSORS",
        BASE_SAFETENSORS_URL,
        BASE_SAFETENSORS_SHA256,
        &cache_dir().join(BASE_SAFETENSORS_FILE),
    );
    println!(
        "cargo:rustc-env=WHISPER_BASE_SAFETENSORS_PATH={}",
        checkpoint.display()
    );
}

/// Fetches both `.tiktoken` rank files and names them as compile-time
/// constants.
#[cfg(feature = "vocab")]
fn fetch_vocab() {
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
        let vocab = resolve_asset(override_var, url, sha, &cache_path(sha, file));
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
        out_dir().display()
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
    out_dir()
}

/// Where a pretrained asset lands: `OUT_DIR` laid out as a pretrained cache,
/// `pretrained/whisper/openai/<sha256>/<file>`. The digest in the path is
/// the pin, as it is in bunsen's cache, so a cache rooted at `OUT_DIR`
/// trusts what it finds there.
fn cache_path(
    sha256: &str,
    file: &str,
) -> PathBuf {
    pretrained_cache_path(&cache_dir(), WHISPER_KIT, OPENAI_NAMESPACE, sha256, file)
}

/// A path to an asset: its override, or `dest`, fetched and verified if it
/// is not already there, by `bunsen_fetch`'s [`Asset`] rules.
fn resolve_asset(
    override_var: &str,
    url: &str,
    sha256: &str,
    dest: &std::path::Path,
) -> PathBuf {
    Asset {
        override_var,
        url,
        sha256,
    }
    .resolve(dest)
}
