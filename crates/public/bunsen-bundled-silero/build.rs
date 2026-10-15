//! Fetches the Silero VAD assets.
//!
//! Two sets, each behind its own feature: `checkpoint` is upstream's 16 kHz
//! `silero_vad_16k.safetensors`, laid out as a bunsen pretrained cache, and
//! `onnx_gen` is upstream's `silero_vad_op18_ifless.onnx`, from which
//! `burn-onnx` generates the reference model and its burnpack. Both come
//! from one pinned revision of `snakers4/silero-vad`.
//!
//! Fetching, pinning and the `SILERO_VAD_16K_SAFETENSORS` and
//! `SILERO_VAD_ONNX` overrides follow `bunsen-fetch`'s rules.

// Each feature uses its own constants; the rest are dead in that build.
#![allow(unused)]

/// The revision of `snakers4/silero-vad` every asset is fetched at.
const REVISION: &str = "https://raw.githubusercontent.com/snakers4/silero-vad/fb7d7c7f5df63235f0afd037d8397593829aeb41/src/silero_vad/data";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "checkpoint")]
    fetch_checkpoint();

    #[cfg(feature = "onnx_gen")]
    generate_reference();
}

/// Fetches upstream's 16 kHz safetensors into `OUT_DIR`, laid out as a
/// pretrained cache, `pretrained/silero_vad/silero/<sha256>/<file>`, at the
/// revision bunsen's `silero/vad-16k` row pins.
#[cfg(feature = "checkpoint")]
fn fetch_checkpoint() {
    const SHA256: &str = "c59271c284ae9c8335d795d60e0bfdb71aaaceec578d9bd9ffc1b8153c319ea1";
    const FILE: &str = "silero_vad_16k.safetensors";
    let out = bunsen_fetch::out_dir();
    let url = format!("{REVISION}/{FILE}");
    let asset = bunsen_fetch::Asset {
        override_var: "SILERO_VAD_16K_SAFETENSORS",
        url: &url,
        sha256: SHA256,
    };
    let dest = bunsen_fetch::pretrained_cache_path(&out, "silero_vad", "silero", SHA256, FILE);
    let path = asset.resolve(&dest);
    println!("cargo:rustc-env=SILERO_CACHE_DIR={}", out.display());
    println!(
        "cargo:rustc-env=SILERO_VAD_16K_SAFETENSORS_PATH={}",
        path.display()
    );
}

/// Fetches the ONNX graph and generates the reference model from it.
#[cfg(feature = "onnx_gen")]
fn generate_reference() {
    const FILE: &str = "silero_vad_op18_ifless.onnx";
    let out = bunsen_fetch::out_dir();
    println!("cargo:rustc-env=SILERO_ONNX_OUT_DIR={}", out.display());
    let url = format!("{REVISION}/{FILE}");
    let asset = bunsen_fetch::Asset {
        override_var: "SILERO_VAD_ONNX",
        url: &url,
        sha256: "7671cd04b004e9076da0d4a7b1a5aec36adf161c39230c1cb94a4fd5db6bbd28",
    };
    // A real file under its local name: `ModelGen` names the generated
    // module after it.
    let dest = out.join(FILE);
    let found = asset.resolve(&dest);
    bunsen_fetch::place_file(&found, &dest).unwrap_or_else(|e| panic!("{e}"));
    println!("cargo:rerun-if-changed={}", dest.display());
    burn_onnx::ModelGen::new()
        .input(dest.to_str().expect("asset path is UTF-8"))
        .out_dir("./")
        .run_from_script();
}
