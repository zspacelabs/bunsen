//! Generates the burnpack from the committed ONNX graph, and names it; with
//! `checkpoint`, fetches upstream's 16 kHz safetensors.
//!
//! `burn-onnx` writes the weights to `OUT_DIR`; the crate links them in
//! with `include_bytes!`. The build also hashes the file it generated and
//! exports the digest, since the burnpack is this build's artifact, not
//! upstream's: bunsen's Silero kit lists it as `bundled:silero/vad`, pinned
//! to that digest, so a rebuilt burnpack is cached in its own slot.

use std::{
    env,
    fs,
    path::PathBuf,
};

use burn_onnx::ModelGen;
use sha2::{
    Digest,
    Sha256,
};

/// The generated burnpack's file name, under `OUT_DIR`.
const BURNPACK_FILE: &str = "silero_vad_op18_ifless.bpk";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=onnx/silero_vad_op18_ifless.onnx");

    ModelGen::new()
        .input("onnx/silero_vad_op18_ifless.onnx")
        .out_dir("./")
        .run_from_script();

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set for a build script"));
    let bytes = fs::read(out_dir.join(BURNPACK_FILE)).expect("the generated burnpack");
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("cargo:rustc-env=SILERO_BURNPACK_SHA256={digest}");

    #[cfg(feature = "checkpoint")]
    fetch_checkpoint(&out_dir);
}

/// Fetches upstream's 16 kHz safetensors into `OUT_DIR`, laid out as a
/// pretrained cache, `pretrained/silero_vad/silero/<sha256>/<file>`, at the
/// revision bunsen's `silero/vad-16k` row pins.
#[cfg(feature = "checkpoint")]
fn fetch_checkpoint(out_dir: &std::path::Path) {
    const SHA256: &str = "c59271c284ae9c8335d795d60e0bfdb71aaaceec578d9bd9ffc1b8153c319ea1";
    const FILE: &str = "silero_vad_16k.safetensors";
    let asset = bunsen_fetch::Asset {
        override_var: "SILERO_VAD_16K_SAFETENSORS",
        url: "https://raw.githubusercontent.com/snakers4/silero-vad/fb7d7c7f5df63235f0afd037d8397593829aeb41/src/silero_vad/data/silero_vad_16k.safetensors",
        sha256: SHA256,
    };
    let dest = bunsen_fetch::pretrained_cache_path(out_dir, "silero_vad", "silero", SHA256, FILE);
    let path = asset.resolve(&dest);
    println!("cargo:rustc-env=SILERO_CACHE_DIR={}", out_dir.display());
    println!(
        "cargo:rustc-env=SILERO_VAD_16K_SAFETENSORS_PATH={}",
        path.display()
    );
}
