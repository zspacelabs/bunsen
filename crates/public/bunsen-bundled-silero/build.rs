//! Generates the burnpack from the committed ONNX graph, and names it.
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
}
