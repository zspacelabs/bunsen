//! Fetches the Parakeet TDT reference: under `onnx_gen`, the `istupakov`
//! ONNX export of NVIDIA's `parakeet-tdt-0.6b-v3` that
//! `parakeet-model-validation` compares bunsen against: its encoder graph and
//! its prediction-and-joint graph. The checkpoints themselves are not fetched
//! here: the validation crate's tests load them through bunsen's own pinned
//! rows, at run time, into the weights cache.
//!
//! The export's feature-extractor graph (`nemo128.onnx`) is not generated:
//! `burn-onnx` 0.22 cannot infer its types (`ArgType::Shape has no DType`).
//!
//! Fetching, pinning and the overrides (`PARAKEET_ONNX_ENCODER`,
//! `PARAKEET_ONNX_ENCODER_DATA`, `PARAKEET_ONNX_DECODER_JOINT`) follow
//! `bunsen-fetch`'s rules.

// Each feature uses its own constants; the rest are dead in that build.
#![allow(unused)]

use bunsen_fetch::{
    Asset,
    out_dir,
    place_file,
};

/// The `FastConformer` encoder graph, `encoder-model.onnx`. Its weights are
/// external, in [`ENCODER_DATA`].
const ENCODER: Asset = Asset {
    override_var: "PARAKEET_ONNX_ENCODER",
    url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/encoder-model.onnx",
    sha256: "98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1",
};

/// The encoder's external weights. The graph names this file by its remote
/// name, relative to the graph's directory, so it keeps that name locally.
const ENCODER_DATA: Asset = Asset {
    override_var: "PARAKEET_ONNX_ENCODER_DATA",
    url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/encoder-model.onnx.data",
    sha256: "9a22d372c51455c34f13405da2520baefb7125bd16981397561423ed32d24f36",
};

/// The TDT prediction network and joint, one step at a time.
const DECODER_JOINT: Asset = Asset {
    override_var: "PARAKEET_ONNX_DECODER_JOINT",
    url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/decoder_joint-model.onnx",
    sha256: "e978ddf6688527182c10fde2eb4b83068421648985ef23f7a86be732be8706c1",
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // `cfg` rather than a `CARGO_FEATURE_*` lookup: `burn_onnx` is an
    // optional build-dependency, so without the feature it is not linked.
    #[cfg(feature = "onnx_gen")]
    generate_reference();
}

/// Fetches the ONNX export and generates Rust models from it.
#[cfg(feature = "onnx_gen")]
fn generate_reference() {
    let out = out_dir();
    // The generated weights are loaded from `OUT_DIR` at run time rather than
    // embedded: the encoder's alone are 2.4 GB.
    println!("cargo:rustc-env=PARAKEET_ONNX_OUT_DIR={}", out.display());

    // Every graph is a real file under its local name: `ModelGen` names the
    // generated module after the file, and the encoder reads its weights from
    // beside itself under their remote name. `onnx-ir` canonicalizes that
    // data path and refuses one that resolves outside the graph's directory,
    // so neither may be a symlink.
    let local = |asset: &Asset, name: &str| {
        let dest = out.join(name);
        let found = asset.resolve(&dest);
        place_file(&found, &dest).unwrap_or_else(|e| panic!("{e}"));
        dest
    };
    local(&ENCODER_DATA, "encoder-model.onnx.data");
    let encoder = local(&ENCODER, "parakeet_encoder.onnx");
    let decoder = local(&DECODER_JOINT, "parakeet_decoder_joint.onnx");

    for onnx in [encoder, decoder] {
        println!("cargo:rerun-if-changed={}", onnx.display());
        burn_onnx::ModelGen::new()
            .input(onnx.to_str().expect("asset path is UTF-8"))
            .out_dir("./")
            .run_from_script();
    }
}
