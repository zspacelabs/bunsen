//! Fetches the Qwen3 reference: under `onnx_gen`, `onnx-community`'s fp32
//! ONNX export of `Qwen3-0.6B`, which `qwen3-model-validation` compares
//! bunsen against. The checkpoints themselves are not fetched here: the
//! validation crate's tests load them through bunsen's own pinned rows.
//!
//! `burn-onnx` 0.22 cannot import the export as published, so the graph is
//! rewritten first, by `rewrite_genai_onnx::rewrite_for_burn_onnx`.
//!
//! Fetching, pinning and the overrides (`QWEN3_ONNX_MODEL`,
//! `QWEN3_ONNX_MODEL_DATA`) follow `bunsen-fetch`'s rules.

// Each feature uses its own constants; the rest are dead in that build.
#![allow(unused)]

use bunsen_fetch::{
    Asset,
    out_dir,
    place_file,
};

/// The export's graph: an onnxruntime-genai export of `Qwen3-0.6B`, fp32,
/// opset 14 with `com.microsoft` ops, a 28-layer KV cache; part of its
/// weights inline, the rest in [`ONNX_DATA`].
const ONNX_MODEL: Asset = Asset {
    override_var: "QWEN3_ONNX_MODEL",
    url: "https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/58ca03a5e7b354b480b1a82b34bde8dc7b854cc0/onnx/model.onnx",
    sha256: "0a8f1cfe531fc7b15d26db55090f0667863f0c62c952dff128cf23cd421e3fbc",
};

/// The graph's external weights, under the name the graph gives them.
const ONNX_DATA: Asset = Asset {
    override_var: "QWEN3_ONNX_MODEL_DATA",
    url: "https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/58ca03a5e7b354b480b1a82b34bde8dc7b854cc0/onnx/model.onnx_data",
    sha256: "e0927a7de2fe7c2a9b50a121905cf2edda78fb2900c85c1790db98188bbfe4be",
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // `cfg` rather than a `CARGO_FEATURE_*` lookup: `burn_onnx` is an
    // optional build-dependency, so without the feature it is not linked.
    #[cfg(feature = "onnx_gen")]
    generate_reference();
}

/// Fetches the export, rewrites it, and generates the Rust model from it.
#[cfg(feature = "onnx_gen")]
fn generate_reference() {
    use rewrite_genai_onnx::onnx_ir::{
        Message,
        ModelProto,
    };

    let out = out_dir();
    // The generated weights are loaded from `OUT_DIR` at run time rather than
    // embedded.
    println!("cargo:rustc-env=QWEN3_ONNX_OUT_DIR={}", out.display());

    let fetched = ONNX_MODEL.resolve(&out.join("model.onnx"));
    println!("cargo:rerun-if-changed={}", fetched.display());
    // The rewritten graph reads its external weights from beside itself,
    // under their remote name, which `onnx-ir` refuses through a symlink.
    let data = out.join("model.onnx_data");
    let found = ONNX_DATA.resolve(&data);
    place_file(&found, &data).unwrap_or_else(|e| panic!("{e}"));

    let bytes =
        std::fs::read(&fetched).unwrap_or_else(|e| panic!("read {}: {e}", fetched.display()));
    let mut model = ModelProto::parse_from_bytes(&bytes)
        .unwrap_or_else(|e| panic!("parse {}: {e}", fetched.display()));
    drop(bytes);
    rewrite_genai_onnx::rewrite_for_burn_onnx(
        &mut model,
        &[
            ("GroupQueryAttention", 28),
            ("RotaryEmbedding", 56),
            ("SimplifiedLayerNormalization", 57),
            ("SkipSimplifiedLayerNormalization", 56),
        ],
    );

    // `ModelGen` names the generated module after the file.
    let patched = out.join("qwen3.onnx");
    std::fs::write(
        &patched,
        model.write_to_bytes().expect("serialize the graph"),
    )
    .unwrap_or_else(|e| panic!("write {}: {e}", patched.display()));
    drop(model);

    burn_onnx::ModelGen::new()
        .input(patched.to_str().expect("asset path is UTF-8"))
        .out_dir("./")
        .run_from_script();
}
