//! Fetches the Qwen3-ASR reference: under `onnx_gen`, the `OpenVoiceOS` ONNX
//! export of `Qwen3-ASR-0.6B` that `qwen3-asr-model-validation` compares
//! bunsen against: its audio encoder, its token embedding, and its merged
//! prefill-and-step text decoder. The checkpoints themselves are not fetched
//! here: the validation crate's tests load them through bunsen's own pinned
//! rows, at run time, into the weights cache.
//!
//! Fetching, pinning and the overrides (`QWEN3_ASR_ONNX_ENCODER`,
//! `QWEN3_ASR_ONNX_EMBED`, `QWEN3_ASR_ONNX_DECODER`,
//! `QWEN3_ASR_ONNX_DECODER_DATA`) follow `bunsen-fetch`'s rules.

// Each feature uses its own constants; the rest are dead in that build.
#![allow(unused)]

use bunsen_fetch::{
    Asset,
    out_dir,
    place_file,
};

/// The audio encoder: padded log-mels, the valid post-convolution frames,
/// and the windowed attention bias, to projected audio embeddings.
const ENCODER: Asset = Asset {
    override_var: "QWEN3_ASR_ONNX_ENCODER",
    url: "https://huggingface.co/OpenVoiceOS/qwen3-asr-0.6b-onnx/resolve/05ac7b041a8e2acbfe8ffc895664ced67a2a4c58/encoder.onnx",
    sha256: "4136a528795bb59780fcf0a36f9429893b75f1e578bbfc86edeb189a0de56428",
};

/// The token embedding.
const EMBED: Asset = Asset {
    override_var: "QWEN3_ASR_ONNX_EMBED",
    url: "https://huggingface.co/OpenVoiceOS/qwen3-asr-0.6b-onnx/resolve/05ac7b041a8e2acbfe8ffc895664ced67a2a4c58/embed_tokens.onnx",
    sha256: "ec55d3a81939196fc64cb585af2adc67e0167ef122a8c692259c884ecfd9c1b0",
};

/// The text decoder, prefill and step in one graph, with a 28-layer KV
/// cache. Its weights are external, in [`DECODER_DATA`].
const DECODER: Asset = Asset {
    override_var: "QWEN3_ASR_ONNX_DECODER",
    url: "https://huggingface.co/OpenVoiceOS/qwen3-asr-0.6b-onnx/resolve/05ac7b041a8e2acbfe8ffc895664ced67a2a4c58/decoder.onnx",
    sha256: "9333188e5b7a55001ba315ff15095b32285f0dbfdba1f85d776d58bc0c527b9c",
};

/// The decoder's external weights, under the name the graph gives them.
const DECODER_DATA: Asset = Asset {
    override_var: "QWEN3_ASR_ONNX_DECODER_DATA",
    url: "https://huggingface.co/OpenVoiceOS/qwen3-asr-0.6b-onnx/resolve/05ac7b041a8e2acbfe8ffc895664ced67a2a4c58/decoder.onnx_data",
    sha256: "981567de7321b02d2f15cfdf65993fddd74e31f167aa0a1f43fcbbb381884157",
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
    println!("cargo:rustc-env=QWEN3_ASR_ONNX_OUT_DIR={}", out.display());

    // Every graph is a real file under its local name: `ModelGen` names the
    // generated module after the file, and the decoder reads its weights from
    // beside itself under their remote name, which `onnx-ir` refuses through
    // a symlink.
    let local = |asset: &Asset, name: &str| {
        let dest = out.join(name);
        let found = asset.resolve(&dest);
        place_file(&found, &dest).unwrap_or_else(|e| panic!("{e}"));
        dest
    };
    local(&DECODER_DATA, "decoder.onnx_data");
    let graphs = [
        local(&ENCODER, "qwen3_asr_encoder.onnx"),
        local(&EMBED, "qwen3_asr_embed.onnx"),
        local(&DECODER, "qwen3_asr_decoder.onnx"),
    ];
    for onnx in graphs {
        println!("cargo:rerun-if-changed={}", onnx.display());
        burn_onnx::ModelGen::new()
            .input(onnx.to_str().expect("asset path is UTF-8"))
            .out_dir("./")
            .run_from_script();
    }
}
