//! Fetches the Voxtral Realtime assets.
//!
//! Two sets, each behind its own feature: `checkpoint` is
//! `mistralai/Voxtral-Mini-4B-Realtime-2602` as `transformers` ships it, laid
//! out as a bunsen pretrained cache; `onnx_gen` is `onnx-community`'s fp32
//! ONNX export of it, three graphs (the audio encoder, the decoder, the token
//! embedding), which `voxtral-model-validation` compares bunsen against.
//!
//! `burn-onnx` 0.22 cannot import the export as published, so each graph is
//! rewritten first, by `rewrite_genai_onnx::rewrite_for_burn_onnx`.
//!
//! Fetching, pinning and the overrides (one per file, named for the file:
//! `VOXTRAL_MODEL_SAFETENSORS`, `VOXTRAL_ONNX_AUDIO_ENCODER_ONNX`,
//! `VOXTRAL_ONNX_DECODER_MODEL_MERGED_ONNX_DATA_3`, ...) follow
//! `bunsen-fetch`'s rules.

// Each feature uses its own constants; the rest are dead in that build.
#![allow(unused)]

use bunsen_fetch::{
    Asset,
    out_dir,
    place_file,
    pretrained_cache_path,
};

/// The Voxtral kit's segment of the pretrained cache layout: `VOXTRAL_KIT`
/// in bunsen.
const VOXTRAL_KIT: &str = "voxtral";

/// The namespace of Mistral's rows.
const MISTRAL_NAMESPACE: &str = "mistralai";

/// `mistralai/Voxtral-Mini-4B-Realtime-2602`'s files, at the revision
/// bunsen's `mistralai/voxtral-mini-4b-realtime-2602` row pins.
const CHECKPOINT: &[(&str, Asset<'static>)] = &[
    (
        "model.safetensors",
        Asset {
            override_var: "VOXTRAL_MODEL_SAFETENSORS",
            url: "https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/resolve/2769294da9567371363522aac9bbcfdd19447add/model.safetensors",
            sha256: "e745e4902df6a4c48f29f2f8dc1f6d0fb4cc73c7156bc45923451a5bcdfcd1d6",
        },
    ),
    (
        "tekken.json",
        Asset {
            override_var: "VOXTRAL_TEKKEN_JSON",
            url: "https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/resolve/2769294da9567371363522aac9bbcfdd19447add/tekken.json",
            sha256: "8434af1d39eba99f0ef46cf1450bf1a63fa941a26933a1ef5dbbf4adf0d00e44",
        },
    ),
];

/// One graph of the export: the graph, its external weights under the names
/// the graph gives them, and how many of each rewritten op it holds.
struct Graph {
    /// The generated module's name, and the rewritten graph's file stem.
    stem: &'static str,
    graph: Asset<'static>,
    data: &'static [(&'static str, Asset<'static>)],
    counts: &'static [(&'static str, usize)],
}

/// The audio encoder: the causal conv stem over a packed carry, 32 `RoPE`
/// layers over a KV cache, the norm and the projector, to one embedding per
/// four encoder frames.
const VOXTRAL_AUDIO_ENCODER: Graph = Graph {
    stem: "voxtral_audio_encoder",
    graph: Asset {
        override_var: "VOXTRAL_ONNX_AUDIO_ENCODER_ONNX",
        url: "https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX/resolve/db34633130c460bd650644fc18dc90710d9f3e38/onnx/audio_encoder.onnx",
        sha256: "c615696db022f66a9cce3a7d547a3b2275b08c254c5fb994f49021878cb8eddb",
    },
    data: &[
        (
            "audio_encoder.onnx_data",
            Asset {
                override_var: "VOXTRAL_ONNX_AUDIO_ENCODER_ONNX_DATA",
                url: "https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX/resolve/db34633130c460bd650644fc18dc90710d9f3e38/onnx/audio_encoder.onnx_data",
                sha256: "bd564af284bf083a028f43256857dc7b1e96b5dfb5bed6aaa5f9fdbf184ec814",
            },
        ),
        (
            "audio_encoder.onnx_data_1",
            Asset {
                override_var: "VOXTRAL_ONNX_AUDIO_ENCODER_ONNX_DATA_1",
                url: "https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX/resolve/db34633130c460bd650644fc18dc90710d9f3e38/onnx/audio_encoder.onnx_data_1",
                sha256: "11946900469504753acd4763d3e544d87bd992675708817a2da691c5e46dcd0f",
            },
        ),
    ],
    counts: &[
        ("GroupQueryAttention", 32),
        ("RotaryEmbedding", 64),
        ("SimplifiedLayerNormalization", 65),
    ],
};

/// The text decoder: 26 layers over a KV cache, their adaptive scales baked
/// in at the export's delay (6 tokens), to the logits.
const VOXTRAL_DECODER: Graph = Graph {
    stem: "voxtral_decoder",
    graph: Asset {
        override_var: "VOXTRAL_ONNX_DECODER_MODEL_MERGED_ONNX",
        url: "https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX/resolve/db34633130c460bd650644fc18dc90710d9f3e38/onnx/decoder_model_merged.onnx",
        sha256: "8296fb23f5bb3637001ac9f5e3ce9ff9a4017010ef2fcf27b4dd64cb412cfde0",
    },
    data: &[
        (
            "decoder_model_merged.onnx_data",
            Asset {
                override_var: "VOXTRAL_ONNX_DECODER_MODEL_MERGED_ONNX_DATA",
                url: "https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX/resolve/db34633130c460bd650644fc18dc90710d9f3e38/onnx/decoder_model_merged.onnx_data",
                sha256: "810eb1f1ca855cb51d03acd4f823105d89b672813906d6ad97cc53f6e136a700",
            },
        ),
        (
            "decoder_model_merged.onnx_data_1",
            Asset {
                override_var: "VOXTRAL_ONNX_DECODER_MODEL_MERGED_ONNX_DATA_1",
                url: "https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX/resolve/db34633130c460bd650644fc18dc90710d9f3e38/onnx/decoder_model_merged.onnx_data_1",
                sha256: "0612da08998f4f1f0497c44b3d0fdf1167c7163d7b4da1f434277db26e147aa8",
            },
        ),
        (
            "decoder_model_merged.onnx_data_2",
            Asset {
                override_var: "VOXTRAL_ONNX_DECODER_MODEL_MERGED_ONNX_DATA_2",
                url: "https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX/resolve/db34633130c460bd650644fc18dc90710d9f3e38/onnx/decoder_model_merged.onnx_data_2",
                sha256: "10d0337efea9dc3f6fe04755ddca53de5afd51bccad5619b99bbeb3c438001d2",
            },
        ),
        (
            "decoder_model_merged.onnx_data_3",
            Asset {
                override_var: "VOXTRAL_ONNX_DECODER_MODEL_MERGED_ONNX_DATA_3",
                url: "https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX/resolve/db34633130c460bd650644fc18dc90710d9f3e38/onnx/decoder_model_merged.onnx_data_3",
                sha256: "283904c14f2780781ed44709738c2b5adcb5408a6673382e255bf3b716deca11",
            },
        ),
        (
            "decoder_model_merged.onnx_data_4",
            Asset {
                override_var: "VOXTRAL_ONNX_DECODER_MODEL_MERGED_ONNX_DATA_4",
                url: "https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX/resolve/db34633130c460bd650644fc18dc90710d9f3e38/onnx/decoder_model_merged.onnx_data_4",
                sha256: "d5d7743bcb9f8d3f81fbbf2c546c038d88052324bdded171817bcacc17576efe",
            },
        ),
        (
            "decoder_model_merged.onnx_data_5",
            Asset {
                override_var: "VOXTRAL_ONNX_DECODER_MODEL_MERGED_ONNX_DATA_5",
                url: "https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX/resolve/db34633130c460bd650644fc18dc90710d9f3e38/onnx/decoder_model_merged.onnx_data_5",
                sha256: "c02a4acf0e349c15ebadc4e55524aa5cd132d5d720f77f7a7f3657fb5c42d0f8",
            },
        ),
        (
            "decoder_model_merged.onnx_data_6",
            Asset {
                override_var: "VOXTRAL_ONNX_DECODER_MODEL_MERGED_ONNX_DATA_6",
                url: "https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX/resolve/db34633130c460bd650644fc18dc90710d9f3e38/onnx/decoder_model_merged.onnx_data_6",
                sha256: "bbaeaa6af7b301cb557fe00e2ae0b77b4c42f4b700e62061c3900048ecf922ee",
            },
        ),
    ],
    counts: &[
        ("GroupQueryAttention", 26),
        ("SimplifiedLayerNormalization", 1),
        ("SkipSimplifiedLayerNormalization", 52),
    ],
};

/// The token embedding.
const VOXTRAL_EMBED_TOKENS: Graph = Graph {
    stem: "voxtral_embed_tokens",
    graph: Asset {
        override_var: "VOXTRAL_ONNX_EMBED_TOKENS_ONNX",
        url: "https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX/resolve/db34633130c460bd650644fc18dc90710d9f3e38/onnx/embed_tokens.onnx",
        sha256: "c8da5ab11bdc58efa6773d9a460e44a7c0c3c84b7cbf5b4fe2d92cb524fc78b5",
    },
    data: &[(
        "embed_tokens.onnx_data",
        Asset {
            override_var: "VOXTRAL_ONNX_EMBED_TOKENS_ONNX_DATA",
            url: "https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX/resolve/db34633130c460bd650644fc18dc90710d9f3e38/onnx/embed_tokens.onnx_data",
            sha256: "fe317ed76f6ed59eb6d32f740795691c3cd010a13b6433fe71fbd7b98cadc15b",
        },
    )],
    counts: &[],
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=VOXTRAL_CACHE_DIR={}", out_dir().display());

    #[cfg(feature = "checkpoint")]
    for (file, asset) in CHECKPOINT {
        let dest = pretrained_cache_path(
            &out_dir(),
            VOXTRAL_KIT,
            MISTRAL_NAMESPACE,
            asset.sha256,
            file,
        );
        asset.resolve(&dest);
    }

    // `cfg` rather than a `CARGO_FEATURE_*` lookup: `burn_onnx` is an
    // optional build-dependency, so without the feature it is not linked.
    #[cfg(feature = "onnx_gen")]
    generate_reference();
}

/// Fetches the export, rewrites each graph, and generates the Rust models
/// from them.
#[cfg(feature = "onnx_gen")]
fn generate_reference() {
    let out = out_dir();
    // The generated weights are loaded from `OUT_DIR` at run time rather than
    // embedded.
    println!("cargo:rustc-env=VOXTRAL_ONNX_OUT_DIR={}", out.display());

    let mut generate = burn_onnx::ModelGen::new();
    for g in [
        &VOXTRAL_AUDIO_ENCODER,
        &VOXTRAL_DECODER,
        &VOXTRAL_EMBED_TOKENS,
    ] {
        let patched = rewrite(g, &out);
        generate.input(patched.to_str().expect("asset path is UTF-8"));
    }
    generate.out_dir("./").run_from_script();
}

/// Fetches one graph and its weights, rewrites it, and writes the rewritten
/// graph beside them as `<stem>.onnx`, the name `ModelGen` names its module
/// after.
#[cfg(feature = "onnx_gen")]
fn rewrite(
    g: &Graph,
    out: &std::path::Path,
) -> std::path::PathBuf {
    use rewrite_genai_onnx::onnx_ir::{
        Message,
        ModelProto,
    };

    let fetched = g
        .graph
        .resolve(&out.join(format!("{}.published.onnx", g.stem)));
    println!("cargo:rerun-if-changed={}", fetched.display());
    // The rewritten graph reads its external weights from beside itself,
    // under their remote names, which `onnx-ir` refuses through a symlink.
    for (name, asset) in g.data {
        let data = out.join(name);
        let found = asset.resolve(&data);
        place_file(&found, &data).unwrap_or_else(|e| panic!("{e}"));
    }

    let bytes =
        std::fs::read(&fetched).unwrap_or_else(|e| panic!("read {}: {e}", fetched.display()));
    let mut model = ModelProto::parse_from_bytes(&bytes)
        .unwrap_or_else(|e| panic!("parse {}: {e}", fetched.display()));
    drop(bytes);
    rewrite_genai_onnx::rewrite_for_burn_onnx(&mut model, g.counts);

    let patched = out.join(format!("{}.onnx", g.stem));
    std::fs::write(
        &patched,
        model.write_to_bytes().expect("serialize the graph"),
    )
    .unwrap_or_else(|e| panic!("write {}: {e}", patched.display()));
    patched
}
