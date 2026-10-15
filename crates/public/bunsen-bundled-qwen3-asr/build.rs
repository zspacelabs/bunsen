//! Fetches the Qwen3-ASR assets.
//!
//! Four sets, each behind its own feature: `checkpoint` is
//! `FermionResearch`'s Phonon-1 archive, `variant_checkpoints` its Micro and
//! Big builds, `base_checkpoint` Qwen's `Qwen3-ASR-0.6B` as `transformers`
//! ships it, all laid out as a bunsen pretrained cache, and `onnx_gen` is the
//! `OpenVoiceOS` ONNX export of `Qwen3-ASR-0.6B` that
//! `qwen3-asr-model-validation` compares bunsen against: its audio encoder,
//! its token embedding, and its merged prefill-and-step text decoder.
//!
//! Fetching, pinning and the `PHONON_1_ARCHIVE`, `PHONON_1_MICRO_ARCHIVE`,
//! `PHONON_1_BIG_ARCHIVE`, `QWEN3_ASR_SAFETENSORS`, `QWEN3_ASR_TOKENIZER`,
//! `QWEN3_ASR_ONNX_ENCODER`,
//! `QWEN3_ASR_ONNX_EMBED`, `QWEN3_ASR_ONNX_DECODER` and
//! `QWEN3_ASR_ONNX_DECODER_DATA` overrides follow `bunsen-asset-fetch`'s
//! rules.

// Each feature uses its own constants; the rest are dead in that build.
#![allow(unused)]

use bunsen_asset_fetch::{
    Asset,
    out_dir,
    place_file,
    pretrained_cache_path,
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

/// The Qwen3-ASR kit's segment of the pretrained cache layout:
/// `QWEN3_ASR_KIT` in bunsen.
const QWEN3_ASR_KIT: &str = "qwen3-asr";

/// A pretrained asset: its namespace, file name, and the compile-time
/// variable naming its path.
struct Pretrained {
    asset: Asset<'static>,
    namespace: &'static str,
    file: &'static str,
    path_var: &'static str,
}

/// `FermionResearch/Phonon-1`'s archive, at the revision bunsen's
/// `fermion/phonon-1` row pins.
const PHONON_1: Pretrained = Pretrained {
    asset: Asset {
        override_var: "PHONON_1_ARCHIVE",
        url: "https://huggingface.co/FermionResearch/Phonon-1/resolve/68f96e309cd19bf5b9b72c0ead80eed92a499eae/phonon-audio6.bps.tar.zst",
        sha256: "214c3b45aa57257013811a53f99a905848466ad4ab21f2b2f8368f7ac79427b2",
    },
    namespace: "fermion",
    file: "phonon-audio6.bps.tar.zst",
    path_var: "PHONON_1_ARCHIVE_PATH",
};

/// `FermionResearch/Phonon-1-Micro`'s archive, at the revision bunsen's
/// `fermion/phonon-1-micro` row pins.
const PHONON_1_MICRO: Pretrained = Pretrained {
    asset: Asset {
        override_var: "PHONON_1_MICRO_ARCHIVE",
        url: "https://huggingface.co/FermionResearch/Phonon-1-Micro/resolve/e916a4f8e30ed5c3ddfc1808429f049669bd8bf3/phonon-micro.bps.tar.zst",
        sha256: "8a03e55d78e65c10f6ce22aff7e32359ca6a7aaa27973435b9ebc3cdbff13f9c",
    },
    namespace: "fermion",
    file: "phonon-micro.bps.tar.zst",
    path_var: "PHONON_1_MICRO_ARCHIVE_PATH",
};

/// `FermionResearch/Phonon-1-Big`'s archive, at the revision bunsen's
/// `fermion/phonon-1-big` row pins.
const PHONON_1_BIG: Pretrained = Pretrained {
    asset: Asset {
        override_var: "PHONON_1_BIG_ARCHIVE",
        url: "https://huggingface.co/FermionResearch/Phonon-1-Big/resolve/74ec0333ac8a5ede5e855489c069ef763bc99edb/phonon-parity.bps.tar.zst",
        sha256: "5dbd566d022a05bac045d1e84f0bf6188999494070e2a00dc5fd1bac6caffdb9",
    },
    namespace: "fermion",
    file: "phonon-parity.bps.tar.zst",
    path_var: "PHONON_1_BIG_ARCHIVE_PATH",
};

/// `Qwen/Qwen3-ASR-0.6B-hf`'s checkpoint, at the revision bunsen's
/// `qwen/qwen3-asr-0.6b` row pins.
const BASE_SAFETENSORS: Pretrained = Pretrained {
    asset: Asset {
        override_var: "QWEN3_ASR_SAFETENSORS",
        url: "https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/resolve/7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c/model.safetensors",
        sha256: "d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35",
    },
    namespace: "qwen",
    file: "model.safetensors",
    path_var: "QWEN3_ASR_SAFETENSORS_PATH",
};

/// `Qwen/Qwen3-ASR-0.6B-hf`'s vocabulary.
const BASE_TOKENIZER: Pretrained = Pretrained {
    asset: Asset {
        override_var: "QWEN3_ASR_TOKENIZER",
        url: "https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/resolve/7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c/tokenizer.json",
        sha256: "fe1fad59be22a41ee293363fcf95fdedbc7c93f3b49270b1d2e18bd1399a7a05",
    },
    namespace: "qwen",
    file: "tokenizer.json",
    path_var: "QWEN3_ASR_TOKENIZER_PATH",
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rustc-env=QWEN3_ASR_CACHE_DIR={}",
        out_dir().display()
    );

    #[cfg(feature = "checkpoint")]
    fetch_pretrained(&PHONON_1);

    #[cfg(feature = "variant_checkpoints")]
    for p in [&PHONON_1_MICRO, &PHONON_1_BIG] {
        fetch_pretrained(p);
    }

    #[cfg(feature = "base_checkpoint")]
    for p in [&BASE_SAFETENSORS, &BASE_TOKENIZER] {
        fetch_pretrained(p);
    }

    // `cfg` rather than a `CARGO_FEATURE_*` lookup: `burn_onnx` is an
    // optional build-dependency, so without the feature it is not linked.
    #[cfg(feature = "onnx_gen")]
    generate_reference();
}

/// Fetches a pretrained asset into the cache layout and names its path as a
/// compile-time variable.
fn fetch_pretrained(p: &Pretrained) {
    let dest = pretrained_cache_path(
        &out_dir(),
        QWEN3_ASR_KIT,
        p.namespace,
        p.asset.sha256,
        p.file,
    );
    let path = p.asset.resolve(&dest);
    println!("cargo:rustc-env={}={}", p.path_var, path.display());
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
        place_file(&found, &dest);
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
