//! Fetches the Parakeet TDT assets.
//!
//! Three sets, each behind its own feature: `checkpoint` is
//! `FermionResearch`'s Phonon-2 archive, `base_checkpoint` is NVIDIA's
//! `parakeet-tdt-0.6b-v3` as `transformers` ships it, both laid out as a
//! bunsen pretrained cache, and `onnx_gen` is the `istupakov` ONNX export of
//! NVIDIA's `parakeet-tdt-0.6b-v3` that `parakeet-model-validation` compares
//! bunsen against: its encoder graph and its prediction-and-joint graph. The
//! export's feature-extractor graph (`nemo128.onnx`) is not generated:
//! `burn-onnx` 0.22 cannot infer its types (`ArgType::Shape has no DType`).
//!
//! Fetching, pinning and the `PHONON_2_ARCHIVE`, `PARAKEET_SAFETENSORS`,
//! `PARAKEET_TOKENIZER`, `PARAKEET_ONNX_ENCODER`, `PARAKEET_ONNX_ENCODER_DATA`
//! and `PARAKEET_ONNX_DECODER_JOINT` overrides follow `bunsen-fetch`'s
//! rules.

// Each feature uses its own constants; the rest are dead in that build.
#![allow(unused)]

use bunsen_fetch::{
    Asset,
    out_dir,
    place_file,
    pretrained_cache_path,
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

/// The Parakeet kit's segment of the pretrained cache layout: `PARAKEET_KIT`
/// in bunsen.
const PARAKEET_KIT: &str = "parakeet";

/// A pretrained asset: its namespace, file name, and the compile-time
/// variable naming its path.
struct Pretrained {
    asset: Asset<'static>,
    namespace: &'static str,
    file: &'static str,
    path_var: &'static str,
}

/// `FermionResearch/Phonon-2`'s archive, at the revision bunsen's
/// `fermion/phonon-2` row pins.
const PHONON_2: Pretrained = Pretrained {
    asset: Asset {
        override_var: "PHONON_2_ARCHIVE",
        url: "https://huggingface.co/FermionResearch/Phonon-2/resolve/7e153e4054c0a10db6d47d6fff5c062d4c77c154/phonon-2.bps.tar.zst",
        sha256: "98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e",
    },
    namespace: "fermion",
    file: "phonon-2.bps.tar.zst",
    path_var: "PHONON_2_ARCHIVE_PATH",
};

/// `nvidia/parakeet-tdt-0.6b-v3`'s checkpoint, at the revision bunsen's
/// `nvidia/parakeet-tdt-0.6b-v3` row pins.
const BASE_SAFETENSORS: Pretrained = Pretrained {
    asset: Asset {
        override_var: "PARAKEET_SAFETENSORS",
        url: "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/model.safetensors",
        sha256: "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11",
    },
    namespace: "nvidia",
    file: "model.safetensors",
    path_var: "PARAKEET_SAFETENSORS_PATH",
};

/// `nvidia/parakeet-tdt-0.6b-v3`'s vocabulary.
const BASE_TOKENIZER: Pretrained = Pretrained {
    asset: Asset {
        override_var: "PARAKEET_TOKENIZER",
        url: "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/tokenizer.json",
        sha256: "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91",
    },
    namespace: "nvidia",
    file: "tokenizer.json",
    path_var: "PARAKEET_TOKENIZER_PATH",
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=PARAKEET_CACHE_DIR={}", out_dir().display());

    #[cfg(feature = "checkpoint")]
    fetch_pretrained(&PHONON_2);

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
        PARAKEET_KIT,
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
