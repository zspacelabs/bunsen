//! Bunsen Qwen3-ASR bundled model assets.
//!
//! **This crate hosts nothing.** `FermionResearch`'s Phonon-1 builds are
//! 285 to 581 MB, Qwen's `Qwen3-ASR-0.6B` 1.6 GB, and its ONNX export 3.8 GB,
//! all far too large to commit, so this crate *fetches* them, pins each file
//! to a SHA-256, and keeps them in `OUT_DIR`.
//!
//! The pretrained assets are laid out in `OUT_DIR` as a **pretrained
//! cache**, `pretrained/qwen3-asr/<namespace>/<sha256>/<file>`, which is how
//! bunsen's `data::pretrained::PretrainedCache` roots a resource. A cache
//! pointed at `cache_dir()`, offline, hits the fetched rows without knowing
//! they were bundled.
//!
//! ## What it exposes
//!
//! Each item exists only under the feature that fetches its asset:
//!
//! | Item | Feature | What it is |
//! |---|---|---|
//! | `cache_dir()` | any of the checkpoint features | `OUT_DIR`, laid out as a pretrained cache |
//! | `phonon_1_archive()` | `checkpoint` | Phonon-1's `phonon-audio6.bps.tar.zst` |
//! | `phonon_1_micro_archive()`, `phonon_1_big_archive()` | `variant_checkpoints` | Phonon-1-Micro's and Phonon-1-Big's archives |
//! | `qwen3_asr_safetensors()`, `qwen3_asr_tokenizer()` | `base_checkpoint` | `Qwen3-ASR-0.6B-hf`'s `model.safetensors` and `tokenizer.json` |
//! | `onnx_gen::*` | `onnx_gen` | reference models generated from `OpenVoiceOS/qwen3-asr-0.6b-onnx` |
//!
//! ## Using local copies
//!
//! Each asset has a variable naming a local file to use instead, read only
//! when the feature that fetches it is on: `PHONON_1_ARCHIVE`,
//! `PHONON_1_MICRO_ARCHIVE`, `PHONON_1_BIG_ARCHIVE`, `QWEN3_ASR_SAFETENSORS`,
//! `QWEN3_ASR_TOKENIZER`, `QWEN3_ASR_ONNX_ENCODER`, `QWEN3_ASR_ONNX_EMBED`,
//! `QWEN3_ASR_ONNX_DECODER` and `QWEN3_ASR_ONNX_DECODER_DATA`. A file that is
//! the pinned asset is put where the download would have gone; any other file
//! is used where it is, with a warning.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

/// `OUT_DIR` laid out as a pretrained cache: the root a
/// `bunsen::data::pretrained::PretrainedCache` is pointed at to read the
/// bundled assets. A cache that fetches other models into it will change it,
/// so a test wants it offline.
#[cfg(any(
    feature = "checkpoint",
    feature = "variant_checkpoints",
    feature = "base_checkpoint"
))]
pub fn cache_dir() -> &'static std::path::Path {
    std::path::Path::new(env!("QWEN3_ASR_CACHE_DIR"))
}

/// The fetched `phonon-audio6.bps.tar.zst`: `FermionResearch`'s Phonon-1.
///
/// Resolved at build time: the digest-pinned download, or whatever
/// `PHONON_1_ARCHIVE` pointed at.
#[cfg(feature = "checkpoint")]
pub fn phonon_1_archive() -> &'static std::path::Path {
    std::path::Path::new(env!("PHONON_1_ARCHIVE_PATH"))
}

/// The fetched `phonon-micro.bps.tar.zst`: Phonon-1-Micro.
///
/// Resolved at build time, as [`phonon_1_archive`] (`PHONON_1_MICRO_ARCHIVE`).
#[cfg(feature = "variant_checkpoints")]
pub fn phonon_1_micro_archive() -> &'static std::path::Path {
    std::path::Path::new(env!("PHONON_1_MICRO_ARCHIVE_PATH"))
}

/// The fetched `phonon-parity.bps.tar.zst`: Phonon-1-Big.
///
/// Resolved at build time, as [`phonon_1_archive`] (`PHONON_1_BIG_ARCHIVE`).
#[cfg(feature = "variant_checkpoints")]
pub fn phonon_1_big_archive() -> &'static std::path::Path {
    std::path::Path::new(env!("PHONON_1_BIG_ARCHIVE_PATH"))
}

/// The fetched `model.safetensors` of Qwen's `Qwen3-ASR-0.6B-hf`.
///
/// Resolved at build time, as [`phonon_1_archive`] (`QWEN3_ASR_SAFETENSORS`).
#[cfg(feature = "base_checkpoint")]
pub fn qwen3_asr_safetensors() -> &'static std::path::Path {
    std::path::Path::new(env!("QWEN3_ASR_SAFETENSORS_PATH"))
}

/// The fetched `tokenizer.json` of Qwen's `Qwen3-ASR-0.6B-hf`.
///
/// Resolved at build time, as [`phonon_1_archive`] (`QWEN3_ASR_TOKENIZER`).
#[cfg(feature = "base_checkpoint")]
pub fn qwen3_asr_tokenizer() -> &'static std::path::Path {
    std::path::Path::new(env!("QWEN3_ASR_TOKENIZER_PATH"))
}

/// The ONNX-generated reference models.
#[cfg(feature = "onnx_gen")]
pub mod onnx_gen {
    use burn::prelude::*;

    fn out_dir() -> &'static std::path::Path {
        std::path::Path::new(env!("QWEN3_ASR_ONNX_OUT_DIR"))
    }

    /// The reference audio encoder.
    // Machine-generated: not held to this crate's lint bar.
    #[allow(warnings, clippy::all)]
    mod encoder {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/qwen3_asr_encoder.rs"));

        impl Model {
            /// Loads the reference encoder from the generated weights.
            pub fn load_pretrained(device: &Device) -> Self {
                Self::from_file(out_dir().join("qwen3_asr_encoder.bpk"), device)
            }
        }
    }
    pub use encoder::Model as EncoderModel;

    /// The reference token embedding.
    #[allow(warnings, clippy::all)]
    mod embed {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/qwen3_asr_embed.rs"));

        impl Model {
            /// Loads the reference embedding from the generated weights.
            pub fn load_pretrained(device: &Device) -> Self {
                Self::from_file(out_dir().join("qwen3_asr_embed.bpk"), device)
            }
        }
    }
    pub use embed::Model as EmbedModel;

    /// The reference text decoder.
    #[allow(warnings, clippy::all)]
    mod decoder {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/qwen3_asr_decoder.rs"));

        impl Model {
            /// Loads the reference decoder from the generated weights.
            pub fn load_pretrained(device: &Device) -> Self {
                Self::from_file(out_dir().join("qwen3_asr_decoder.bpk"), device)
            }
        }
    }
    pub use decoder::Model as DecoderModel;
}
