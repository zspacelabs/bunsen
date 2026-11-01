//! Bunsen Qwen3-ASR reference assets.
//!
//! **This crate hosts nothing.** The ONNX export of Qwen's `Qwen3-ASR-0.6B`
//! is 3.8 GB, far too large to commit, so this crate *fetches* it under
//! `onnx_gen`, pins each file to a SHA-256, keeps them in `OUT_DIR`, and
//! generates the reference models from it: `onnx_gen::{EncoderModel,
//! EmbedModel, DecoderModel}`. The checkpoints are not here;
//! `qwen3-asr-model-validation`'s tests load them through bunsen's pinned
//! rows, at run time, into the weights cache.
//!
//! `QWEN3_ASR_ONNX_ENCODER`, `QWEN3_ASR_ONNX_EMBED`, `QWEN3_ASR_ONNX_DECODER`
//! and `QWEN3_ASR_ONNX_DECODER_DATA` name local copies of the export's files
//! instead. A file that is the pinned asset is put where the download would
//! have gone; any other file is used where it is, with a warning.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

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
