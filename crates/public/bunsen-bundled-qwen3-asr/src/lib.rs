//! Bunsen Qwen3-ASR bundled model assets.
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
