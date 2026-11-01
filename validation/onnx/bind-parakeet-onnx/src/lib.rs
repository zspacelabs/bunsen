//! Bunsen Parakeet TDT reference assets.
//!
//! **This crate hosts nothing.** The ONNX export of NVIDIA's
//! `parakeet-tdt-0.6b-v3` is 2.5 GB, far too large to commit, so this crate
//! *fetches* it under `onnx_gen`, pins each file to a SHA-256, keeps them in
//! `OUT_DIR`, and generates the reference models from it:
//! `onnx_gen::{EncoderModel, DecoderJointModel}`. The checkpoints are not
//! here; `parakeet-model-validation`'s tests load them through bunsen's
//! pinned rows, at run time, into the weights cache.
//!
//! ## Using local copies: the `PARAKEET_ONNX_*` variables
//!
//! A build without the network, or one that should not download gigabytes
//! again after a `cargo clean`, can point the build script at local files.
//! Each variable replaces one asset, and is read only when `onnx_gen` is on:
//!
//! | Variable | Replaces |
//! |---|---|
//! | `PARAKEET_ONNX_ENCODER` | `encoder-model.onnx`, behind `onnx_gen::EncoderModel` |
//! | `PARAKEET_ONNX_ENCODER_DATA` | `encoder-model.onnx.data`, the encoder's external weights |
//! | `PARAKEET_ONNX_DECODER_JOINT` | `decoder_joint-model.onnx`, behind `onnx_gen::DecoderJointModel` |
//!
//! The variable must name a file, or the build fails. Nothing is downloaded
//! for that asset. A file that is the pinned asset is put where the download
//! would have gone (a symlink on Unix, a copy elsewhere); any other file is
//! used where it is, with a warning.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

/// The ONNX-generated reference models.
///
/// A transliteration of `istupakov/parakeet-tdt-0.6b-v3-onnx`, `NeMo`'s own
/// export of NVIDIA's `parakeet-tdt-0.6b-v3`. The two graphs are the two
/// model stages of a transcription; the export's feature extractor is not
/// generated, as `burn-onnx` cannot import it.
///
/// | Model | Inputs | Outputs |
/// |---|---|---|
/// | [`EncoderModel`](onnx_gen::EncoderModel) | `audio_signal [B, 128, T]`, `length [B]` | `outputs [B, 1024, T/8]`, `encoded_lengths [B]` |
/// | [`DecoderJointModel`](onnx_gen::DecoderJointModel) | `encoder_outputs [B, 1024, 1]`, `targets [B, 1]`, `target_length [B]`, two LSTM states `[2, B, 640]` | `outputs [B, 1, 1, 8198]`, `prednet_lengths [B]`, two LSTM states |
///
/// The joint's 8198 outputs are 8192 tokens, the blank (8192), and the five
/// TDT duration logits for durations 0 through 4.
///
/// Weights are loaded from `OUT_DIR` at run time rather than embedded.
#[cfg(feature = "onnx_gen")]
pub mod onnx_gen {
    use burn::prelude::*;

    /// The directory the build generated the weights into.
    fn out_dir() -> &'static std::path::Path {
        std::path::Path::new(env!("PARAKEET_ONNX_OUT_DIR"))
    }

    /// The reference `FastConformer` encoder.
    // Machine-generated: not held to this crate's lint bar.
    #[allow(warnings, clippy::all)]
    mod encoder {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/parakeet_encoder.rs"));

        impl Model {
            /// Loads the reference encoder from the generated weights.
            pub fn load_pretrained(device: &Device) -> Self {
                Self::from_file(out_dir().join("parakeet_encoder.bpk"), device)
            }
        }
    }
    pub use encoder::Model as EncoderModel;

    /// The reference TDT prediction network and joint, one step at a time.
    #[allow(warnings, clippy::all)]
    mod decoder_joint {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/parakeet_decoder_joint.rs"));

        impl Model {
            /// Loads the reference prediction network and joint from the
            /// generated weights.
            pub fn load_pretrained(device: &Device) -> Self {
                Self::from_file(out_dir().join("parakeet_decoder_joint.bpk"), device)
            }
        }
    }
    pub use decoder_joint::Model as DecoderJointModel;
}

#[cfg(test)]
mod tests {
    /// Without a feature the crate is deliberately empty; this documents that
    /// rather than leaving a suite that silently has nothing in it.
    #[cfg(not(feature = "onnx_gen"))]
    #[test]
    #[ignore = "no assets fetched; rerun with --features onnx_gen"]
    fn test_assets_need_a_feature() {}
}
