//! Bunsen Qwen3 reference assets.
//!
//! **This crate hosts nothing.** The ONNX export of `Qwen3-0.6B` is 2.4 GB,
//! far too large to commit, so this crate *fetches* it under `onnx_gen`, pins
//! each file to a SHA-256, keeps them in `OUT_DIR`, and generates the
//! reference model from it: `onnx_gen::Qwen3OnnxModel`. The checkpoints are
//! not here; `qwen3-model-validation` loads them through bunsen's pinned rows.
//!
//! `QWEN3_ONNX_MODEL` and `QWEN3_ONNX_MODEL_DATA` name local copies of the
//! export's files instead. A file that is the pinned asset is put where the
//! download would have gone; any other file is used where it is, with a
//! warning.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

/// The reference model, generated from the rewritten export.
#[cfg(feature = "onnx_gen")]
pub mod onnx_gen {
    use burn::prelude::*;

    fn out_dir() -> &'static std::path::Path {
        std::path::Path::new(env!("QWEN3_ONNX_OUT_DIR"))
    }

    /// The reference decoder: `input_ids`, `attention_mask` and
    /// `position_ids`, then each layer's past key and value, to the logits
    /// and each layer's present key and value.
    // Machine-generated: not held to this crate's lint bar.
    #[allow(warnings, clippy::all)]
    mod qwen3 {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/qwen3.rs"));

        impl Model {
            /// Loads the reference model from the generated weights.
            pub fn load_pretrained(device: &Device) -> Self {
                Self::from_file(out_dir().join("qwen3.bpk"), device)
            }
        }
    }
    pub use qwen3::Model as Qwen3OnnxModel;
}
