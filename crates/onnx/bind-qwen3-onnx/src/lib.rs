//! Bunsen Qwen3 bundled model assets.
//!
//! **This crate hosts nothing.** Qwen's `Qwen3-0.6B` is 1.5 GB, its larger
//! checkpoints 4 to 8 GB, and the ONNX export of `Qwen3-0.6B` 2.4 GB, all far
//! too large to commit, so this crate *fetches* them, pins each file to a
//! SHA-256, and keeps them in `OUT_DIR`.
//!
//! The pretrained assets are laid out in `OUT_DIR` as a **pretrained
//! cache**, `pretrained/qwen3/qwen/<sha256>/<file>`, which is how bunsen's
//! `data::pretrained::PretrainedCache` roots a resource. A cache pointed at
//! `cache_dir()`, offline, hits the fetched rows without knowing they were
//! bundled.
//!
//! ## What it exposes
//!
//! Each item exists only under the feature that fetches its asset:
//!
//! | Item | Feature | What it is |
//! |---|---|---|
//! | `cache_dir()` | `checkpoint` or `large_checkpoints` | `OUT_DIR`, laid out as a pretrained cache |
//! | `onnx_gen::Qwen3OnnxModel` | `onnx_gen` | the reference model generated from `onnx-community/Qwen3-0.6B-ONNX` |
//!
//! ## Using local copies
//!
//! Each file has a variable naming a local file to use instead, read only
//! when the feature that fetches it is on: the row and the file, in capitals
//! with `_` for `-` and `.` (`QWEN3_0_6B_MODEL_SAFETENSORS`,
//! `QWEN3_4B_MODEL_00001_OF_00003_SAFETENSORS`), and `QWEN3_ONNX_MODEL` and
//! `QWEN3_ONNX_MODEL_DATA` for the export. A file that is the pinned asset is
//! put where the download would have gone; any other file is used where it
//! is, with a warning.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

/// `OUT_DIR` laid out as a pretrained cache: the root a
/// `bunsen::data::pretrained::PretrainedCache` is pointed at to read the
/// bundled rows. A cache that fetches other models into it will change it,
/// so a test wants it offline.
#[cfg(any(feature = "checkpoint", feature = "large_checkpoints"))]
pub fn cache_dir() -> &'static std::path::Path {
    std::path::Path::new(env!("QWEN3_CACHE_DIR"))
}

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
