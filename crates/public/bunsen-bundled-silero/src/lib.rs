//! Bunsen `silero_vad` bundled model assets.
//!
//! **This crate hosts nothing.** It *fetches* Silero VAD's files from a
//! pinned revision of [`snakers4/silero-vad`](https://github.com/snakers4/silero-vad)
//! (MIT) at build time, pins each to a SHA-256, and keeps them in
//! `OUT_DIR`, each behind its own feature:
//!
//! | Item | Feature | What it is |
//! |---|---|---|
//! | `cache_dir()`, `silero_vad_16k_safetensors()` | `checkpoint` | upstream's 16 kHz `silero_vad_16k.safetensors`, laid out as a pretrained cache |
//! | `onnx_gen::{Model, burnpack_path}` | `onnx_gen` | the reference model `burn-onnx` generates from upstream's `silero_vad_op18_ifless.onnx`, and its burnpack |
//!
//! The checkpoint is laid out as a **pretrained cache**,
//! `pretrained/silero_vad/silero/<sha256>/<file>`, which is how bunsen's
//! `data::pretrained::PretrainedCache` roots a resource: a cache pointed at
//! `cache_dir()`, offline, resolves `silero/vad-16k` without knowing it was
//! bundled. `silero-model-validation`'s `weights` feature turns on
//! `checkpoint`.
//!
//! The reference is a direct transliteration of the graph, both sample-rate
//! branches, so `silero-model-validation` checks bunsen's hand-written Silero
//! VAD against it.
//!
//! `SILERO_VAD_16K_SAFETENSORS` and `SILERO_VAD_ONNX` name local copies
//! instead: a file that is the pinned asset is laid out where the download
//! would have gone, and any other is used where it is, with a warning.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

extern crate alloc;

/// `OUT_DIR` laid out as a pretrained cache: the root a
/// `bunsen::data::pretrained::PretrainedCache` is pointed at to resolve
/// `silero/vad-16k` offline.
#[cfg(feature = "checkpoint")]
pub fn cache_dir() -> &'static std::path::Path {
    std::path::Path::new(env!("SILERO_CACHE_DIR"))
}

/// The fetched `silero_vad_16k.safetensors`: the digest-pinned download, or
/// whatever `SILERO_VAD_16K_SAFETENSORS` pointed at.
#[cfg(feature = "checkpoint")]
pub fn silero_vad_16k_safetensors() -> &'static std::path::Path {
    std::path::Path::new(env!("SILERO_VAD_16K_SAFETENSORS_PATH"))
}

/// The reference model `burn-onnx` generates from upstream's
/// `silero_vad_op18_ifless.onnx`.
#[cfg(feature = "onnx_gen")]
pub mod onnx_gen {
    use burn::prelude::*;

    /// The burnpack `burn-onnx` generated from the graph, both sample-rate
    /// branches: what [`Model::load_pretrained`] reads, and what bunsen's
    /// burnpack loaders read to run bunsen's Silero VAD on the reference's
    /// weights.
    pub fn burnpack_path() -> std::path::PathBuf {
        std::path::Path::new(env!("SILERO_ONNX_OUT_DIR")).join("silero_vad_op18_ifless.bpk")
    }

    // Machine-generated: not held to this crate's lint bar.
    #[allow(warnings, clippy::all)]
    mod generated {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/silero_vad_op18_ifless.rs"));
    }
    pub use generated::Model;

    impl Model {
        /// Loads the reference model from the generated burnpack.
        pub fn load_pretrained(device: &Device) -> Model {
            Model::from_file(burnpack_path(), device)
        }
    }
}
