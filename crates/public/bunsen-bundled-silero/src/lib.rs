//! Bunsen `silero_vad` bundled model assets: Silero VAD's pretrained
//! weights, linked into the binary.
//!
//! The crate ships Silero's ONNX graph, `onnx/silero_vad_op18_ifless.onnx`
//! (2.8 MB, from [`snakers4/silero-vad`](https://github.com/snakers4/silero-vad),
//! MIT). Its build script converts the graph to a burnpack with
//! `burn-onnx`, offline: unlike `bunsen-bundled-whisper`, nothing is
//! fetched, and no feature is needed. The burnpack lands in `OUT_DIR` and is
//! linked in as [`BURNPACK_WEIGHTS`]. One burnpack holds both of the model's
//! sample-rate branches, 16 kHz and 8 kHz.
//!
//! The burnpack is this build's artifact, not upstream's, so the build also
//! hashes it, as [`BURNPACK_SHA256`]. bunsen's `silero-weights` feature
//! pulls this crate in, and bunsen's Silero kit serves the bytes as the
//! `bundled:silero/vad` row, pinned to that digest. The pretrained cache
//! writes them under `<cache>/pretrained/silero_vad/bunsen/<sha256>/` on
//! first use, so a rebuilt burnpack with different bytes gets its own slot.
//!
//! ## The reference model: `onnx_gen`
//!
//! `burn-onnx` also generates a Rust model from the graph. The `onnx_gen`
//! feature compiles it in as the `onnx_gen` module, whose
//! `Model::load_pretrained` reads [`BURNPACK_WEIGHTS`]. It is a direct
//! transliteration of the graph, so `silero-model-validation` checks
//! bunsen's hand-written Silero VAD against it. The burnpack is generated
//! either way; the feature decides only whether the generated model is
//! compiled. It is off by default, so docs.rs does not show the module.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

extern crate alloc;

/// The pretrained burnpack: both sample-rate branches, 16 kHz and 8 kHz,
/// generated from the ONNX graph when the crate was built.
pub const BURNPACK_WEIGHTS: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/silero_vad_op18_ifless.bpk"));

/// The burnpack's file name: what a pretrained resource calls it.
pub const BURNPACK_FILE: &str = "silero_vad_op18_ifless.bpk";

/// SHA-256 of [`BURNPACK_WEIGHTS`], computed when the crate was built.
///
/// The burnpack is generated from the ONNX graph by `burn-onnx`, so its
/// digest is this build's rather than upstream's; bunsen's Silero kit pins
/// its `bundled:silero/vad` row to it, so a rebuilt burnpack is cached in
/// its own slot.
pub const BURNPACK_SHA256: &str = env!("SILERO_BURNPACK_SHA256");

/// [`BURNPACK_WEIGHTS`], copied into burn's [`Bytes`](burn::tensor::Bytes).
pub fn burnpack_as_burn_bytes() -> burn::tensor::Bytes {
    burn::tensor::Bytes::from_bytes_vec(BURNPACK_WEIGHTS.to_vec())
}

#[cfg(feature = "onnx_gen")]
pub mod onnx_gen {
    #![allow(clippy::type_complexity)]
    //! The Rust model `burn-onnx` generates from the ONNX graph: the
    //! reference bunsen's Silero VAD is validated against.
    use super::*;

    include!(concat!(env!("OUT_DIR"), "/silero_vad_op18_ifless.rs"));

    impl<B: Backend> Model<B> {
        /// Load the pretrained model.
        pub fn load_pretrained(device: &B::Device) -> Model<B> {
            Model::from_bytes(burnpack_as_burn_bytes(), device)
        }
    }
}
