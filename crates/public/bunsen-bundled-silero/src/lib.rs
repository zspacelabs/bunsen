//! Bunsen `silero_vad` bundled model assets.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

extern crate alloc;

/// Pretrained dual branch 16khz/8khs model.
pub const BURNPACK_WEIGHTS: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/silero_vad_op18_ifless.bpk"));

/// Convert the burnpack weights to a burn bytes.
pub fn burnpack_as_burn_bytes() -> burn::tensor::Bytes {
    burn::tensor::Bytes::from_bytes_vec(BURNPACK_WEIGHTS.to_vec())
}

#[cfg(feature = "onnx_gen")]
pub mod onnx_gen {
    #![allow(clippy::type_complexity)]
    //! Bundled ONNX generated Model.
    use super::*;

    include!(concat!(env!("OUT_DIR"), "/silero_vad_op18_ifless.rs"));

    impl<B: Backend> Model<B> {
        /// Load the pretrained model.
        pub fn load_pretrained(device: &B::Device) -> Model<B> {
            Model::from_bytes(burnpack_as_burn_bytes(), device)
        }
    }
}
