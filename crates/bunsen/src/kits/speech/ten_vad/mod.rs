//! ten-vad model.

/// The reference model.
#[cfg(feature = "store")]
pub mod reference {
    pub use bunsen_onnx_gen::ten::*;

    /// Reference ONNX Model.
    pub type ReferenceModel<B> = Model<B>;
}

#[cfg(feature = "store")]
mod cross_test;
#[cfg(feature = "store")]
pub mod pretrained;

pub mod blocks;

#[doc(inline)]
pub use blocks::*;
