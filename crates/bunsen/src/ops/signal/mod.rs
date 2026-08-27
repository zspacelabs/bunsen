//! Tensor signal operations.

pub mod mels;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

mod cosine_window;
mod sliding_stft;
mod stft_window;
mod window_builder;

#[doc(inline)]
pub use cosine_window::*;
#[doc(inline)]
pub use sliding_stft::*;
#[doc(inline)]
pub use stft_window::*;
#[doc(inline)]
pub use window_builder::*;
