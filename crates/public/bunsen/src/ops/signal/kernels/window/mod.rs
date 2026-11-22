//! # Analysis windows
//!
//! [`SamplingWindowBuilder`] materializes a window as `N` points, as a
//! `Vec<f64>` on the host or as a tensor on a device. [`CosineWindowConfig`]
//! (Hann, Hamming) and [`DualCosineWindow`] (Blackman) are the families, and
//! [`StftWindowConfig`] picks one as a config value. [`WindowTaper`] is each
//! family as a function on `[-1, 1]`, of which a table is a sampling: what a
//! windowed-sinc kernel reads at fractional positions.
//!
//! The `testing` submodule (with the `testing` feature) holds the cross-test
//! of a builder's two materializations.

#[cfg(any(test, feature = "testing"))]
pub mod testing;

mod cosine_window;
mod stft_window;
mod taper;
mod window_builder;

pub use cosine_window::*;
pub use stft_window::*;
pub use taper::*;
pub use window_builder::*;
