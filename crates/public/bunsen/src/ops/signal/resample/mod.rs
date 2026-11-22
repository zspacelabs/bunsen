//! # Rate conversion
//!
//! The rate converters, each placing the windowed-sinc kernels of
//! [`kernels::sinc`](super::kernels::sinc) its own way.
//! [`HannSincResampler`]: `torchaudio`'s `sinc_interp_hann` conversion of a
//! whole signal, for a model trained behind it; it reproduces the reference,
//! sample for sample.

mod hann_sinc;

pub use hann_sinc::*;
