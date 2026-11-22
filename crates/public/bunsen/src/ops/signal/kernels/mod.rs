//! # Signal kernels
//!
//! The coefficient math the signal operators apply, one family to a module:
//! each family is defined once, built on the host (`f64`, the reference) and
//! on a device (a [`Tensor`](burn::Tensor)), and the two are cross-tested in
//! the family's own module. The operators that apply a kernel
//! ([`SlidingStft`](super::SlidingStft), [`KaldiFbank`](super::KaldiFbank),
//! ...) live in [`ops::signal`](super), beside this tree.
//!
//! - [`window`]: analysis windows, the Hann, Hamming and Blackman families,
//!   materialized as `N`-point tables by [`window::SamplingWindowBuilder`] and
//!   picked as a config value by [`window::StftWindowConfig`], and as a
//!   function on `[-1, 1]` by [`window::WindowTaper`].
//! - [`sinc`]: windowed-sinc low-pass kernels over any taper, and the phase
//!   tables a rational rate conversion applies.
//! - [`mel`]: the mel scales and the mel filterbanks over an FFT's bins
//!   (`librosa`'s, Kaldi's).

pub mod mel;
pub mod sinc;
pub mod window;
