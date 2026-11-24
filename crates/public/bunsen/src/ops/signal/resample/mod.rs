//! # Rate conversion
//!
//! The rate converters, each placing the windowed-sinc kernels of
//! [`kernels::sinc`](super::kernels::sinc) its own way:
//!
//! - [`StreamingResampler`]: a Blackman-windowed sinc, fed in any pieces, each
//!   output phase-aligned to the output rate and normalized to unit DC gain;
//!   [`StreamingResampler::resample`] converts a whole signal. A microphone
//!   opened at another rate, or one model's output read by another.
//! - [`HannSincResampler`]: `torchaudio`'s `sinc_interp_hann` conversion of a
//!   whole signal, for a model trained behind it; it reproduces the reference,
//!   sample for sample. Its kernels also run as a stream, the whole
//!   conversion's output in pieces
//!   ([`SincPhaseKernels::resample_stream`](super::kernels::sinc::SincPhaseKernels::resample_stream)).

mod hann_sinc;
mod streaming;

pub use hann_sinc::*;
pub use streaming::*;
