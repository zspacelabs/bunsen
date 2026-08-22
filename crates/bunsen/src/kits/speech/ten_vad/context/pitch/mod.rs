//! # ten-vad pitch estimation.
//!
//! Feature `40` of the ten-vad feature vector: a pitch estimate in Hz, `0.0`
//! when unvoiced (`ALGO_TRACE.md` §3.5).
//!
//! Unlike the rest of the front end this branch does not vectorize. The
//! reference estimator is a deeply serial recurrence over scalars — an LPC
//! fit, an IIR cascade, a lag search, and a Viterbi tracker, each carrying
//! state across hops — so it runs host-side, per stream, behind the
//! [`TenVadPitchSource`] seam.
//!
//! ## The pieces
//!
//! * [`TenVadPitchSource`] — the seam the driver calls through.
//! * [`TenVadPitchEstimator`] — the port of the reference estimator.
//! * [`ZeroPitch`] — a constant stub that skips the branch entirely.
//! * [`BiquadCascade`] — the anti-alias filter before decimation.
//! * [`coeff`](self) — the reference constants.
//!
//! [`lpc`] holds the pre-filter design: band folding, the cepstrum, the
//! autocorrelation, and the Levinson-Durbin solve.
//!
//! ## Choosing a source
//!
//! [`TenVadPitchEstimator`] is the faithful choice and what
//! [`TenVadContext`](super::TenVadContext) should carry for reference
//! parity. It costs a device-to-host readback of the raw hop and the bin
//! powers on every frame, which pins the sequence path to a host-side walk.
//!
//! [`ZeroPitch`] pins feature `40` to a constant and reports
//! [`inspects_input`](TenVadPitchSource::inspects_input) as `false`, letting
//! the driver keep the whole sequence path on-device. The other 40 features
//! are exact either way.

mod biquad;
mod coeff;
mod estimator;
mod lpc;
mod source;

#[doc(inline)]
pub use biquad::*;
#[doc(inline)]
pub use coeff::*;
#[doc(inline)]
pub use estimator::*;
#[doc(inline)]
pub use lpc::*;
#[doc(inline)]
pub use source::*;
