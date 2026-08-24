//! # ten-vad pitch estimation.
//!
//! Feature `40` of the ten-vad feature vector: a pitch estimate in Hz, `0.0`
//! when unvoiced (`ALGO_TRACE.md` §3.5).
//!
//! The driver reaches this branch through [`TenVadPitchSource`], which is
//! tensor-in, tensor-out so the rest of the front end can stay
//! device-resident.
//!
//! ## The pieces
//!
//! * [`TenVadPitchSource`] — the device seam the driver calls through, built by
//!   [`TenVadPitchSourceInit`].
//! * [`TenVadPitchEstimator`] — the port of the reference estimator, and the
//!   permanent oracle the rest of the branch is validated against.
//! * [`HostPitch`] — adapts a host-side [`TenVadPitchScalarSource`] to the
//!   device seam. **The only place in the front end that synchronizes.**
//! * [`ZeroPitch`] — a constant stub that skips the branch entirely.
//! * [`coeff`](self) — the reference constants.
//! * [`tensor`] — the device-side port, built stage by stage against the host
//!   estimator as its oracle.
//!
//! [`lpc`] holds the pre-filter design: band folding, the cepstrum, the
//! autocorrelation, and the Levinson-Durbin solve.
//!
//! ## Choosing a source
//!
//! [`TenVadPitchEstimator`] is the faithful choice: all 41 features then match
//! the reference. Because it is a serial recurrence over scalars, stepping it
//! means reading the raw hops and the bin powers back from the device — once
//! per `forward_sequence` call for the whole sequence, or once per hop on the
//! single-step path, which pays that cost anyway.
//!
//! [`ZeroPitch`] pins feature `40` to a constant and never inspects its
//! arguments, so the sequence path stays entirely on-device. The other 40
//! features are exact either way.
//!
//! ## Why the estimator is not a tensor op
//!
//! The reference estimator is four stages, and only the first is free of
//! carried state: an LPC fit (stateless), an excitation branch carrying a FIFO
//! and an IIR cascade, a lag search carrying a correlation ring, and a Viterbi
//! tracker carrying its accumulator across hops. The last is a genuine
//! recurrence over 56 states with two steps per hop.

mod coeff;
mod estimator;
mod host;
mod lpc;
mod source;

pub mod tensor;

#[doc(inline)]
pub use coeff::*;
#[doc(inline)]
pub use estimator::*;
#[doc(inline)]
pub use host::*;
#[doc(inline)]
pub use lpc::*;
#[doc(inline)]
pub use source::*;
