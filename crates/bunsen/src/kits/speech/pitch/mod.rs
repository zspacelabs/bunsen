//! # Time-domain pitch estimation.
//!
//! A pitch estimate in Hz per hop, `0.0` when unvoiced. Four stages:
//!
//! 1. **pre-filter design** — fold the power spectrum into bands, round-trip
//!    through a cepstrum, and solve for an LPC whitening filter. Stateless.
//! 2. **excitation** — whiten the raw hop with that filter, smooth, and
//!    decimate by four. Carries a FIFO and the anti-alias filter's state.
//! 3. **lag search** — normalized cross-correlation over every candidate
//!    period, with octave suppression. Carries a correlation ring.
//! 4. **tracking** — a Viterbi pass over the period candidates, then a weighted
//!    fit over the recovered contour. Carries the accumulator.
//!
//! The design follows `LPCNet`'s `compute_frame_features` /
//! `process_superframe` (Xiph/Mozilla, BSD-2-Clause), and the coefficient
//! tables in [`coeff`] come from that lineage. They are reference data rather
//! than tunables: a consumer whose downstream model was fitted against these
//! values cannot change them without refitting.
//!
//! ## Host and device
//!
//! Both forms are in the tree and cross-tested against each other:
//!
//! * [`HostPitchEstimator`] — scalar, one stream, a serial recurrence. The
//!   oracle every device stage is validated against, and the cheaper choice for
//!   a caller stepping hop by hop.
//! * [`tensor`] — the device port, built stage by stage against that oracle.
//!   Batches over streams and keeps the whole estimate resident; it is built
//!   for sequences, so a single-hop call pays sequence-shaped setup.
//!
//! [`PitchSource`] is the seam both satisfy — tensor in, tensor out, so a
//! caller's pipeline can stay device-resident — with [`HostPitch`] adapting a
//! scalar [`PitchScalarSource`] to it at the cost of one readback per call.
//! [`ZeroPitch`] reports silence and never inspects its arguments, for callers
//! that want the branch gone.
//!
//! ## What is shared, and what is here
//!
//! The reusable machinery lives in [`ops::signal`](crate::ops::signal) and
//! [`ops::seq`](crate::ops::seq): the biquad cascade and decimating FIR, the
//! autocorrelator and both Levinson forms, the LPC analysis filter, the
//! triangular filterbank, the normalized lag search, and the Viterbi decoder.
//! What remains here is the *assembly* — the geometry that ties them together,
//! the band tables, the octave suppression, and the period fit.

pub mod coeff;
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
