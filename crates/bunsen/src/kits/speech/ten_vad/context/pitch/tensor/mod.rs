//! # The ten-vad pitch estimator, as tensor ops.
//!
//! A device-side implementation of the reference pitch estimator, built
//! stage by stage against
//! [`TenVadPitchEstimator`](super::TenVadPitchEstimator) as its oracle. The
//! host implementation stays: it is pinned to the C reference by
//! `testdata/ten/pitch.json`, and every stage here is validated differentially
//! against the corresponding stage there.
//!
//! ## Why this exists
//!
//! The host estimator is a serial recurrence over scalars, so the driver has
//! to read the raw hops and the bin powers back from the device to step it —
//! a host island in the middle of an otherwise device-resident front end. It
//! is also one instance per stream walked serially, which will not scale when
//! the batch axis opens up. A tensor implementation batches over streams for
//! free.
//!
//! ## The shape of the problem
//!
//! Four stages, and they are not alike:
//!
//! | stage | carried state | parallel across a sequence |
//! |---|---|---|
//! | 1 · pre-filter design | **none** | **yes, fully** |
//! | 2 · excitation | FIFO, smoother, anti-alias filter | partly |
//! | 3 · correlation | correlation ring, per-slot energies | yes, given the history |
//! | 4 · tracking | Viterbi accumulator and backpointers | **no** |
//!
//! Of the state that stages 2-4 carry, most is a sliding window and folds
//! into the same "prepend the carry, run the batched op once, re-slice the
//! tail" idiom [`SlidingStftContext`](crate::ops::signal::SlidingStftContext)
//! already uses. Two things genuinely are not: the anti-alias filter's IIR
//! state, and the Viterbi triple. Do not assume the whole port is a `cat`.
//!
//! ## Numerics
//!
//! Stage boundaries are `f32` tensors; internal precision is each stage's own
//! business. The formulations here were chosen to add no error of their own —
//! see [`prefilter`] for why the Levinson recursion is masked rather than
//! branched, and [`tables`] for why the autocorrelation is folded on the host
//! in `f64` rather than contracted over 513 bins on the device.
//!
//! That matters more than it looks. The tracker's output passes through an
//! `argmax` over 56 states and a threshold on the frame correlation, and a
//! single `argmax` step of ±1 moves the reported pitch by roughly half a
//! percent — some fifty times the golden's tolerance. There is no such thing
//! as a small error in a discrete decision, so the design goal is to add
//! *none*, leaving only the perturbation the golden has already been shown to
//! survive.
//!
//! Tests here run on `PerformanceBackend` and assert relative tolerances, not
//! bit-exact equality: a dev may be on a backend that enables fast-math, where
//! exact equality against a host scalar reference cannot hold.

pub mod prefilter;
pub mod tables;

#[cfg(test)]
pub(crate) mod hybrid;

#[doc(inline)]
pub use prefilter::*;
#[doc(inline)]
pub use tables::*;
