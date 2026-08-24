//! # Sequence and trellis operations.
//!
//! Algorithms that find or score a path through a sequence, as opposed to the
//! per-sample filtering in [`signal`](super::signal).
//!
//! * [`Viterbi`] — batched maximum-scoring path through a trellis, streaming
//!   across calls, with a backtrace that costs `depth` gathers rather than
//!   `steps * depth`.

mod viterbi;

#[doc(inline)]
pub use viterbi::*;
