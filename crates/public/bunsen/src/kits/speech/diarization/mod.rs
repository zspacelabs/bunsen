//! # Speaker diarization: the parts that do not depend on the model
//!
//! A diarizer answers "who spoke when". Everything about that which does not
//! depend on the model lives here; a model plugs in as a
//! [`SpeakerActivityModel`].
//!
//! - [`SpeakerActivity`]: what a diarizer emits, the probability that each
//!   speaker is talking in each frame, `[frames, speakers]`. Speakers are
//!   numbered by the diarizer, in the order it first heard them.
//! - [`SpeakerActivityModel`] and [`SpeakerActivityStream`]: the diarizer seam.
//!   The model is shared and immutable; each stream opens its own stream state,
//!   takes samples in pieces of any size, and hands back the frames it has
//!   finalized.
//! - [`SpeakerTurn`] and [`TurnFilter`]: activity to turns, one speaker's
//!   stretch of speech each, by `NeMo`'s diarization post-processing
//!   ([`TurnFilterConfig`]: hysteresis thresholds, padding, and the short-turn
//!   and short-gap filters), streamed or over a whole recording.
//! - [`RttmSegment`]: turns as NIST RTTM lines, written and read.
//!
//! A diarizer's activity and its turns are on the stream's sample clock: a
//! turn's times are seconds from the first frame the filter saw.

mod activity;
mod rttm;
mod turns;

pub use activity::*;
pub use rttm::*;
pub use turns::*;
