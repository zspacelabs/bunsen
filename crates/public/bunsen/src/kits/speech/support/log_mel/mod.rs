//! # The log-mel front end the Whisper family shares
//!
//! Whisper, Qwen3-ASR and Voxtral read log-mels on one grid, fixed in time
//! and floored under a level:
//!
//! - [`LogMelFrontEndConfig`]: the grid (rate, hop, window) and the clamp range
//!   a checkpoint was trained with; it builds the mel converter and packages a
//!   window into model input. [`drop_last_frame`] is upstream's trailing-frame
//!   drop, and [`log_mels_of`] a whole waveform's log-mels.
//! - [`LogMelFeatures`]: the same log-mels streamed, a
//!   [`StreamingFeatures`](super::StreamingFeatures) front end.

mod features;
mod front_end;

pub use features::*;
pub use front_end::*;
