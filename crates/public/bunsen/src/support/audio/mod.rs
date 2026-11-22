//! # Audio Support
//!
//! Audio files: [`load_audio_mono_sr`], a mono file at the rate a model
//! needs, decoded, never resampled; [`load_audio`], a file as it is, at its
//! own rate and channels ([`DecodedAudio`]); [`save_wav_mono`], mono samples
//! to a WAV file. Rate conversion, a signal-processing decision the caller
//! makes, is [`ops::signal::resample`](crate::ops::signal::resample).

mod load;
mod save;

pub use load::*;
pub use save::*;
