//! Internals of the Whisper driver: [`drop_last_frame`], the trailing-frame
//! drop of upstream's spectrogram, and the crate-private splitting of a
//! timestamped decode into segments. Nothing here is part of the driver's
//! API; the speech regions a voice-activity filter produces are
//! [`SpeechRegion`](super::SpeechRegion)s, in the driver itself.

pub(crate) mod segments;
mod util;

pub use util::*;
