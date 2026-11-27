//! Internals of the Whisper driver: the crate-private splitting of a
//! timestamped decode into segments. Nothing here is part of the driver's
//! API; the speech regions a voice-activity filter produces are
//! [`SpeechRegion`](crate::kits::speech::vad::SpeechRegion)s, in the
//! driver itself.

pub(crate) mod segments;
