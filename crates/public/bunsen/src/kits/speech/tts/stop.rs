//! # Why a take ended

/// Why a take ended: the vocabulary every TTS model's take and the driver
/// share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TtsStop {
    /// The model ended it: Qwen3-TTS's talker chose its end token, a ZONOS3
    /// codebook sampled the end of audio.
    End,

    /// It ran out of frames: the take's frame budget, the request's
    /// [`max_seconds`](super::driver::TtsRequest::max_seconds) or the model's
    /// context.
    Budget,
}
