//! The Voxtral Realtime stream driver: audio pushed in chunks, timed
//! transcript out.
//!
//! The driver and its stream context are the
//! [`asr_driver`](crate::kits::speech::asr::driver) core's
//! [`AsrStreamDriver`](crate::kits::speech::asr::driver::AsrStreamDriver) and
//! [`AsrStreamContext`](crate::kits::speech::asr::driver::AsrStreamContext)
//! over a [`VoxtralEngine`]: what does not depend on the model (scheduling,
//! the emission policy, the clock, the voice-activity gate, speakers) lives
//! there, as it does for every speech-to-text kit, and `bunsen-tool` drives
//! them all alike.
//!
//! # Lifecycle
//!
//! 1. A [`VoxtralConstruct`](super::VoxtralConstruct): from a name, through
//!    [`default_voxtral_factory`](super::pretrained::default_voxtral_factory)
//!    and a [`ResourceCache`](crate::data::resources::ResourceCache) (features
//!    `store` and `cache`), or over a model in hand;
//!    [`with_delay_ms`](super::VoxtralConstruct::with_delay_ms) sets how far
//!    the text runs behind the audio.
//! 2. A [`VoxtralEngine`] over it, its window set by
//!    [`with_window_seconds`](VoxtralEngine::with_window_seconds).
//! 3. A [`VoxtralStreamDriver`] over the engine and an
//!    [`EmissionPolicy`](crate::kits::speech::asr::driver::EmissionPolicy).
//! 4. One [`VoxtralStreamContext`] per stream, samples in through `write_read`,
//!    [`TranscriptEvent`](crate::kits::speech::asr::driver::TranscriptEvent)s
//!    out.
//!
//! # Windows
//!
//! The engine decodes a window whole, as `transformers` decodes an
//! utterance: padded with silence, then one decoder step per 80 ms of it.
//! Voxtral times every word itself (its `[STREAMING_WORD]` marker is written
//! where the word starts, a fixed delay after the audio), so a window is laid
//! out a segment per word, and a long stream is cut between words. The
//! model's natively incremental decode, one step per 80 ms written, without
//! windows, is a context of its own: it carries the convolution stem's left
//! frames and both stacks' caches across writes.

mod engine;

pub use engine::*;

#[cfg(test)]
mod tests;
