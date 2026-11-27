//! # Voice-activity models: per-chunk speech probabilities.

use core::fmt::Debug;

use dyn_clone::DynClone;

/// A voice-activity model: the probability that each chunk of audio holds
/// speech, which a
/// [`VoiceActivityFilter`](super::VoiceActivityFilter)
/// turns into speech regions.
///
/// Shared and immutable, like an ASR stream driver it is attached to; each
/// stream opens its own [`VoiceActivityStream`] for its state.
pub trait VoiceActivityModel: Debug + Send + Sync {
    /// The sample rate it runs at, in Hz.
    fn sample_rate(&self) -> usize;

    /// The samples in one chunk: one probability per chunk.
    fn chunk_size(&self) -> usize;

    /// Opens a stream.
    fn new_stream(&self) -> Box<dyn VoiceActivityStream>;
}

/// One stream's voice-activity state.
///
/// `Clone`, through [`DynClone`], because the stream context that holds one
/// boxed is.
pub trait VoiceActivityStream: Debug + Send + Sync + DynClone {
    /// The speech probability of each chunk of `samples`, in order.
    ///
    /// # Arguments
    /// * `samples` - a whole number of chunks, following the samples of the
    ///   previous call.
    fn probabilities(
        &mut self,
        samples: Vec<f32>,
    ) -> Vec<f32>;
}

dyn_clone::clone_trait_object!(<> VoiceActivityStream);
