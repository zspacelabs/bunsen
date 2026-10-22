//! A voice-activity model that replays a recording.
use std::sync::Arc;

use serde::{
    Deserialize,
    Serialize,
};

use crate::kits::speech::asr_driver::{
    VoiceActivityModel,
    VoiceActivityStream,
};

/// A voice-activity model that replays recorded probabilities: chunk `i` of
/// a stream scores `probabilities[i]`, and every chunk past the recording
/// scores `0.0`.
///
/// For tests of what a driver or a gate does with speech probabilities,
/// without a model. [`silero_jfk_moon_4s`](Self::silero_jfk_moon_4s) is
/// Silero's, over a real clip; `silero-model-validation` holds it to the
/// model it was taken from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordedVoiceActivity {
    /// The sample rate, in Hz.
    pub sample_rate: usize,

    /// The samples in one chunk.
    pub chunk_size: usize,

    /// The clip's length in samples, before it was padded to whole chunks.
    pub samples: usize,

    /// One per chunk, in order.
    pub probabilities: Vec<f32>,
}

impl RecordedVoiceActivity {
    /// Upstream Silero VAD's 16 kHz model (`silero/vad-16k`) over
    /// `bunsen_test_audio::JFK_MOON_4S`: 4.0 s of "...the highest
    /// mountain?", a pause, and "Why, thirty-five years ago...", in 125
    /// chunks of 512 samples.
    ///
    /// Committed as `testdata/silero/jfk_moon_4s.json`.
    pub fn silero_jfk_moon_4s() -> Self {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/silero/jfk_moon_4s.json"
        )))
        .expect("the committed recording parses")
    }
}

impl VoiceActivityModel for RecordedVoiceActivity {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn new_stream(&self) -> Box<dyn VoiceActivityStream> {
        Box::new(RecordedStream {
            recording: Arc::new(self.clone()),
            chunks: 0,
        })
    }
}

#[derive(Clone, Debug)]
struct RecordedStream {
    recording: Arc<RecordedVoiceActivity>,

    /// The chunks scored so far.
    chunks: usize,
}

impl VoiceActivityStream for RecordedStream {
    fn probabilities(
        &mut self,
        samples: Vec<f32>,
    ) -> Vec<f32> {
        let steps = samples.len() / self.recording.chunk_size;
        let recorded = &self.recording.probabilities;
        let out = (self.chunks..self.chunks + steps)
            .map(|i| recorded.get(i).copied().unwrap_or(0.0))
            .collect();
        self.chunks += steps;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replays_in_pieces_then_silence() {
        let vad = RecordedVoiceActivity {
            sample_rate: 16_000,
            chunk_size: 4,
            samples: 12,
            probabilities: vec![0.1, 0.9, 0.8],
        };
        let mut stream = vad.new_stream();
        assert_eq!(stream.probabilities(vec![0.0; 8]), vec![0.1, 0.9]);
        assert_eq!(stream.probabilities(vec![0.0; 3]), Vec::<f32>::new());
        assert_eq!(stream.probabilities(vec![0.0; 8]), vec![0.8, 0.0]);
    }

    #[test]
    fn test_the_silero_recording_is_the_clip() {
        let vad = RecordedVoiceActivity::silero_jfk_moon_4s();
        assert_eq!(vad.sample_rate, 16_000);
        assert_eq!(vad.chunk_size, 512);
        assert_eq!(vad.samples, 64_000);
        assert_eq!(vad.probabilities.len(), 125);
        assert!(vad.probabilities.iter().all(|p| (0.0..=1.0).contains(p)));
    }
}
