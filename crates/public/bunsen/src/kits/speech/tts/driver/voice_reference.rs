//! [`TtsVoiceReference`]: what a cloned voice is made from, whatever the model.

use alloc::{
    string::String,
    vec::Vec,
};

use serde::{
    Deserialize,
    Serialize,
};

/// The version of the `reference.json` layout this reads and writes.
pub const TTS_VOICE_REFERENCE_VERSION: u32 = 1;

/// An edit to a clip as a reference reads it; the clip's file itself is kept
/// as given.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsClipEdit {
    /// Only the samples `start..end` (per channel, at the clip's rate) are
    /// read: silence trimmed.
    Trim {
        /// The first sample kept.
        start: usize,

        /// One past the last sample kept.
        end: usize,
    },
}

/// One clip of a reference, as `reference.json` records it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsClipRecord {
    /// Its file's name in the reference's directory.
    pub file: String,

    /// Its file's sha256, in hex.
    pub sha256: String,

    /// Its sample rate, in Hz.
    pub sample_rate: u32,

    /// Its channels.
    pub channels: usize,

    /// Its samples per channel, as decoded, before any edit.
    pub samples: usize,

    /// The edits a read applies, in order.
    #[serde(default)]
    pub edits: Vec<TtsClipEdit>,
}

/// What a reference says of itself: `reference.json`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsVoiceReferenceRecord {
    /// The layout's version, [`TTS_VOICE_REFERENCE_VERSION`].
    pub version: u32,

    /// The voice's name, as given.
    #[serde(default)]
    pub name: Option<String>,

    /// What the clips say, as given or transcribed.
    #[serde(default)]
    pub transcript: Option<String>,

    /// The clips' language, as given: a language tag or an engine's key.
    #[serde(default)]
    pub language: Option<String>,

    /// The speaker's gender, as given.
    #[serde(default)]
    pub gender: Option<String>,

    /// The speaker's style, as given.
    #[serde(default)]
    pub style: Option<String>,

    /// When it was made, in seconds since the Unix epoch.
    #[serde(default)]
    pub created_unix: Option<u64>,

    /// The clips, in order.
    pub clips: Vec<TtsClipRecord>,
}

/// One clip of a reference, decoded: its record, and its samples per channel
/// at its own rate, its edits applied.
#[derive(Clone, Debug, PartialEq)]
pub struct TtsReferenceClip {
    /// What the reference records of it.
    pub record: TtsClipRecord,

    /// The samples, `[channels][samples]`, in `[-1, 1]`.
    pub channels: Vec<Vec<f32>>,
}

impl TtsReferenceClip {
    /// The clip's rate, in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.record.sample_rate
    }

    /// The clip as one channel: the channels' mean, sample by sample (the
    /// mix both ZONOS3's and Qwen3-TTS's encoders take).
    pub fn mono(&self) -> Vec<f32> {
        match self.channels.as_slice() {
            [] => Vec::new(),
            [only] => only.clone(),
            many => {
                let n = many.iter().map(Vec::len).min().unwrap_or(0);
                let scale = 1.0 / many.len() as f32;
                (0..n)
                    .map(|i| many.iter().map(|c| c[i]).sum::<f32>() * scale)
                    .collect()
            }
        }
    }

    /// Its length after its edits, in seconds.
    pub fn seconds(&self) -> f64 {
        self.channels.first().map_or(0, Vec::len) as f64 / f64::from(self.record.sample_rate)
    }
}

/// A voice reference: what a voice is made from, model-free. Its reference
/// clips at their own rates and channels (an engine resamples them for its
/// encoder as it builds them), what they say, and what was given of the
/// speaker.
///
/// An engine builds a voice file of its own from a reference
/// ([`TtsEngine::build_voice`](super::TtsEngine::build_voice)); one reference
/// builds a voice for any engine and mode it satisfies, and again when a
/// checkpoint changes. A reference is saved as a directory: its clips' files as
/// given, and `reference.json` ([`TtsVoiceReferenceRecord`]).
#[derive(Clone, Debug, PartialEq)]
pub struct TtsVoiceReference {
    /// What the reference says of itself.
    pub record: TtsVoiceReferenceRecord,

    /// The clips, decoded, in the record's order.
    pub clips: Vec<TtsReferenceClip>,

    /// The reference's identity: the sha256 of its `reference.json`, in hex,
    /// which a voice file built from it names.
    pub digest: String,
}

impl TtsVoiceReference {
    /// Its transcript, when it has one.
    pub fn transcript(&self) -> Option<&str> {
        self.record.transcript.as_deref()
    }

    /// The clips' total length, in seconds.
    pub fn seconds(&self) -> f64 {
        self.clips.iter().map(TtsReferenceClip::seconds).sum()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn clip(channels: Vec<Vec<f32>>) -> TtsReferenceClip {
        TtsReferenceClip {
            record: TtsClipRecord {
                file: "a.wav".into(),
                sha256: "00".into(),
                sample_rate: 4,
                channels: channels.len(),
                samples: channels[0].len(),
                edits: vec![],
            },
            channels,
        }
    }

    #[test]
    fn test_mono_and_seconds() {
        let stereo = clip(vec![vec![1.0, 0.0, 0.5, 0.5], vec![0.0, 1.0, 0.5, -0.5]]);
        assert_eq!(stereo.mono(), vec![0.5, 0.5, 0.5, 0.0]);
        assert_eq!(stereo.seconds(), 1.0);
        let mono = clip(vec![vec![0.25; 8]]);
        assert_eq!(mono.mono(), vec![0.25; 8]);
    }

    /// The record's JSON form: its edits tagged by kind, the optional
    /// fields omittable.
    #[test]
    fn test_record_json() {
        let record: TtsVoiceReferenceRecord = serde_json::from_str(
            r#"{"version": 1, "transcript": "hello",
                "clips": [{"file": "a.wav", "sha256": "ab", "sample_rate": 16000,
                           "channels": 2, "samples": 32000,
                           "edits": [{"trim": {"start": 10, "end": 20}}]}]}"#,
        )
        .unwrap();
        assert_eq!(record.name, None);
        assert_eq!(
            record.clips[0].edits,
            vec![TtsClipEdit::Trim { start: 10, end: 20 }]
        );
        let again: TtsVoiceReferenceRecord =
            serde_json::from_str(&serde_json::to_string(&record).unwrap()).unwrap();
        assert_eq!(again, record);
    }
}
