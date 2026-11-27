//! The feature extractor's settings files.

use alloc::{
    format,
    string::String,
};

use serde::Deserialize;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::{
        asr::qwen3_asr::{
            Qwen3AsrConfig,
            Qwen3AsrFrontEndConfig,
        },
        support::log_mel::LogMelFrontEndConfig,
    },
};

/// A `transformers` `processor_config.json` of a Qwen3-ASR model, as far as
/// the front end reads it: its `feature_extractor` section.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Qwen3AsrProcessorMirror {
    /// The feature extractor's settings.
    pub feature_extractor: Qwen3AsrFeatureExtractorMirror,
}

/// A Qwen3-ASR feature extractor's settings, under upstream's names:
/// `transformers`' `Qwen3ASRFeatureExtractor` (a `processor_config.json`'s
/// `feature_extractor` section), or the original library's
/// `WhisperFeatureExtractor` (a `preprocessor_config.json`, as a Phonon-1
/// archive carries it).
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Qwen3AsrFeatureExtractorMirror {
    /// `Qwen3ASRFeatureExtractor` or `WhisperFeatureExtractor`.
    pub feature_extractor_type: String,
    /// The mel channels.
    pub feature_size: usize,
    /// The sample rate, in Hz; absent, `WhisperFeatureExtractor`'s 16 kHz.
    #[serde(default = "whisper_sampling_rate")]
    pub sampling_rate: usize,
    /// The hop between frames, in samples.
    pub hop_length: usize,
    /// The analysis window, in samples.
    pub n_fft: usize,
    /// The chunk, in seconds: shorter audio is padded to it.
    pub chunk_length: usize,
    /// The chunk, in samples: `chunk_length · sampling_rate`.
    #[serde(default)]
    pub n_samples: Option<usize>,
    /// The chunk, in frames: `n_samples / hop_length`.
    #[serde(default)]
    pub nb_max_frames: Option<usize>,
    /// The value audio is padded with.
    #[serde(default)]
    pub padding_value: f64,
    /// The noise added before the STFT; 0 is none.
    #[serde(default)]
    pub dither: f64,
    /// Where audio is padded: `right`.
    #[serde(default)]
    pub padding_side: Option<String>,
    /// Half the encoder's chunk, in frames, which the mel axis is padded to a
    /// multiple of twice; the model's audio `n_window`.
    #[serde(default)]
    pub n_window: Option<usize>,
    /// The shortest audio, in samples, whose frames are all valid
    /// (`Qwen3ASRFeatureExtractor`'s); absent, the original library's.
    #[serde(default)]
    pub min_length: Option<usize>,
}

/// `sampling_rate`'s default: `WhisperFeatureExtractor`'s.
fn whisper_sampling_rate() -> usize {
    16_000
}

impl Qwen3AsrFeatureExtractorMirror {
    /// The [`Qwen3AsrFrontEndConfig`] these settings describe: the mel count,
    /// the grid in time, the chunk, and the shortest valid audio
    /// ([`min_length`](Self::min_length), or the front end's default).
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming
    /// the field, for a feature extractor this front end is not: another
    /// type, padding other than zeros on the right, dither, or a hop or window
    /// that is not whole milliseconds;
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when `n_samples` or `nb_max_frames` disagree with the chunk.
    pub fn front_end_config(&self) -> BunsenResult<Qwen3AsrFrontEndConfig> {
        let unsupported = |what: String| Err(BunsenError::unsupported(what));
        let kind = self.feature_extractor_type.as_str();
        if !matches!(kind, "Qwen3ASRFeatureExtractor" | "WhisperFeatureExtractor") {
            return unsupported(format!("a `feature_extractor_type` of {kind:?}"));
        }
        if self.padding_value != 0.0 {
            return unsupported(format!("a `padding_value` of {}", self.padding_value));
        }
        if self.dither != 0.0 {
            return unsupported(format!("a `dither` of {}", self.dither));
        }
        if let Some(side) = &self.padding_side
            && side != "right"
        {
            return unsupported(format!("a `padding_side` of {side:?}"));
        }
        let sr = self.sampling_rate;
        let ms = |field: &str, samples: usize| {
            if sr == 0 || !(samples * 1000).is_multiple_of(sr) {
                return Err(BunsenError::unsupported(format!(
                    "a `{field}` of {samples} samples at {sr} Hz: not whole milliseconds"
                )));
            }
            Ok(samples * 1000 / sr)
        };
        let whisper = LogMelFrontEndConfig::new()
            .with_sample_rate(sr)
            .with_hop_ms(ms("hop_length", self.hop_length)?)
            .with_window_ms(ms("n_fft", self.n_fft)?);
        let n_samples = self.chunk_length * sr;
        for (field, stated, derived) in [
            ("n_samples", self.n_samples, n_samples),
            (
                "nb_max_frames",
                self.nb_max_frames,
                n_samples / self.hop_length,
            ),
        ] {
            if let Some(stated) = stated
                && stated != derived
            {
                return Err(BunsenError::invalid_resource(format!(
                    "a feature extractor whose `{field}`, {stated}, is not its chunk's, {derived}"
                )));
            }
        }
        let config = Qwen3AsrFrontEndConfig::new()
            .with_n_mels(self.feature_size)
            .with_chunk_seconds(self.chunk_length)
            .with_log_mel(whisper);
        Ok(match self.min_length {
            Some(min) => config.with_min_valid_samples(min),
            None => config,
        })
    }

    /// Checks these settings against the model they extract for: its mel
    /// count, and its audio chunk (`n_window`), when stated.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
    /// naming the field, when either is not the model's.
    pub fn check_model(
        &self,
        config: &Qwen3AsrConfig,
    ) -> BunsenResult<()> {
        let mismatch = |field: &str, stated: usize, model: usize| {
            Err(BunsenError::invalid_resource(format!(
                "a feature extractor whose `{field}` is {stated}, for a model whose is {model}"
            )))
        };
        if self.feature_size != config.audio.n_mels {
            return mismatch("feature_size", self.feature_size, config.audio.n_mels);
        }
        if let Some(n_window) = self.n_window
            && n_window != config.audio.n_window
        {
            return mismatch("n_window", n_window, config.audio.n_window);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        kits::speech::asr::qwen3_asr::Qwen3AsrFrontEndMeta,
        support::testing::assert_config_eq,
    };

    /// `Qwen/Qwen3-ASR-0.6B-hf`'s `processor_config.json`.
    const HF: &str = r#"{
        "feature_extractor": {
            "chunk_length": 30, "dither": 0.0,
            "feature_extractor_type": "Qwen3ASRFeatureExtractor", "feature_size": 128,
            "hop_length": 160, "min_length": 8000, "n_fft": 400, "n_samples": 480000,
            "n_window": 50, "nb_max_frames": 3000, "padding_side": "right",
            "padding_value": 0.0, "return_attention_mask": true, "sampling_rate": 16000
        },
        "processor_class": "Qwen3ASRProcessor",
        "timestamp_segment_time": 80
    }"#;

    /// `FermionResearch`'s Phonon-1 archive's `preprocessor_config.json`.
    const PHONON: &str = r#"{
        "chunk_length": 30, "dither": 0.0, "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": 128, "hop_length": 160, "n_fft": 400, "n_samples": 480000,
        "nb_max_frames": 3000, "padding_side": "right", "padding_value": 0.0,
        "processor_class": "Qwen3ASRProcessor", "return_attention_mask": true
    }"#;

    fn hf() -> Qwen3AsrFeatureExtractorMirror {
        serde_json::from_str::<Qwen3AsrProcessorMirror>(HF)
            .unwrap()
            .feature_extractor
    }

    /// Both files are the default front end: the release's.
    #[test]
    fn test_release_files() {
        let phonon: Qwen3AsrFeatureExtractorMirror = serde_json::from_str(PHONON).unwrap();
        for mirror in [hf(), phonon] {
            let config = mirror.front_end_config().unwrap();
            assert_config_eq("the config", &config, &Qwen3AsrFrontEndConfig::new());
            assert_eq!(config.min_samples(), 480_000);
            mirror
                .check_model(&Qwen3AsrConfig::qwen3_asr_0_6b())
                .unwrap();
        }
    }

    /// The file's values reach the front end.
    #[test]
    fn test_values_read() {
        let mirror = Qwen3AsrFeatureExtractorMirror {
            feature_size: 80,
            sampling_rate: 8000,
            hop_length: 80,
            n_fft: 200,
            chunk_length: 20,
            n_samples: None,
            nb_max_frames: None,
            min_length: Some(1000),
            ..hf()
        };
        let config = mirror.front_end_config().unwrap();
        assert_eq!(
            (
                config.n_mels,
                config.chunk_seconds,
                config.min_valid_samples,
                config.log_mel.sample_rate,
                config.log_mel.hop_ms,
                config.log_mel.window_ms
            ),
            (80, 20, 1000, 8000, 10, 25)
        );
    }

    /// What the front end does not do is refused, naming the field; counts
    /// that disagree with the chunk, or with the model, are invalid.
    #[test]
    fn test_refusals() {
        let release = hf();
        for (mirror, field) in [
            (
                Qwen3AsrFeatureExtractorMirror {
                    feature_extractor_type: "ParakeetFeatureExtractor".into(),
                    ..release.clone()
                },
                "feature_extractor_type",
            ),
            (
                Qwen3AsrFeatureExtractorMirror {
                    dither: 1e-5,
                    ..release.clone()
                },
                "dither",
            ),
            (
                Qwen3AsrFeatureExtractorMirror {
                    padding_side: Some("left".into()),
                    ..release.clone()
                },
                "padding_side",
            ),
            (
                Qwen3AsrFeatureExtractorMirror {
                    hop_length: 161,
                    n_samples: None,
                    nb_max_frames: None,
                    ..release.clone()
                },
                "hop_length",
            ),
        ] {
            ErrorMatcher::unsupported()
                .message_contains(field)
                .assert_err(&mirror.front_end_config());
        }
        ErrorMatcher::invalid_resource()
            .message_contains("nb_max_frames")
            .assert_err(
                &Qwen3AsrFeatureExtractorMirror {
                    nb_max_frames: Some(2999),
                    ..release.clone()
                }
                .front_end_config(),
            );
        for (mirror, field) in [
            (
                Qwen3AsrFeatureExtractorMirror {
                    feature_size: 80,
                    ..release.clone()
                },
                "feature_size",
            ),
            (
                Qwen3AsrFeatureExtractorMirror {
                    n_window: Some(25),
                    ..release
                },
                "n_window",
            ),
        ] {
            ErrorMatcher::invalid_resource()
                .message_contains(field)
                .assert_err(&mirror.check_model(&Qwen3AsrConfig::qwen3_asr_0_6b()));
        }
    }
}
