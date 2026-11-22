//! The release's `config.json`.

use alloc::{
    collections::BTreeMap,
    string::String,
    vec::Vec,
};

use serde::Deserialize;

use super::{
    Zonos3CodecMirror,
    Zonos3LmMirror,
};
use crate::{
    errors::BunsenResult,
    kits::{
        speech::tts::zonos3::{
            driver::Zonos3EngineConfig,
            lm::Zonos3LmConfig,
            prompt::Zonos3PromptConfig,
            sampler::Zonos3SamplerConfig,
        },
        support::text::BunsenLanguageTag,
    },
};

/// The `generation` section of `config.json`, under upstream's names: how
/// the release generates and outputs a take.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Zonos3GenerationMirror {
    /// Frames of context a streamed decode carries.
    pub decode_context_frames: usize,
    /// The output gain's headroom, in dB.
    pub headroom_db: f64,
    /// Request language codes to text-normalization grammars.
    pub languages: BTreeMap<String, String>,
    /// The markup tags the normalizer keeps.
    pub markup_tags: Vec<String>,
    /// The sampler's min-p.
    pub min_p: f64,
    /// The output limiter's knee, in dBFS.
    pub output_limit_dbfs: f64,
    /// Frames a streamed decode overlaps.
    pub overlap_frames: usize,
    /// The codebooks the repetition penalty covers.
    pub repetition_codebooks: usize,
    /// The repetition penalty.
    pub repetition_penalty: f64,
    /// The rows the repetition penalty looks back over.
    pub repetition_window: usize,
    /// The codec's silence frame.
    pub silence_frame: Vec<u32>,
    /// The default silence prefill.
    pub silence_prefill_frames: usize,
    /// The longest silence prefill.
    pub silence_prefill_max_frames: usize,
    /// The sampler's temperature.
    pub temperature: f64,
    /// The sampler's top-p.
    pub top_p: f64,
    /// The sampler's top-k.
    pub topk: usize,
}

impl Zonos3GenerationMirror {
    /// Each request code of the [`languages`](Self::languages) table
    /// (`en_us`) with the tag of its grammar (`en`).
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a grammar
    /// that is not a language tag.
    pub fn language_tags(&self) -> BunsenResult<BTreeMap<String, BunsenLanguageTag>> {
        self.languages
            .iter()
            .map(|(code, grammar)| Ok((code.clone(), BunsenLanguageTag::parse(grammar)?)))
            .collect()
    }

    /// The release's sampler settings.
    pub fn sampler_config(&self) -> Zonos3SamplerConfig {
        Zonos3SamplerConfig {
            temperature: self.temperature,
            top_k: self.topk,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            repetition_window: self.repetition_window,
            repetition_codebooks: self.repetition_codebooks,
        }
    }

    /// The release's engine settings: its sampler, silence prefill, decode
    /// overlap and output levels (its markup tags aside: the normalizer
    /// keeps none, as the release's reference).
    pub fn engine_config(&self) -> Zonos3EngineConfig {
        Zonos3EngineConfig::new()
            .with_sampler(self.sampler_config())
            .with_silence_prefill_frames(self.silence_prefill_frames)
            .with_silence_prefill_max_frames(self.silence_prefill_max_frames)
            .with_overlap_frames(self.overlap_frames)
            .with_headroom_db(self.headroom_db)
            .with_output_limit_dbfs(self.output_limit_dbfs)
    }
}

/// `Zyphra/ZONOS3-Mini-hf`'s `config.json`: its language model's, codec
/// decoder's and generation's sections (the speaker encoder's is not read).
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Zonos3Mirror {
    /// The language model.
    pub lm: Zonos3LmMirror,
    /// The codec decoder.
    pub codec: Zonos3CodecMirror,
    /// How a take is generated and output.
    pub generation: Zonos3GenerationMirror,
    /// The output's sample rate.
    pub sampling_rate: u32,
    /// The speaker encoder, when the release ships one.
    #[serde(default)]
    pub speaker_encoder: Option<super::Zonos3SpeakerEncoderMirror>,
}

impl Zonos3Mirror {
    /// The prompt layout of a language model built from `lm` (the
    /// [`lm`](Self::lm) section's
    /// [`lm_config`](Zonos3LmMirror::lm_config)): its text vocabulary, the
    /// codec's silence frame, and the longest reference prefix.
    pub fn prompt_config(
        &self,
        lm: &Zonos3LmConfig,
    ) -> Zonos3PromptConfig {
        Zonos3PromptConfig::new(
            lm,
            self.lm.text_vocab as u32,
            self.generation.silence_frame.clone(),
            self.lm.reference_max_frames,
        )
    }
}
