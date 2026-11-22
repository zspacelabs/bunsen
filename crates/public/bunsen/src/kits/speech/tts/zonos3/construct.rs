//! A loaded ZONOS3 model.

use alloc::{
    collections::BTreeMap,
    format,
    string::String,
    vec::Vec,
};
use std::path::PathBuf;

use super::{
    blocks::{
        Zonos3Codec,
        Zonos3CodecConfig,
        Zonos3Lm,
        Zonos3LmConfig,
        Zonos3SpeakerEncoder,
        Zonos3SpeakerEncoderConfig,
    },
    driver::Zonos3EngineConfig,
    prompt::Zonos3PromptConfig,
    voices::Zonos3Voice,
};
use crate::{
    errors::BunsenResult,
    kits::support::text::BunsenLanguageTag,
};

/// What a ZONOS3 pretrained loads to: the language model and the codec
/// decoder with their configs, the prompt layout, the release's engine
/// settings and languages, the stock voices, and the tokenizer's file,
/// unread.
#[derive(Debug)]
pub struct Zonos3Construct {
    /// The language model.
    pub lm: Zonos3Lm,

    /// Its config.
    pub lm_config: Zonos3LmConfig,

    /// The codec decoder.
    pub codec: Zonos3Codec,

    /// Its config.
    pub codec_config: Zonos3CodecConfig,

    /// The prompt layout.
    pub prompt: Zonos3PromptConfig,

    /// The release's engine settings: `config.json`'s `generation`
    /// section's sampler, silence prefill, decode overlap and output levels.
    pub engine_config: Zonos3EngineConfig,

    /// The release's languages: each request code (`en_us`) with the tag of
    /// its grammar (`en`).
    pub language_tags: BTreeMap<String, BunsenLanguageTag>,

    /// The output's sample rate.
    pub sample_rate: u32,

    /// The stock voices, by name.
    pub voices: BTreeMap<String, Zonos3Voice>,

    /// The conditions a prompt carries, in table order.
    pub conditions: Vec<super::prompt::Zonos3Condition>,

    /// The `tokenizer.json`: ids already offset past the four special ids
    /// ([`zonos3_text_ids`](super::prompt::zonos3_text_ids)).
    pub tokenizer: PathBuf,

    /// The speaker encoder, when the release ships one (`config.json`'s
    /// `speaker_encoder` section).
    pub speaker: Option<Zonos3SpeakerPart>,
}

/// The name of the release's speaker encoder, upstream's: the first part of
/// a [`Zonos3SpeakerPart::build`].
pub const ZONOS3_SPEAKER_ENCODER: &str = "ownmect_s2";

/// A construct's speaker encoder: the encoder (always in f32, as upstream
/// runs it whatever the language model's dtype), its config, and the build
/// a voice it makes names.
#[derive(Debug)]
pub struct Zonos3SpeakerPart {
    /// The encoder.
    pub encoder: Zonos3SpeakerEncoder,

    /// Its config.
    pub config: Zonos3SpeakerEncoderConfig,

    /// The build a voice it makes is of: the encoder and the checkpoint it
    /// came in (`ownmect_s2@sha256:...`). A cloned embedding is read through
    /// that checkpoint's speaker projection, so a voice speaks only in it.
    pub build: String,
}

impl Zonos3Construct {
    /// The language tag of `language`: a request code of the release's
    /// [`language_tags`](Self::language_tags) (`en_us`), as its grammar's tag
    /// (`en`); or else a language tag itself (`fr`, `de-DE`).
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a language
    /// that is neither, naming the release's codes.
    pub fn language_tag(
        &self,
        language: &str,
    ) -> BunsenResult<BunsenLanguageTag> {
        match self.language_tags.get(language) {
            Some(tag) => Ok(tag.clone()),
            None => BunsenLanguageTag::parse(language).map_err(|error| {
                let codes: Vec<&str> = self.language_tags.keys().map(String::as_str).collect();
                error.with_details(format!("the release's request codes: {}", codes.join(", ")))
            }),
        }
    }
}
