//! [`Qwen3TtsEngine`]: Qwen3-TTS as a TTS engine.

use alloc::{
    format,
    string::String,
    sync::Arc,
    vec::Vec,
};

use burn::{
    config::Config,
    tensor::{
        DType,
        Device,
    },
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
    },
    kits::{
        speech::tts::{
            driver::{
                TtsCapabilities,
                TtsDriver,
                TtsDriverConfig,
                TtsEngine,
                TtsLanguage,
                TtsStop,
                TtsTakeInfo,
                TtsTakeRequest,
                TtsVoiceChoice,
                TtsVoiceInfo,
            },
            qwen3_tts::{
                Qwen3TtsCodes,
                Qwen3TtsConstruct,
                Qwen3TtsModelKind,
                Qwen3TtsStop,
                codec::{
                    QWEN3_TTS_DECODE_CHUNK,
                    QWEN3_TTS_DECODE_CONTEXT,
                    Qwen3TtsCodecDecoderMeta,
                },
                prompt::{
                    Qwen3TtsPromptRequest,
                    Qwen3TtsPromptRows,
                    Qwen3TtsTextPlacement,
                },
            },
        },
        support::{
            text::BunsenLanguageTag,
            tokens::Tokenizer,
        },
    },
};

/// Qwen3-TTS's TTS driver: [`TtsDriver`] over [`Qwen3TtsEngine`].
pub type Qwen3TtsDriver<T> = TtsDriver<Qwen3TtsEngine<T>>;

/// The driver of `engine`. Its text is spoken as typed, as upstream speaks
/// it: the release has no normalizer.
///
/// # Errors
/// As [`TtsDriver::new`].
pub fn qwen3_tts_driver<T: Tokenizer + Send + Sync>(
    engine: Qwen3TtsEngine<T>
) -> BunsenResult<Qwen3TtsDriver<T>> {
    TtsDriver::new(engine, TtsDriverConfig::new())
}

/// Config for [`Qwen3TtsEngine`]; its defaults are upstream's
/// (`generate_custom_voice`'s and `generate_voice_design`'s).
#[derive(Config, Debug)]
pub struct Qwen3TtsEngineConfig {
    /// The stock speaker a request that names no voice speaks in; `None`, a
    /// request must name one (upstream has no default speaker).
    #[config(default = "None")]
    pub default_voice: Option<String>,

    /// The language a request that names none is in: a model key
    /// (`english`), or `auto`, the speaker's own dialect or no language.
    #[config(default = "String::from(\"auto\")")]
    pub default_language: String,

    /// Where the text goes in the prompt: upfront (upstream's
    /// `non_streaming_mode`, its default) or interleaved with the frames.
    #[config(default = "Qwen3TtsTextPlacement::Upfront")]
    pub placement: Qwen3TtsTextPlacement,

    /// The frames the codec decodes at once.
    #[config(default = "QWEN3_TTS_DECODE_CHUNK")]
    pub decode_chunk: usize,

    /// The frames of left context a later chunk is decoded with.
    #[config(default = "QWEN3_TTS_DECODE_CONTEXT")]
    pub decode_context: usize,
}

/// A voice, as [`Qwen3TtsEngine`] speaks in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Qwen3TtsVoice {
    /// A stock speaker, by its key, and its codec row.
    Speaker {
        /// Its key (`ryan`).
        name: String,
        /// Its codec row.
        row: u32,
    },

    /// A voice described in words (a `VoiceDesign` model).
    Described(String),
}

/// Qwen3-TTS as a TTS engine ([`TtsEngine`]): the talker and code
/// predictor's take of a chat-wrapped text, rendered by the codec in chunks.
///
/// - A voice is a stock speaker (a `CustomVoice` model; matched without case,
///   as upstream matches), or a description (a `VoiceDesign` model, upstream's
///   `instruct`).
/// - A language is one of the model's keys, or `auto`: the speaker's dialect,
///   or no language row. Its tag, for a normalizer, is the catalog row's
///   (`language_keys`).
/// - An instruction is taken where the construct says the model takes one
///   ([`Qwen3TtsConstruct::instruction`]); upstream drops one on the 0.6B
///   models silently, where the driver refuses it unless told to strip it.
/// - Sampling overrides apply to the talker's sampler (codebook 0); the code
///   predictor keeps the release's.
/// - [`max_seconds`](TtsTakeRequest::max_seconds) bounds the take's frames at
///   the codec's frame rate (12.5 a second).
#[derive(Debug)]
pub struct Qwen3TtsEngine<T: Tokenizer> {
    construct: Arc<Qwen3TtsConstruct>,
    tokenizer: T,
    capabilities: TtsCapabilities,
    config: Qwen3TtsEngineConfig,
}

impl<T: Tokenizer> Qwen3TtsEngine<T> {
    /// The engine of a loaded model, its text read by `tokenizer` (the
    /// construct's [`tokenizer`](Qwen3TtsConstruct::tokenizer) files).
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a default voice
    /// or language the model does not have;
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a decode
    /// chunk of no frames.
    pub fn new(
        construct: Arc<Qwen3TtsConstruct>,
        tokenizer: T,
        config: Qwen3TtsEngineConfig,
    ) -> BunsenResult<Self> {
        if config.decode_chunk == 0 {
            return Err(BunsenError::illegal("a decode chunk of no frames"));
        }
        if let Some(voice) = &config.default_voice {
            construct.prompt.speaker(&voice.to_lowercase())?;
        }
        construct
            .prompt
            .language(&config.default_language.to_lowercase(), None)?;
        let capabilities = TtsCapabilities {
            languages: construct
                .prompt
                .languages
                .keys()
                .filter_map(|key| {
                    construct.language_tags.get(key).map(|tag| TtsLanguage {
                        key: key.clone(),
                        tag: tag.clone(),
                    })
                })
                .collect(),
            default_voice: config.default_voice.clone(),
            stock_voices: !construct.prompt.speakers.is_empty(),
            described_voices: construct.kind == Qwen3TtsModelKind::VoiceDesign,
            instruction: construct.instruction,
        };
        Ok(Self {
            construct,
            tokenizer,
            capabilities,
            config,
        })
    }

    /// The model.
    pub fn construct(&self) -> &Qwen3TtsConstruct {
        &self.construct
    }

    /// The config.
    pub fn config(&self) -> &Qwen3TtsEngineConfig {
        &self.config
    }

    fn device(&self) -> Device {
        self.construct.model.talker.text_embedding.weight.device()
    }

    /// The codec's frame rate, frames a second.
    fn frame_rate(&self) -> f64 {
        f64::from(self.construct.sample_rate)
            / self.construct.codec_config.samples_per_frame() as f64
    }

    /// `text`, chat-wrapped as `role`'s turn, then the assistant's opened
    /// when `open`, and tokenized.
    fn wrapped_ids(
        &self,
        role: &str,
        text: &str,
        open: bool,
    ) -> BunsenResult<Vec<u32>> {
        let mut wrapped = format!("<|im_start|>{role}\n{text}<|im_end|>\n");
        if open {
            wrapped.push_str("<|im_start|>assistant\n");
        }
        self.tokenizer
            .encode(&wrapped)?
            .into_iter()
            .map(|id| {
                u32::try_from(id)
                    .map_err(|_| BunsenError::invalid_resource(format!("a text id {id}")))
            })
            .collect()
    }

    /// The take's prompt request: the text wrapped as the assistant's turn,
    /// the instruction (or a described voice) as the user's, the speaker's
    /// and the language's rows.
    ///
    /// # Errors
    /// The tokenizer's; [`Lookup`](crate::errors::BunsenErrorKind::Lookup)
    /// for a language the model does not have.
    pub fn prompt_request(
        &self,
        take: &TtsTakeRequest<'_, Qwen3TtsVoice>,
    ) -> BunsenResult<Qwen3TtsPromptRequest> {
        let (speaker, name, instruct) = match take.voice {
            Qwen3TtsVoice::Speaker { name, row } => {
                (Some(*row), Some(name.as_str()), take.instruction)
            }
            Qwen3TtsVoice::Described(description) => (None, None, Some(description.as_str())),
        };
        let key = take
            .language
            .unwrap_or(&self.config.default_language)
            .to_lowercase();
        Ok(Qwen3TtsPromptRequest {
            input_ids: self.wrapped_ids("assistant", take.text, true)?,
            instruct_ids: instruct
                .filter(|text| !text.is_empty())
                .map(|text| self.wrapped_ids("user", text, false))
                .transpose()?,
            speaker,
            language: self.construct.prompt.language(&key, name)?,
            placement: self.config.placement,
        })
    }
}

impl<T: Tokenizer + Send + Sync> TtsEngine for Qwen3TtsEngine<T> {
    type Take = Qwen3TtsCodes;
    type Voice = Qwen3TtsVoice;

    fn capabilities(&self) -> &TtsCapabilities {
        &self.capabilities
    }

    fn sample_rate(&self) -> u32 {
        self.construct.sample_rate
    }

    fn voices(&self) -> Vec<TtsVoiceInfo> {
        self.construct.prompt.voices()
    }

    fn resolve_voice(
        &self,
        choice: &TtsVoiceChoice,
    ) -> BunsenResult<Qwen3TtsVoice> {
        let speaker = |name: &str| {
            let name = name.to_lowercase();
            let row = self.construct.prompt.speaker(&name)?;
            Ok(Qwen3TtsVoice::Speaker { name, row })
        };
        match choice {
            TtsVoiceChoice::Default => match &self.config.default_voice {
                Some(name) => speaker(name),
                None => Err(BunsenError::lookup(
                    LookupError::missing("voice", "the default")
                        .with_candidates(self.construct.prompt.speakers.keys().cloned()),
                )
                .with_details("Qwen3-TTS has no default speaker: name one")),
            },
            TtsVoiceChoice::Stock(name) => speaker(name),
            TtsVoiceChoice::Described(text) if self.capabilities.described_voices => {
                Ok(Qwen3TtsVoice::Described(text.clone()))
            }
            other => Err(BunsenError::unsupported(format!(
                "this Qwen3-TTS model ({:?}) cannot speak in {other:?}",
                self.construct.kind
            ))),
        }
    }

    fn language_tag(
        &self,
        language: Option<&str>,
        voice: &Qwen3TtsVoice,
    ) -> BunsenResult<Option<BunsenLanguageTag>> {
        let key = language
            .unwrap_or(&self.config.default_language)
            .to_lowercase();
        let speaker = match voice {
            Qwen3TtsVoice::Speaker { name, .. } => Some(name.as_str()),
            Qwen3TtsVoice::Described(_) => None,
        };
        let prompt = &self.construct.prompt;
        prompt.language(&key, speaker)?;
        let key = match key.as_str() {
            "auto" => match speaker.and_then(|s| prompt.speaker_languages.get(s)) {
                Some(dialect) => dialect.clone(),
                None => return Ok(None),
            },
            _ => key,
        };
        Ok(self.construct.language_tags.get(&key).cloned())
    }

    fn generate(
        &self,
        take: &TtsTakeRequest<'_, Qwen3TtsVoice>,
    ) -> BunsenResult<Qwen3TtsCodes> {
        let sampling = take.sampling;
        if sampling.min_p.is_some() {
            return Err(BunsenError::unsupported("Qwen3-TTS's sampler has no min-p"));
        }
        let mut config = self.construct.take.clone();
        let talker = &mut config.talker;
        if let Some(t) = sampling.temperature {
            talker.temperature = t;
            talker.sample = t > 0.0;
        }
        if let Some(k) = sampling.top_k {
            talker.top_k = k;
        }
        if let Some(p) = sampling.top_p {
            talker.top_p = p;
        }
        if let Some(r) = sampling.repetition_penalty {
            talker.repetition_penalty = r;
        }
        if let Some(seconds) = take.max_seconds {
            let frames = (seconds * self.frame_rate()).ceil().max(1.0) as usize;
            config.max_frames = config.max_frames.min(frames);
        }
        let request = self.prompt_request(take)?;
        let device = self.device();
        let prompt = Qwen3TtsPromptRows::build(&self.construct.prompt, &request)?
            .embed(&self.construct.model.talker, &device);
        if let Some(seed) = take.seed {
            device.seed(seed);
        }
        Ok(self
            .construct
            .model
            .generate_codes(&prompt, &self.construct.prompt, &config))
    }

    fn render(
        &self,
        take: &Qwen3TtsCodes,
    ) -> BunsenResult<Vec<f32>> {
        if take.frames.is_empty() {
            return Ok(Vec::new());
        }
        let audio = self.construct.codec.chunked_decode(
            take.codec_input(&self.device()),
            self.config.decode_chunk,
            self.config.decode_context,
        );
        let samples: Vec<f32> = audio
            .cast(DType::F32)
            .into_data()
            .try_into_vec_as::<f32>()
            .map_err(|e| BunsenError::internal(format!("the codec's audio: {e:?}")))?;
        if samples.iter().any(|x| !x.is_finite()) {
            return Err(BunsenError::internal("the codec's audio is not finite"));
        }
        Ok(samples)
    }

    fn take_info(
        &self,
        take: &Qwen3TtsCodes,
    ) -> TtsTakeInfo {
        TtsTakeInfo {
            frames: take.frames.len(),
            steps: take.frames.len(),
            stop: match take.stop {
                Qwen3TtsStop::Eos => TtsStop::End,
                Qwen3TtsStop::MaxFrames => TtsStop::Budget,
            },
        }
    }
}
