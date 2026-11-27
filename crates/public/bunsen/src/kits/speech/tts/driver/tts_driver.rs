//! [`TtsDriver`]: a request through its stages and an engine, to speech.

use alloc::{
    boxed::Box,
    format,
    string::String,
    sync::Arc,
    vec,
    vec::Vec,
};
use core::fmt::Debug;

use burn::config::Config;

use super::{
    StripInstruction,
    TtsCapabilities,
    TtsChunkPolicy,
    TtsEngine,
    TtsRequest,
    TtsRequestStage,
    TtsSpeechStream,
    TtsStageNote,
    TtsTakeInfo,
    TtsTakeRequest,
    TtsTextStage,
    TtsVoiceInfo,
    TtsVoicePrint,
    TtsVoiceReference,
    stream::{
        TtsWholeSpeechStream,
        driver_speech_stream,
    },
    tts_split_text,
};
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::support::text::{
        TextNormalizationConfig,
        TextNormalizer,
    },
};

/// Config for [`TtsDriver`]: what the driver does to a request before its
/// engine sees it.
#[derive(Config, Debug)]
pub struct TtsDriverConfig {
    /// How a request's text is normalized. A model's release has its own
    /// (ZONOS3's
    /// [`zonos3_release_normalization`](crate::kits::speech::tts::zonos3::driver::zonos3_release_normalization));
    /// the default is the text as typed.
    #[config(default = "TextNormalizationConfig::None")]
    pub normalization: TextNormalizationConfig,

    /// Whether a request's instruction is dropped, with a note, rather than
    /// refused by an engine that takes none. An engine that takes one is
    /// still given none.
    #[config(default = false)]
    pub strip_instruction: bool,

    /// Whether a long text is spoken as several takes
    /// ([`tts_split_text`]), each at most
    /// [`max_take_chars`](Self::max_take_chars) (or the engine's
    /// [limit](TtsEngine::max_take_chars), if lower); else one take, however
    /// long.
    #[config(default = true)]
    pub split: bool,

    /// The most characters a take speaks, after normalization: about 20
    /// seconds of speech.
    #[config(default = 300)]
    pub max_take_chars: usize,

    /// The silence between takes, in seconds.
    #[config(default = 0.1)]
    pub take_gap_seconds: f64,
}

/// What a [`TtsDriver`] spoke.
#[derive(Clone, Debug, PartialEq)]
pub struct TtsSpeech {
    /// The samples, mono, in `[-1, 1]`.
    pub samples: Vec<f32>,

    /// Their rate, in Hz.
    pub sample_rate: u32,

    /// Each take, in order.
    pub takes: Vec<TtsTakeInfo>,

    /// What the request's stages did worth saying.
    pub notes: Vec<TtsStageNote>,
}

impl TtsSpeech {
    /// Its length, in seconds.
    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }
}

/// Text to speech, whatever the model: what a tool or a test holds, as
/// `Arc<dyn TtsSynthesizer>`. Every [`TtsDriver`] is one.
pub trait TtsSynthesizer: Debug + Send + Sync {
    /// What a request may ask of it.
    fn capabilities(&self) -> &TtsCapabilities;

    /// The output's sample rate, in Hz.
    fn sample_rate(&self) -> u32;

    /// Its stock voices.
    fn voices(&self) -> Vec<TtsVoiceInfo>;

    /// `request`, spoken.
    ///
    /// # Errors
    /// As [`TtsDriver::synthesize`].
    fn synthesize(
        &self,
        request: &TtsRequest,
    ) -> BunsenResult<TtsSpeech>;

    /// `request`, spoken as a stream of events ([`TtsSpeechStream`]),
    /// each take's final frames decoded as `chunks` groups them; joined, its
    /// audio is [`synthesize`](Self::synthesize)'s.
    ///
    /// The default synthesizes the request whole at the call and says it as
    /// events: a correct stream, not an early one. A [`TtsDriver`]'s is its
    /// own ([`TtsDriver::stream`]).
    ///
    /// # Errors
    /// As [`TtsDriver::stream`]; the default's as `synthesize`.
    fn stream<'a>(
        &'a self,
        request: &TtsRequest,
        chunks: TtsChunkPolicy,
    ) -> BunsenResult<Box<dyn TtsSpeechStream + 'a>> {
        chunks.check()?;
        Ok(Box::new(TtsWholeSpeechStream::new(
            self.synthesize(request)?,
        )))
    }

    /// The voice built from `reference` for this model
    /// ([`TtsEngine::build_voice`]); the default clones none.
    ///
    /// # Errors
    /// As [`TtsEngine::build_voice`].
    fn build_voice(
        &self,
        reference: &TtsVoiceReference,
        mode: Option<&str>,
    ) -> BunsenResult<TtsVoicePrint> {
        self.capabilities().clone_mode(reference, mode)?;
        Err(BunsenError::unsupported("this model clones no voices"))
    }
}

/// The TTS driver: a request through its stages (text normalization, then
/// any others), its voice and language resolved by the engine, then the
/// engine's take, rendered.
///
/// One driver serves every model: an engine ([`TtsEngine`]) supplies what
/// depends on the model, and nothing here does.
#[derive(Debug)]
pub struct TtsDriver<E: TtsEngine> {
    engine: E,
    stages: Vec<Arc<dyn TtsRequestStage>>,
    config: TtsDriverConfig,
}

impl<E: TtsEngine> TtsDriver<E> {
    /// The driver of `engine`, its stages from `config`: the text stage,
    /// then [`StripInstruction`] when it is asked for.
    ///
    /// # Errors
    /// The normalizer's, building it;
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for takes of no
    /// characters, or a gap that is negative or not finite.
    pub fn new(
        engine: E,
        config: TtsDriverConfig,
    ) -> BunsenResult<Self> {
        if config.max_take_chars == 0 {
            return Err(BunsenError::illegal("a take of no characters"));
        }
        if !(config.take_gap_seconds.is_finite() && config.take_gap_seconds >= 0.0) {
            return Err(BunsenError::illegal(format!(
                "a gap between takes of {} s",
                config.take_gap_seconds
            )));
        }
        let normalizer = config.normalization.try_init()?;
        let mut stages: Vec<Arc<dyn TtsRequestStage>> =
            vec![Arc::new(TtsTextStage::new(normalizer))];
        if config.strip_instruction {
            stages.push(Arc::new(StripInstruction));
        }
        Ok(Self {
            engine,
            stages,
            config,
        })
    }

    /// With `normalizer` as its text stage, in place of the config's: a
    /// normalizer bunsen does not ship, plugged in.
    pub fn with_normalizer(
        mut self,
        normalizer: Arc<dyn TextNormalizer>,
    ) -> Self {
        self.stages[0] = Arc::new(TtsTextStage::new(normalizer));
        self
    }

    /// With `stage` run after the others.
    pub fn with_stage(
        mut self,
        stage: Arc<dyn TtsRequestStage>,
    ) -> Self {
        self.stages.push(stage);
        self
    }

    /// The engine.
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// The config.
    pub fn config(&self) -> &TtsDriverConfig {
        &self.config
    }

    /// The request's stages, in order.
    pub fn stages(&self) -> &[Arc<dyn TtsRequestStage>] {
        &self.stages
    }

    /// The text's takes: the text whole, or split
    /// ([`split`](TtsDriverConfig::split)) at the lower of the config's and
    /// the engine's limits.
    pub fn takes(
        &self,
        text: &str,
    ) -> Vec<String> {
        if !self.config.split {
            let text = text.trim();
            return if text.is_empty() {
                Vec::new()
            } else {
                vec![text.into()]
            };
        }
        let limit = self
            .engine
            .max_take_chars()
            .map_or(self.config.max_take_chars, |engine| {
                engine.min(self.config.max_take_chars)
            });
        tts_split_text(text, limit.max(1))
    }

    /// `request`, spoken: its voice resolved, its stages run in its
    /// language, its text split into takes ([`takes`](Self::takes)), then
    /// each take the engine's, rendered, the takes joined by the config's
    /// gap. Take `i`'s seed is the request's plus `i`, so a seeded request
    /// repeats; [`max_seconds`](TtsRequest::max_seconds) bounds each take.
    ///
    /// # Errors
    /// The engine's, resolving the voice or the language, generating or
    /// rendering; a stage's; and
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for no text to
    /// speak, or an instruction to an engine that takes none, naming the
    /// opt-in.
    pub fn synthesize(
        &self,
        request: &TtsRequest,
    ) -> BunsenResult<TtsSpeech> {
        let TtsPrepared {
            voice,
            request,
            notes,
            texts,
        } = self.prepare(request)?;
        let sample_rate = self.engine.sample_rate();
        let gap = self.gap_samples();
        let mut samples = Vec::new();
        let mut takes = Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
            let take = self.engine.generate(&TtsTakeRequest {
                text,
                voice: &voice,
                language: request.language.as_deref(),
                instruction: request.instruction.as_deref(),
                seed: request.seed.map(|seed| seed.wrapping_add(i as u64)),
                max_seconds: request.max_seconds,
                sampling: &request.sampling,
            })?;
            takes.push(self.engine.take_info(&take));
            if i > 0 {
                samples.resize(samples.len() + gap, 0.0);
            }
            samples.extend(self.engine.render(&take)?);
        }
        Ok(TtsSpeech {
            samples,
            sample_rate,
            takes,
            notes,
        })
    }

    /// The silence between two takes, in samples.
    pub fn gap_samples(&self) -> usize {
        (self.config.take_gap_seconds * f64::from(self.engine.sample_rate())).round() as usize
    }

    /// `request` as [`synthesize`](Self::synthesize) and
    /// [`stream`](Self::stream) take it: its voice resolved, its stages run
    /// in its language, the instruction checked, its text split into takes.
    pub(crate) fn prepare(
        &self,
        request: &TtsRequest,
    ) -> BunsenResult<TtsPrepared<E::Voice>> {
        let voice = self.engine.resolve_voice(&request.voice)?;
        let tag = self
            .engine
            .language_tag(request.language.as_deref(), &voice)?;
        let mut request = request.clone();
        let mut notes = Vec::new();
        for stage in &self.stages {
            stage.check_language(tag.as_ref())?;
            let (next, said) = stage.apply(request, tag.as_ref())?;
            request = next;
            notes.extend(said);
        }
        if request.instruction.is_some() && !self.engine.capabilities().instruction {
            return Err(BunsenError::illegal(
                "this model takes no instruction: drop it, or have the driver strip it \
                 (`strip_instruction`)",
            ));
        }
        let texts = self.takes(&request.text);
        if texts.is_empty() {
            return Err(BunsenError::illegal("no text to speak"));
        }
        Ok(TtsPrepared {
            voice,
            request,
            notes,
            texts,
        })
    }
}

/// A request prepared for its takes ([`TtsDriver::prepare`]).
pub(crate) struct TtsPrepared<V> {
    /// The voice, resolved.
    pub(crate) voice: V,

    /// The request, after its stages.
    pub(crate) request: TtsRequest,

    /// What the stages said.
    pub(crate) notes: Vec<TtsStageNote>,

    /// The takes' texts.
    pub(crate) texts: Vec<String>,
}

impl<E: TtsEngine> TtsSynthesizer for TtsDriver<E> {
    fn capabilities(&self) -> &TtsCapabilities {
        self.engine.capabilities()
    }

    fn sample_rate(&self) -> u32 {
        self.engine.sample_rate()
    }

    fn voices(&self) -> Vec<TtsVoiceInfo> {
        self.engine.voices()
    }

    fn synthesize(
        &self,
        request: &TtsRequest,
    ) -> BunsenResult<TtsSpeech> {
        TtsDriver::synthesize(self, request)
    }

    fn stream<'a>(
        &'a self,
        request: &TtsRequest,
        chunks: TtsChunkPolicy,
    ) -> BunsenResult<Box<dyn TtsSpeechStream + 'a>> {
        driver_speech_stream(self, request, chunks)
    }

    fn build_voice(
        &self,
        reference: &TtsVoiceReference,
        mode: Option<&str>,
    ) -> BunsenResult<TtsVoicePrint> {
        self.engine.build_voice(reference, mode)
    }
}
