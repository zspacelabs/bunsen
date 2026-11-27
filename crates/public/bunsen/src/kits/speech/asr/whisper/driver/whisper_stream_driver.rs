//! # The Whisper stream driver: the core's driver over a Whisper engine.

use alloc::sync::Arc;

use burn::{
    config::Config,
    module::Module,
    tensor::Device,
};

use crate::{
    errors::{
        BunsenResult,
        ConstraintError,
        ResultContext,
    },
    kits::{
        speech::{
            asr::{
                driver::{
                    AsrStreamDriver,
                    SpeakerCutConfig,
                },
                whisper::{
                    DecodeConfig,
                    Whisper,
                    WhisperFallbackConfig,
                    WhisperMeta,
                    blocks::AUDIO_ENCODER_STRIDE,
                    driver::{
                        WhisperConstruct,
                        WhisperPolicy,
                        WhisperTask,
                        WhisperTokenLayout,
                    },
                    logit_filters::ApplyTimestampRules,
                },
            },
            support::log_mel::LogMelFrontEndConfig,
        },
        support::{
            decoding::{
                LogitFilter,
                SearchConfig,
            },
            tokens::Detokenizer,
        },
    },
    ops::signal::perceptive_audio::{
        PerceptiveAudioConverter,
        PerceptiveAudioConverterMeta,
    },
};

/// Config for a [`WhisperEngine`]: what each window is decoded as.
///
/// Set the language, task, timestamps, search and fallback ladder here, then
/// build the engine over a loaded [`WhisperConstruct`] with
/// [`try_init`](Self::try_init), and the driver over the engine and an
/// [`EmissionPolicy`](crate::kits::speech::asr::driver::EmissionPolicy) with [`AsrStreamDriver::new`], as for every ASR engine.
/// A bare model is a construct through [`WhisperConstruct::from_model`], or
/// [`WhisperConstruct::new`] with an explicit token layout. The engine keeps
/// a copy of this config; the driver opens a
/// [`WhisperStreamContext`](super::WhisperStreamContext) per stream with
/// [`new_context`](WhisperStreamDriver::new_context).
#[derive(Config, Debug)]
pub struct WhisperEngineConfig {
    /// The language of the speech, as a
    /// [`LANGUAGES`](crate::kits::speech::asr::whisper::blocks::LANGUAGES)
    /// code.
    ///
    /// `None` on a multilingual checkpoint detects the language per stream
    /// from its first committed window, as upstream's `transcribe()` does
    /// from its first; a draft before that detects one for its own decode
    /// and keeps nothing. Must be `None` for an English-only checkpoint,
    /// which takes no language token.
    #[config(default = "None")]
    pub language: Option<String>,

    /// Transcribe, or translate to English. Ignored by an English-only
    /// checkpoint, which takes no task token.
    #[config(default = "WhisperTask::Transcribe")]
    pub task: WhisperTask,

    /// Let the model emit timestamp tokens, under upstream's timestamp
    /// rules. Emissions are then split on them, and the seek pointer
    /// advances to the last closed timestamp rather than a whole window.
    #[config(default = "false")]
    pub timestamps: bool,

    /// Under `timestamps`, the latest time the first timestamp of a window
    /// may name, in seconds; upstream's default is one second.
    #[config(default = "Some(1.0)")]
    pub max_initial_timestamp: Option<f64>,

    /// Cap on tokens generated per window.
    #[config(default = "224")]
    pub max_tokens: usize,

    /// The search of each window: its beams, patience and ranking. Its
    /// temperature and trajectories are each rung's of the
    /// [`fallback`](Self::fallback) ladder.
    #[config(default = "SearchConfig::new()")]
    pub search: SearchConfig,

    /// Prompt each window with the tail of the transcript so far, after
    /// `<|startofprev|>`, as upstream's `condition_on_previous_text` does.
    #[config(default = "true")]
    pub condition_on_previous_text: bool,

    /// With a diarizer attached, a window is cut at a change of speaker
    /// before it is decoded
    /// ([`SpeakerCutConfig::cut`](crate::kits::speech::asr::driver::SpeakerCutConfig::cut)), so a
    /// segment is one speaker's as far as the turns go; `None` decodes
    /// windows as the driver's emission policy cuts them.
    #[config(default = "Some(SpeakerCutConfig::new())")]
    pub speaker_cut: Option<SpeakerCutConfig>,

    /// The temperature ladder and its thresholds. The default ladder is
    /// temperature zero alone; [`WhisperFallbackConfig::upstream`] is
    /// `transcribe()`'s.
    #[config(default = "WhisperFallbackConfig::new()")]
    pub fallback: WhisperFallbackConfig,
}

impl WhisperEngineConfig {
    /// Sets the decode fields from `policy`: the task, the language, the
    /// timestamps, the beam size (the rest of the search as it is), the token
    /// cap, the prompt carry, and the fallback (upstream's ladder, or
    /// temperature zero alone). A construct's
    /// [`policy`](WhisperConstruct::policy) is what its presets resolved to.
    pub fn with_policy(
        self,
        policy: &WhisperPolicy,
    ) -> Self {
        let search = self.search.clone().with_beam_size(policy.beam_size);
        self.with_task(policy.task)
            .with_language(policy.language.clone())
            .with_timestamps(policy.timestamps)
            .with_search(search)
            .with_max_tokens(policy.max_tokens)
            .with_condition_on_previous_text(policy.condition_on_previous_text)
            .with_fallback(if policy.fallback {
                WhisperFallbackConfig::upstream()
            } else {
                WhisperFallbackConfig::new()
            })
    }

    /// Builds the engine over a loaded construct, sharing it.
    ///
    /// See the [errors convention](crate::errors#convention-try_x-and-x).
    ///
    /// The model and its layout come from the construct. So do, when the
    /// construct has a vocabulary, upstream's default logit filters and, under
    /// the `tokenizer` feature, a detokenizer, so emissions carry text;
    /// the driver's
    /// [`with_logit_filters`](WhisperStreamDriver::with_logit_filters) and
    /// [`with_detokenizer`](WhisperStreamDriver::with_detokenizer) replace
    /// either. Several engines over one construct are `Arc::clone`s of it.
    ///
    /// # Errors
    /// - [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if the layout
    ///   does not fit the model or its vocabulary (see
    ///   [`WhisperConstruct::try_validate`]), if a language or task is given
    ///   for an English-only layout, or if the configuration breaks a rule: a
    ///   zero `search.beam_size` or `fallback.ladder.best_of`, no fallback
    ///   temperature, or a `search.patience` that collects no candidates. A
    ///   broken field rule carries a [`ConstraintError`].
    /// - [`Lookup`](crate::errors::BunsenErrorKind::Lookup) if the language is
    ///   not one the layout has (see [`WhisperTokenLayout::sot_sequence`]).
    /// - As [`LogMelFrontEndConfig::try_init_audio_converter`] for the model's
    ///   front end.
    pub fn try_init(
        &self,
        construct: Arc<WhisperConstruct>,
        device: &Device,
    ) -> BunsenResult<WhisperEngine> {
        construct.try_validate()?;
        let special_ids = *construct.layout.ids();

        if !special_ids.is_multilingual() && self.language.is_some() {
            return Err(crate::errors::ConstraintError::custom(
                "WhisperEngineConfig",
                "language",
                "an English-only checkpoint takes no language",
            )
            .into());
        }
        let task = special_ids.is_multilingual().then_some(self.task);
        // A multilingual checkpoint with no language detects it per
        // stream; its prompt is built when the language is known.
        let prompt = match (special_ids.is_multilingual(), self.language.as_deref()) {
            (true, None) => Vec::new(),
            (_, language) => construct
                .layout
                .sot_sequence(language, task, self.timestamps)?,
        };
        let max_initial_timestamp_index = self.max_initial_timestamp.map(|seconds| {
            (seconds / construct.layout.layout().timestamp_step_seconds).round() as usize
        });
        let mut filters: Vec<Arc<dyn LogitFilter>> = construct.default_filters();
        if self.timestamps {
            filters.push(Arc::new(ApplyTimestampRules::new(
                &special_ids,
                max_initial_timestamp_index,
            )));
        }

        const OWNER: &str = "WhisperEngineConfig";
        self.search
            .try_validate()
            .context("WhisperEngineConfig.search")?;
        if self.fallback.ladder.temperatures.is_empty() {
            return Err(
                ConstraintError::zero_or_empty(OWNER, "fallback.ladder.temperatures").into(),
            );
        }
        if self.fallback.ladder.best_of == Some(0) {
            return Err(ConstraintError::zero_or_empty(OWNER, "fallback.ladder.best_of").into());
        }

        let audio_converter = construct
            .model
            .front_end()
            .try_init_audio_converter(construct.model.n_mels(), device)?;

        #[cfg(feature = "tokenizer")]
        let detokenizer: Option<Arc<dyn Detokenizer>> = construct
            .detokenizer()?
            .map(|detokenizer| Arc::new(detokenizer) as Arc<dyn Detokenizer>);
        #[cfg(not(feature = "tokenizer"))]
        let detokenizer: Option<Arc<dyn Detokenizer>> = None;

        let engine = WhisperEngine {
            config: self.clone(),
            construct,
            audio_converter,
            prompt,
            task,
            max_initial_timestamp_index,
            filters,
            detokenizer,
        };
        Ok(engine)
    }

    /// [`try_init`](Self::try_init), panicking.
    ///
    /// # Panics
    /// As [`try_init`](Self::try_init) errs.
    pub fn init(
        &self,
        construct: Arc<WhisperConstruct>,
        device: &Device,
    ) -> WhisperEngine {
        crate::errors::WithOkOrPanic::ok_or_panic(self.try_init(construct, device))
    }
}

/// The Whisper stream driver: the shared, immutable half of a
/// transcription, the core's [`AsrStreamDriver`] over a [`WhisperEngine`].
///
/// Built as every ASR driver is: [`AsrStreamDriver::new`] over a
/// [`WhisperEngine`] ([`WhisperEngineConfig::try_init`]) and an
/// [`EmissionPolicy`](crate::kits::speech::asr::driver::EmissionPolicy), then optionally given a voice-activity model with
/// [`with_vad`](AsrStreamDriver::with_vad). Opens streams with
/// [`new_context`](AsrStreamDriver::new_context); each
/// [`WhisperStreamContext`](super::WhisperStreamContext) holds a clone of
/// the driver, which is cheap (the construct is an `Arc`, and tensors are
/// shared), so one driver serves any number of streams in one process and
/// [`advance_ready`](crate::kits::speech::asr::driver::AsrStreamDriver::advance_ready) batches
/// their decodes.
pub type WhisperStreamDriver = AsrStreamDriver<WhisperEngine>;

/// Whisper as an [`AsrEngine`](crate::kits::speech::asr::driver::AsrEngine):
/// the model and its token layout (a [`WhisperConstruct`] behind an `Arc`),
/// the mel front end, the logit filters, the decode settings and the
/// detokenizer.
///
/// The engine streams log-mels a hop at a time, floors each window against
/// a [`MaxLevel`](crate::kits::speech::support::MaxLevel), decodes up
/// the fallback ladder, splits a timestamped decode into segments, carries
/// the transcript's tail into the next prompt, and detects a stream's
/// language from its first committed window.
#[derive(Clone, Debug)]
pub struct WhisperEngine {
    pub(super) config: WhisperEngineConfig,
    pub(super) audio_converter: PerceptiveAudioConverter,
    /// The model, its layout and its vocabulary, shared.
    pub(super) construct: Arc<WhisperConstruct>,
    /// The sot sequence every window's decode opens with; empty when the
    /// language is detected per stream.
    pub(super) prompt: Vec<i64>,
    /// The task token's meaning; `None` for an English-only layout.
    pub(super) task: Option<WhisperTask>,
    pub(super) max_initial_timestamp_index: Option<usize>,
    /// Applied to the logits every step, in order: the caller's, then the
    /// timestamp rules when timestamps are on.
    pub(super) filters: Vec<Arc<dyn LogitFilter>>,
    pub(super) detokenizer: Option<Arc<dyn Detokenizer>>,
}

impl WhisperEngine {
    /// The driver configuration.
    pub fn config(&self) -> &WhisperEngineConfig {
        &self.config
    }

    /// The construct: the model, its layout and its vocabulary.
    pub fn construct(&self) -> &Arc<WhisperConstruct> {
        &self.construct
    }

    /// The model.
    pub fn whisper_model(&self) -> &Whisper {
        &self.construct.model
    }

    /// The mel front end.
    pub fn audio_converter(&self) -> &PerceptiveAudioConverter {
        &self.audio_converter
    }

    /// The token layout, derived from the model.
    pub fn token_layout(&self) -> &WhisperTokenLayout {
        &self.construct.layout
    }

    /// The sot sequence every window's decode opens with.
    pub fn prompt(&self) -> &[i64] {
        &self.prompt
    }

    /// The logit filters every decode applies.
    pub fn filters(&self) -> &[Arc<dyn LogitFilter>] {
        &self.filters
    }

    /// The task; `None` for an English-only layout.
    pub fn task(&self) -> Option<WhisperTask> {
        self.task
    }

    /// Whether streams detect their language, from their first committed
    /// window.
    pub fn detects_language(&self) -> bool {
        self.prompt.is_empty()
    }

    /// The sot sequence for `language`, under this driver's task and
    /// timestamp setting.
    ///
    /// # Errors
    /// As [`WhisperTokenLayout::sot_sequence`].
    pub fn sot_sequence(
        &self,
        language: Option<&str>,
    ) -> BunsenResult<Vec<i64>> {
        self.construct
            .layout
            .sot_sequence(language, self.task, self.config.timestamps)
    }

    /// Audio frames per timestamp index: two.
    pub fn frames_per_timestamp(&self) -> usize {
        AUDIO_ENCODER_STRIDE
    }

    /// The decode of one window under this driver, given its prompt.
    pub fn decode_config(
        &self,
        prompt: Vec<i64>,
    ) -> DecodeConfig {
        let ids = self.construct.layout.ids();
        DecodeConfig::new(prompt, ids.eot)
            .with_max_tokens(self.config.max_tokens)
            .with_search(self.config.search.clone())
            .with_sot_token(Some(ids.sot))
            .with_no_speech_token(Some(ids.no_speech))
    }

    /// The detokenizer, if one was attached.
    pub fn detokenizer(&self) -> Option<&Arc<dyn Detokenizer>> {
        self.detokenizer.as_ref()
    }

    /// Frames per decode window: the model's audio context.
    pub fn window_frames(&self) -> usize {
        self.construct.model.max_audio_ctx()
    }

    /// The audio front end the model's log-mels are computed with.
    pub fn front_end(&self) -> &LogMelFrontEndConfig {
        self.construct.model.front_end()
    }

    /// The encoder grid in samples: one timestamp step, which is
    /// [`frames_per_timestamp`](Self::frames_per_timestamp) mel hops. 320 at
    /// 16 kHz.
    pub fn encoder_grid(&self) -> usize {
        self.frames_per_timestamp() * self.audio_converter.hop()
    }

    /// The devices the model lives on.
    pub fn devices(&self) -> Vec<Device> {
        self.construct.model.devices()
    }
}

impl AsrStreamDriver<WhisperEngine> {
    /// Attaches a detokenizer, so emissions carry text as well as ids.
    pub fn with_detokenizer(
        self,
        detokenizer: Arc<dyn Detokenizer>,
    ) -> Self {
        self.map_engine(|mut engine| {
            engine.detokenizer = Some(detokenizer);
            engine
        })
    }

    /// Sets the logit filters every decode applies, in order, replacing
    /// any set before.
    pub fn with_logit_filters(
        self,
        filters: Vec<Arc<dyn LogitFilter>>,
    ) -> Self {
        self.map_engine(|mut engine| {
            engine.filters = filters;
            if engine.config.timestamps {
                engine.filters.push(Arc::new(ApplyTimestampRules::new(
                    engine.construct.layout.ids(),
                    engine.max_initial_timestamp_index,
                )));
            }
            engine
        })
    }
}
