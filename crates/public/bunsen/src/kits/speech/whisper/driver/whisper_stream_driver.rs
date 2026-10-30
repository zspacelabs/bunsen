//! # The Whisper stream driver: the core's driver over a Whisper engine.

use std::sync::Arc;

use burn::{
    config::Config,
    module::Module,
    tensor::Device,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
    },
    kits::{
        decoding::{
            LogitFilter,
            SearchConfig,
        },
        speech::{
            asr_driver::{
                AsrStreamDriver,
                EmissionPolicy,
                SpeakerCutConfig,
            },
            whisper::{
                DecodeConfig,
                Whisper,
                WhisperFallbackConfig,
                WhisperMeta,
                blocks::{
                    AUDIO_ENCODER_STRIDE,
                    WhisperFrontEndConfig,
                },
                driver::{
                    WhisperBundle,
                    WhisperTask,
                    WhisperTokenLayout,
                },
                logit_filters::ApplyTimestampRules,
            },
        },
        tokens::Detokenizer,
    },
    ops::signal::perceptive_audio::{
        PerceptiveAudioConverter,
        PerceptiveAudioConverterMeta,
    },
};

/// Config for [`WhisperStreamDriver`]: what each window is decoded as, and
/// when a stream emits.
///
/// Set the language, task, timestamps, search, [`EmissionPolicy`] and
/// fallback ladder here, then build the driver over a loaded
/// [`WhisperBundle`] with [`init_from_bundle`](Self::init_from_bundle),
/// over a bare model with [`init`](Self::init), or over a model and an
/// explicit token layout with [`init_with_layout`](Self::init_with_layout).
/// The driver keeps a copy of this config, and opens a
/// [`WhisperStreamContext`](super::WhisperStreamContext) per stream with
/// [`new_context`](WhisperStreamDriver::new_context).
#[derive(Config, Debug)]
pub struct WhisperStreamDriverConfig {
    /// The language of the speech, as a
    /// [`LANGUAGES`](crate::kits::speech::whisper::blocks::LANGUAGES)
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

    /// When to decode, and when a decode is final.
    ///
    /// A policy with the `endpoint` trigger needs a voice-activity model,
    /// attached with [`with_vad`](WhisperStreamDriver::with_vad).
    #[config(default = "EmissionPolicy::offline()")]
    pub emission: EmissionPolicy,

    /// With a diarizer attached, a window is cut at a change of speaker
    /// before it is decoded
    /// ([`speaker_cut`](crate::kits::speech::asr_driver::speaker_cut)), so a
    /// segment is one speaker's as far as the turns go; `None` decodes
    /// windows as the emission policy cuts them.
    #[config(default = "Some(SpeakerCutConfig::new())")]
    pub speaker_cut: Option<SpeakerCutConfig>,

    /// The temperature ladder and its thresholds. The default ladder is
    /// temperature zero alone; [`WhisperFallbackConfig::upstream`] is
    /// `transcribe()`'s.
    #[config(default = "WhisperFallbackConfig::new()")]
    pub fallback: WhisperFallbackConfig,
}

impl WhisperStreamDriverConfig {
    /// Builds the driver over a model alone, deriving the token layout from
    /// the model's vocabulary size: ids only, with no vocabulary, no text
    /// and no default suppress list. See
    /// [`init_from_bundle`](Self::init_from_bundle).
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, if the vocabulary size is not a Whisper
    /// layout, or as [`init_from_bundle`](Self::init_from_bundle).
    pub fn init(
        &self,
        model: Whisper,
        device: &Device,
    ) -> BunsenResult<WhisperStreamDriver> {
        self.init_from_bundle(Arc::new(WhisperBundle::from_model(model)?), device)
    }

    /// Builds the driver over a model with an explicit token layout, and no
    /// vocabulary.
    ///
    /// For a model whose vocabulary is not one of Whisper's &mdash; a test
    /// model &mdash; or to override what [`init`](Self::init) would derive.
    ///
    /// # Errors
    /// As [`init_from_bundle`](Self::init_from_bundle).
    pub fn init_with_layout(
        &self,
        whisper_model: Whisper,
        token_layout: WhisperTokenLayout,
        device: &Device,
    ) -> BunsenResult<WhisperStreamDriver> {
        self.init_from_bundle(
            Arc::new(WhisperBundle::new(whisper_model, token_layout)),
            device,
        )
    }

    /// Builds the driver over a loaded bundle, sharing it.
    ///
    /// The model and its layout come from the bundle. So do, when the
    /// bundle has a vocabulary, upstream's default logit filters and, under
    /// the `tokenizer` feature, a detokenizer, so emissions carry text;
    /// [`with_logit_filters`](WhisperStreamDriver::with_logit_filters) and
    /// [`with_detokenizer`](WhisperStreamDriver::with_detokenizer) replace
    /// either. Several drivers over one bundle are `Arc::clone`s of it.
    ///
    /// # Errors
    /// - [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if the layout
    ///   does not fit the model or its vocabulary (see
    ///   [`WhisperBundle::validate`]), if a language or task is given for an
    ///   English-only layout, or if the configuration breaks a rule: no trigger
    ///   that decodes, a zero `interval`, the `interval` trigger without
    ///   `endpoint` (which could never draft), a zero `search.beam_size` or
    ///   `fallback.ladder.best_of`, no fallback temperature, or a
    ///   `search.patience` that collects no candidates. A broken field rule
    ///   carries a [`ConstraintError`].
    /// - [`Lookup`](crate::errors::BunsenErrorKind::Lookup) if the language is
    ///   not one the layout has (see [`WhisperTokenLayout::sot_sequence`]).
    /// - [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for
    ///   [`CommitRule::Agreement`](crate::kits::speech::asr_driver::CommitRule::Agreement),
    ///   which is not implemented.
    /// - As [`WhisperFrontEndConfig::try_init_audio_converter`] for the model's
    ///   front end.
    pub fn init_from_bundle(
        &self,
        bundle: Arc<WhisperBundle>,
        device: &Device,
    ) -> BunsenResult<WhisperStreamDriver> {
        bundle.validate()?;
        let special_ids = *bundle.layout.ids();

        if !special_ids.is_multilingual() && self.language.is_some() {
            return Err(BunsenError::illegal(
                "WhisperStreamDriverConfig.language: an English-only checkpoint takes no language",
            ));
        }
        let task = special_ids.is_multilingual().then_some(self.task);
        // A multilingual checkpoint with no language detects it per
        // stream; its prompt is built when the language is known.
        let prompt = match (special_ids.is_multilingual(), self.language.as_deref()) {
            (true, None) => Vec::new(),
            (_, language) => bundle
                .layout
                .sot_sequence(language, task, self.timestamps)?,
        };
        let max_initial_timestamp_index = self.max_initial_timestamp.map(|seconds| {
            (seconds / bundle.layout.layout().timestamp_step_seconds).round() as usize
        });
        let mut filters: Vec<Arc<dyn LogitFilter>> = bundle.default_filters();
        if self.timestamps {
            filters.push(Arc::new(ApplyTimestampRules::new(
                &special_ids,
                max_initial_timestamp_index,
            )));
        }

        const OWNER: &str = "WhisperStreamDriverConfig";
        let search = &self.search;
        if search.beam_size == 0 {
            return Err(ConstraintError::zero_or_empty(OWNER, "search.beam_size").into());
        }
        if self.fallback.ladder.temperatures.is_empty() {
            return Err(
                ConstraintError::zero_or_empty(OWNER, "fallback.ladder.temperatures").into(),
            );
        }
        if self.fallback.ladder.best_of == Some(0) {
            return Err(ConstraintError::zero_or_empty(OWNER, "fallback.ladder.best_of").into());
        }
        if (search.beam_size as f64 * search.patience.unwrap_or(1.0)).round() < 1.0 {
            // `round(beam_size * patience) >= 1` is `patience >= 0.5 /
            // beam_size`.
            return Err(ConstraintError::out_of_range(
                OWNER,
                "search.patience",
                search.patience.unwrap_or(1.0),
                format!(
                    "[{}, ..) with {} beams, to collect a candidate",
                    0.5 / search.beam_size as f64,
                    search.beam_size,
                ),
            )
            .into());
        }

        let audio_converter = bundle
            .model
            .front_end()
            .try_init_audio_converter(bundle.model.n_mels(), device)?;

        #[cfg(feature = "tokenizer")]
        let detokenizer: Option<Arc<dyn Detokenizer>> = bundle
            .detokenizer()?
            .map(|detokenizer| Arc::new(detokenizer) as Arc<dyn Detokenizer>);
        #[cfg(not(feature = "tokenizer"))]
        let detokenizer: Option<Arc<dyn Detokenizer>> = None;

        let engine = WhisperEngine {
            config: self.clone(),
            bundle,
            audio_converter,
            prompt,
            task,
            max_initial_timestamp_index,
            filters,
            detokenizer,
        };
        AsrStreamDriver::new(engine, self.emission.clone())
    }
}

/// The Whisper stream driver: the shared, immutable half of a
/// transcription, the core's [`AsrStreamDriver`] over a [`WhisperEngine`].
///
/// Built by [`WhisperStreamDriverConfig`]
/// ([`init_from_bundle`](WhisperStreamDriverConfig::init_from_bundle) and
/// its siblings), then optionally given a voice-activity model with
/// [`with_vad`](AsrStreamDriver::with_vad). Opens streams with
/// [`new_context`](AsrStreamDriver::new_context); each
/// [`WhisperStreamContext`](super::WhisperStreamContext) holds a clone of
/// the driver, which is cheap (the bundle is an `Arc`, and tensors are
/// shared), so one driver serves any number of streams in one process and
/// [`advance_ready`](crate::kits::speech::asr_driver::advance_ready) batches
/// their decodes.
pub type WhisperStreamDriver = AsrStreamDriver<WhisperEngine>;

/// Whisper as an [`AsrEngine`](crate::kits::speech::asr_driver::AsrEngine):
/// the model and its token layout (a [`WhisperBundle`] behind an `Arc`),
/// the mel front end, the logit filters, the decode settings and the
/// detokenizer.
///
/// The engine streams log-mels a hop at a time, floors each window against
/// a [`MaxLevel`](crate::kits::speech::asr_driver::MaxLevel), decodes up
/// the fallback ladder, splits a timestamped decode into segments, carries
/// the transcript's tail into the next prompt, and detects a stream's
/// language from its first committed window.
#[derive(Clone, Debug)]
pub struct WhisperEngine {
    pub(super) config: WhisperStreamDriverConfig,
    pub(super) audio_converter: PerceptiveAudioConverter,
    /// The model, its layout and its vocabulary, shared.
    pub(super) bundle: Arc<WhisperBundle>,
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
    pub fn config(&self) -> &WhisperStreamDriverConfig {
        &self.config
    }

    /// The bundle: the model, its layout and its vocabulary.
    pub fn bundle(&self) -> &Arc<WhisperBundle> {
        &self.bundle
    }

    /// The model.
    pub fn whisper_model(&self) -> &Whisper {
        &self.bundle.model
    }

    /// The mel front end.
    pub fn audio_converter(&self) -> &PerceptiveAudioConverter {
        &self.audio_converter
    }

    /// The token layout, derived from the model.
    pub fn token_layout(&self) -> &WhisperTokenLayout {
        &self.bundle.layout
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
        self.bundle
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
        let ids = self.bundle.layout.ids();
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
        self.bundle.model.max_audio_ctx()
    }

    /// The audio front end the model's log-mels are computed with.
    pub fn front_end(&self) -> &WhisperFrontEndConfig {
        self.bundle.model.front_end()
    }

    /// The encoder grid in samples: one timestamp step, which is
    /// [`frames_per_timestamp`](Self::frames_per_timestamp) mel hops. 320 at
    /// 16 kHz.
    pub fn encoder_grid(&self) -> usize {
        self.frames_per_timestamp() * self.audio_converter.hop()
    }

    /// The devices the model lives on.
    pub fn devices(&self) -> Vec<Device> {
        self.bundle.model.devices()
    }
}

impl AsrStreamDriver<WhisperEngine> {
    /// The driver configuration.
    pub fn config(&self) -> &WhisperStreamDriverConfig {
        self.engine().config()
    }

    /// The bundle: the model, its layout and its vocabulary.
    pub fn bundle(&self) -> &Arc<WhisperBundle> {
        self.engine().bundle()
    }

    /// The model.
    pub fn whisper_model(&self) -> &Whisper {
        self.engine().whisper_model()
    }

    /// The mel front end.
    pub fn audio_converter(&self) -> &PerceptiveAudioConverter {
        self.engine().audio_converter()
    }

    /// The token layout, derived from the model.
    pub fn token_layout(&self) -> &WhisperTokenLayout {
        self.engine().token_layout()
    }

    /// The sot sequence every window's decode opens with.
    pub fn prompt(&self) -> &[i64] {
        self.engine().prompt()
    }

    /// The logit filters every decode applies.
    pub fn filters(&self) -> &[Arc<dyn LogitFilter>] {
        self.engine().filters()
    }

    /// Whether streams detect their language, from their first committed
    /// window.
    pub fn detects_language(&self) -> bool {
        self.engine().detects_language()
    }

    /// Audio frames per timestamp index: two.
    pub fn frames_per_timestamp(&self) -> usize {
        self.engine().frames_per_timestamp()
    }

    /// The decode of one window under this driver, given its prompt.
    pub fn decode_config(
        &self,
        prompt: Vec<i64>,
    ) -> DecodeConfig {
        self.engine().decode_config(prompt)
    }

    /// Frames per decode window: the model's audio context.
    pub fn window_frames(&self) -> usize {
        self.engine().window_frames()
    }

    /// The audio front end the model's log-mels are computed with.
    pub fn front_end(&self) -> &WhisperFrontEndConfig {
        self.engine().front_end()
    }

    /// The devices the model lives on.
    pub fn devices(&self) -> Vec<Device> {
        self.engine().devices()
    }

    /// The encoder grid in samples: one timestamp step. 320 at 16 kHz.
    pub fn encoder_grid(&self) -> usize {
        self.engine().encoder_grid()
    }

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
                    engine.bundle.layout.ids(),
                    engine.max_initial_timestamp_index,
                )));
            }
            engine
        })
    }
}
