use std::sync::Arc;

use burn::{
    config::Config,
    module::Module,
    prelude::Backend,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
        Rule,
    },
    kits::{
        speech::{
            silero_vad::{
                SileroVad,
                SileroVadMeta,
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
                    CommitRule,
                    EmissionPolicy,
                    StreamClampPolicy,
                    StreamClock,
                    VoiceActivityFilterConfig,
                    WhisperBundle,
                    WhisperStreamContext,
                    WhisperTask,
                    WhisperTokenLayout,
                },
                logit_filters::{
                    ApplyTimestampRules,
                    LogitFilter,
                },
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
/// [`WhisperStreamContext`] per stream with
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

    /// Beams per window; one is greedy.
    #[config(default = "1")]
    pub beam_size: usize,

    /// Finished candidates a beam search collects before stopping, as a
    /// multiple of the beam size; `None` is one.
    #[config(default = "None")]
    pub patience: Option<f64>,

    /// The exponent of the ranker's length penalty; `None` normalizes by
    /// length.
    #[config(default = "None")]
    pub length_penalty: Option<f64>,

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
    pub fn init<B: Backend>(
        &self,
        model: Whisper<B>,
        device: &B::Device,
    ) -> BunsenResult<WhisperStreamDriver<B>> {
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
    pub fn init_with_layout<B: Backend>(
        &self,
        whisper_model: Whisper<B>,
        token_layout: WhisperTokenLayout,
        device: &B::Device,
    ) -> BunsenResult<WhisperStreamDriver<B>> {
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
    ///   `endpoint` (which could never draft), a zero `beam_size` or `best_of`,
    ///   no fallback temperature, or a `patience` that collects no candidates.
    ///   A broken field rule carries a [`ConstraintError`].
    /// - [`Lookup`](crate::errors::BunsenErrorKind::Lookup) if the language is
    ///   not one the layout has (see [`WhisperTokenLayout::sot_sequence`]).
    /// - [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for
    ///   [`CommitRule::Agreement`], which is not implemented.
    /// - As [`WhisperFrontEndConfig::try_init_audio_converter`] for the model's
    ///   front end.
    pub fn init_from_bundle<B: Backend>(
        &self,
        bundle: Arc<WhisperBundle<B>>,
        device: &B::Device,
    ) -> BunsenResult<WhisperStreamDriver<B>> {
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
        let mut filters: Vec<Arc<dyn LogitFilter<B>>> = bundle.default_filters();
        if self.timestamps {
            filters.push(Arc::new(ApplyTimestampRules::new(
                &special_ids,
                max_initial_timestamp_index,
            )));
        }

        const OWNER: &str = "WhisperStreamDriverConfig";
        let triggers = &self.emission.triggers;
        if !triggers.window_full && !triggers.endpoint {
            return Err(BunsenError::illegal(
                "WhisperStreamDriverConfig.emission.triggers: with neither the window_full nor \
                 the endpoint trigger nothing would ever decode",
            ));
        }
        if triggers.interval.is_some_and(|i| i.is_zero()) {
            return Err(BunsenError::from(ConstraintError::zero_or_empty(
                OWNER,
                "emission.triggers.interval",
            ))
            .with_details("an interval of zero would draft on every push"));
        }
        if triggers.interval.is_some() && !triggers.endpoint {
            return Err(BunsenError::illegal(
                "WhisperStreamDriverConfig.emission.triggers: the interval trigger drafts only \
                 while speech is in progress, which only the endpoint trigger's voice-activity \
                 gate tracks; turn endpoint on as well",
            ));
        }
        if let CommitRule::Agreement { .. } = self.emission.commit {
            return Err(BunsenError::unsupported(
                "WhisperStreamDriverConfig.emission.commit: the Agreement commit rule is not \
                 implemented; use Complete or LastTimestamp",
            ));
        }

        if self.beam_size == 0 {
            return Err(ConstraintError::zero_or_empty(OWNER, "beam_size").into());
        }
        if self.fallback.temperatures.is_empty() {
            return Err(ConstraintError::zero_or_empty(OWNER, "fallback.temperatures").into());
        }
        if self.fallback.best_of == Some(0) {
            return Err(ConstraintError::zero_or_empty(OWNER, "fallback.best_of").into());
        }
        if (self.beam_size as f64 * self.patience.unwrap_or(1.0)).round() < 1.0 {
            // `round(beam_size * patience) >= 1` is `patience >= 0.5 /
            // beam_size`.
            return Err(ConstraintError::out_of_range(
                OWNER,
                "patience",
                self.patience.unwrap_or(1.0),
                format!(
                    "[{}, ..) with {} beams, to collect a candidate",
                    0.5 / self.beam_size as f64,
                    self.beam_size,
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

        Ok(WhisperStreamDriver {
            config: self.clone(),
            bundle,
            audio_converter,
            vad_model: None,
            prompt,
            task,
            max_initial_timestamp_index,
            filters,
            va_filter: None,
            detokenizer,
        })
    }
}

/// The shared, immutable half of a transcription: what every stream needs
/// and none of them mutates.
///
/// Built by [`WhisperStreamDriverConfig`]
/// ([`init_from_bundle`](WhisperStreamDriverConfig::init_from_bundle) and
/// its siblings), then optionally given a voice-activity model with
/// [`with_vad`](Self::with_vad). Opens streams with
/// [`new_context`](Self::new_context): each [`WhisperStreamContext`] holds
/// a clone of the driver, which is cheap (the bundle is an `Arc`, and
/// tensors are shared), and all the state of its stream. So one driver
/// serves any number of streams in one process, and
/// [`advance_ready`](super::advance_ready) batches their decodes.
#[derive(Clone, Debug)]
pub struct WhisperStreamDriver<B: Backend> {
    config: WhisperStreamDriverConfig,

    audio_converter: PerceptiveAudioConverter<B>,

    /// The model, its layout and its vocabulary, shared.
    bundle: Arc<WhisperBundle<B>>,

    /// The voice-activity model, when one was attached.
    vad_model: Option<SileroVad<B>>,
    va_filter: Option<VoiceActivityFilterConfig>,

    /// The sot sequence every window's decode opens with; empty when the
    /// language is detected per stream.
    prompt: Vec<i64>,

    /// The task token's meaning; `None` for an English-only layout.
    task: Option<WhisperTask>,

    max_initial_timestamp_index: Option<usize>,

    /// Applied to the logits every step, in order: the caller's, then the
    /// timestamp rules when timestamps are on.
    filters: Vec<Arc<dyn LogitFilter<B>>>,

    detokenizer: Option<Arc<dyn Detokenizer>>,
}

impl<B: Backend> WhisperStreamDriver<B> {
    /// Attaches a detokenizer, so emissions carry text as well as ids.
    pub fn with_detokenizer(
        mut self,
        detokenizer: Arc<dyn Detokenizer>,
    ) -> Self {
        self.detokenizer = Some(detokenizer);
        self
    }

    /// Sets the logit filters every decode applies, in order, replacing
    /// any set before.
    pub fn with_logit_filters(
        mut self,
        filters: Vec<Arc<dyn LogitFilter<B>>>,
    ) -> Self {
        self.filters = filters;
        if self.config.timestamps {
            self.filters.push(Arc::new(ApplyTimestampRules::new(
                self.bundle.layout.ids(),
                self.max_initial_timestamp_index,
            )));
        }
        self
    }

    /// Attaches a voice-activity model and the filter that turns its
    /// probabilities into regions.
    ///
    /// Needed by any emission policy with the `endpoint` trigger; ignored by
    /// one without.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, if the model or the filter runs at a rate
    /// other than this driver's model, or the filter's chunk is not the
    /// model's.
    pub fn with_vad(
        mut self,
        vad: SileroVad<B>,
        filter: VoiceActivityFilterConfig,
    ) -> BunsenResult<Self> {
        self.check_vad(&vad, &filter)?;
        self.vad_model = Some(vad);
        self.va_filter = Some(filter);
        Ok(self)
    }

    /// The model and the filter must agree with the driver on the rate, and
    /// with each other on the chunk.
    fn check_vad(
        &self,
        vad: &SileroVad<B>,
        filter: &VoiceActivityFilterConfig,
    ) -> BunsenResult<()> {
        const OWNER: &str = "WhisperStreamDriver::with_vad";
        let must_equal = |lhs: (&str, usize), rhs: (&str, usize)| -> BunsenError {
            ConstraintError::new(
                OWNER,
                "",
                Rule::Relation {
                    lhs: (lhs.0.into(), lhs.1.to_string()),
                    op: "==",
                    rhs: (rhs.0.into(), rhs.1.to_string()),
                },
            )
            .into()
        };
        let rate = self.sample_rate();
        if vad.sample_rate() != rate {
            return Err(must_equal(
                ("vad.sample_rate()", vad.sample_rate()),
                ("the driver's sample_rate()", rate),
            ));
        }
        if filter.sample_rate != rate {
            return Err(must_equal(
                ("filter.sample_rate", filter.sample_rate),
                ("the driver's sample_rate()", rate),
            ));
        }
        if filter.samples_per_chunk != vad.chunk_size() {
            return Err(must_equal(
                ("filter.samples_per_chunk", filter.samples_per_chunk),
                ("vad.chunk_size()", vad.chunk_size()),
            ));
        }
        Ok(())
    }

    /// The driver configuration.
    pub fn config(&self) -> &WhisperStreamDriverConfig {
        &self.config
    }

    /// The bundle: the model, its layout and its vocabulary.
    pub fn bundle(&self) -> &Arc<WhisperBundle<B>> {
        &self.bundle
    }

    /// The model.
    pub fn whisper_model(&self) -> &Whisper<B> {
        &self.bundle.model
    }

    /// The mel front end.
    pub fn audio_converter(&self) -> &PerceptiveAudioConverter<B> {
        &self.audio_converter
    }

    /// The voice-activity model, if one was attached.
    pub fn silero_vad_model(&self) -> Option<&SileroVad<B>> {
        self.vad_model.as_ref()
    }

    /// The filter config, if a VAD is attached.
    pub fn va_filter_config(&self) -> Option<VoiceActivityFilterConfig> {
        self.va_filter.clone()
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
    pub fn filters(&self) -> &[Arc<dyn LogitFilter<B>>] {
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
            .with_beam_size(self.config.beam_size)
            .with_patience(self.config.patience)
            .with_length_penalty(self.config.length_penalty)
            .with_sot_token(Some(ids.sot))
            .with_no_speech_token(Some(ids.no_speech))
    }

    /// The draft interval in samples of media time, when the policy has
    /// one.
    pub fn interval_samples(&self) -> Option<usize> {
        self.config
            .emission
            .triggers
            .interval
            .map(|i| (i.as_secs_f64() * self.sample_rate() as f64).round() as usize)
    }

    /// The detokenizer, if one was attached.
    pub fn detokenizer(&self) -> Option<&Arc<dyn Detokenizer>> {
        self.detokenizer.as_ref()
    }

    /// Frames per decode window: the model's audio context.
    pub fn window_frames(&self) -> usize {
        self.bundle.model.max_audio_ctx()
    }

    /// The sample rate the model's front end runs at, in Hz. The stream's
    /// clock must run at it too.
    pub fn sample_rate(&self) -> usize {
        self.bundle.model.sample_rate()
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
    pub fn devices(&self) -> Vec<B::Device> {
        self.bundle.model.devices()
    }

    /// Opens a stream.
    ///
    /// # Arguments
    /// * `clock` - the stream's sample-to-time map. A bare stream gets
    ///   [`StreamClock::uniform`] at [`sample_rate`](Self::sample_rate).
    /// * `clamp` - where each window's dynamic-range reference comes from: a
    ///   concrete policy, or a `Box<dyn StreamClampPolicy<B>>` chosen at run
    ///   time.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if the clock
    /// does not run at the model's [`sample_rate`](Self::sample_rate) (with
    /// a [`ConstraintError`] cause), or if the emission policy wants
    /// endpoints and no VAD was attached.
    pub fn new_context<C: StreamClampPolicy<B> + 'static>(
        &self,
        clock: StreamClock,
        clamp: C,
    ) -> BunsenResult<WhisperStreamContext<B>> {
        if clock.rate() != self.sample_rate() {
            return Err(ConstraintError::new(
                "WhisperStreamDriver::new_context",
                "",
                Rule::Relation {
                    lhs: ("clock.rate()".into(), clock.rate().to_string()),
                    op: "==",
                    rhs: (
                        "the model's sample_rate()".into(),
                        self.sample_rate().to_string(),
                    ),
                },
            )
            .into());
        }
        if self.config.emission.triggers.endpoint && self.vad_model.is_none() {
            return Err(BunsenError::illegal(
                "the endpoint trigger needs a voice-activity model; attach one with with_vad",
            ));
        }

        Ok(WhisperStreamContext::init(
            self.clone(),
            clock,
            Box::new(clamp),
        ))
    }
}
