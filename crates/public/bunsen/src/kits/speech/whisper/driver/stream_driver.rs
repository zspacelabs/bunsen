use std::sync::Arc;

use burn::{
    config::Config,
    module::Module,
    prelude::Backend,
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::{
            silero_vad::{
                SileroVad,
                SileroVadMeta,
            },
            whisper::{
                ApplyTimestampRules,
                DecodeConfig,
                LogitFilter,
                Whisper,
                WhisperFallbackConfig,
                WhisperMeta,
                blocks::{
                    AUDIO_ENCODER_STRIDE,
                    WhisperFrontEndConfig,
                },
                driver::{
                    ClampPolicy,
                    EmissionPolicy,
                    StreamClock,
                    VoiceActivityFilterConfig,
                    WhisperStreamContext,
                    WhisperTask,
                    WhisperTokenLayout,
                },
            },
        },
        tokens::Detokenizer,
    },
    ops::signal::mels::{
        MelConverter,
        MelConverterMeta,
    },
};

/// Config for [`WhisperStreamDriver`].
#[derive(Config, Debug)]
pub struct WhisperStreamDriverConfig {
    /// The language of the speech, as a
    /// [`LANGUAGES`](crate::kits::speech::whisper::blocks::LANGUAGES)
    /// code.
    ///
    /// `None` on a multilingual checkpoint detects the language per stream
    /// from its first decoded window, as upstream's `transcribe()` does;
    /// must be `None` for an English-only one, which takes no language
    /// token.
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
    #[config(default = "EmissionPolicy::offline()")]
    pub emission: EmissionPolicy,

    /// The temperature ladder and its thresholds. The default ladder is
    /// temperature zero alone; [`WhisperFallbackConfig::upstream`] is
    /// `transcribe()`'s.
    #[config(default = "WhisperFallbackConfig::new()")]
    pub fallback: WhisperFallbackConfig,
}

impl WhisperStreamDriverConfig {
    /// Builds the driver over a model, deriving the token layout from the
    /// model's vocabulary size.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the vocabulary size is not a Whisper
    /// layout, or as [`init_with_policy`](Self::init_with_layout).
    pub fn init<B: Backend>(
        &self,
        model: Whisper<B>,
        device: &B::Device,
    ) -> BunsenResult<WhisperStreamDriver<B>> {
        let token_layout = model.token_layout().policy_for_vocab(model.vocab_size())?;
        self.init_with_layout(model, token_layout, device)
    }

    /// Builds the driver over a model with an explicit token layout.
    ///
    /// For a model whose vocabulary is not one of Whisper's &mdash; a test
    /// model &mdash; or to override what [`init`](Self::init) would derive.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the language and task do not fit the
    /// layout, or if the configuration asks for something this slice of the
    /// driver does not support yet.
    pub fn init_with_layout<B: Backend>(
        &self,
        model: Whisper<B>,
        token_layout: WhisperTokenLayout,
        device: &B::Device,
    ) -> BunsenResult<WhisperStreamDriver<B>> {
        let ids = token_layout.ids();
        assert!(
            ids.n_vocab() <= model.vocab_size(),
            "the token layout has {} ids but the model's vocabulary has {}",
            ids.n_vocab(),
            model.vocab_size(),
        );

        if !ids.is_multilingual() && self.language.is_some() {
            return Err(BunsenError::Invalid(
                "an English-only checkpoint takes no language".to_string(),
            ));
        }
        let task = ids.is_multilingual().then_some(self.task);
        // A multilingual checkpoint with no language detects it per
        // stream; its prompt is built when the language is known.
        let prompt = match (ids.is_multilingual(), self.language.as_deref()) {
            (true, None) => Vec::new(),
            (_, language) => token_layout.sot_sequence(language, task, self.timestamps)?,
        };
        let max_initial_timestamp_index = self.max_initial_timestamp.map(|seconds| {
            (seconds / token_layout.layout().timestamp_step_seconds).round() as usize
        });
        let filters: Vec<Arc<dyn LogitFilter<B>>> = if self.timestamps {
            vec![Arc::new(ApplyTimestampRules::new(
                ids,
                max_initial_timestamp_index,
            ))]
        } else {
            Vec::new()
        };

        let triggers = &self.emission.triggers;
        if !triggers.window_full && !triggers.endpoint {
            return Err(BunsenError::Invalid(
                "with neither the window_full nor the endpoint trigger nothing would ever decode"
                    .to_string(),
            ));
        }
        if triggers.interval.is_some_and(|i| i.is_zero()) {
            return Err(BunsenError::Invalid(
                "an interval of zero would draft on every push".to_string(),
            ));
        }

        if self.beam_size == 0 {
            return Err(BunsenError::Invalid(
                "beam_size must be at least one".to_string(),
            ));
        }
        if self.fallback.temperatures.is_empty() {
            return Err(BunsenError::Invalid(
                "the fallback ladder needs at least one temperature".to_string(),
            ));
        }
        if self.fallback.best_of == Some(0) {
            return Err(BunsenError::Invalid(
                "best_of must be at least one".to_string(),
            ));
        }
        if (self.beam_size as f64 * self.patience.unwrap_or(1.0)).round() < 1.0 {
            return Err(BunsenError::Invalid(format!(
                "a patience of {:?} with {} beams collects no candidates",
                self.patience, self.beam_size
            )));
        }

        let mel = model
            .front_end()
            .mel_converter_options(model.n_mels())?
            .try_init(device)?;

        Ok(WhisperStreamDriver {
            config: self.clone(),
            whisper_model: model,
            mel_converter: mel,
            vad_model: None,
            policy: token_layout,
            prompt,
            task,
            max_initial_timestamp_index,
            filters,
            va_filter: None,
            detokenizer: None,
        })
    }
}

/// The shared, immutable half of a transcription: what every stream needs
/// and none of them mutates.
///
/// Built by [`WhisperStreamDriverConfig::init`]. Opens streams with
/// [`new_context`](Self::new_context).
#[derive(Clone, Debug)]
pub struct WhisperStreamDriver<B: Backend> {
    config: WhisperStreamDriverConfig,

    mel_converter: MelConverter<B>,
    whisper_model: Whisper<B>,

    /// The voice-activity model, when one was attached.
    vad_model: Option<SileroVad<B>>,
    va_filter: Option<VoiceActivityFilterConfig>,

    policy: WhisperTokenLayout,

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
                self.policy.ids(),
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
    /// [`BunsenError::Invalid`] if the model or the filter runs at a rate
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
        let rate = self.sample_rate();
        if vad.sample_rate() != rate {
            return Err(BunsenError::Invalid(format!(
                "the voice-activity model runs at {} Hz; this driver's model at {rate}",
                vad.sample_rate(),
            )));
        }
        if filter.sample_rate != rate {
            return Err(BunsenError::Invalid(format!(
                "the voice-activity filter is configured at {} Hz; this driver's model at {rate}",
                filter.sample_rate,
            )));
        }
        if filter.samples_per_chunk != vad.chunk_size() {
            return Err(BunsenError::Invalid(format!(
                "the voice-activity filter expects {}-sample chunks; the model emits {}",
                filter.samples_per_chunk,
                vad.chunk_size(),
            )));
        }
        Ok(())
    }

    /// The driver configuration.
    pub fn config(&self) -> &WhisperStreamDriverConfig {
        &self.config
    }

    /// The model.
    pub fn whisper_model(&self) -> &Whisper<B> {
        &self.whisper_model
    }

    /// The mel front end.
    pub fn mel_converter(&self) -> &MelConverter<B> {
        &self.mel_converter
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
        &self.policy
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

    /// Whether streams detect their language from their first window.
    pub fn detects_language(&self) -> bool {
        self.prompt.is_empty()
    }

    /// The sot sequence for `language`, under this driver's task and
    /// timestamp setting.
    pub fn sot_sequence(
        &self,
        language: Option<&str>,
    ) -> BunsenResult<Vec<i64>> {
        self.policy
            .sot_sequence(language, self.task, self.config.timestamps)
    }

    /// Mel frames per timestamp index: two.
    pub fn frames_per_timestamp(&self) -> usize {
        AUDIO_ENCODER_STRIDE
    }

    /// The decode of one window under this driver, given its prompt.
    pub fn decode_config(
        &self,
        prompt: Vec<i64>,
    ) -> DecodeConfig {
        let ids = self.policy.ids();
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
        self.whisper_model.max_audio_ctx()
    }

    /// The sample rate the model's front end runs at, in Hz. The stream's
    /// clock must run at it too.
    pub fn sample_rate(&self) -> usize {
        self.whisper_model.sample_rate()
    }

    /// The audio front end the model's log-mels are computed with.
    pub fn front_end(&self) -> &WhisperFrontEndConfig {
        self.whisper_model.front_end()
    }

    /// The encoder grid in samples: one timestamp step, which is
    /// [`frames_per_timestamp`](Self::frames_per_timestamp) mel hops. 320 at
    /// 16 kHz.
    pub fn encoder_grid(&self) -> usize {
        self.frames_per_timestamp() * self.mel_converter.hop()
    }

    /// The devices the model lives on.
    pub fn devices(&self) -> Vec<B::Device> {
        self.whisper_model.devices()
    }

    /// Opens a stream.
    ///
    /// # Arguments
    /// * `clock` - the stream's sample-to-time map. A bare stream gets
    ///   [`StreamClock::uniform`] at [`sample_rate`](Self::sample_rate).
    /// * `clamp` - where each window's dynamic-range reference comes from: a
    ///   concrete policy, or a `Box<dyn ClampPolicy<B>>` chosen at run time.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the clock does not run at the model's
    /// [`sample_rate`](Self::sample_rate), or if the emission policy wants
    /// endpoints and no VAD was attached.
    pub fn new_context<C: ClampPolicy<B> + 'static>(
        &self,
        clock: StreamClock,
        clamp: C,
    ) -> BunsenResult<WhisperStreamContext<B>> {
        if clock.rate() != self.sample_rate() {
            return Err(BunsenError::Invalid(format!(
                "the stream clock runs at {} Hz; the model's front end at {}",
                clock.rate(),
                self.sample_rate(),
            )));
        }
        if self.config.emission.triggers.endpoint && self.vad_model.is_none() {
            return Err(BunsenError::Invalid(
                "the endpoint trigger needs a voice-activity model; attach one with with_vad"
                    .to_string(),
            ));
        }

        Ok(WhisperStreamContext::open(
            self.clone(),
            clock,
            Box::new(clamp),
        ))
    }
}
