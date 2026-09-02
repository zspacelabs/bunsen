//! # The stream driver: one configured baseline, deployed three ways.
//!
//! Two concrete types carry the whole design. [`WhisperDriver`] is shared
//! and immutable: the model, the mel front end, the token layout, the
//! emission policy, an optional voice-activity model, and an optional
//! detokenizer, built once and cheap to share. [`WhisperStreamContext`] is
//! one per stream and the only stateful type: it takes samples in through
//! [`push`](WhisperStreamContext::push) and hands
//! [`Emission`](support::Emission)s back. Everything that varies by
//! deployment enters as an injected object behind a trait at a construction
//! point &mdash; the clock and the clamp policy at
//! [`new_context`](WhisperDriver::new_context), the detokenizer at
//! [`with_detokenizer`](WhisperDriver::with_detokenizer), the VAD at
//! [`with_vad`](WhisperDriver::with_vad) &mdash; and is never something the
//! driver reasons about.
//!
//! The driver derives its token layout from the model it is given, through
//! [`TokenPolicy::from_vocab_size`], and builds the prompt with
//! [`sot_sequence`](TokenPolicy::sot_sequence). Ids are never configuration.
//!
//! ## What this slice supports
//!
//! Two of the three deployments. **Offline batch**: the `window_full`
//! trigger and the `Complete` commit rule, one stream at a time through
//! [`push`](WhisperStreamContext::push), or many at once through
//! [`advance_ready`]. **Conservative real time**: the `endpoint` trigger as
//! well, with a Silero VAD attached by [`with_vad`](WhisperDriver::with_vad);
//! each speech region is decoded as it closes, in the parent stream's frame
//! of reference, and full windows of silence are never decoded at all.
//! **Responsive real time**: the `interval` trigger as well; while speech
//! is in progress, every interval of media time a draft of everything past
//! the seek pointer is emitted, superseding the previous draft, and the
//! commits are exactly conservative's. Without timestamps, `LastTimestamp`
//! commits whole, as upstream does when a decode emits none.

mod batch;
mod context;
pub mod support;

use std::sync::Arc;

pub use batch::advance_ready;
use burn::{
    config::Config,
    module::Module,
    prelude::Backend,
};
pub use context::WhisperStreamContext;
use support::{
    ClampPolicy,
    EmissionPolicy,
    TIMESTAMP_STEP_SAMPLES,
    TIMESTAMP_STEP_SECONDS,
    Task,
    TimestampHistory,
    TokenPolicy,
    VoiceActivityFilterConfig,
    mel_options,
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::{
            silero_vad::SileroVad,
            whisper::{
                blocks::{
                    Whisper,
                    WhisperMeta,
                },
                decode::{
                    ApplyTimestampRules,
                    DecodeConfig,
                    FallbackConfig,
                    LogitFilter,
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

/// The sample rate Whisper's front end is defined at, in Hz.
pub const SAMPLE_RATE: usize = 16_000;

/// How to transcribe: the things a caller decides once, for every stream.
///
/// Everything here is about the *decode*. What varies per stream &mdash; the
/// clock, the clamp reference &mdash; is injected at
/// [`WhisperDriver::new_context`] instead.
#[derive(Config, Debug)]
pub struct WhisperDriverConfig {
    /// The language of the speech, as a [`LANGUAGES`](support::LANGUAGES)
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
    #[config(default = "Task::Transcribe")]
    pub task: Task,

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
    ///
    /// Off is what makes [`advance_ready`] batch: windows are batched by
    /// prompt, and a carried prompt is a different prompt per stream.
    #[config(default = "true")]
    pub condition_on_previous_text: bool,

    /// When to decode, and when a decode is final.
    #[config(default = "EmissionPolicy::offline()")]
    pub emission: EmissionPolicy,

    /// The temperature ladder and its thresholds. The default ladder is
    /// temperature zero alone; [`FallbackConfig::upstream`] is
    /// `transcribe()`'s.
    #[config(default = "FallbackConfig::new()")]
    pub fallback: FallbackConfig,
}

impl WhisperDriverConfig {
    /// Builds the driver over a model, deriving the token layout from the
    /// model's vocabulary size.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the vocabulary size is not a Whisper
    /// layout, or as [`init_with_policy`](Self::init_with_policy).
    pub fn init<B: Backend>(
        &self,
        model: Whisper<B>,
        device: &B::Device,
    ) -> BunsenResult<WhisperDriver<B>> {
        let policy = TokenPolicy::from_vocab_size(model.vocab_size())?;
        self.init_with_policy(model, policy, device)
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
    pub fn init_with_policy<B: Backend>(
        &self,
        model: Whisper<B>,
        policy: TokenPolicy,
        device: &B::Device,
    ) -> BunsenResult<WhisperDriver<B>> {
        let ids = policy.ids();
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
            (_, language) => policy.sot_sequence(language, task, self.timestamps)?,
        };
        let max_initial_timestamp_index = self
            .max_initial_timestamp
            .map(|seconds| (seconds / TIMESTAMP_STEP_SECONDS).round() as usize);
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

        let mel = mel_options(SAMPLE_RATE, model.n_mels()).try_init(device)?;

        Ok(WhisperDriver {
            model,
            mel,
            vad: None,
            policy,
            prompt,
            language: self.language.clone(),
            task,
            timestamps: self.timestamps,
            max_initial_timestamp_index,
            max_tokens: self.max_tokens,
            beam_size: self.beam_size,
            patience: self.patience,
            length_penalty: self.length_penalty,
            filters,
            carry_prompt: self.condition_on_previous_text,
            emission: self.emission.clone(),
            fallback: self.fallback.clone(),
            filter_config: None,
            detokenizer: None,
        })
    }
}

/// The shared, immutable half of a transcription: what every stream needs
/// and none of them mutates.
///
/// Built by [`WhisperDriverConfig::init`]. Opens streams with
/// [`new_context`](Self::new_context).
#[derive(Clone, Debug)]
pub struct WhisperDriver<B: Backend> {
    model: Whisper<B>,
    mel: MelConverter<B>,

    /// The voice-activity model, when one was attached.
    vad: Option<SileroVad<B>>,

    policy: TokenPolicy,

    /// The sot sequence every window's decode opens with; empty when the
    /// language is detected per stream.
    prompt: Vec<i64>,

    language: Option<String>,

    /// The task token's meaning; `None` for an English-only layout.
    task: Option<Task>,

    timestamps: bool,

    max_initial_timestamp_index: Option<usize>,

    max_tokens: usize,

    beam_size: usize,

    patience: Option<f64>,

    length_penalty: Option<f64>,

    /// Applied to the logits every step, in order: the caller's, then the
    /// timestamp rules when timestamps are on.
    filters: Vec<Arc<dyn LogitFilter<B>>>,

    carry_prompt: bool,

    emission: EmissionPolicy,

    fallback: FallbackConfig,

    filter_config: Option<VoiceActivityFilterConfig>,

    detokenizer: Option<Arc<dyn Detokenizer>>,
}

impl<B: Backend> WhisperDriver<B> {
    /// Attaches a detokenizer, so emissions carry text as well as ids.
    pub fn with_detokenizer(
        mut self,
        detokenizer: Arc<dyn Detokenizer>,
    ) -> Self {
        self.detokenizer = Some(detokenizer);
        self
    }

    /// Sets the logit filters every decode applies, in order, replacing
    /// any set before. Upstream's defaults are
    /// [`default_filters`](super::decode::default_filters), which need the
    /// vocabulary and so are not built here. Under `timestamps` the
    /// timestamp rules are appended, as upstream applies them last.
    pub fn with_logit_filters(
        mut self,
        filters: Vec<Arc<dyn LogitFilter<B>>>,
    ) -> Self {
        self.filters = filters;
        if self.timestamps {
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
    pub fn with_vad(
        mut self,
        vad: SileroVad<B>,
        filter_config: VoiceActivityFilterConfig,
    ) -> Self {
        self.vad = Some(vad);
        self.filter_config = Some(filter_config);
        self
    }

    /// The model.
    pub fn model(&self) -> &Whisper<B> {
        &self.model
    }

    /// The mel front end.
    pub fn mel(&self) -> &MelConverter<B> {
        &self.mel
    }

    /// The voice-activity model, if one was attached.
    pub fn vad(&self) -> Option<&SileroVad<B>> {
        self.vad.as_ref()
    }

    /// The filter config, if a VAD is attached.
    pub fn filter_config(&self) -> Option<VoiceActivityFilterConfig> {
        self.filter_config.clone()
    }

    /// The token layout, derived from the model.
    pub fn policy(&self) -> &TokenPolicy {
        &self.policy
    }

    /// The sot sequence every window's decode opens with.
    pub fn prompt(&self) -> &[i64] {
        &self.prompt
    }

    /// Cap on tokens generated per window.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Whether windows are prompted with the transcript so far.
    pub fn carries_prompt(&self) -> bool {
        self.carry_prompt
    }

    /// When to decode, and when a decode is final.
    pub fn emission(&self) -> &EmissionPolicy {
        &self.emission
    }

    /// Beams per window; one is greedy.
    pub fn beam_size(&self) -> usize {
        self.beam_size
    }

    /// The logit filters every decode applies.
    pub fn filters(&self) -> &[Arc<dyn LogitFilter<B>>] {
        &self.filters
    }

    /// Whether decodes are prompted for timestamps.
    pub fn timestamps(&self) -> bool {
        self.timestamps
    }

    /// The configured language, if any.
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    /// The task; `None` for an English-only layout.
    pub fn task(&self) -> Option<Task> {
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
            .sot_sequence(language, self.task, self.timestamps)
    }

    /// Mel frames per timestamp index: two.
    pub fn frames_per_timestamp(&self) -> usize {
        TIMESTAMP_STEP_SAMPLES / self.mel.hop()
    }

    /// The decode of one window under this driver, given its prompt.
    pub fn decode_config(
        &self,
        prompt: Vec<i64>,
    ) -> DecodeConfig {
        let ids = self.policy.ids();
        DecodeConfig::new(prompt, ids.eot)
            .with_max_tokens(self.max_tokens)
            .with_beam_size(self.beam_size)
            .with_patience(self.patience)
            .with_length_penalty(self.length_penalty)
            .with_sot_token(Some(ids.sot))
            .with_no_speech_token(Some(ids.no_speech))
    }

    /// The temperature ladder and its thresholds.
    pub fn fallback(&self) -> &FallbackConfig {
        &self.fallback
    }

    /// The draft interval in samples of media time, when the policy has
    /// one.
    pub fn interval_samples(&self) -> Option<usize> {
        self.emission
            .triggers
            .interval
            .map(|i| (i.as_secs_f64() * SAMPLE_RATE as f64).round() as usize)
    }

    /// The detokenizer, if one was attached.
    pub fn detokenizer(&self) -> Option<&Arc<dyn Detokenizer>> {
        self.detokenizer.as_ref()
    }

    /// Frames per decode window: the model's audio context.
    pub fn window_frames(&self) -> usize {
        self.model.max_audio_ctx()
    }

    /// The devices the model lives on.
    pub fn devices(&self) -> Vec<B::Device> {
        self.model.devices()
    }

    /// Opens a stream.
    ///
    /// # Arguments
    /// * `clock` - the stream's sample-to-time map. A bare stream gets
    ///   [`TimestampHistory::uniform`] at [`SAMPLE_RATE`].
    /// * `clamp` - where each window's dynamic-range reference comes from: a
    ///   concrete policy, or a `Box<dyn ClampPolicy<B>>` chosen at run time.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the clock does not run at
    /// [`SAMPLE_RATE`], or if the emission policy wants endpoints and no VAD
    /// was attached.
    pub fn new_context<C: ClampPolicy<B> + 'static>(
        &self,
        clock: TimestampHistory,
        clamp: C,
    ) -> BunsenResult<WhisperStreamContext<B>> {
        if clock.rate() != SAMPLE_RATE {
            return Err(BunsenError::Invalid(format!(
                "the stream clock runs at {} Hz; Whisper's front end is defined at {SAMPLE_RATE}",
                clock.rate(),
            )));
        }
        if self.emission.triggers.endpoint && self.vad.is_none() {
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
