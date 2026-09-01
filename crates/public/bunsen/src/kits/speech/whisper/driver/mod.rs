//! # The stream driver: one configured baseline, deployed three ways.
//!
//! Two concrete types carry the whole design. [`WhisperDriver`] is shared
//! and immutable: the model, the mel front end, the token layout, the
//! emission policy, and an optional detokenizer, built once and cheap to
//! share. [`WhisperStreamContext`] is one per stream and the only stateful
//! type: it takes samples in through [`push`](WhisperStreamContext::push)
//! and hands [`Emission`](super::emission::Emission)s back. Everything that
//! varies by deployment enters as an injected object behind a trait at a
//! construction point &mdash; the clock and the clamp policy at
//! [`new_context`](WhisperDriver::new_context), the detokenizer at
//! [`with_detokenizer`](WhisperDriver::with_detokenizer) &mdash; and is never
//! something the driver reasons about.
//!
//! The driver derives its token layout from the model it is given, through
//! [`TokenPolicy::from_vocab_size`], and builds the prompt with
//! [`sot_sequence`](TokenPolicy::sot_sequence). Ids are never configuration.
//!
//! ## What this slice supports
//!
//! The offline deployment: the `window_full` trigger and the `Complete`
//! commit rule, without timestamps. The other presets exist and are refused
//! at [`init`](WhisperDriverConfig::init) with a reason, rather than being
//! silently approximated; voice activity, timestamps and drafts arrive in
//! their own phases and turn them on.

mod context;

use std::sync::Arc;

use burn::{
    config::Config,
    module::Module,
    prelude::Backend,
};
pub use context::WhisperStreamContext;

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::whisper::{
            blocks::{
                Whisper,
                WhisperMeta,
            },
            clamp::ClampPolicy,
            clock::TimestampHistory,
            emission::{
                CommitRule,
                EmissionPolicy,
            },
            mel::mel_options,
            tokens::{
                Task,
                TokenPolicy,
            },
        },
        tokens::Detokenizer,
    },
    ops::signal::mels::MelConverter,
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
    /// The language of the speech, as a [`LANGUAGES`](super::tokens::LANGUAGES)
    /// code.
    ///
    /// Required for a multilingual checkpoint, until language detection
    /// arrives; must be `None` for an English-only one, which takes no
    /// language token.
    #[config(default = "None")]
    pub language: Option<String>,

    /// Transcribe, or translate to English. Ignored by an English-only
    /// checkpoint, which takes no task token.
    #[config(default = "Task::Transcribe")]
    pub task: Task,

    /// Let the model emit timestamp tokens. Not supported yet; must be off.
    #[config(default = "false")]
    pub timestamps: bool,

    /// Cap on tokens generated per window.
    #[config(default = "224")]
    pub max_tokens: usize,

    /// Prompt each window with the tail of the transcript so far, after
    /// `<|startofprev|>`, as upstream's `condition_on_previous_text` does.
    #[config(default = "true")]
    pub condition_on_previous_text: bool,

    /// When to decode, and when a decode is final.
    #[config(default = "EmissionPolicy::offline()")]
    pub emission: EmissionPolicy,
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

        let (language, task) = if ids.is_multilingual() {
            let language = self.language.as_deref().ok_or_else(|| {
                BunsenError::Invalid(
                    "a multilingual checkpoint needs a language; detection is not supported yet"
                        .to_string(),
                )
            })?;
            (Some(language), Some(self.task))
        } else {
            if self.language.is_some() {
                return Err(BunsenError::Invalid(
                    "an English-only checkpoint takes no language".to_string(),
                ));
            }
            (None, None)
        };

        if self.timestamps {
            return Err(BunsenError::Invalid(
                "timestamps are not supported yet".to_string(),
            ));
        }
        let prompt = policy.sot_sequence(language, task, self.timestamps)?;

        let triggers = &self.emission.triggers;
        if !triggers.window_full {
            return Err(BunsenError::Invalid(
                "without the window_full trigger nothing would ever decode".to_string(),
            ));
        }
        if triggers.endpoint || triggers.interval.is_some() {
            return Err(BunsenError::Invalid(
                "the endpoint and interval triggers are not supported yet; use EmissionPolicy::offline()"
                    .to_string(),
            ));
        }
        if self.emission.commit != CommitRule::Complete {
            return Err(BunsenError::Invalid(
                "only CommitRule::Complete is supported yet".to_string(),
            ));
        }

        let mel = mel_options(SAMPLE_RATE, model.n_mels()).try_init(device)?;

        Ok(WhisperDriver {
            model,
            mel,
            policy,
            prompt,
            max_tokens: self.max_tokens,
            carry_prompt: self.condition_on_previous_text,
            emission: self.emission.clone(),
            detokenizer: None,
        })
    }
}

/// The shared, immutable half of a transcription: what every stream needs
/// and none of them mutates.
///
/// Built by [`WhisperDriverConfig::init`]. Opens streams with
/// [`new_context`](Self::new_context).
#[derive(Module, Debug)]
pub struct WhisperDriver<B: Backend> {
    model: Whisper<B>,
    mel: MelConverter<B>,

    #[module(skip)]
    policy: TokenPolicy,

    /// The sot sequence every window's decode opens with.
    #[module(skip)]
    prompt: Vec<i64>,

    #[module(skip)]
    max_tokens: usize,

    #[module(skip)]
    carry_prompt: bool,

    #[module(skip)]
    emission: EmissionPolicy,

    #[module(skip)]
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

    /// The model.
    pub fn model(&self) -> &Whisper<B> {
        &self.model
    }

    /// The mel front end.
    pub fn mel(&self) -> &MelConverter<B> {
        &self.mel
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

    /// The detokenizer, if one was attached.
    pub fn detokenizer(&self) -> Option<&Arc<dyn Detokenizer>> {
        self.detokenizer.as_ref()
    }

    /// Frames per decode window: the model's audio context.
    pub fn window_frames(&self) -> usize {
        self.model.max_audio_ctx()
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
    /// [`SAMPLE_RATE`].
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

        Ok(WhisperStreamContext::open(
            self.clone(),
            clock,
            Box::new(clamp),
        ))
    }
}
