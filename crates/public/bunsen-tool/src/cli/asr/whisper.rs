//! Transcribing with Whisper: the construct through a stream driver.
use std::{
    path::{
        Path,
        PathBuf,
    },
    sync::Arc,
};

use bunsen::{
    data::pretrained::{
        PretrainedCache,
        PretrainedCacheOptions,
        ResourceBundle,
    },
    errors::{
        BunsenResult,
        ResultContext,
    },
    kits::{
        speech::{
            asr_driver::{
                LevelPolicy,
                TranscriptSegment,
            },
            whisper::{
                driver::{
                    WhisperConstruct,
                    WhisperStreamDriver,
                    WhisperStreamDriverConfig,
                    WhisperTask,
                },
                pretrained::{
                    CHECKPOINT,
                    OPENAI_LOCAL_DIR,
                    VOCABULARY,
                    WhisperConstructConfig,
                    default_whisper_factory,
                },
            },
        },
        support::construct::Deferred,
    },
};
use burn::tensor::Device;

use crate::cli::asr::{
    stream::{
        self,
        Emission,
    },
    transcribe_cmd::{
        Segment,
        Transcript,
        segment_speaker,
        speaker_turns,
    },
};

/// What Whisper emits.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    /// The speech in its own language.
    Transcribe,

    /// An English translation of it.
    Translate,
}

/// Whisper's own transcription flags.
#[derive(clap::Args, Debug)]
pub struct WhisperFlags {
    /// A Whisper language code (`en`, `de`, ...); detected from the first
    /// window when omitted. English-only checkpoints ignore it.
    #[arg(long)]
    pub language: Option<String>,

    /// Transcribe, or translate to English; the model's preset when omitted.
    #[arg(long, value_enum)]
    pub task: Option<Task>,

    /// Beams in the search; 1 is greedy. The model's policy when omitted.
    #[arg(long)]
    pub beam_size: Option<usize>,

    /// Emit timestamp tokens and split segments on them, seeking to the
    /// last closed timestamp, as upstream's `transcribe()` does.
    #[arg(long)]
    pub timestamps: bool,

    /// Cap on generated tokens per window; the model's policy when omitted.
    #[arg(long)]
    pub max_tokens: Option<usize>,

    /// Prompt each window with the transcript so far; the model's policy
    /// when omitted.
    #[arg(long, action = clap::ArgAction::Set)]
    pub prompt_carry: Option<bool>,

    /// Climb upstream's temperature ladder when a window's decode fails its
    /// thresholds; with `false`, temperature zero alone. The model's policy
    /// when omitted.
    #[arg(long, action = clap::ArgAction::Set)]
    pub fallback: Option<bool>,

    /// A vocabulary file (`multilingual.tiktoken`, `gpt2.tiktoken`) in place
    /// of the one the checkpoint's layout selects; trusted as given.
    #[arg(long)]
    pub vocab: Option<PathBuf>,

    /// `openai-whisper`'s download cache, whose files are used in place;
    /// `~/.cache/whisper` when omitted.
    #[arg(long)]
    pub upstream_cache_dir: Option<PathBuf>,
}

impl WhisperFlags {
    /// `options`, with `--upstream-cache-dir` as `openai-whisper`'s cache.
    pub fn cache_options(
        &self,
        options: PretrainedCacheOptions,
    ) -> PretrainedCacheOptions {
        match &self.upstream_cache_dir {
            Some(dir) => options.with_local_dir(OPENAI_LOCAL_DIR, dir.clone()),
            None => options,
        }
    }
}

/// What `--model` names: a checkpoint on disk, as a one-resource bundle under
/// [`CHECKPOINT`] (its vocabulary the one its layout selects), or a row of
/// the Whisper factory; `--vocab` over either.
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when the factory
/// does not have the name, or the map cannot take the vocabulary: `spec`
/// and `--vocab` come from the command line.
pub fn resolve(
    spec: &str,
    flags: Option<&WhisperFlags>,
    cache: &PretrainedCache,
) -> BunsenResult<Deferred<WhisperConstructConfig>> {
    let path = Path::new(spec);
    let mut model = if path.is_file() {
        Deferred::from_resource_bundle(ResourceBundle::given(spec, CHECKPOINT, path)).as_policy()?
    } else {
        default_whisper_factory()?
            .resolve(spec, cache)
            .as_policy()?
    };
    if let Some(vocab) = flags.and_then(|f| f.vocab.as_ref()) {
        model = model
            .with_overlay(ResourceBundle::given("--vocab", VOCABULARY, vocab))
            .as_policy()?;
    }
    Ok(model)
}

/// The driver for `bundle` under `flags` and `emission`.
pub fn driver(
    bundle: Arc<WhisperConstruct>,
    flags: &WhisperFlags,
    emission: Emission,
    device: &Device,
) -> BunsenResult<WhisperStreamDriver> {
    let multilingual = bundle.layout.ids().is_multilingual();
    if !multilingual && flags.language.is_some() {
        log::warn!("English-only checkpoint: --language and --task do not apply");
    }
    // The model's policy, its presets resolved, under the flags given.
    let mut policy = bundle.policy.clone();
    if let Some(task) = flags.task {
        policy.task = match task {
            Task::Transcribe => WhisperTask::Transcribe,
            Task::Translate => WhisperTask::Translate,
        };
    }
    policy.language = if multilingual {
        flags.language.clone().or(policy.language)
    } else {
        None
    };
    policy.timestamps |= flags.timestamps;
    if let Some(beam_size) = flags.beam_size {
        policy.beam_size = beam_size;
    }
    if let Some(max_tokens) = flags.max_tokens {
        policy.max_tokens = max_tokens;
    }
    if let Some(prompt_carry) = flags.prompt_carry {
        policy.condition_on_previous_text = prompt_carry;
    }
    if let Some(fallback) = flags.fallback {
        policy.fallback = fallback;
    }
    let driver: WhisperStreamDriver = WhisperStreamDriverConfig::new()
        .with_policy(&policy)
        .with_emission(emission.into())
        .init_from_construct(bundle, device)
        // The settings come from the command line.
        .as_policy()?;
    Ok(driver)
}

/// What each window's features are scaled against: each window floored against
/// the loudest the stream has been.
pub const LEVEL: LevelPolicy = LevelPolicy::Running;

/// Transcribes one recording through `driver`.
pub fn transcribe(
    driver: &WhisperStreamDriver,
    samples: &[f32],
    chunk: usize,
) -> BunsenResult<(Transcript, Vec<TranscriptSegment>)> {
    let (committed, ctx) = stream::stream(driver, samples, LEVEL, chunk)?;
    let segments = committed
        .iter()
        .map(|s| Segment {
            start: s.start,
            end: s.end,
            text: s.text.as_deref().unwrap_or("").trim().to_string(),
            tokens: s.tokens.clone(),
            speaker: segment_speaker(s),
        })
        .collect();
    let transcript = Transcript {
        text: stream::join_text(&committed),
        turns: driver.speakers().map(|_| speaker_turns(&committed)),
        language: ctx.language().map(String::from),
        words: None,
        segments: Some(segments),
    };
    Ok((transcript, committed))
}
