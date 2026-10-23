//! Transcribing with Whisper: the bundle through a stream driver.
use std::{
    path::{
        Path,
        PathBuf,
    },
    sync::Arc,
};

use bunsen::{
    data::pretrained::{
        Deferred,
        PretrainedCache,
        PretrainedCacheOptions,
        ResourceMap,
    },
    errors::{
        BunsenResult,
        ResultContext,
    },
    kits::speech::{
        asr_driver::{
            LevelPolicy,
            TranscriptSegment,
        },
        whisper::{
            WhisperFallbackConfig,
            driver::{
                WhisperBundle,
                WhisperStreamDriver,
                WhisperStreamDriverConfig,
                WhisperTask,
            },
            pretrained::{
                CHECKPOINT,
                OPENAI_LOCAL_DIR,
                VOCABULARY,
                WhisperConstruct,
                default_whisper_factory,
            },
        },
    },
};
use burn::tensor::Device;

use crate::{
    stream::{
        self,
        Preset,
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

    /// Transcribe, or translate to English.
    #[arg(long, value_enum, default_value_t = Task::Transcribe)]
    pub task: Task,

    /// Beams in the search; 1 is greedy.
    #[arg(long, default_value_t = 5)]
    pub beam_size: usize,

    /// Emit timestamp tokens and split segments on them, seeking to the
    /// last closed timestamp, as upstream's `transcribe()` does.
    #[arg(long)]
    pub timestamps: bool,

    /// Cap on generated tokens per window.
    #[arg(long, default_value_t = 224)]
    pub max_tokens: usize,

    /// Prompt each window with the transcript so far.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    pub prompt_carry: bool,

    /// Climb upstream's temperature ladder when a window's decode fails its
    /// thresholds; with `false`, temperature zero alone.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    pub fallback: bool,

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

/// What `--model` names: a checkpoint on disk, as a one-resource map under
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
) -> BunsenResult<Deferred<WhisperConstruct>> {
    let path = Path::new(spec);
    let mut model = if path.is_file() {
        Deferred::from_map(ResourceMap::given(spec, CHECKPOINT, path)).as_policy()?
    } else {
        default_whisper_factory()?
            .resolve(spec, cache)
            .as_policy()?
    };
    if let Some(vocab) = flags.and_then(|f| f.vocab.as_ref()) {
        model = model
            .with_overlay(ResourceMap::given("--vocab", VOCABULARY, vocab))
            .as_policy()?;
    }
    Ok(model)
}

/// The driver for `bundle` under `flags` and `preset`.
pub fn driver(
    bundle: Arc<WhisperBundle>,
    flags: &WhisperFlags,
    preset: Preset,
    cache: &PretrainedCache,
    device: &Device,
) -> BunsenResult<WhisperStreamDriver> {
    let multilingual = bundle.layout.ids().is_multilingual();
    if !multilingual && flags.language.is_some() {
        log::warn!("English-only checkpoint: --language and --task do not apply");
    }
    let task = match flags.task {
        Task::Transcribe => WhisperTask::Transcribe,
        Task::Translate => WhisperTask::Translate,
    };
    let driver: WhisperStreamDriver = WhisperStreamDriverConfig::new()
        .with_language(if multilingual {
            flags.language.clone()
        } else {
            None
        })
        .with_task(task)
        .with_timestamps(flags.timestamps)
        .with_beam_size(flags.beam_size)
        .with_max_tokens(flags.max_tokens)
        .with_condition_on_previous_text(flags.prompt_carry)
        .with_emission(preset.into())
        .with_fallback(if flags.fallback {
            WhisperFallbackConfig::upstream()
        } else {
            WhisperFallbackConfig::new()
        })
        .init_from_bundle(bundle, device)
        // The settings come from the command line.
        .as_policy()?;
    stream::with_vad(driver, preset, cache, device)
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
