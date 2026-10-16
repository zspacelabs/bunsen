//! Transcribing with Whisper: the bundle through a stream driver.
use std::sync::Arc;

use bunsen::{
    data::pretrained::PretrainedCache,
    errors::{
        BunsenResult,
        ResultContext,
    },
    kits::speech::{
        asr_driver::LevelPolicy,
        whisper::{
            WhisperFallbackConfig,
            driver::{
                WhisperBundle,
                WhisperStreamDriver,
                WhisperStreamDriverConfig,
                WhisperTask,
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
        .with_beam_size(flags.beam_size)
        .with_emission(preset.into())
        .with_fallback(WhisperFallbackConfig::upstream())
        .init_from_bundle(bundle, device)
        // The settings come from the command line.
        .as_policy()?;
    stream::with_vad(driver, preset, cache, device)
}

/// Transcribes one recording through `driver`.
pub fn transcribe(
    driver: &WhisperStreamDriver,
    samples: &[f32],
) -> BunsenResult<Transcript> {
    let (committed, ctx) = stream::stream(driver, samples, LevelPolicy::Running)?;
    let segments = committed
        .iter()
        .map(|s| Segment {
            start: s.start,
            end: s.end,
            text: s.text.as_deref().unwrap_or("").trim().to_string(),
        })
        .collect();
    Ok(Transcript {
        text: stream::join_text(&committed),
        language: ctx.language().map(String::from),
        words: None,
        segments: Some(segments),
    })
}
