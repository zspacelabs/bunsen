//! Transcribing with Whisper: the bundle through a stream driver.
use std::sync::Arc;

use bunsen::{
    data::pretrained::PretrainedCache,
    errors::{
        BunsenResult,
        ResultContext,
    },
    kits::speech::{
        silero_vad::{
            SileroVadMeta,
            pretrained::default_silero_factory,
        },
        whisper::{
            WhisperFallbackConfig,
            driver::{
                PresetEmissionPolicy,
                RunningMaxClamp,
                StreamClock,
                WhisperBundle,
                WhisperStreamDriver,
                WhisperStreamDriverConfig,
                WhisperTask,
            },
        },
    },
    ops::signal::perceptive_audio::PerceptiveAudioConverterMeta,
};
use burn::tensor::Device;

use crate::transcribe_cmd::{
    Segment,
    Transcript,
};

/// When the driver decodes.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// Whole windows, one after another.
    Offline,

    /// Speech regions as well, found by Silero's VAD (`silero/vad-16k`,
    /// through the same cache): silence is never decoded.
    Conservative,
}

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

    /// When to decode.
    #[arg(long, value_enum, default_value_t = Preset::Offline)]
    pub preset: Preset,

    /// Beams in the search; 1 is greedy.
    #[arg(long, default_value_t = 5)]
    pub beam_size: usize,
}

/// The driver for `bundle` under `flags`, with Silero's VAD attached under
/// the conservative preset.
pub fn driver(
    bundle: Arc<WhisperBundle>,
    flags: &WhisperFlags,
    cache: &PretrainedCache,
    device: &Device,
) -> BunsenResult<WhisperStreamDriver> {
    let multilingual = bundle.layout.ids().is_multilingual();
    if !multilingual && flags.language.is_some() {
        log::warn!("English-only checkpoint: --language and --task do not apply");
    }
    let preset = match flags.preset {
        Preset::Offline => PresetEmissionPolicy::Offline,
        Preset::Conservative => PresetEmissionPolicy::Conservative,
    };
    let task = match flags.task {
        Task::Transcribe => WhisperTask::Transcribe,
        Task::Translate => WhisperTask::Translate,
    };
    let mut driver: WhisperStreamDriver = WhisperStreamDriverConfig::new()
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
    if preset != PresetEmissionPolicy::Offline {
        let vad = default_silero_factory()?
            .load("silero/vad-16k", cache, device)?
            .handle;
        driver = driver.with_vad(vad.expect_branch(16000).clone(), Default::default())?;
    }
    Ok(driver)
}

/// Transcribes one recording through `driver`.
pub fn transcribe(
    driver: &WhisperStreamDriver,
    samples: &[f32],
) -> BunsenResult<Transcript> {
    let mut ctx = driver.new_context(
        StreamClock::uniform(driver.sample_rate()),
        RunningMaxClamp::new(),
    )?;
    let mut events = Vec::new();
    // About a second of audio a push, a whole number of the driver's grain:
    // its mel hop, its encoder grid and the VAD's chunk, so a push leaves
    // nothing staged behind it.
    let mut grain = lcm(driver.audio_converter().hop(), driver.encoder_grid());
    if let Some(vad) = driver.silero_vad_model() {
        grain = lcm(grain, vad.chunk_size());
    }
    let chunk = driver.sample_rate().div_ceil(grain) * grain;
    for block in samples.chunks(chunk) {
        events.extend(ctx.write_read(block)?);
    }
    events.extend(ctx.end_read()?);

    let segments: Vec<Segment> = events
        .iter()
        .filter(|e| e.is_committed())
        .map(|e| {
            let s = e.segment();
            Segment {
                start: s.start,
                end: s.end,
                text: s.text.as_deref().unwrap_or("").trim().to_string(),
            }
        })
        .collect();
    let text = segments
        .iter()
        .map(|s| s.text.as_str())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    Ok(Transcript {
        text,
        language: ctx.language().map(String::from),
        words: None,
        segments: Some(segments),
    })
}

fn gcd(
    a: usize,
    b: usize,
) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(
    a: usize,
    b: usize,
) -> usize {
    a / gcd(a, b) * b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcm() {
        assert_eq!(lcm(160, 320), 320);
        assert_eq!(lcm(320, 512), 2560);
    }
}
