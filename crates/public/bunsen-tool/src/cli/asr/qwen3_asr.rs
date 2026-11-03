//! Transcribing with Qwen3-ASR: the construct through a stream driver.
use std::sync::Arc;

use bunsen::{
    data::pretrained::PretrainedCache,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::{
        asr_driver::{
            LevelPolicy,
            TranscriptSegment,
        },
        qwen3_asr::{
            Qwen3AsrConstruct,
            Qwen3AsrEngine,
            Qwen3AsrFrontEndMeta,
            Qwen3AsrStreamDriver,
        },
    },
};
use burn::tensor::Device;

use crate::cli::asr::{
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

/// Qwen3-ASR's own transcription flags.
#[derive(clap::Args, Debug)]
pub struct Qwen3AsrFlags {
    /// The longest stretch of audio the model sees at once, in seconds; 30
    /// is the most it was trained on. A longer recording is cut at the
    /// quietest point of each full window's last third. The model's policy
    /// when omitted.
    #[arg(long)]
    pub window_seconds: Option<f64>,
}

/// The driver for `bundle` under `flags` and `preset`.
///
/// # Errors
/// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) if
/// `--window-seconds` is under one frame or over
/// [`MAX_WINDOW_SECONDS`](Qwen3AsrConstruct::MAX_WINDOW_SECONDS); as the driver
/// and the VAD.
pub fn driver(
    bundle: Arc<Qwen3AsrConstruct>,
    flags: &Qwen3AsrFlags,
    preset: Preset,
    cache: &PretrainedCache,
    device: &Device,
) -> BunsenResult<Qwen3AsrStreamDriver> {
    let front_end = &bundle.front_end;
    let frame = front_end.hop() as f64 / front_end.sample_rate() as f64;
    let most = Qwen3AsrConstruct::MAX_WINDOW_SECONDS as f64;
    let window_seconds = flags.window_seconds.unwrap_or(bundle.policy.window_seconds);
    if !(frame..=most).contains(&window_seconds) {
        return Err(BunsenError::illegal(format!(
            "--window-seconds {window_seconds} is outside [{frame}, {most}] s"
        )));
    }
    let engine = Qwen3AsrEngine::new(bundle).with_window_seconds(window_seconds);
    let driver = Qwen3AsrStreamDriver::new(engine, preset.into())?;
    stream::with_vad(driver, preset, cache, device)
}

/// What each window's features are scaled against: each window packaged as the
/// front end packages a whole utterance.
pub const LEVEL: LevelPolicy = LevelPolicy::PerWindow;

/// Transcribes one recording through `driver`.
pub fn transcribe(
    driver: &Qwen3AsrStreamDriver,
    samples: &[f32],
    chunk: usize,
) -> BunsenResult<(Transcript, Vec<TranscriptSegment>)> {
    let (segments, ctx) = stream::stream(driver, samples, LEVEL, chunk)?;
    let transcript = Transcript {
        text: stream::join_text(&segments),
        turns: driver.speakers().map(|_| speaker_turns(&segments)),
        language: ctx.language().map(String::from),
        words: None,
        // A committed span each: a window, cut quiet when it was full.
        segments: Some(
            segments
                .iter()
                .map(|s| Segment {
                    start: s.start,
                    end: s.end,
                    text: s.text.as_deref().unwrap_or("").trim().to_string(),
                    tokens: s.tokens.clone(),
                    speaker: segment_speaker(s),
                })
                .collect(),
        ),
    };
    Ok((transcript, segments))
}
