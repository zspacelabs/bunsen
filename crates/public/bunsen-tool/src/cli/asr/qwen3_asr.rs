//! Transcribing with Qwen3-ASR: the construct through a stream driver.
use std::sync::Arc;

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::asr::{
        driver::{
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

/// Qwen3-ASR's own transcription flags: none. The policy is the model's, its
/// presets and patch resolved (`--model '<name>{window_seconds: 20}'`).
#[derive(clap::Args, Debug, Default)]
pub struct Qwen3AsrFlags {}

/// The driver for `bundle` under its policy and `emission`.
///
/// # Errors
/// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) if
/// the policy's `window_seconds` is under one frame or over
/// [`MAX_WINDOW_SECONDS`](Qwen3AsrConstruct::MAX_WINDOW_SECONDS); as the driver
/// and the VAD.
pub fn driver(
    bundle: Arc<Qwen3AsrConstruct>,
    emission: Emission,
) -> BunsenResult<Qwen3AsrStreamDriver> {
    let front_end = &bundle.front_end;
    let frame = front_end.hop() as f64 / front_end.sample_rate() as f64;
    let most = Qwen3AsrConstruct::MAX_WINDOW_SECONDS as f64;
    let window_seconds = bundle.policy.window_seconds;
    if !(frame..=most).contains(&window_seconds) {
        return Err(BunsenError::illegal(format!(
            "window_seconds {window_seconds} is outside [{frame}, {most}] s"
        )));
    }
    let engine = Qwen3AsrEngine::new(bundle).with_window_seconds(window_seconds);
    let driver = Qwen3AsrStreamDriver::new(engine, emission.into())?;
    Ok(driver)
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
