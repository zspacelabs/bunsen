//! Transcribing with Qwen3-ASR: the construct through a stream driver.
use std::sync::Arc;

use bunsen::{
    errors::{
        BunsenResult,
        ResultContext,
    },
    kits::speech::asr::{
        driver::TranscriptSegment,
        qwen3_asr::{
            Qwen3AsrConstruct,
            Qwen3AsrEngine,
            Qwen3AsrStreamDriver,
        },
    },
};

use crate::asr::{
    Emission,
    Segment,
    Transcript,
    segment_speaker,
    speaker_turns,
    stream,
};

/// Qwen3-ASR's own transcription flags: none. The policy is the model's, its
/// presets and patch resolved (`--model '<name>{window_seconds: 20}'`).
#[derive(clap::Args, Debug, Default)]
pub struct Qwen3AsrFlags {}

/// The driver for `bundle` under its policy and `emission`.
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) if the policy's
/// `window_seconds` is under one frame or over
/// [`MAX_WINDOW_SECONDS`](Qwen3AsrConstruct::MAX_WINDOW_SECONDS); as the driver
/// and the VAD.
pub fn driver(
    bundle: Arc<Qwen3AsrConstruct>,
    emission: Emission,
) -> BunsenResult<Qwen3AsrStreamDriver> {
    let engine = Qwen3AsrEngine::new(bundle)
        .context("the model policy's window_seconds")
        .as_policy()?;
    let driver = Qwen3AsrStreamDriver::new(engine, emission.into())?;
    Ok(driver)
}

/// Transcribes one recording through `driver`.
pub fn transcribe(
    driver: &Qwen3AsrStreamDriver,
    samples: &[f32],
    chunk: usize,
) -> BunsenResult<(Transcript, Vec<TranscriptSegment>)> {
    let (segments, ctx) = stream::stream(driver, samples, chunk)?;
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
