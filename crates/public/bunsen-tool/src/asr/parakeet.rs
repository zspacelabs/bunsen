//! Transcribing with Parakeet: the construct through a stream driver.
use std::sync::Arc;

use bunsen::{
    errors::{
        BunsenResult,
        ResultContext,
    },
    kits::speech::asr::{
        driver::{
            LevelPolicy,
            TranscriptSegment,
        },
        parakeet::{
            ParakeetConstruct,
            ParakeetEngine,
            ParakeetStreamDriver,
        },
    },
};

use crate::asr::{
    Emission,
    Transcript,
    Word,
    millis,
    segment_speaker,
    speaker_turns,
    stream,
};

/// Parakeet's own transcription flags: none. The policy is the model's, its
/// presets and patch resolved (`--model '<name>{window_seconds: 20}'`).
#[derive(clap::Args, Debug, Default)]
pub struct ParakeetFlags {}

/// The driver for `bundle` under its policy and `emission`.
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) if the policy's
/// `window_seconds` is under one encoder frame; as the driver and the VAD.
pub fn driver(
    bundle: Arc<ParakeetConstruct>,
    emission: Emission,
) -> BunsenResult<ParakeetStreamDriver> {
    let engine = ParakeetEngine::new(bundle)
        .context("the model policy's window_seconds")
        .as_policy()?;
    let driver = ParakeetStreamDriver::new(engine, emission.into())?;
    Ok(driver)
}

/// What each window's features are scaled against: each window normalized as
/// the extractor normalizes a whole item.
pub const LEVEL: LevelPolicy = LevelPolicy::PerWindow;

/// Transcribes one recording through `driver`: each committed segment is a
/// word.
pub fn transcribe(
    driver: &ParakeetStreamDriver,
    samples: &[f32],
    chunk: usize,
) -> BunsenResult<(Transcript, Vec<TranscriptSegment>)> {
    let (segments, _) = stream::stream(driver, samples, LEVEL, chunk)?;
    let words = segments
        .iter()
        .filter_map(|s| {
            let text = s.text.as_deref()?.trim();
            (!text.is_empty()).then(|| Word {
                word: text.to_string(),
                start: millis(s.start),
                end: millis(s.end),
                tokens: s.tokens.clone(),
                speaker: segment_speaker(s),
            })
        })
        .collect();
    let transcript = Transcript {
        text: stream::join_text(&segments),
        turns: driver.speakers().map(|_| speaker_turns(&segments)),
        language: None,
        words: Some(words),
        segments: None,
    };
    Ok((transcript, segments))
}
