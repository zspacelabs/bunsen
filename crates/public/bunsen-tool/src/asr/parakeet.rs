//! Transcribing with Parakeet: the construct through a stream driver.
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
        parakeet::{
            ParakeetConstruct,
            ParakeetEngine,
            ParakeetStreamDriver,
        },
    },
};

use crate::asr::{
    stream,
    stream::Emission,
    transcript::{
        Transcript,
        Word,
        millis,
        segment_speaker,
        speaker_turns,
    },
};

/// Parakeet's own transcription flags: none. The policy is the model's, its
/// presets and patch resolved (`--model '<name>{window_seconds: 20}'`).
#[derive(clap::Args, Debug, Default)]
pub struct ParakeetFlags {}

/// The driver for `bundle` under its policy and `emission`.
///
/// # Errors
/// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) if
/// the policy's `window_seconds` is under one encoder frame; as the driver and
/// the VAD.
pub fn driver(
    bundle: Arc<ParakeetConstruct>,
    emission: Emission,
) -> BunsenResult<ParakeetStreamDriver> {
    let frame = bundle.frame_seconds();
    let window_seconds = bundle.policy.window_seconds;
    if window_seconds.is_nan() || window_seconds < frame {
        return Err(BunsenError::illegal(format!(
            "window_seconds {window_seconds} is under one encoder frame, {frame} s"
        )));
    }
    let engine = ParakeetEngine::new(bundle).with_window_seconds(window_seconds);
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
