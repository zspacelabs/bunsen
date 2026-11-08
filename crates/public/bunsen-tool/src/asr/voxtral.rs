//! Transcribing with Voxtral Realtime: the construct through a stream driver.
use std::sync::Arc;

use bunsen::{
    errors::{
        BunsenResult,
        ResultContext,
    },
    kits::speech::asr::{
        driver::TranscriptSegment,
        voxtral::{
            VoxtralConstruct,
            driver::{
                VoxtralEngine,
                VoxtralStreamDriver,
            },
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

/// Voxtral Realtime's own transcription flags: none. The policy is the model's,
/// its presets and patch resolved (`--model '<name>{window_seconds: 20}'`).
#[derive(clap::Args, Debug, Default)]
pub struct VoxtralFlags {}

/// The driver for `bundle` under its policy and `emission`.
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) if the policy's
/// `window_seconds` is under one decoder position; as the driver and the VAD.
/// The policy's `delay_ms` was applied when the model loaded.
pub fn driver(
    bundle: Arc<VoxtralConstruct>,
    emission: Emission,
) -> BunsenResult<VoxtralStreamDriver> {
    log::info!(
        "voxtral: delay {} ms",
        bundle.layout.delay_positions() * bundle.layout.config().position_ms
    );
    let engine = VoxtralEngine::new(bundle)
        .context("the model policy's window_seconds")
        .as_policy()?;
    let driver = VoxtralStreamDriver::new(engine, emission.into())?;
    Ok(driver)
}

/// Transcribes one recording through `driver`: each committed segment is a
/// word.
pub fn transcribe(
    driver: &VoxtralStreamDriver,
    samples: &[f32],
    chunk: usize,
) -> BunsenResult<(Transcript, Vec<TranscriptSegment>)> {
    let (segments, _) = stream::stream(driver, samples, chunk)?;
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
