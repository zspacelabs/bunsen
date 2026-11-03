//! Transcribing with Parakeet: the construct through a stream driver.
use std::sync::Arc;

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::{
        asr_driver::{
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

use crate::cli::asr::{
    stream::{
        self,
        Emission,
    },
    transcribe_cmd::{
        Transcript,
        Word,
        millis,
        segment_speaker,
        speaker_turns,
    },
};

/// Parakeet's own transcription flags.
#[derive(clap::Args, Debug)]
pub struct ParakeetFlags {
    /// The longest stretch of audio the encoder sees at once, in seconds; a
    /// longer one is committed up to a pause between words, and the rest
    /// decoded again with the audio after it. The model's policy when
    /// omitted.
    #[arg(long)]
    pub window_seconds: Option<f64>,
}

/// The driver for `bundle` under `flags` and `emission`.
///
/// # Errors
/// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) if
/// `--window-seconds` is under one encoder frame; as the driver and the VAD.
pub fn driver(
    bundle: Arc<ParakeetConstruct>,
    flags: &ParakeetFlags,
    emission: Emission,
) -> BunsenResult<ParakeetStreamDriver> {
    let frame = bundle.frame_seconds();
    let window_seconds = flags.window_seconds.unwrap_or(bundle.policy.window_seconds);
    if window_seconds.is_nan() || window_seconds < frame {
        return Err(BunsenError::illegal(format!(
            "--window-seconds {window_seconds} is under one encoder frame, {frame} s"
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
