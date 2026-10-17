//! Transcribing with Parakeet: the bundle through a stream driver.
use std::sync::Arc;

use bunsen::{
    data::pretrained::PretrainedCache,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::{
        asr_driver::LevelPolicy,
        parakeet::{
            ParakeetBundle,
            ParakeetEngine,
            ParakeetStreamDriver,
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
        Transcript,
        Word,
        millis,
    },
};

/// Parakeet's own transcription flags.
#[derive(clap::Args, Debug)]
pub struct ParakeetFlags {
    /// The longest stretch of audio the encoder sees at once, in seconds; a
    /// longer one is committed up to a pause between words, and the rest
    /// decoded again with the audio after it.
    #[arg(long, default_value_t = ParakeetEngine::DEFAULT_WINDOW_SECONDS)]
    pub window_seconds: f64,
}

/// The driver for `bundle` under `flags` and `preset`.
///
/// # Errors
/// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) if
/// `--window-seconds` is under one encoder frame; as the driver and the VAD.
pub fn driver(
    bundle: Arc<ParakeetBundle>,
    flags: &ParakeetFlags,
    preset: Preset,
    cache: &PretrainedCache,
    device: &Device,
) -> BunsenResult<ParakeetStreamDriver> {
    let frame = bundle.frame_seconds();
    if flags.window_seconds.is_nan() || flags.window_seconds < frame {
        return Err(BunsenError::illegal(format!(
            "--window-seconds {} is under one encoder frame, {frame} s",
            flags.window_seconds
        )));
    }
    let engine = ParakeetEngine::new(bundle).with_window_seconds(flags.window_seconds);
    let driver = ParakeetStreamDriver::new(engine, preset.into())?;
    stream::with_vad(driver, preset, cache, device)
}

/// Transcribes one recording through `driver`: each committed segment is a
/// word.
pub fn transcribe(
    driver: &ParakeetStreamDriver,
    samples: &[f32],
) -> BunsenResult<Transcript> {
    // Each window is normalized as the extractor normalizes a whole item.
    let (segments, _) = stream::stream(driver, samples, LevelPolicy::PerWindow)?;
    let words = segments
        .iter()
        .filter_map(|s| {
            let text = s.text.as_deref()?.trim();
            (!text.is_empty()).then(|| Word {
                word: text.to_string(),
                start: millis(s.start),
                end: millis(s.end),
            })
        })
        .collect();
    Ok(Transcript {
        text: stream::join_text(&segments),
        language: None,
        words: Some(words),
        segments: None,
    })
}
