//! Transcribing with Voxtral Realtime: the bundle through a stream driver.
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
        voxtral::{
            VoxtralBundle,
            VoxtralFrontEndMeta,
            driver::{
                VoxtralEngine,
                VoxtralStreamDriver,
            },
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
        Transcript,
        Word,
        millis,
        segment_speaker,
        speaker_turns,
    },
};

/// Voxtral Realtime's own transcription flags.
#[derive(clap::Args, Debug)]
pub struct VoxtralFlags {
    /// The longest stretch of audio the model sees at once, in seconds; a
    /// longer one is committed up to a pause between words, and the rest
    /// decoded again with the audio after it.
    #[arg(long, default_value_t = VoxtralEngine::DEFAULT_WINDOW_SECONDS)]
    pub window_seconds: f64,

    /// How far the transcript runs behind the audio, in ms: a multiple of
    /// 80 up to 1200, or 2400. Longer is more accurate; the model's
    /// `tekken.json` names its default, 480.
    #[arg(long)]
    pub delay_ms: Option<usize>,
}

/// The driver for `bundle` under `flags` and `preset`.
///
/// # Errors
/// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) if
/// `--window-seconds` is under one decoder position, or `--delay-ms` is not a
/// delay the model takes; as the driver and the VAD.
pub fn driver(
    bundle: Arc<VoxtralBundle>,
    flags: &VoxtralFlags,
    preset: Preset,
    cache: &PretrainedCache,
    device: &Device,
) -> BunsenResult<VoxtralStreamDriver> {
    let position = bundle.samples_per_position() as f64 / bundle.front_end.sample_rate() as f64;
    if flags.window_seconds.is_nan() || flags.window_seconds < position {
        return Err(BunsenError::illegal(format!(
            "--window-seconds {} is under one decoder position, {position} s",
            flags.window_seconds
        )));
    }
    let bundle = match flags.delay_ms {
        Some(ms) => Arc::new(Arc::unwrap_or_clone(bundle).with_delay_ms(ms)?),
        None => bundle,
    };
    log::info!(
        "voxtral: delay {} ms",
        bundle.layout.delay_positions() * bundle.layout.config().position_ms
    );
    let engine = VoxtralEngine::new(bundle).with_window_seconds(flags.window_seconds);
    let driver = VoxtralStreamDriver::new(engine, preset.into())?;
    stream::with_vad(driver, preset, cache, device)
}

/// What each window's features are scaled against: nothing the stream
/// measures, as Voxtral's front end floors under a fixed level; the window's
/// own, which is not read.
pub const LEVEL: LevelPolicy = LevelPolicy::PerWindow;

/// Transcribes one recording through `driver`: each committed segment is a
/// word.
pub fn transcribe(
    driver: &VoxtralStreamDriver,
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
