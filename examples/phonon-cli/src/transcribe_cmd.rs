//! `phonon transcribe`.
use std::{
    path::PathBuf,
    time::Instant,
};

use bunsen::{
    errors::BunsenResult,
    kits::speech::parakeet::ParakeetFrontEndMeta,
    support::audio::load_audio_mono_sr,
};
use bunsen_app::logging::{
    LogArgs,
    LogLevelNum,
};
use serde::Serialize;

use crate::args::ModelArgs;

/// Transcribes audio files.
///
/// Each file is decoded at the model's rate (16 kHz, mono): bunsen does not
/// resample. A recording longer than `--window-seconds` is transcribed in
/// windows cut at quiet points.
#[derive(clap::Args, Debug)]
pub struct TranscribeCmd {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub model: ModelArgs,

    /// Print one JSON object per file: the text, and each word with its
    /// start and end in seconds.
    #[arg(long)]
    json: bool,

    /// The longest stretch of audio the encoder sees at once, in seconds.
    #[arg(long, default_value_t = 60.0)]
    window_seconds: f64,

    /// Windows transcribed at once.
    #[arg(long, default_value_t = 4)]
    batch: usize,

    /// The audio files.
    #[arg(required = true)]
    files: Vec<PathBuf>,
}

/// One word, as `--json` prints it.
#[derive(Serialize)]
struct JsonWord<'a> {
    word: &'a str,
    start: f64,
    end: f64,
}

/// One file, as `--json` prints it.
#[derive(Serialize)]
struct JsonTranscript<'a> {
    file: String,
    text: &'a str,
    words: Vec<JsonWord<'a>>,
}

impl TranscribeCmd {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging.init(Some(LogLevelNum::Warn))?;
        let device = self.model.device()?;
        let t0 = Instant::now();
        let bundle = self.model.load_bundle(&device)?;
        log::info!("loaded {} in {:.1?}", self.model.model, t0.elapsed());

        let rate = bundle.front_end.sample_rate();
        let window = (self.window_seconds * rate as f64) as usize;
        for path in &self.files {
            let samples = load_audio_mono_sr(path, rate)?;
            let t0 = Instant::now();
            let transcript = bundle.transcribe_long(&samples, window, self.batch, &device)?;
            let elapsed = t0.elapsed();
            log::info!(
                "{}: {:.1} s of audio in {:.2?} ({:.1}x realtime)",
                path.display(),
                samples.len() as f64 / rate as f64,
                elapsed,
                samples.len() as f64 / rate as f64 / elapsed.as_secs_f64()
            );

            if self.json {
                let words = transcript.words(&bundle.vocab, bundle.frame_seconds())?;
                let json = JsonTranscript {
                    file: path.display().to_string(),
                    text: &transcript.text,
                    words: words
                        .iter()
                        .map(|w| JsonWord {
                            word: &w.text,
                            start: millis(w.start),
                            end: millis(w.end),
                        })
                        .collect(),
                };
                println!(
                    "{}",
                    serde_json::to_string(&json).expect("a transcript serializes")
                );
            } else {
                println!("{}", transcript.text);
            }
        }
        Ok(())
    }
}

/// Seconds, rounded to the millisecond.
fn millis(seconds: f64) -> f64 {
    (seconds * 1000.0).round() / 1000.0
}
