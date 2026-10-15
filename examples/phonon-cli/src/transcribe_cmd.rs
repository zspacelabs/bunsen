//! `phonon transcribe`.
use std::{
    path::PathBuf,
    time::Instant,
};

use bunsen::{
    errors::BunsenResult,
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
/// Each file is decoded at the model's rate (16 kHz, mono). A recording
/// longer than `--window-seconds` is transcribed in windows cut at quiet
/// points.
#[derive(clap::Args, Debug)]
pub struct TranscribeCmd {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub model: ModelArgs,

    /// Print one JSON object per file: the text, the language when the model
    /// names one (Phonon-1), and each word with its start and end in seconds
    /// when the model times them (Phonon-2).
    #[arg(long)]
    json: bool,

    /// The longest stretch of audio the model sees at once, in seconds: 60
    /// for Phonon-2, 30 (the most it was trained on) for Phonon-1, when
    /// omitted.
    #[arg(long)]
    window_seconds: Option<f64>,

    /// Windows transcribed at once, where the model batches (Phonon-2).
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
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    words: Option<Vec<JsonWord<'a>>>,
}

impl TranscribeCmd {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging.init(Some(LogLevelNum::Warn))?;
        let device = self.model.device()?;
        let t0 = Instant::now();
        let model = self.model.load(&device)?;
        log::info!("loaded {} in {:.1?}", self.model.model, t0.elapsed());

        let rate = model.sample_rate();
        let seconds = self
            .window_seconds
            .unwrap_or_else(|| model.default_window_seconds());
        let window = (seconds * rate as f64) as usize;
        for path in &self.files {
            let samples = load_audio_mono_sr(path, rate)?;
            let t0 = Instant::now();
            let transcript = model.transcribe(&samples, window, self.batch, &device)?;
            let elapsed = t0.elapsed();
            log::info!(
                "{}: {:.1} s of audio in {:.2?} ({:.1}x realtime)",
                path.display(),
                samples.len() as f64 / rate as f64,
                elapsed,
                samples.len() as f64 / rate as f64 / elapsed.as_secs_f64()
            );

            if self.json {
                let json = JsonTranscript {
                    file: path.display().to_string(),
                    text: &transcript.text,
                    language: transcript.language.as_deref(),
                    words: transcript.words.as_ref().map(|words| {
                        words
                            .iter()
                            .map(|(word, start, end)| JsonWord {
                                word,
                                start: millis(*start),
                                end: millis(*end),
                            })
                            .collect()
                    }),
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
