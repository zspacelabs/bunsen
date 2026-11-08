//! `asr diarize run`.

use std::{
    path::PathBuf,
    time::Instant,
};

use bunsen::{
    errors::BunsenResult,
    kits::speech::diarization::RttmSegment,
    support::audio::load_audio_mono_sr,
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    device::BackendArgs,
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};

use super::{
    AsrDiarizeOutputArgs,
    AsrDiarizerArgs,
    JsonDiarization,
    JsonTurn,
    speaker_totals,
    text_line,
};
use crate::asr::recording_id;

/// Diarizes audio files: each one's speaker turns, as lines of
/// `{start} {end} speaker_{k}` (seconds), RTTM, or JSON.
///
/// Each file is decoded at the model's rate (16 kHz, mono) and streamed
/// through the diarizer at its `latency` preset; speakers are numbered in the
/// order they first speak. The model's info card, and after each file how
/// long it took, are logged at `INFO` (`-v`).
#[derive(clap::Args, Debug)]
pub struct AsrDiarizeRunCommand {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub diarizer: AsrDiarizerArgs,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    #[clap(flatten)]
    pub device: BackendArgs,

    #[clap(flatten)]
    pub output: AsrDiarizeOutputArgs,

    /// The audio files.
    #[arg(required = true)]
    pub files: Vec<PathBuf>,
}

impl AsrDiarizeRunCommand {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let device = self.device.open()?;
        let cache = self.cache.init()?;
        let bundle = self.diarizer.load(&cache, &device)?;
        let streaming = bundle.streaming()?;
        let filter = bundle.turns().clone();
        let rate = bundle.sample_rate();

        for path in &self.files {
            let samples = load_audio_mono_sr(path, rate)?;
            let t0 = Instant::now();
            let turns = bundle.diarize(&samples, streaming.clone(), &filter)?;
            let elapsed = t0.elapsed();

            if self.output.json {
                let json = JsonDiarization {
                    file: path.display().to_string(),
                    turns: turns.iter().map(JsonTurn::from).collect(),
                    speakers: speaker_totals(&turns),
                };
                println!(
                    "{}",
                    serde_json::to_string(&json).expect("a diarization serializes")
                );
            } else if self.output.rttm {
                let id = recording_id(path);
                for turn in &turns {
                    println!("{}", RttmSegment::from_turn(&id, turn));
                }
            } else {
                if self.files.len() > 1 {
                    println!("{}:", path.display());
                }
                for turn in &turns {
                    println!("{}", text_line(turn));
                }
            }

            let seconds = samples.len() as f64 / rate as f64;
            log::info!(
                "{}: {seconds:.1} s of audio in {elapsed:.2?} ({:.1}x realtime)",
                path.display(),
                seconds / elapsed.as_secs_f64()
            );
        }
        Ok(())
    }
}
