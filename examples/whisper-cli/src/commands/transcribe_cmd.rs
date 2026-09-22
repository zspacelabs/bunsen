use std::{
    path::PathBuf,
    time::{
        Duration,
        Instant,
    },
};

use bunsen::{
    errors::BunsenResult,
    kits::speech::whisper::driver::PresetEmissionPolicy,
    support::audio::load_audio_mono_sr,
};
use burn::prelude::Backend;
use clap_common::logging::{
    LogArgs,
    LogLevelNum,
};

use crate::whisper_clap::WhisperDriverArgs;

/// Milliseconds of audio per push when `--chunk-ms` is omitted.
const CHUNK_MS: usize = 1000;

/// The emission preset when `--preset` is omitted: a file is read whole,
/// so nothing is gained by decoding before a window fills.
const PRESET: PresetEmissionPolicy = PresetEmissionPolicy::Offline;

/// Transcribes an audio file with a Whisper checkpoint and its vocabulary,
/// through the stream driver: the audio is pushed in chunks as a live loop
/// would feed it, and segments are printed as they become final — or,
/// under the responsive preset, as drafts first.
///
/// The checkpoint and the `.tiktoken` vocabulary that matches it come
/// through bunsen's pretrained cache, fetched on first use or found where
/// a deployment put them; the vocabulary is what gives text rather than
/// ids and upstream's default suppress list. `silero-weights` bundles the
/// VAD the real-time presets need.
/// The backend is [`bunsen::support::testing::PerformanceBackend`],
/// chosen by bunsen's backend feature at build time (`--features
/// bunsen/wgpu`; see the README).
#[derive(clap::Args, Debug)]
pub struct TranscribeCmd {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub whisper: WhisperDriverArgs,

    /// Display the filename before each transcript.
    #[arg(long)]
    print_filename: bool,

    /// Print only the basename of the filename.
    #[arg(
      long,
      action=clap::ArgAction::Set,
      default_value_t = true,
      default_missing_value = "true"
    )]
    strip_filename: bool,

    /// Display the index of the filename.
    #[arg(long)]
    print_index: bool,

    /// Path to the audio files.
    files: Vec<PathBuf>,
}

impl TranscribeCmd {
    pub fn run<B: Backend>(&self) -> BunsenResult<()> {
        self.logging.init(Some(LogLevelNum::Warn))?;

        let device = B::Device::default();
        let driver = self.whisper.init_driver::<B>(&device, PRESET)?;
        let chunk = self.whisper.chunk_samples(&driver, CHUNK_MS);

        let num_files = self.files.len();
        let mut timings: Vec<(Duration, Duration)> = Vec::with_capacity(num_files);
        for (idx, path) in self.files.iter().enumerate() {
            // The audio is decoded at the model's rate: the checkpoint's to
            // declare, not the caller's.
            let wav = load_audio_mono_sr(path, driver.sample_rate())?;

            log::debug!("path: {}", path.display());
            log::debug!(
                "audio: {} samples, {:.2} s",
                wav.len(),
                wav.len() as f64 / driver.sample_rate() as f64
            );

            if self.print_index {
                print!("{:>6}/{:<6} ", idx + 1, num_files);
            }
            if self.print_filename {
                if self.strip_filename {
                    print!("{}\t", path.file_name().unwrap().display());
                } else {
                    print!("{}\t", path.display());
                }
            }

            let t0 = Instant::now();

            let mut stream = self.whisper.open_stream(&driver)?;
            for block in wav.chunks(chunk) {
                stream.write_read(block)?;
            }
            stream.end_read()?;

            let t1 = Instant::now();
            let decode_time = t1.duration_since(t0);

            let sample_time = Duration::from_secs_f64(stream.last_end());

            timings.push((sample_time, decode_time));

            log::debug!("sample: {:10.1?}", sample_time);
            log::debug!("decode: {:10.1?}", decode_time);

            let ratio = sample_time.as_secs_f64() / decode_time.as_secs_f64();
            log::debug!("sample/decode: {ratio:.2}");
        }

        let mean_sample_time = timings
            .iter()
            .map(|(sample_time, _)| sample_time)
            .sum::<Duration>()
            / timings.len() as u32;
        let mean_decode_time = timings
            .iter()
            .map(|(_, decode_time)| decode_time)
            .sum::<Duration>()
            / timings.len() as u32;
        let mean_ratio = timings
            .iter()
            .map(|(sample_time, decode_time)| sample_time.as_secs_f64() / decode_time.as_secs_f64())
            .sum::<f64>()
            / timings.len() as f64;

        log::info!("mean sample: {:10.1?}", mean_sample_time);
        log::info!("mean decode: {:10.1?}", mean_decode_time);
        log::info!("mean sample/decode: {mean_ratio:.2}");

        Ok(())
    }
}
