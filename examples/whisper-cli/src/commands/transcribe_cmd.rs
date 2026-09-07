use std::{
    path::PathBuf,
    time::{
        Duration,
        Instant,
    },
};

use bunsen::{
    errors::BunsenResult,
    kits::speech::whisper::driver::{
        RunningMaxClamp,
        StreamClock,
        TranscriptEvent,
    },
    support::audio::load_audio_mono_sr,
};
use burn::prelude::Backend;
use clap_common::logging::{
    LogArgs,
    LogLevelNum,
};

use crate::whisper_clap::WhisperDriverArgs;

/// Transcribes an audio file with the bundled Whisper `base` checkpoint and
/// its vocabulary, through the stream driver: the audio is pushed in chunks
/// as a live loop would feed it, and segments are printed as they become
/// final — or, under the responsive preset, as drafts first.
///
/// Everything comes from bunsen's own features. `whisper-weights` bundles
/// the checkpoint and the `.tiktoken` vocabulary that matches it, which
/// is what gives text rather than ids and upstream's default suppress
/// list; `silero-weights` bundles the VAD the real-time presets need.
/// The backend is [`bunsen::support::testing::PerformanceBackend`],
/// chosen by bunsen's backend feature at build time (`--features
/// bunsen/wgpu`; see the README).
#[derive(clap::Args, Debug)]
pub struct TranscribeCmd {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub whisper: WhisperDriverArgs,

    /// Milliseconds of audio per push, as a live loop would feed it.
    #[arg(long, default_value = "1000")]
    chunk_ms: usize,

    /// Print each segment's ids beside its text.
    #[arg(long)]
    ids: bool,

    /// Display the filename before each transcript.
    #[arg(long)]
    print_filename: bool,

    /// Display the index of the filename.
    #[arg(long)]
    print_index: bool,

    /// Path to the audio files.
    files: Vec<PathBuf>,
}

impl TranscribeCmd {
    pub fn run<B: Backend>(&self) -> BunsenResult<()> {
        self.logging.init(Some(LogLevelNum::Info))?;

        let device = B::Device::default();

        let driver = self.whisper.init_driver::<B>(&device)?;

        let num_files = self.files.len();
        let mut timings: Vec<(Duration, Duration)> = Vec::with_capacity(num_files);
        for (idx, path) in self.files.iter().enumerate() {
            // The audio is decoded at the model's rate: the checkpoint's to
            // declare, not the caller's.
            let wav = load_audio_mono_sr(path, driver.sample_rate())?;

            if self.print_index {
                print!("{:>6}/{:<6} ", idx + 1, num_files);
            }
            if self.print_filename {
                print!("{}\t", path.display());
            }

            log::info!(
                "path: {}\naudio: {} samples, {:.2} s",
                path.display(),
                wav.len(),
                wav.len() as f64 / driver.sample_rate() as f64
            );

            let t0 = Instant::now();

            // A bare stream: a clock from zero at the model's rate, and the
            // running maximum as the mel clamp reference.
            let mut ctx = driver.new_context(
                StreamClock::uniform(driver.sample_rate()),
                RunningMaxClamp::new(),
            )?;
            let chunk = (self.chunk_ms * driver.sample_rate() / 1000).max(1);
            let mut announced = false;
            for block in wav.chunks(chunk) {
                let emissions = ctx.write_read(block)?;
                // Detection runs on the first window decoded, so the language
                // is known once anything has been emitted; say
                // so before the text.
                if !announced
                    && driver.detects_language()
                    && let Some(code) = ctx.language()
                {
                    log::info!("language: {code}");
                    announced = true;
                }
                for emission in emissions {
                    report(&emission, self.ids);
                }
            }
            let mut sample_decode_time = 0.0;
            for emission in ctx.end_read()? {
                sample_decode_time = emission.segment().end;
                report(&emission, self.ids);
            }
            let t1 = Instant::now();
            let decode_time = t1.duration_since(t0);

            let sample_time = Duration::from_secs_f64(sample_decode_time);

            timings.push((sample_time, decode_time));

            log::info!("sample: {:10.1?}", sample_time);
            log::info!("decode: {:10.1?}", decode_time);

            let ratio = sample_time.as_secs_f64() / decode_time.as_secs_f64();
            log::info!("sample/decode: {ratio:.2}");
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

/// One line per emission: a draft is marked `~`, a commit is not.
fn report(
    emission: &TranscriptEvent,
    ids: bool,
) {
    let segment = emission.segment();
    let mark = if emission.is_committed() { ' ' } else { '~' };

    log::info!("{mark}[{:>8.2} --> {:>8.2}]", segment.start, segment.end,);
    println!("{}", segment.text.as_deref().unwrap_or("").trim(),);

    if ids {
        log::info!("ids: {:?}", segment.tokens);
    }
}
