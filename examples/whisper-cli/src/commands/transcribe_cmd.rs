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
use clap_common::logging::LogArgs;

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

    /// Path to the audio file; decoded to mono at the model's rate.
    #[arg(long)]
    audio: String,

    /// Milliseconds of audio per push, as a live loop would feed it.
    #[arg(long, default_value = "1000")]
    chunk_ms: usize,

    /// Print each segment's ids beside its text.
    #[arg(long)]
    ids: bool,
}

impl TranscribeCmd {
    pub fn run<B: Backend>(&self) -> BunsenResult<()> {
        let device = B::Device::default();

        let driver = self.whisper.init_driver::<B>(&device)?;

        // The audio is decoded at the model's rate: the checkpoint's to
        // declare, not the caller's.
        let wav = load_audio_mono_sr(&self.audio, driver.sample_rate())?;
        log::info!(
            "audio: {} samples, {:.2} s",
            wav.len(),
            wav.len() as f64 / driver.sample_rate() as f64
        );

        // A bare stream: a clock from zero at the model's rate, and the running
        // maximum as the mel clamp reference.
        let mut ctx = driver.new_context(
            StreamClock::uniform(driver.sample_rate()),
            RunningMaxClamp::new(),
        )?;
        let chunk = (self.chunk_ms * driver.sample_rate() / 1000).max(1);
        let mut announced = false;
        for block in wav.chunks(chunk) {
            let emissions = ctx.write_read(block)?;
            // Detection runs on the first window decoded, so the language is
            // known once anything has been emitted; say so before the text.
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
        for emission in ctx.end_read()? {
            report(&emission, self.ids);
        }

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
