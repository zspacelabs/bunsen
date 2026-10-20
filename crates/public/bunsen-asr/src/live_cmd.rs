//! `bunsen-asr <family> live`.
use std::io::{
    IsTerminal,
    Write,
};

use bunsen::{
    data::pretrained::PretrainedCache,
    errors::{
        BunsenResult,
        ResultContext,
    },
    kits::speech::asr_driver::TranscriptEvent,
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};

use crate::{
    args::DeviceArgs,
    capture::CaptureArgs,
    driver::{
        Driver,
        Stream,
    },
    family::Family,
    stream::Preset,
    transcribe_cmd::FamilyFlags,
};

/// The most of a draft shown on a terminal: its end, where it changes.
const DRAFT_CHARS: usize = 100;

/// Transcribes the microphone, as it speaks, through the family's stream
/// driver.
///
/// The host's default input device (or `--input-device`) is opened at the
/// model's rate when it offers that, and at its own default rate
/// otherwise, resampled here. The capture callback hands its buffers,
/// downmixed to mono, to the main thread, which batches them into pushes,
/// anchors the stream's clock at each push's capture time, and prints what
/// the driver emits: each decode's committed text as a line on stdout, and,
/// under `--preset responsive`, a draft of the speech so far on stderr,
/// redrawn in place on a terminal. Ctrl-C, or `--seconds`, ends the stream
/// cleanly: the speech past the last commit is decoded before the process
/// exits.
#[derive(clap::Args, Debug)]
pub struct LiveCmd<F: clap::Args> {
    #[clap(flatten)]
    pub logging: LogArgs,

    /// The model, by name: `list` shows them; the family's default when
    /// omitted. Whisper also takes a checkpoint's path.
    #[arg(long)]
    pub model: Option<String>,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    #[clap(flatten)]
    pub device: DeviceArgs,

    /// When to decode: a microphone has no end, so by default each speech
    /// region is decoded as it closes.
    #[arg(long, value_enum, default_value_t = Preset::Conservative)]
    pub preset: Preset,

    /// Milliseconds of audio per push: the capture callbacks, which arrive
    /// every few milliseconds, are batched to this.
    #[arg(long, default_value_t = 250)]
    pub chunk_ms: usize,

    #[clap(flatten)]
    pub flags: F,

    #[clap(flatten)]
    pub capture: CaptureArgs,
}

impl<F: clap::Args + FamilyFlags> LiveCmd<F> {
    pub fn run(
        &self,
        family: Family,
    ) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let Some(capture) = self.capture.open()? else {
            return Ok(());
        };

        let device = self.device.device()?;
        // The options come from the command line.
        let cache =
            PretrainedCache::new(self.flags.cache_options(self.cache.options())).as_policy()?;
        let model = self.model.as_deref().unwrap_or(family.default_model());
        let loaded = family.load(model, &self.flags, &cache, &device)?;
        let driver = Driver::build(loaded, &self.flags, self.preset, &cache, &device)?;
        let rate = driver.sample_rate();
        let chunk = driver.push_samples(self.chunk_ms);
        log::debug!(
            "push: {chunk} samples ({:.0} ms)",
            chunk as f64 * 1000.0 / rate as f64
        );

        let mut stream = driver.open()?;
        let mut printer = Printer::new();
        capture.run(rate, chunk, |samples, time| {
            let events = push(&mut stream, samples, time)?;
            printer.report(&stream, &events);
            Ok(())
        })?;
        let events = stream.end_read()?;
        printer.report(&stream, &events);
        printer.clear_draft();
        log::info!("{} samples pushed", stream.samples_seen());
        Ok(())
    }
}

/// Pushes `samples` as one block, anchored at `time` when the host gave
/// one.
fn push(
    stream: &mut Stream,
    samples: &[f32],
    time: Option<f64>,
) -> BunsenResult<Vec<TranscriptEvent>> {
    log::trace!(
        "push: {} samples{}",
        samples.len(),
        time.map(|t| format!(" captured at {t:.3} s"))
            .unwrap_or_default()
    );
    match time {
        Some(time) => stream.anchor_write_read(time, samples),
        None => stream.write_read(samples),
    }
}

/// Prints what a stream emits: each decode's committed text as one line on
/// stdout (Parakeet commits a segment per word, so a decode's words share a
/// line); the latest draft on stderr, redrawn in place on a terminal and a
/// line of its own, marked `~`, otherwise.
struct Printer {
    terminal: bool,
    draft_shown: bool,
    announced: bool,
}

impl Printer {
    fn new() -> Self {
        Self {
            terminal: std::io::stderr().is_terminal(),
            draft_shown: false,
            announced: false,
        }
    }

    fn report(
        &mut self,
        stream: &Stream,
        events: &[TranscriptEvent],
    ) {
        if !self.announced
            && let Some(language) = stream.language()
        {
            log::info!("language: {language}");
            self.announced = true;
        }
        let committed: Vec<&TranscriptEvent> = events.iter().filter(|e| e.is_committed()).collect();
        let text: Vec<&str> = committed
            .iter()
            .filter_map(|e| e.segment().text.as_deref().map(str::trim))
            .filter(|t| !t.is_empty())
            .collect();
        if let (Some(first), Some(last)) = (committed.first(), committed.last()) {
            log::debug!(
                "[{:>8.2} --> {:>8.2}]",
                first.segment().start,
                last.segment().end
            );
        }
        if !text.is_empty() {
            self.clear_draft();
            println!("{}", text.join(" "));
        }
        if let Some(draft) = events.iter().rev().find(|e| !e.is_committed()) {
            self.show_draft(draft.segment().text.as_deref().unwrap_or("").trim());
        }
    }

    fn show_draft(
        &mut self,
        text: &str,
    ) {
        if !self.terminal {
            eprintln!("~ {text}");
            return;
        }
        let skip = text.chars().count().saturating_sub(DRAFT_CHARS);
        let tail: String = text.chars().skip(skip).collect();
        eprint!("\r\x1b[2K~ {tail}");
        let _ = std::io::stderr().flush();
        self.draft_shown = true;
    }

    /// Erases a draft shown in place.
    fn clear_draft(&mut self) {
        if self.draft_shown {
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
            self.draft_shown = false;
        }
    }
}
