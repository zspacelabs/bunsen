//! `bunsen-asr family <family> live`, and what `bunsen-asr live` shares.
use bunsen::{
    data::pretrained::PretrainedCache,
    errors::{
        BunsenResult,
        ResultContext,
    },
    kits::speech::asr_driver::{
        TranscriptEvent,
        TranscriptSegment,
    },
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

use crate::{
    capture::CaptureArgs,
    diarize::SpeakerArgs,
    driver::{
        Driver,
        Stream,
    },
    family::Family,
    publish::{
        Clock,
        Event,
        PublishArgs,
        Publisher,
        statements,
    },
    stream::Preset,
    transcribe_cmd::FamilyFlags,
};

/// Transcribes the microphone, as it speaks, through the family's stream
/// driver.
///
/// The host's default input device (or `--input-device`) is opened at the
/// model's rate when it offers that, and at its own default rate
/// otherwise, resampled here. The capture callback hands its buffers,
/// downmixed to mono, to the main thread, which batches them into pushes,
/// anchors the stream's clock at each push's capture time, and publishes
/// what the driver emits (`--publish`): each decode's statements, by default
/// a line each on stdout, and, under `--preset responsive`, a draft of the
/// speech so far, on stderr and redrawn in place on a terminal. Times are
/// the wall clock's: the stream's start is when capture began. Ctrl-C, or
/// `--seconds`, ends the stream
/// cleanly: the speech past the last commit is decoded before the process
/// exits.
#[derive(clap::Args, Debug)]
pub struct LiveCmd<F: clap::Args> {
    /// The model, by name: `list` shows them; the family's default when
    /// omitted. Whisper also takes a checkpoint's path.
    #[arg(long)]
    pub model: Option<String>,

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
    pub io: LiveIo,
}

/// What `live` listens to and runs on, whichever model it runs: the options
/// a family's command and a profile's share.
#[derive(clap::Args, Debug)]
pub struct LiveIo {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    #[clap(flatten)]
    pub device: BackendArgs,

    #[clap(flatten)]
    pub publish: PublishArgs,

    #[clap(flatten)]
    pub speakers: SpeakerArgs,

    #[clap(flatten)]
    pub capture: CaptureArgs,
}

impl<F: clap::Args + FamilyFlags> LiveCmd<F> {
    pub fn run(
        &self,
        family: Family,
    ) -> BunsenResult<()> {
        self.io.init_logging()?;
        let model = self.model.as_deref().unwrap_or(family.default_model());
        self.io
            .run(family, model, &self.flags, self.preset, self.chunk_ms)
    }
}

impl LiveIo {
    /// Logs as `-v`/`-q` say, warnings by default.
    pub fn init_logging(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))
    }

    /// Transcribes the microphone with `model` of `family` under `flags`,
    /// `preset` and pushes of `chunk_ms`.
    pub fn run<F: FamilyFlags>(
        &self,
        family: Family,
        model: &str,
        flags: &F,
        preset: Preset,
        chunk_ms: usize,
    ) -> BunsenResult<()> {
        let Some(capture) = self.capture.open()? else {
            return Ok(());
        };
        // The targets open first: a bad config, or a broker, fails before
        // the model loads. The wall clock's origin is set at the first push.
        let mut publisher = self
            .publish
            .init(Clock::wall_now(), self.speakers.diarize.is_some())?;

        let device = self.device.open()?;
        // The options come from the command line.
        let cache = PretrainedCache::new(flags.cache_options(self.cache.options())).as_policy()?;
        let loaded = family.load(model, flags, &cache, &device)?;
        let speakers = self.speakers.diarizer(&cache, &device)?;
        let driver = Driver::build(loaded, flags, preset, speakers, &cache, &device)?;
        let rate = driver.sample_rate();
        let chunk = driver.push_samples(chunk_ms);
        log::debug!(
            "push: {chunk} samples ({:.0} ms)",
            chunk as f64 * 1000.0 / rate as f64
        );

        let mut stream = driver.open()?;
        let mut reporter = Reporter { announced: false };
        let mut anchored = false;
        capture.run(rate, chunk, |samples, time| {
            if !anchored {
                // The first push's last sample was captured about now.
                let past = time.unwrap_or(0.0) + samples.len() as f64 / rate as f64;
                publisher.set_clock(Clock::wall_since(past));
                anchored = true;
            }
            let events = push(&mut stream, samples, time)?;
            reporter.report(&stream, &events, &mut publisher)
        })?;
        let events = stream.end_read()?;
        reporter.report(&stream, &events, &mut publisher)?;
        log::info!("{} samples pushed", stream.samples_seen());
        publisher.finish()
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

/// Publishes what a stream emits: each decode's committed segments as its
/// statements (one per run of a speaker), then its latest draft; and logs
/// the language once the model names one.
struct Reporter {
    announced: bool,
}

impl Reporter {
    fn report(
        &mut self,
        stream: &Stream,
        events: &[TranscriptEvent],
        publisher: &mut Publisher,
    ) -> BunsenResult<()> {
        if !self.announced
            && let Some(language) = stream.language()
        {
            log::info!("language: {language}");
            self.announced = true;
        }
        let committed: Vec<TranscriptSegment> = events
            .iter()
            .filter(|e| e.is_committed())
            .map(|e| e.segment().clone())
            .collect();
        if let (Some(first), Some(last)) = (committed.first(), committed.last()) {
            log::debug!("[{:>8.2} --> {:>8.2}]", first.start, last.end);
        }
        for statement in statements(&committed) {
            publisher.push(&Event::Statement {
                file: None,
                statement: &statement,
            })?;
        }
        if let Some(draft) = events.iter().rev().find(|e| !e.is_committed()) {
            let segment = draft.segment();
            publisher.push(&Event::Draft {
                start: segment.start,
                end: segment.end,
                text: segment.text.as_deref().unwrap_or(""),
            })?;
        }
        Ok(())
    }
}
