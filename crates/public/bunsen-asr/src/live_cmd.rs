//! `bunsen-asr <family> live`.
use std::{
    io::{
        IsTerminal,
        Write,
    },
    sync::{
        Arc,
        atomic::{
            AtomicBool,
            Ordering,
        },
        mpsc::{
            self,
            RecvTimeoutError,
        },
    },
    time::{
        Duration,
        Instant,
    },
};

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::asr_driver::TranscriptEvent,
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LogArgs,
        LogLevelNum,
    },
};
use cpal::{
    Data,
    Device,
    FromSample,
    Host,
    InputCallbackInfo,
    Sample,
    SampleFormat,
    SizedSample,
    StreamConfig,
    StreamInstant,
    SupportedStreamConfig,
    traits::{
        DeviceTrait,
        HostTrait,
        StreamTrait,
    },
};

use crate::{
    args::DeviceArgs,
    driver::{
        Driver,
        Stream,
    },
    family::Family,
    resample::Resampler,
    stream::Preset,
    transcribe_cmd::FamilyFlags,
};

/// How long the main loop waits for a capture callback before checking
/// whether it has been asked to stop.
const POLL: Duration = Duration::from_millis(100);

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
    /// omitted.
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

    /// The input device, by a case-insensitive substring of its id or its
    /// name (see `--list-devices`); the host's default input device when
    /// omitted.
    #[arg(long)]
    pub input_device: Option<String>,

    /// List the input devices, id and name, and exit. Loads nothing.
    #[arg(long)]
    pub list_devices: bool,

    /// Stop after this many seconds of audio; on Ctrl-C when omitted.
    #[arg(long)]
    pub seconds: Option<f64>,
}

/// One capture callback's audio: mono at the device's rate, and when its
/// first sample was captured.
struct Block {
    /// Seconds since the first callback's capture, when the host's clock
    /// could say.
    time: Option<f64>,
    samples: Vec<f32>,
}

impl<F: clap::Args + FamilyFlags> LiveCmd<F> {
    pub fn run(
        &self,
        family: Family,
    ) -> BunsenResult<()> {
        self.logging.init(Some(LogLevelNum::Warn))?;
        let host = cpal::default_host();
        if self.list_devices {
            return list_devices(&host);
        }
        let input = self.pick_device(&host)?;
        let input_name = device_name(&input);

        let device = self.device.device()?;
        let cache = self.cache.init()?;
        let model = self.model.as_deref().unwrap_or(family.default_model());
        let loaded = family.load(model, &cache, &device)?;
        let driver = Driver::build(loaded, &self.flags, self.preset, &cache, &device)?;
        let model_rate = driver.sample_rate();
        let chunk = driver.push_samples(self.chunk_ms);
        log::debug!(
            "push: {chunk} samples ({:.0} ms)",
            chunk as f64 * 1000.0 / model_rate as f64
        );

        let supported = pick_config(&input, model_rate as u32)?;
        let capture_rate = supported.sample_rate() as usize;
        let mut resampler = Resampler::new(capture_rate, model_rate);
        log::info!(
            "capture: {input_name}: {} ch, {capture_rate} Hz, {}{}",
            supported.channels(),
            supported.sample_format(),
            if resampler.is_identity() {
                String::new()
            } else {
                format!(", resampled to {model_rate} Hz")
            }
        );

        let stop = Arc::new(AtomicBool::new(false));
        {
            let stop = stop.clone();
            ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst)).map_err(|e| {
                BunsenError::sys("could not install the Ctrl-C handler").with_cause(e)
            })?;
        }

        let (tx, rx) = mpsc::channel::<Block>();
        let capture = build_stream(&input, &supported, tx)?;
        let mut stream = driver.open()?;
        let mut printer = Printer::new();
        let limit = self.seconds.map(|s| (s * capture_rate as f64) as usize);

        capture.play().map_err(|e| {
            BunsenError::sys(format!("could not start capture on {input_name}")).with_cause(e)
        })?;
        eprintln!(
            "listening on {input_name}; {}",
            match self.seconds {
                Some(s) => format!("stopping after {s} s"),
                None => "Ctrl-C to stop".to_string(),
            }
        );
        let started = Instant::now();

        // Model-rate audio waiting for a push, and the capture time of its
        // first sample. Pushes are whole chunks, so each has the driver's
        // grain; what a block leaves over waits for the next.
        let mut pending: Vec<f32> = Vec::with_capacity(2 * chunk);
        let mut pending_time: Option<f64> = None;
        let mut captured = 0usize;

        loop {
            if stop.load(Ordering::SeqCst) || limit.is_some_and(|n| captured >= n) {
                break;
            }
            let block = match rx.recv_timeout(POLL) {
                Ok(block) => take(block, &mut captured, limit),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            };
            if pending.is_empty() {
                pending_time = block.time;
            }
            resampler.process(&block.samples, &mut pending);
            while pending.len() >= chunk {
                let events = push(&mut stream, &pending[..chunk], pending_time)?;
                printer.report(&stream, &events);
                pending.drain(..chunk);
                // The remainder is contiguous with what was pushed, so its
                // first sample's time follows at the rate.
                pending_time = pending_time.map(|t| t + chunk as f64 / model_rate as f64);
            }
        }

        // Stop capturing before the flush, so the tail is what was heard
        // before the stop; the callback's sender goes with the stream, so
        // the drain below ends.
        let _ = capture.pause();
        drop(capture);
        for block in rx.try_iter() {
            if limit.is_some_and(|n| captured >= n) {
                break;
            }
            let block = take(block, &mut captured, limit);
            if pending.is_empty() {
                pending_time = block.time;
            }
            resampler.process(&block.samples, &mut pending);
        }
        if !pending.is_empty() {
            let events = push(&mut stream, &pending, pending_time)?;
            printer.report(&stream, &events);
        }
        let events = stream.end_read()?;
        printer.report(&stream, &events);
        printer.clear_draft();

        log::info!(
            "captured: {:.1} s of audio in {:.1?}; {} samples pushed",
            captured as f64 / capture_rate as f64,
            started.elapsed(),
            stream.samples_seen(),
        );
        Ok(())
    }

    /// `--input-device`, or the host's default input device.
    fn pick_device(
        &self,
        host: &Host,
    ) -> BunsenResult<Device> {
        match &self.input_device {
            None => host.default_input_device().ok_or_else(|| {
                BunsenError::policy("no default input device; name one with --input-device")
            }),
            Some(want) => {
                let needle = want.to_lowercase();
                host.input_devices()
                    .map_err(list_error)?
                    .find(|device| {
                        device_id(device).to_lowercase().contains(&needle)
                            || device_name(device).to_lowercase().contains(&needle)
                    })
                    .ok_or_else(|| {
                        BunsenError::policy(format!(
                            "no input device matches {want:?}; see --list-devices"
                        ))
                    })
            }
        }
    }
}

/// `block`, cut to what is left of `limit` samples, counted into
/// `captured`: `--seconds` is audio taken, however far capture ran ahead
/// of the decode.
fn take(
    mut block: Block,
    captured: &mut usize,
    limit: Option<usize>,
) -> Block {
    if let Some(n) = limit {
        block.samples.truncate(n.saturating_sub(*captured));
    }
    *captured += block.samples.len();
    block
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

/// Sorts a failure to enumerate the host's input devices.
fn list_error(e: cpal::DevicesError) -> BunsenError {
    BunsenError::sys("could not list the input devices").with_cause(e)
}

/// The device's human-readable name; `?` when the host cannot say.
fn device_name(device: &Device) -> String {
    device
        .description()
        .map(|description| description.name().to_string())
        .unwrap_or_else(|_| "?".to_string())
}

/// The device's id within its host (ALSA's PCM name, for one); `?` when
/// the host cannot say.
fn device_id(device: &Device) -> String {
    device
        .id()
        .map(|id| id.1)
        .unwrap_or_else(|_| "?".to_string())
}

/// Prints the host's default input device, then each input device that
/// opens, as `id  name: default configuration`. One that does not open
/// (ALSA lists plugins with no device behind them) is logged at debug.
fn list_devices(host: &Host) -> BunsenResult<()> {
    match host.default_input_device() {
        Some(device) => println!("default: {}  {}", device_id(&device), device_name(&device)),
        None => println!("default: none"),
    }
    for device in host.input_devices().map_err(list_error)? {
        let id = device_id(&device);
        let name = device_name(&device);
        match device.default_input_config() {
            Ok(config) => println!(
                "  {id}  {name}: {} ch, {} Hz, {}",
                config.channels(),
                config.sample_rate(),
                config.sample_format(),
            ),
            Err(err) => log::debug!("{id} ({name}): {err}"),
        }
    }
    Ok(())
}

/// The capture configuration: the fewest channels the device offers at
/// `rate` in a format this can read, else its default configuration, to be
/// resampled.
fn pick_config(
    device: &Device,
    rate: u32,
) -> BunsenResult<SupportedStreamConfig> {
    let native = device
        .supported_input_configs()
        .map_err(|e| {
            BunsenError::sys("could not list the device's input configurations").with_cause(e)
        })?
        .filter(|range| {
            format_rank(range.sample_format()).is_some()
                && range.min_sample_rate() <= rate
                && rate <= range.max_sample_rate()
        })
        .min_by_key(|range| (range.channels(), format_rank(range.sample_format())))
        .and_then(|range| range.try_with_sample_rate(rate));
    if let Some(config) = native {
        return Ok(config);
    }

    let config = device.default_input_config().map_err(|e| {
        BunsenError::sys("could not read the device's default input configuration").with_cause(e)
    })?;
    if format_rank(config.sample_format()).is_none() {
        return Err(BunsenError::unsupported(format!(
            "the device captures {}, which this cannot read",
            config.sample_format()
        )));
    }
    Ok(config)
}

/// Preference among the formats [`downmix`] reads; `None` for one it does
/// not.
fn format_rank(format: SampleFormat) -> Option<u8> {
    Some(match format {
        SampleFormat::F32 => 0,
        SampleFormat::I16 => 1,
        SampleFormat::I32 => 2,
        SampleFormat::I24 => 3,
        SampleFormat::F64 => 4,
        SampleFormat::U16 => 5,
        SampleFormat::U32 => 6,
        SampleFormat::U24 => 7,
        SampleFormat::I8 => 8,
        SampleFormat::U8 => 9,
        SampleFormat::I64 => 10,
        SampleFormat::U64 => 11,
        _ => return None,
    })
}

/// Opens the input stream: each callback downmixes its buffer and sends it
/// with its capture time.
fn build_stream(
    device: &Device,
    supported: &SupportedStreamConfig,
    tx: mpsc::Sender<Block>,
) -> BunsenResult<cpal::Stream> {
    let config: StreamConfig = supported.config();
    let channels = config.channels as usize;
    let mut origin: Option<StreamInstant> = None;

    let data_callback = move |data: &Data, info: &InputCallbackInfo| {
        let capture = info.timestamp().capture;
        let origin = *origin.get_or_insert(capture);
        let time = capture
            .duration_since(&origin)
            .map(|since| since.as_secs_f64());
        let mut samples = Vec::with_capacity(data.len() / channels);
        downmix(data, channels, &mut samples);
        // A closed receiver means the main loop has stopped: nothing to do.
        let _ = tx.send(Block { time, samples });
    };
    let error_callback = |err: cpal::StreamError| log::error!("capture: {err}");

    device
        .build_input_stream_raw(
            &config,
            supported.sample_format(),
            data_callback,
            error_callback,
            None,
        )
        .map_err(|e| BunsenError::sys("could not open the input stream").with_cause(e))
}

/// Averages each frame's channels into one `f32` sample in `[-1, 1]`.
fn downmix(
    data: &Data,
    channels: usize,
    out: &mut Vec<f32>,
) {
    fn frames<T>(
        data: &Data,
        channels: usize,
        out: &mut Vec<f32>,
    ) where
        T: SizedSample,
        f32: FromSample<T>,
    {
        let samples = data
            .as_slice::<T>()
            .expect("the format the stream was built with");
        out.extend(samples.chunks_exact(channels).map(|frame| {
            frame.iter().map(|&s| f32::from_sample(s)).sum::<f32>() / channels as f32
        }));
    }

    match data.sample_format() {
        SampleFormat::I8 => frames::<i8>(data, channels, out),
        SampleFormat::I16 => frames::<i16>(data, channels, out),
        SampleFormat::I24 => frames::<cpal::I24>(data, channels, out),
        SampleFormat::I32 => frames::<i32>(data, channels, out),
        SampleFormat::I64 => frames::<i64>(data, channels, out),
        SampleFormat::U8 => frames::<u8>(data, channels, out),
        SampleFormat::U16 => frames::<u16>(data, channels, out),
        SampleFormat::U24 => frames::<cpal::U24>(data, channels, out),
        SampleFormat::U32 => frames::<u32>(data, channels, out),
        SampleFormat::U64 => frames::<u64>(data, channels, out),
        SampleFormat::F32 => frames::<f32>(data, channels, out),
        SampleFormat::F64 => frames::<f64>(data, channels, out),
        // `pick_config` admits only the formats above.
        other => log::error!("capture: unreadable sample format {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Blocks are taken whole until the limit, then cut to it, then empty.
    #[test]
    fn test_take() {
        let block = |n: usize| Block {
            time: None,
            samples: vec![0.0; n],
        };
        let mut captured = 0;
        assert_eq!(take(block(4), &mut captured, Some(10)).samples.len(), 4);
        assert_eq!(take(block(8), &mut captured, Some(10)).samples.len(), 6);
        assert_eq!(take(block(8), &mut captured, Some(10)).samples.len(), 0);
        assert_eq!(captured, 10);
        assert_eq!(take(block(8), &mut captured, None).samples.len(), 8);
    }
}
