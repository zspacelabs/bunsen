use std::{
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
    kits::speech::whisper::driver::PresetEmissionPolicy,
};
use burn::prelude::Backend;
use clap_common::logging::{
    LogArgs,
    LogLevelNum,
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
    resample::Resampler,
    whisper_clap::{
        TranscriptStream,
        WhisperDriverArgs,
    },
};

/// Milliseconds of audio per push when `--chunk-ms` is omitted: the
/// capture callbacks, which arrive every few milliseconds, are batched to
/// this before the front end sees them.
const CHUNK_MS: usize = 250;

/// The emission preset when `--preset` is omitted: a microphone has no
/// end, so each speech region is decoded as it closes, and every line is
/// final.
const PRESET: PresetEmissionPolicy = PresetEmissionPolicy::Conservative;

/// How long the main loop waits for a capture callback before checking
/// whether it has been asked to stop.
const POLL: Duration = Duration::from_millis(100);

/// Transcribes the microphone, as it speaks, through the same stream
/// driver `transcribe` feeds a file into.
///
/// The host's default input device is opened at the model's rate when it
/// offers that, and at its own default rate otherwise, resampled here. The
/// capture callback hands its buffers, downmixed to mono, to the main
/// thread, which batches them into pushes, anchors the stream's clock at
/// each push's capture time, and prints what the driver emits. Ctrl-C, or
/// `--seconds`, ends the stream cleanly: the tail past the last endpoint
/// is decoded before the process exits.
///
/// Under the default `conservative` preset a line appears when a speech
/// region closes; `--preset responsive` adds a draft of the region so far
/// every 600 ms of speech.
#[derive(clap::Args, Debug)]
pub struct LiveCmd {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub whisper: WhisperDriverArgs,

    /// The input device, by a case-insensitive substring of its id or its
    /// name (see `--list-devices`); the host's default input device when
    /// omitted.
    #[arg(long)]
    device: Option<String>,

    /// List the input devices, id and name, and exit. Loads nothing.
    #[arg(long)]
    list_devices: bool,

    /// Stop after this many seconds of audio; on Ctrl-C when omitted.
    #[arg(long)]
    seconds: Option<f64>,
}

/// One capture callback's audio: mono at the device's rate, and when its
/// first sample was captured.
struct Block {
    /// Seconds since the first callback's capture, when the host's clock
    /// could say.
    time: Option<f64>,
    samples: Vec<f32>,
}

impl LiveCmd {
    pub fn run<B: Backend>(&self) -> BunsenResult<()> {
        self.logging.init(Some(LogLevelNum::Warn))?;

        let host = cpal::default_host();
        if self.list_devices {
            return list_devices(&host);
        }
        let device = self.pick_device(&host)?;
        let device_name = device_name(&device);

        let compute = B::Device::default();
        let driver = self.whisper.init_driver::<B>(&compute, PRESET)?;
        let model_rate = driver.sample_rate();
        let chunk = self.whisper.chunk_samples(&driver, CHUNK_MS);
        log::debug!(
            "push: {chunk} samples ({:.0} ms)",
            chunk as f64 * 1000.0 / model_rate as f64
        );

        let supported = pick_config(&device, model_rate as u32)?;
        let capture_rate = supported.sample_rate() as usize;
        let mut resampler = Resampler::new(capture_rate, model_rate);
        log::info!(
            "capture: {device_name}: {} ch, {capture_rate} Hz, {}{}",
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
            ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst))
                .map_err(BunsenError::external)?;
        }

        let (tx, rx) = mpsc::channel::<Block>();
        let stream = build_stream(&device, &supported, tx)?;
        let mut transcript = self.whisper.open_stream(&driver)?;
        let limit = self.seconds.map(|s| (s * capture_rate as f64) as usize);

        stream.play().map_err(BunsenError::external)?;
        eprintln!(
            "listening on {device_name}; {}",
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
                Ok(block) => block,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            };
            captured += block.samples.len();
            if pending.is_empty() {
                pending_time = block.time;
            }
            resampler.process(&block.samples, &mut pending);
            while pending.len() >= chunk {
                push(&mut transcript, &pending[..chunk], pending_time)?;
                pending.drain(..chunk);
                // The remainder is contiguous with what was pushed, so its
                // first sample's time follows at the rate.
                pending_time = pending_time.map(|t| t + chunk as f64 / model_rate as f64);
            }
        }

        // Stop capturing before the flush, so the tail is what was heard
        // before the stop; the callback's sender goes with the stream, so
        // the drain below ends.
        let _ = stream.pause();
        drop(stream);
        for block in rx.try_iter() {
            captured += block.samples.len();
            if pending.is_empty() {
                pending_time = block.time;
            }
            resampler.process(&block.samples, &mut pending);
        }
        if !pending.is_empty() {
            push(&mut transcript, &pending, pending_time)?;
        }
        transcript.end_read()?;

        let elapsed = started.elapsed();
        log::info!(
            "captured: {:.1} s of audio in {:.1?}; {} samples pushed",
            captured as f64 / capture_rate as f64,
            elapsed,
            transcript.samples_seen(),
        );
        Ok(())
    }

    /// `--device`, or the host's default input device.
    fn pick_device(
        &self,
        host: &Host,
    ) -> BunsenResult<Device> {
        match &self.device {
            None => host
                .default_input_device()
                .ok_or_else(|| BunsenError::Invalid("no default input device".to_string())),
            Some(want) => {
                let needle = want.to_lowercase();
                host.input_devices()
                    .map_err(BunsenError::external)?
                    .find(|device| {
                        device_id(device).to_lowercase().contains(&needle)
                            || device_name(device).to_lowercase().contains(&needle)
                    })
                    .ok_or_else(|| {
                        BunsenError::Invalid(format!(
                            "no input device matches {want:?}; see --list-devices"
                        ))
                    })
            }
        }
    }
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

/// Pushes `samples` as one block, anchored at `time` when the host gave
/// one.
fn push<B: Backend>(
    transcript: &mut TranscriptStream<B>,
    samples: &[f32],
    time: Option<f64>,
) -> BunsenResult<()> {
    log::trace!(
        "push: {} samples{}",
        samples.len(),
        time.map(|t| format!(" captured at {t:.3} s"))
            .unwrap_or_default()
    );
    match time {
        Some(time) => transcript.anchor_write_read(time, samples),
        None => transcript.write_read(samples),
    }
}

/// Prints the host's default input device, then each input device that
/// opens, as `id  name: default configuration`. One that does not open
/// (ALSA lists plugins with no device behind them) is logged at debug.
fn list_devices(host: &Host) -> BunsenResult<()> {
    match host.default_input_device() {
        Some(device) => println!("default: {}  {}", device_id(&device), device_name(&device)),
        None => println!("default: none"),
    }
    for device in host.input_devices().map_err(BunsenError::external)? {
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
        .map_err(BunsenError::external)?
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

    let config = device
        .default_input_config()
        .map_err(BunsenError::external)?;
    if format_rank(config.sample_format()).is_none() {
        return Err(BunsenError::Invalid(format!(
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
        .map_err(BunsenError::external)
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
