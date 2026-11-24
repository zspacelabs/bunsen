//! A take played on an output device, for `tts run --play`.

use std::{
    sync::{
        Arc,
        atomic::{
            AtomicBool,
            Ordering,
        },
    },
    time::Duration,
};

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    ops::signal::resample::StreamingResampler,
};
use cpal::{
    Data,
    Device,
    FromSample,
    OutputCallbackInfo,
    Sample,
    SampleFormat,
    SizedSample,
    StreamConfig,
    SupportedStreamConfig,
    traits::{
        DeviceTrait,
        StreamTrait,
    },
};

use crate::support::{
    AudioDirection,
    audio_device_name,
    find_audio_device,
    list_audio_devices,
    sample_format_rank,
};

/// How often playback checks whether it has finished, or been stopped.
pub(super) const POLL: Duration = Duration::from_millis(50);

/// What playback waits, after the last sample is handed over, for the
/// device to sound it.
pub(super) const DRAIN: Duration = Duration::from_millis(250);

/// Whether to play the take, and on which output device.
#[derive(clap::Args, Debug)]
pub struct PlaybackArgs {
    /// Play the speech on an output device.
    ///
    /// Once it is synthesized; as it is, with `--stream`.
    #[arg(long)]
    pub play: bool,

    /// The output device, by a substring of its id or name.
    ///
    /// Matched case-insensitively (`--list-output-devices`); the host's
    /// default output device when omitted.
    #[arg(long)]
    pub output_device: Option<String>,

    /// List the output devices, id and name, and exit. Loads nothing.
    #[arg(long)]
    pub list_output_devices: bool,

    /// With `--stream`, the audio queued before playback starts, in
    /// milliseconds.
    ///
    /// Playback starts once this much is queued, and again after an
    /// underrun (the queue running dry before the speech ends, which the
    /// summary counts). Raise it on a device that decodes slower than real
    /// time.
    #[arg(long, default_value_t = 200)]
    pub prebuffer_ms: u64,
}

impl PlaybackArgs {
    /// Prints the output devices, when `--list-output-devices` asked;
    /// whether it did.
    ///
    /// # Errors
    /// As the host listing its devices.
    pub fn list(&self) -> BunsenResult<bool> {
        if self.list_output_devices {
            list_audio_devices(&cpal::default_host(), AudioDirection::Output)?;
        }
        Ok(self.list_output_devices)
    }

    /// The output device, opened, when `--play` asked for one.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when no device
    /// matches; as the host listing its devices.
    pub fn open(&self) -> BunsenResult<Option<Playback>> {
        if !self.play {
            return Ok(None);
        }
        let device = find_audio_device(
            &cpal::default_host(),
            AudioDirection::Output,
            self.output_device.as_deref(),
        )?;
        Ok(Some(Playback {
            name: audio_device_name(&device),
            device,
        }))
    }
}

/// An output device to play a take on.
pub struct Playback {
    pub(super) device: Device,
    pub(super) name: String,
}

impl Playback {
    /// The device's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Plays `samples`, mono at `rate`, to the end or Ctrl-C: at the
    /// device's rate (resampled when it cannot open at `rate`), every
    /// channel the same.
    ///
    /// # Errors
    /// As opening and starting the stream.
    pub fn play(
        &self,
        samples: &[f32],
        rate: u32,
    ) -> BunsenResult<()> {
        let supported = pick_config(&self.device, rate)?;
        let device_rate = supported.sample_rate();
        let samples = StreamingResampler::resample(samples, rate as usize, device_rate as usize);
        log::info!(
            "playback: {}: {} ch, {device_rate} Hz, {}{}",
            self.name,
            supported.channels(),
            supported.sample_format(),
            if device_rate == rate {
                String::new()
            } else {
                format!(", resampled from {rate} Hz")
            }
        );

        let stop = Arc::new(AtomicBool::new(false));
        {
            let stop = stop.clone();
            ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst)).map_err(|e| {
                BunsenError::sys("could not install the Ctrl-C handler").with_cause(e)
            })?;
        }
        let done = Arc::new(AtomicBool::new(false));
        let stream = build_stream(&self.device, &supported, samples, done.clone())?;
        stream.play().map_err(|e| {
            BunsenError::sys(format!("could not start playback on {}", self.name)).with_cause(e)
        })?;
        while !done.load(Ordering::SeqCst) && !stop.load(Ordering::SeqCst) {
            std::thread::sleep(POLL);
        }
        if !stop.load(Ordering::SeqCst) {
            std::thread::sleep(DRAIN);
        }
        let _ = stream.pause();
        Ok(())
    }
}

/// The playback configuration: the device's at `rate` in a format this
/// writes, the fewest channels first; else its default, resampled to.
pub(super) fn pick_config(
    device: &Device,
    rate: u32,
) -> BunsenResult<SupportedStreamConfig> {
    let native = device
        .supported_output_configs()
        .map_err(|e| {
            BunsenError::sys("could not list the device's output configurations").with_cause(e)
        })?
        .filter(|range| {
            sample_format_rank(range.sample_format()).is_some()
                && range.min_sample_rate() <= rate
                && rate <= range.max_sample_rate()
        })
        .min_by_key(|range| (range.channels(), sample_format_rank(range.sample_format())))
        .and_then(|range| range.try_with_sample_rate(rate));
    if let Some(config) = native {
        return Ok(config);
    }
    let config = AudioDirection::Output.default_config(device)?;
    if sample_format_rank(config.sample_format()).is_none() {
        return Err(BunsenError::unsupported(format!(
            "the device plays {}, which this cannot write",
            config.sample_format()
        )));
    }
    Ok(config)
}

/// Opens the output stream: each callback takes its frames from `samples`,
/// then silence; `done` is set once every sample is handed over.
fn build_stream(
    device: &Device,
    supported: &SupportedStreamConfig,
    samples: Vec<f32>,
    done: Arc<AtomicBool>,
) -> BunsenResult<cpal::Stream> {
    let config: StreamConfig = supported.config();
    let channels = config.channels as usize;
    let mut at = 0;
    let data_callback = move |data: &mut Data, _: &OutputCallbackInfo| {
        if fill_data(data, channels, &samples, &mut at) {
            done.store(true, Ordering::SeqCst);
        }
    };
    let error_callback = |err: cpal::Error| log::error!("playback: {err}");
    device
        .build_output_stream_raw(
            config,
            supported.sample_format(),
            data_callback,
            error_callback,
            None,
        )
        .map_err(|e| BunsenError::sys("could not open the output stream").with_cause(e))
}

/// Fills `data` in its sample format: [`fill`], by format.
fn fill_data(
    data: &mut Data,
    channels: usize,
    samples: &[f32],
    at: &mut usize,
) -> bool {
    fn typed<T>(
        data: &mut Data,
        channels: usize,
        samples: &[f32],
        at: &mut usize,
    ) -> bool
    where
        T: SizedSample + FromSample<f32>,
    {
        let out = data
            .as_slice_mut::<T>()
            .expect("the format the stream was built with");
        fill(out, channels, samples, at)
    }

    match data.sample_format() {
        SampleFormat::I8 => typed::<i8>(data, channels, samples, at),
        SampleFormat::I16 => typed::<i16>(data, channels, samples, at),
        SampleFormat::I24 => typed::<cpal::I24>(data, channels, samples, at),
        SampleFormat::I32 => typed::<i32>(data, channels, samples, at),
        SampleFormat::I64 => typed::<i64>(data, channels, samples, at),
        SampleFormat::U8 => typed::<u8>(data, channels, samples, at),
        SampleFormat::U16 => typed::<u16>(data, channels, samples, at),
        SampleFormat::U24 => typed::<cpal::U24>(data, channels, samples, at),
        SampleFormat::U32 => typed::<u32>(data, channels, samples, at),
        SampleFormat::U64 => typed::<u64>(data, channels, samples, at),
        SampleFormat::F32 => typed::<f32>(data, channels, samples, at),
        SampleFormat::F64 => typed::<f64>(data, channels, samples, at),
        // `pick_config` admits only the formats above.
        other => {
            log::error!("playback: unwritable sample format {other}");
            true
        }
    }
}

/// Fills `out`, interleaved frames of `channels`, from mono `samples` at
/// `at` (every channel the same sample, clipped to `[-1, 1]`), then
/// silence; advances `at`, and says whether every sample is now out.
fn fill<T>(
    out: &mut [T],
    channels: usize,
    samples: &[f32],
    at: &mut usize,
) -> bool
where
    T: Sample + FromSample<f32>,
{
    for frame in out.chunks_mut(channels) {
        let value = match samples.get(*at) {
            Some(&s) => {
                *at += 1;
                T::from_sample(s.clamp(-1.0, 1.0))
            }
            None => T::EQUILIBRIUM,
        };
        frame.fill(value);
    }
    *at >= samples.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mono fans out to every channel, then silence; `at` says how far.
    #[test]
    fn test_fill() {
        let samples = [0.5, -0.5, 2.0];
        let mut at = 0;
        let mut out = [9.0f32; 4];
        assert!(!fill(&mut out, 2, &samples, &mut at));
        assert_eq!(out, [0.5, 0.5, -0.5, -0.5]);
        assert!(fill(&mut out, 2, &samples, &mut at), "the last sample out");
        assert_eq!(out, [1.0, 1.0, 0.0, 0.0], "clipped, then silence");

        // Integer formats: their own scale, silence at their midpoint.
        let mut at = 0;
        let mut out = [0u16; 3];
        assert!(fill(&mut out, 1, &[0.0, 1.0], &mut at));
        assert_eq!(out[0], u16::EQUILIBRIUM);
        assert_eq!(out[1], u16::MAX);
        assert_eq!(out[2], u16::EQUILIBRIUM);
    }
}
