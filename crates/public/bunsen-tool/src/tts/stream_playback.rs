//! Speech played as it is synthesized, for `tts run --stream --play`.

use std::{
    collections::VecDeque,
    sync::{
        Arc,
        Mutex,
        atomic::{
            AtomicBool,
            Ordering,
        },
    },
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
    FromSample,
    OutputCallbackInfo,
    Sample,
    SampleFormat,
    SizedSample,
    StreamConfig,
    traits::{
        DeviceTrait,
        StreamTrait,
    },
};

use super::{
    Playback,
    playback::{
        DRAIN,
        POLL,
        pick_config,
    },
};

/// What the device plays from: the samples queued, at its rate, and whether
/// it is playing or (re)filling its prebuffer.
///
/// It starts by filling: silence until its prebuffer's samples are queued
/// (or the speech has ended). When the queue runs dry
/// before the speech ends, that is an underrun: silence, counted, and
/// filling again.
#[derive(Debug)]
pub struct PlaybackQueue {
    samples: VecDeque<f32>,
    prebuffer: usize,
    playing: bool,
    closed: bool,
    underruns: usize,
}

impl PlaybackQueue {
    /// An empty queue that plays once `prebuffer` samples are in it.
    pub fn new(prebuffer: usize) -> Self {
        Self {
            samples: VecDeque::new(),
            prebuffer,
            playing: false,
            closed: false,
            underruns: 0,
        }
    }

    /// `samples` after the queued ones.
    pub fn push(
        &mut self,
        samples: &[f32],
    ) {
        self.samples.extend(samples);
    }

    /// The speech has ended: no more samples come.
    pub fn close(&mut self) {
        self.closed = true;
    }

    /// The device's next sample; `None`, silence (filling, an underrun, or
    /// the end).
    pub fn next_sample(&mut self) -> Option<f32> {
        if !self.playing {
            if self.samples.len() < self.prebuffer && !self.closed {
                return None;
            }
            self.playing = true;
        }
        match self.samples.pop_front() {
            Some(sample) => Some(sample),
            None => {
                if !self.closed {
                    self.underruns += 1;
                    self.playing = false;
                }
                None
            }
        }
    }

    /// Whether every sample has been played and none will come.
    pub fn drained(&self) -> bool {
        self.closed && self.samples.is_empty()
    }

    /// The underruns so far.
    pub fn underruns(&self) -> usize {
        self.underruns
    }
}

/// Speech playing as it arrives ([`Playback::start_stream`]): samples
/// pushed at the speech's rate, resampled to the device's, queued.
pub struct StreamingPlayback {
    stream: cpal::Stream,
    queue: Arc<Mutex<PlaybackQueue>>,
    resampler: StreamingResampler,
    stop: Arc<AtomicBool>,
}

impl Playback {
    /// Starts playing speech that will arrive, mono at `rate`, once
    /// `prebuffer_ms` of it is queued: at the device's rate (resampled when it
    /// cannot open at `rate`), every channel the same. Ctrl-C stops it.
    ///
    /// # Errors
    /// As opening and starting the stream; installing the Ctrl-C handler.
    pub fn start_stream(
        &self,
        rate: u32,
        prebuffer_ms: u64,
    ) -> BunsenResult<StreamingPlayback> {
        let supported = pick_config(&self.device, rate)?;
        let device_rate = supported.sample_rate();
        log::info!(
            "playback: {}: {} ch, {device_rate} Hz, {}, streamed, {prebuffer_ms} ms prebuffer",
            self.name,
            supported.channels(),
            supported.sample_format(),
        );
        let prebuffer = (u64::from(device_rate) * prebuffer_ms / 1000) as usize;
        let queue = Arc::new(Mutex::new(PlaybackQueue::new(prebuffer)));
        let stop = Arc::new(AtomicBool::new(false));
        {
            let stop = stop.clone();
            ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst)).map_err(|e| {
                BunsenError::sys("could not install the Ctrl-C handler").with_cause(e)
            })?;
        }
        let config: StreamConfig = supported.config();
        let channels = config.channels as usize;
        let callback_queue = queue.clone();
        let data_callback = move |data: &mut Data, _: &OutputCallbackInfo| {
            let mut queue = callback_queue.lock().expect("the playback queue");
            fill_queue_data(data, channels, &mut queue);
        };
        let error_callback = |err: cpal::Error| log::error!("playback: {err}");
        let stream = self
            .device
            .build_output_stream_raw(
                config,
                supported.sample_format(),
                data_callback,
                error_callback,
                None,
            )
            .map_err(|e| BunsenError::sys("could not open the output stream").with_cause(e))?;
        stream.play().map_err(|e| {
            BunsenError::sys(format!("could not start playback on {}", self.name)).with_cause(e)
        })?;
        Ok(StreamingPlayback {
            stream,
            queue,
            resampler: StreamingResampler::new(rate as usize, device_rate as usize),
            stop,
        })
    }
}

impl StreamingPlayback {
    /// Queues `samples`, at the speech's rate.
    pub fn push(
        &mut self,
        samples: &[f32],
    ) {
        let mut out = Vec::with_capacity(samples.len());
        self.resampler.process(samples, &mut out);
        self.queue.lock().expect("the playback queue").push(&out);
    }

    /// Whether Ctrl-C stopped it.
    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    /// The speech has ended: plays what is queued to the end (or Ctrl-C),
    /// then stops.
    ///
    /// # Returns
    /// The underruns: times the queue ran dry before the speech ended.
    pub fn finish(self) -> usize {
        let Self {
            stream,
            queue,
            resampler,
            stop,
        } = self;
        let mut out = Vec::new();
        resampler.finish(&mut out);
        {
            let mut queue = queue.lock().expect("the playback queue");
            queue.push(&out);
            queue.close();
        }
        let stopped = || stop.load(Ordering::SeqCst);
        while !stopped() && !queue.lock().expect("the playback queue").drained() {
            std::thread::sleep(POLL);
        }
        if !stopped() {
            std::thread::sleep(DRAIN);
        }
        let _ = stream.pause();
        queue.lock().expect("the playback queue").underruns()
    }
}

/// Fills `data` in its sample format from `queue`: [`fill_queue`], by
/// format.
fn fill_queue_data(
    data: &mut Data,
    channels: usize,
    queue: &mut PlaybackQueue,
) {
    fn typed<T>(
        data: &mut Data,
        channels: usize,
        queue: &mut PlaybackQueue,
    ) where
        T: SizedSample + FromSample<f32>,
    {
        let out = data
            .as_slice_mut::<T>()
            .expect("the format the stream was built with");
        fill_queue(out, channels, queue);
    }

    match data.sample_format() {
        SampleFormat::I8 => typed::<i8>(data, channels, queue),
        SampleFormat::I16 => typed::<i16>(data, channels, queue),
        SampleFormat::I24 => typed::<cpal::I24>(data, channels, queue),
        SampleFormat::I32 => typed::<i32>(data, channels, queue),
        SampleFormat::I64 => typed::<i64>(data, channels, queue),
        SampleFormat::U8 => typed::<u8>(data, channels, queue),
        SampleFormat::U16 => typed::<u16>(data, channels, queue),
        SampleFormat::U24 => typed::<cpal::U24>(data, channels, queue),
        SampleFormat::U32 => typed::<u32>(data, channels, queue),
        SampleFormat::U64 => typed::<u64>(data, channels, queue),
        SampleFormat::F32 => typed::<f32>(data, channels, queue),
        SampleFormat::F64 => typed::<f64>(data, channels, queue),
        // `pick_config` admits only the formats above.
        other => log::error!("playback: unwritable sample format {other}"),
    }
}

/// Fills `out`, interleaved frames of `channels`, from `queue` (every
/// channel the same sample, clipped to `[-1, 1]`; silence where it gives
/// none).
fn fill_queue<T>(
    out: &mut [T],
    channels: usize,
    queue: &mut PlaybackQueue,
) where
    T: Sample + FromSample<f32>,
{
    for frame in out.chunks_mut(channels) {
        let value = queue
            .next_sample()
            .map_or(T::EQUILIBRIUM, |s| T::from_sample(s.clamp(-1.0, 1.0)));
        frame.fill(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(
        queue: &mut PlaybackQueue,
        n: usize,
    ) -> Vec<Option<f32>> {
        (0..n).map(|_| queue.next_sample()).collect()
    }

    /// Silence until the prebuffer is queued; then the samples; a dry queue
    /// before the end is an underrun, silence, and filling again.
    #[test]
    fn test_prebuffer_and_underrun() {
        let mut queue = PlaybackQueue::new(3);
        queue.push(&[0.1, 0.2]);
        assert_eq!(drain(&mut queue, 2), [None, None], "filling");
        queue.push(&[0.3]);
        assert_eq!(
            drain(&mut queue, 4),
            [Some(0.1), Some(0.2), Some(0.3), None]
        );
        assert_eq!(queue.underruns(), 1);
        queue.push(&[0.4, 0.5]);
        assert_eq!(drain(&mut queue, 1), [None], "filling again");
        assert!(!queue.drained());
    }

    /// The end plays what is left, prebuffer or not; then silence, drained,
    /// no underrun.
    #[test]
    fn test_close_plays_the_rest() {
        let mut queue = PlaybackQueue::new(10);
        queue.push(&[0.5, 0.25]);
        queue.close();
        assert_eq!(drain(&mut queue, 3), [Some(0.5), Some(0.25), None]);
        assert!(queue.drained());
        assert_eq!(queue.underruns(), 0);
    }

    /// Mono fans out to every channel, clipped; silence at the format's
    /// midpoint.
    #[test]
    fn test_fill_queue() {
        let mut queue = PlaybackQueue::new(0);
        queue.push(&[2.0]);
        queue.close();
        let mut out = [9.0f32; 4];
        fill_queue(&mut out, 2, &mut queue);
        assert_eq!(out, [1.0, 1.0, 0.0, 0.0]);
        let mut out = [0u16; 1];
        fill_queue(&mut out, 1, &mut queue);
        assert_eq!(out[0], u16::EQUILIBRIUM);
    }
}
