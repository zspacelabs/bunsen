//! A family's stream driver, and one stream through it.
use std::sync::Arc;

#[cfg(feature = "live")]
use bunsen::kits::speech::{
    asr_driver::{
        StreamClock,
        TranscriptEvent,
    },
    parakeet::ParakeetStreamContext,
    qwen3_asr::Qwen3AsrStreamContext,
    whisper::driver::WhisperStreamContext,
};
use bunsen::{
    data::pretrained::PretrainedCache,
    errors::BunsenResult,
    kits::speech::{
        diarization::SpeakerActivityModel,
        parakeet::ParakeetStreamDriver,
        qwen3_asr::Qwen3AsrStreamDriver,
        whisper::driver::WhisperStreamDriver,
    },
};
use burn::tensor::Device;

use crate::{
    family::Loaded,
    parakeet,
    qwen3_asr,
    stream::{
        self,
        Preset,
    },
    transcribe_cmd::{
        FamilyFlags,
        Transcript,
    },
    whisper,
};

/// A family's stream driver; Whisper's, much the largest, boxed.
pub enum Driver {
    Parakeet(ParakeetStreamDriver),
    Qwen3Asr(Qwen3AsrStreamDriver),
    Whisper(Box<WhisperStreamDriver>),
}

impl Driver {
    /// The driver for `loaded` under its family's `flags` and `preset`, with
    /// Silero's VAD attached when the preset finds speech regions, and
    /// `speakers` when there is a diarizer.
    ///
    /// # Panics
    /// If `flags` are not `loaded`'s family's.
    pub fn build<F: FamilyFlags>(
        loaded: Loaded,
        flags: &F,
        preset: Preset,
        speakers: Option<Arc<dyn SpeakerActivityModel>>,
        cache: &PretrainedCache,
        device: &Device,
    ) -> BunsenResult<Self> {
        Ok(match loaded {
            Loaded::Parakeet(bundle) => Driver::Parakeet(stream::with_speakers(
                parakeet::driver(
                    bundle,
                    flags.parakeet().expect("Parakeet's flags"),
                    preset,
                    cache,
                    device,
                )?,
                speakers,
            )?),
            Loaded::Qwen3Asr(bundle) => Driver::Qwen3Asr(stream::with_speakers(
                qwen3_asr::driver(
                    bundle,
                    flags.qwen3_asr().expect("Qwen3-ASR's flags"),
                    preset,
                    cache,
                    device,
                )?,
                speakers,
            )?),
            Loaded::Whisper(bundle) => Driver::Whisper(Box::new(stream::with_speakers(
                whisper::driver(
                    bundle,
                    flags.whisper().expect("Whisper's flags"),
                    preset,
                    cache,
                    device,
                )?,
                speakers,
            )?)),
        })
    }

    /// The model's sample rate, in Hz.
    pub fn sample_rate(&self) -> usize {
        match self {
            Driver::Parakeet(d) => d.sample_rate(),
            Driver::Qwen3Asr(d) => d.sample_rate(),
            Driver::Whisper(d) => d.sample_rate(),
        }
    }

    /// Samples per push of `ms` milliseconds ([`stream::push_samples`]).
    pub fn push_samples(
        &self,
        ms: usize,
    ) -> usize {
        match self {
            Driver::Parakeet(d) => stream::push_samples(d, ms),
            Driver::Qwen3Asr(d) => stream::push_samples(d, ms),
            Driver::Whisper(d) => stream::push_samples(d, ms),
        }
    }

    /// Transcribes one recording, `chunk` samples a push.
    pub fn transcribe(
        &self,
        samples: &[f32],
        chunk: usize,
    ) -> BunsenResult<Transcript> {
        match self {
            Driver::Parakeet(d) => parakeet::transcribe(d, samples, chunk),
            Driver::Qwen3Asr(d) => qwen3_asr::transcribe(d, samples, chunk),
            Driver::Whisper(d) => whisper::transcribe(d, samples, chunk),
        }
    }

    /// Opens a stream, its clock at the model's rate from 0, its windows
    /// scaled as its family's are.
    #[cfg(feature = "live")]
    pub fn open(&self) -> BunsenResult<Stream> {
        let clock = || StreamClock::uniform(self.sample_rate());
        Ok(match self {
            Driver::Parakeet(d) => {
                Stream::Parakeet(Box::new(d.new_context(clock(), parakeet::LEVEL)?))
            }
            Driver::Qwen3Asr(d) => {
                Stream::Qwen3Asr(Box::new(d.new_context(clock(), qwen3_asr::LEVEL)?))
            }
            Driver::Whisper(d) => {
                Stream::Whisper(Box::new(d.new_context(clock(), whisper::LEVEL)?))
            }
        })
    }
}

/// One stream through a family's driver, boxed: a context holds its
/// feature ring and its engine's state.
#[cfg(feature = "live")]
pub enum Stream {
    Parakeet(Box<ParakeetStreamContext>),
    Qwen3Asr(Box<Qwen3AsrStreamContext>),
    Whisper(Box<WhisperStreamContext>),
}

/// `$body` on the stream's context, whichever family's.
#[cfg(feature = "live")]
macro_rules! each {
    ($stream:expr, $ctx:ident => $body:expr) => {
        match $stream {
            Stream::Parakeet($ctx) => $body,
            Stream::Qwen3Asr($ctx) => $body,
            Stream::Whisper($ctx) => $body,
        }
    };
}

#[cfg(feature = "live")]
impl Stream {
    /// Pushes samples at the model's rate; what the driver emitted.
    pub fn write_read(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<Vec<TranscriptEvent>> {
        each!(self, ctx => ctx.write_read(samples))
    }

    /// Anchors the clock, the first of `samples` captured at media time
    /// `time` in seconds, then pushes them.
    pub fn anchor_write_read(
        &mut self,
        time: f64,
        samples: &[f32],
    ) -> BunsenResult<Vec<TranscriptEvent>> {
        each!(self, ctx => ctx.anchor_write_read(time, samples))
    }

    /// Ends the stream; what was left past the seek pointer.
    pub fn end_read(&mut self) -> BunsenResult<Vec<TranscriptEvent>> {
        each!(self, ctx => ctx.end_read())
    }

    /// Samples pushed so far.
    pub fn samples_seen(&self) -> usize {
        each!(self, ctx => ctx.samples_seen())
    }

    /// The language, once the model has named one; Parakeet names none.
    pub fn language(&self) -> Option<&str> {
        match self {
            Stream::Parakeet(_) => None,
            Stream::Qwen3Asr(ctx) => ctx.language(),
            Stream::Whisper(ctx) => ctx.language(),
        }
    }
}
