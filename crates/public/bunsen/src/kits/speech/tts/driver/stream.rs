//! Speech as a stream: takes' samples as they become final, as events.

use alloc::{
    boxed::Box,
    collections::VecDeque,
    string::String,
    vec,
    vec::Vec,
};

use burn::config::Config;

use super::{
    TtsDriver,
    TtsEngine,
    TtsRequest,
    TtsSpeech,
    TtsStageNote,
    TtsTakeInfo,
    TtsTakeRequest,
};
use crate::errors::BunsenResult;

/// How a take's stream groups its final frames into codec decodes: the
/// first decode as soon as [`first`](Self::first) frames are final, then
/// each decode waiting for as many more frames as were decoded before it,
/// up to [`max`](Self::max): 1, 2, 4, 8, 15, 15, … by default (upstream
/// ZONOS3's prefix stream).
///
/// The codecs' streams are exact, so the grouping changes only when the
/// samples come out: fewer, larger decodes cost fewer launches, a small
/// first one starts the audio sooner.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub struct TtsChunkPolicy {
    /// The frames of the first decode; at least one.
    #[config(default = 1)]
    pub first: usize,

    /// The most frames of one decode; at least [`first`](Self::first).
    #[config(default = 15)]
    pub max: usize,
}

impl TtsChunkPolicy {
    /// The final frames the next decode waits for, after `decoded` frames:
    /// `min(max, first + decoded)`.
    pub fn need(
        &self,
        decoded: usize,
    ) -> usize {
        self.max.min(self.first + decoded).max(1)
    }

    /// Refuses a policy of no first frames, or a `max` under `first`.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), naming the
    /// field.
    pub fn check(&self) -> BunsenResult<()> {
        if self.first == 0 {
            return Err(
                crate::errors::ConstraintError::zero_or_empty("TtsChunkPolicy", "first").into(),
            );
        }
        if self.max < self.first {
            return Err(crate::errors::ConstraintError::new(
                "TtsChunkPolicy",
                "max",
                crate::errors::Rule::Relation {
                    lhs: ("max".into(), self.max.to_string()),
                    op: ">=",
                    rhs: ("first".into(), self.first.to_string()),
                },
            )
            .into());
        }
        Ok(())
    }
}

/// One step of a take's stream ([`TtsTakeStream::step`]).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TtsStreamStep {
    /// The samples the step made final, mono, in `[-1, 1]`; often none.
    pub samples: Vec<f32>,

    /// At the take's last step, what the take was; the stream has then
    /// returned all of its samples.
    pub end: Option<TtsTakeInfo>,
}

/// One take, generated and rendered as a stream
/// ([`TtsEngine::start_stream`]): each [`step`](Self::step) runs the
/// engine's next generation step and returns the samples it made final.
/// Joined, a take's samples are its whole render's
/// ([`TtsEngine::render`] of [`TtsEngine::generate`]).
pub trait TtsTakeStream: Send {
    /// The next generation step.
    ///
    /// # Errors
    /// The engine's, generating or decoding.
    ///
    /// # Panics
    /// May panic after the step that ended the take.
    fn step(&mut self) -> BunsenResult<TtsStreamStep>;
}

/// [`TtsEngine::start_stream`]'s default: the take generated whole when the
/// stream starts, and rendered whole at its one step.
pub(crate) struct TtsWholeTakeStream<'a, E: TtsEngine> {
    pub(crate) engine: &'a E,
    pub(crate) take: Option<E::Take>,
}

impl<E: TtsEngine> TtsTakeStream for TtsWholeTakeStream<'_, E> {
    fn step(&mut self) -> BunsenResult<TtsStreamStep> {
        let take = self.take.take().expect("a whole take's one step");
        Ok(TtsStreamStep {
            samples: self.engine.render(&take)?,
            end: Some(self.engine.take_info(&take)),
        })
    }
}

/// What a request's stream says, in order
/// ([`TtsSpeechStream::next_event`]).
#[derive(Clone, Debug, PartialEq)]
pub enum TtsEvent {
    /// What the request's stages did worth saying; first, when there is any.
    Notes(Vec<TtsStageNote>),

    /// A take starts: its index, and its text.
    TakeStart {
        /// The take's index.
        index: usize,

        /// Its text, normalized.
        text: String,
    },

    /// Samples, final: a take's, or the silence between two takes.
    Audio(Vec<f32>),

    /// A take ended: its index, and what it was.
    TakeEnd {
        /// The take's index.
        index: usize,

        /// What it was.
        info: TtsTakeInfo,
    },
}

/// A request's speech as a stream: its events, pulled one at a time. The
/// caller paces it; nothing runs between pulls. Joined, its
/// [`Audio`](TtsEvent::Audio) is
/// [`TtsSynthesizer::synthesize`](super::TtsSynthesizer::synthesize)'s samples.
pub trait TtsSpeechStream: Send {
    /// The samples' rate, in Hz.
    fn sample_rate(&self) -> u32;

    /// The next event, running the engine as far as it needs; `None` once
    /// the last take has ended.
    ///
    /// # Errors
    /// The engine's, generating or decoding.
    fn next_event(&mut self) -> BunsenResult<Option<TtsEvent>>;
}

impl TtsSpeech {
    /// The speech `stream` says, gathered: its audio joined, its takes and
    /// notes in order.
    ///
    /// # Errors
    /// The stream's.
    pub fn from_stream(stream: &mut dyn TtsSpeechStream) -> BunsenResult<Self> {
        let mut speech = TtsSpeech {
            samples: Vec::new(),
            sample_rate: stream.sample_rate(),
            takes: Vec::new(),
            notes: Vec::new(),
        };
        while let Some(event) = stream.next_event()? {
            match event {
                TtsEvent::Notes(notes) => speech.notes.extend(notes),
                TtsEvent::Audio(samples) => speech.samples.extend(samples),
                TtsEvent::TakeEnd { info, .. } => speech.takes.push(info),
                TtsEvent::TakeStart { .. } => {}
            }
        }
        Ok(speech)
    }
}

/// [`TtsSynthesizer::stream`](super::TtsSynthesizer::stream)'s default: the request synthesized whole at
/// the first pull, then said as events.
pub(crate) struct TtsWholeSpeechStream {
    pub(crate) sample_rate: u32,
    pub(crate) events: VecDeque<TtsEvent>,
}

impl TtsWholeSpeechStream {
    /// `speech`'s events: its notes, its audio, its takes' ends.
    pub(crate) fn new(speech: TtsSpeech) -> Self {
        let mut events = VecDeque::new();
        if !speech.notes.is_empty() {
            events.push_back(TtsEvent::Notes(speech.notes));
        }
        events.push_back(TtsEvent::Audio(speech.samples));
        for (index, info) in speech.takes.into_iter().enumerate() {
            events.push_back(TtsEvent::TakeEnd { index, info });
        }
        Self {
            sample_rate: speech.sample_rate,
            events,
        }
    }
}

impl TtsSpeechStream for TtsWholeSpeechStream {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn next_event(&mut self) -> BunsenResult<Option<TtsEvent>> {
        Ok(self.events.pop_front())
    }
}

/// A [`TtsDriver`]'s request as a stream ([`TtsDriver::stream`]): its
/// takes in turn, each through the engine's
/// [`start_stream`](TtsEngine::start_stream), the driver's gap between
/// them.
pub struct TtsDriverStream<'a, E: TtsEngine> {
    driver: &'a TtsDriver<E>,
    voice: E::Voice,
    request: TtsRequest,
    texts: Vec<String>,
    chunks: TtsChunkPolicy,
    next_take: usize,
    current: Option<(usize, Box<dyn TtsTakeStream + 'a>)>,
    events: VecDeque<TtsEvent>,
}

impl<E: TtsEngine> core::fmt::Debug for TtsDriverStream<'_, E> {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("TtsDriverStream")
            .field("texts", &self.texts)
            .field("next_take", &self.next_take)
            .field("in_take", &self.current.as_ref().map(|(i, _)| *i))
            .finish_non_exhaustive()
    }
}

impl<E: TtsEngine> TtsDriver<E> {
    /// `request`, spoken as a stream: as
    /// [`synthesize`](Self::synthesize) prepares it (its voice, its stages,
    /// its takes), each take then streamed by the engine, its final frames
    /// decoded as `chunks` groups them. Joined, the stream's audio is
    /// `synthesize`'s.
    ///
    /// # Errors
    /// As `synthesize` prepares a request; and
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a chunk
    /// policy [`check`](TtsChunkPolicy::check) refuses.
    pub fn stream(
        &self,
        request: &TtsRequest,
        chunks: TtsChunkPolicy,
    ) -> BunsenResult<TtsDriverStream<'_, E>> {
        chunks.check()?;
        let prepared = self.prepare(request)?;
        let mut events = VecDeque::new();
        if !prepared.notes.is_empty() {
            events.push_back(TtsEvent::Notes(prepared.notes));
        }
        Ok(TtsDriverStream {
            driver: self,
            voice: prepared.voice,
            request: prepared.request,
            texts: prepared.texts,
            chunks,
            next_take: 0,
            current: None,
            events,
        })
    }
}

impl<E: TtsEngine> TtsSpeechStream for TtsDriverStream<'_, E> {
    fn sample_rate(&self) -> u32 {
        self.driver.engine().sample_rate()
    }

    fn next_event(&mut self) -> BunsenResult<Option<TtsEvent>> {
        loop {
            if let Some(event) = self.events.pop_front() {
                return Ok(Some(event));
            }
            match &mut self.current {
                Some((index, take)) => {
                    let step = take.step()?;
                    if !step.samples.is_empty() {
                        self.events.push_back(TtsEvent::Audio(step.samples));
                    }
                    if let Some(info) = step.end {
                        self.events.push_back(TtsEvent::TakeEnd {
                            index: *index,
                            info,
                        });
                        self.current = None;
                    }
                }
                None if self.next_take == self.texts.len() => return Ok(None),
                None => {
                    let i = self.next_take;
                    self.next_take += 1;
                    if i > 0 {
                        let gap = self.driver.gap_samples();
                        if gap > 0 {
                            self.events.push_back(TtsEvent::Audio(vec![0.0; gap]));
                        }
                    }
                    self.events.push_back(TtsEvent::TakeStart {
                        index: i,
                        text: self.texts[i].clone(),
                    });
                    let request = &self.request;
                    let take = self.driver.engine().start_stream(
                        &TtsTakeRequest {
                            text: &self.texts[i],
                            voice: &self.voice,
                            language: request.language.as_deref(),
                            instruction: request.instruction.as_deref(),
                            seed: request.seed.map(|seed| seed.wrapping_add(i as u64)),
                            max_seconds: request.max_seconds,
                            sampling: &request.sampling,
                        },
                        self.chunks,
                    )?;
                    self.current = Some((i, take));
                }
            }
        }
    }
}

/// [`TtsSynthesizer::stream`](super::TtsSynthesizer::stream) for a driver: its own stream.
pub(crate) fn driver_speech_stream<'a, E: TtsEngine>(
    driver: &'a TtsDriver<E>,
    request: &TtsRequest,
    chunks: TtsChunkPolicy,
) -> BunsenResult<Box<dyn TtsSpeechStream + 'a>> {
    Ok(Box::new(driver.stream(request, chunks)?))
}
