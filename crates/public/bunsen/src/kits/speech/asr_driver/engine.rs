//! # The engine: what a model gives the stream driver core.

use std::fmt::Debug;

use burn::Tensor;

use crate::{
    errors::BunsenResult,
    kits::speech::asr_driver::LevelStatistic,
};

/// A speech-to-text model, as an
/// [`AsrStreamDriver`](crate::kits::speech::asr_driver::AsrStreamDriver)
/// drives it.
///
/// The core owns everything about a stream that does not depend on the
/// model: buffering, the voice-activity gate and its regions, silence
/// skipping, when to decode, what to commit and what to draft, the clock,
/// and batching across streams. The engine owns the rest: the features and
/// how they stream, how a window is packaged against its level, the decode,
/// how a decode splits into timed segments, the text, and any per-stream
/// carry (Whisper's prompt and language).
///
/// Time inside the core is in feature frames, `hop` samples each; a window
/// is at most `window_frames` of them.
pub trait AsrEngine: Clone + Debug + Send + Sync {
    /// The streaming feature front end each stream opens.
    type Features: StreamingFeatures;

    /// What a window is packaged against
    /// ([`LevelStatistic`](crate::kits::speech::asr_driver::LevelStatistic)).
    type Level: LevelStatistic;

    /// What the engine carries from one window of a stream to the next
    /// (Whisper: where the prompt carry starts, the language).
    type StreamState: Clone + Debug + Send + Sync;

    /// What a decode hands back to [`commit`](Self::commit) when the core
    /// commits it.
    type Carry: Clone + Debug + Send + Sync;

    /// The sample rate, in Hz.
    fn sample_rate(&self) -> usize;

    /// Samples per feature frame.
    fn hop(&self) -> usize;

    /// The most frames one decode takes.
    fn window_frames(&self) -> usize;

    /// The frames speech regions are snapped onto: the decode's timing unit.
    fn grid_frames(&self) -> usize;

    /// Frames dropped from the end of the stream once its input ends
    /// (Whisper: the end padding's frame).
    fn end_frames_dropped(&self) -> usize {
        0
    }

    /// Opens a stream's feature front end.
    ///
    /// # Errors
    /// As the front end's construction.
    fn new_features(&self) -> BunsenResult<Self::Features>;

    /// A new stream's carry.
    fn new_stream_state(&self) -> Self::StreamState;

    /// Decodes windows, one per request, in order.
    ///
    /// The requests may come from different streams; the engine batches them
    /// as it can. It must not let one request's result depend on another's.
    ///
    /// # Errors
    /// As the decode.
    fn decode(
        &self,
        requests: Vec<DecodeRequest<'_, Self>>,
    ) -> BunsenResult<Vec<WindowDecode<Self::Carry>>>;

    /// Records a committed decode in the stream's carry: the only place the
    /// engine changes a stream.
    ///
    /// # Arguments
    /// * `state` - the stream's carry.
    /// * `transcript` - every id the stream has committed, this decode's
    ///   included.
    /// * `carry` - what [`decode`](Self::decode) handed back for it.
    fn commit(
        &self,
        state: &mut Self::StreamState,
        transcript: &[i64],
        carry: Self::Carry,
    );

    /// The text of some decoded ids, when the engine can say.
    ///
    /// # Errors
    /// As the detokenizer.
    fn text(
        &self,
        tokens: &[i64],
    ) -> BunsenResult<Option<String>>;
}

/// A front end that turns audio, arriving in pieces of any size, into
/// feature frames.
pub trait StreamingFeatures: Clone + Debug + Send + Sync {
    /// Takes samples in, and returns the frames they complete.
    ///
    /// # Returns
    /// `[1, frames, features]`, or `None` when no frame completes.
    ///
    /// # Errors
    /// As the front end.
    fn push(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<Option<Tensor<3>>>;

    /// Ends the input, and returns the frames left.
    ///
    /// # Returns
    /// `[1, frames, features]`, or `None` when none are.
    ///
    /// # Errors
    /// As the front end.
    fn finish(&mut self) -> BunsenResult<Option<Tensor<3>>>;
}

/// One window to decode.
#[derive(Debug)]
pub struct DecodeRequest<'a, E: AsrEngine> {
    /// The stream's carry, read-only: a decode never changes a stream.
    pub state: &'a E::StreamState,

    /// Every id the stream has committed.
    pub transcript: &'a [i64],

    /// `[1, frames, features]`, at most
    /// [`window_frames`](AsrEngine::window_frames).
    pub window: Tensor<3>,

    /// What the window is packaged against, under the stream's
    /// [`LevelPolicy`](crate::kits::speech::asr_driver::LevelPolicy).
    pub level: E::Level,

    /// A provisional decode, emitted as a draft and never committed.
    pub draft: bool,
}

/// Ids over a run of frames, relative to the start of their window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimedTokens {
    /// The first frame.
    pub start: usize,

    /// One past the last frame.
    pub end: usize,

    /// The ids.
    pub tokens: Vec<i64>,
}

/// A window's decode, laid out for the core.
#[derive(Clone, Debug)]
pub struct WindowDecode<C> {
    /// Everything decoded, in order: what a draft of the window shows.
    pub tokens: Vec<i64>,

    /// The window holds no speech: nothing is committed, and the stream
    /// moves past it whole.
    pub skip: bool,

    /// The complete segments, in order.
    pub segments: Vec<TimedTokens>,

    /// The frames the stream moves past when this decode is committed; the
    /// core clamps it to at least one, and at most the window.
    pub advance: usize,

    /// The unfinished segment past `advance`, if any: decoded again with
    /// more audio behind it, and drafted first under
    /// [`CommitRule::LastTimestamp`](crate::kits::speech::asr_driver::CommitRule::LastTimestamp).
    /// Its `end` is the window's end.
    pub tail: Option<TimedTokens>,

    /// Handed to [`AsrEngine::commit`] when this decode is committed.
    pub carry: C,
}

impl<C> WindowDecode<C> {
    /// A decode with no timing of its own: one segment over the whole window,
    /// which the stream moves past.
    ///
    /// # Arguments
    /// * `frames` - the window's frames.
    /// * `tokens` - the ids.
    /// * `carry` - for [`AsrEngine::commit`].
    pub fn whole(
        frames: usize,
        tokens: Vec<i64>,
        carry: C,
    ) -> Self {
        Self {
            segments: vec![TimedTokens {
                start: 0,
                end: frames,
                tokens: tokens.clone(),
            }],
            tokens,
            skip: false,
            advance: frames,
            tail: None,
            carry,
        }
    }
}
