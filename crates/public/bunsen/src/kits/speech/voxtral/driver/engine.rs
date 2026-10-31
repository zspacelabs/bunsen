//! Voxtral Realtime as the stream driver core's engine.
use std::sync::Arc;

use burn::{
    module::Module,
    tensor::Device,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::{
        asr_driver::{
            AsrEngine,
            AsrStreamContext,
            AsrStreamDriver,
            DecodeRequest,
            LogMelFeatures,
            MaxLevel,
            TimedTokens,
            WindowDecode,
            WordSpan,
            pause_cut,
            word_segments,
        },
        voxtral::{
            VoxtralBundle,
            VoxtralFrontEndMeta,
            VoxtralTranscript,
            blocks::VoxtralRealtimeMeta,
        },
    },
};

/// The stream driver of a Voxtral Realtime model: the core's
/// [`AsrStreamDriver`] over a [`VoxtralEngine`].
pub type VoxtralStreamDriver = AsrStreamDriver<VoxtralEngine>;

/// One Voxtral Realtime stream: the core's [`AsrStreamContext`] over a
/// [`VoxtralEngine`].
pub type VoxtralStreamContext = AsrStreamContext<VoxtralEngine>;

/// A [`VoxtralBundle`] as an [`AsrEngine`]: what the stream driver core needs
/// of the model.
///
/// - **Features:** the bundle's log-mels, streamed ([`LogMelFeatures`]), and
///   packaged per window under the front end's fixed level: no statistic of the
///   stream enters, so the window's [`MaxLevel`] is not read. A window is at
///   most [`DEFAULT_WINDOW_SECONDS`](Self::DEFAULT_WINDOW_SECONDS) of frames by
///   default ([`with_window_seconds`](Self::with_window_seconds)).
/// - **Decode:** [`transcribe_features`](VoxtralBundle::transcribe_features),
///   one window at a time: the window padded with silence as an utterance is,
///   and decoded greedily, one id per 80 ms.
/// - **Timing:** each word starts at its `[STREAMING_WORD]`'s position, less
///   the left padding and the delay, and ends where its last text id is
///   written, less the same: on the decoder's grid, 8 mel frames. Each word is
///   a segment. A window the core cut at its full length commits the words
///   before a pause in its last third ([`pause_cut`]), and the rest is decoded
///   again with the audio after it; any other window is committed whole. A
///   window with no words is skipped.
///
/// A stream's last frame is the end padding's, and goes, as the front end's
/// extractor drops it.
#[derive(Clone, Debug)]
pub struct VoxtralEngine {
    bundle: Arc<VoxtralBundle>,
    window_frames: usize,
    device: Device,
}

impl VoxtralEngine {
    /// The longest window by default, in seconds: twice the audio encoder's
    /// attention window, so its sliding window is in play.
    pub const DEFAULT_WINDOW_SECONDS: f64 = 30.0;

    /// The engine of `bundle`, on the model's device, with windows of at most
    /// [`DEFAULT_WINDOW_SECONDS`](Self::DEFAULT_WINDOW_SECONDS).
    pub fn new(bundle: Arc<VoxtralBundle>) -> Self {
        let device = bundle.model.devices()[0].clone();
        let engine = Self {
            bundle,
            window_frames: 0,
            device,
        };
        engine.with_window_seconds(Self::DEFAULT_WINDOW_SECONDS)
    }

    /// The engine with windows of at most `seconds`, rounded down to whole
    /// decoder positions.
    ///
    /// # Panics
    /// If that is under one position.
    pub fn with_window_seconds(
        mut self,
        seconds: f64,
    ) -> Self {
        let front_end = &self.bundle.front_end;
        let per = self.bundle.model.mel_frames_per_position();
        let frames = (seconds * front_end.sample_rate() as f64 / front_end.hop() as f64) as usize;
        let frames = frames / per * per;
        assert!(frames > 0, "a window of {seconds} s is under one position");
        self.window_frames = frames;
        self
    }

    /// The bundle.
    pub fn bundle(&self) -> &Arc<VoxtralBundle> {
        &self.bundle
    }

    /// A transcript's words, in mel frames of the window, and their text ids,
    /// joined.
    fn spans(
        &self,
        transcript: &VoxtralTranscript,
    ) -> (Vec<WordSpan>, Vec<i64>) {
        let per = self.bundle.model.mel_frames_per_position();
        let config = self.bundle.layout.config();
        let lead = config.left_pad_positions + self.bundle.layout.delay_positions();
        let frame = |position: usize| position.saturating_sub(lead) * per;
        let mut tokens = Vec::new();
        let mut words = Vec::new();
        for word in &transcript.words {
            let first = tokens.len();
            tokens.extend_from_slice(&word.ids);
            // The word's text runs to the position its last id is written at.
            let last = word.position + word.ids.len();
            words.push(WordSpan {
                first_token: first,
                start: frame(word.position),
                end: frame(last + 1),
            });
        }
        (words, tokens)
    }

    /// Lays a transcript out for the core: a segment per word.
    fn lay_out(
        &self,
        transcript: &VoxtralTranscript,
        frames: usize,
    ) -> WindowDecode<()> {
        let (words, tokens) = self.spans(transcript);
        let grid = self.grid_frames();
        let cut = (frames >= self.window_frames)
            .then(|| pause_cut(&words, frames, grid))
            .flatten();
        let (advance, count) = cut.unwrap_or((frames, words.len()));
        let committed = words.get(count).map_or(tokens.len(), |w| w.first_token);
        WindowDecode {
            segments: word_segments(&words[..count], &tokens[..committed], advance),
            tail: cut.map(|(cut, _)| TimedTokens {
                start: cut,
                end: frames,
                tokens: tokens[committed..].to_vec(),
            }),
            skip: tokens.is_empty(),
            tokens,
            advance,
            carry: (),
        }
    }
}

impl AsrEngine for VoxtralEngine {
    type Carry = ();
    type Features = LogMelFeatures;
    type Level = MaxLevel;
    type StreamState = ();

    fn sample_rate(&self) -> usize {
        self.bundle.front_end.sample_rate()
    }

    fn hop(&self) -> usize {
        self.bundle.front_end.hop()
    }

    fn window_frames(&self) -> usize {
        self.window_frames
    }

    /// The decoder position: when a word can be written.
    fn grid_frames(&self) -> usize {
        self.bundle.model.mel_frames_per_position()
    }

    /// The extractor's `stft[..., :-1]`: the stream's last frame is the end
    /// padding's.
    fn end_frames_dropped(&self) -> usize {
        1
    }

    fn new_features(&self) -> BunsenResult<LogMelFeatures> {
        Ok(LogMelFeatures::new(
            self.bundle.front_end.converter.clone(),
            self.device.clone(),
        ))
    }

    fn new_stream_state(&self) {}

    fn decode(
        &self,
        requests: Vec<DecodeRequest<'_, Self>>,
    ) -> BunsenResult<Vec<WindowDecode<()>>> {
        requests
            .into_iter()
            .map(|request| {
                let frames = request.window.dims()[1];
                let features = self.bundle.front_end.package(request.window);
                let transcript = self.bundle.transcribe_features(features)?;
                Ok(self.lay_out(&transcript, frames))
            })
            .collect()
    }

    fn commit(
        &self,
        _state: &mut (),
        _transcript: &[i64],
        _carry: (),
    ) {
    }

    fn text(
        &self,
        tokens: &[i64],
    ) -> BunsenResult<Option<String>> {
        let ids = tokens
            .iter()
            .filter(|&&t| !self.bundle.layout.is_control(t))
            .map(|&t| {
                usize::try_from(t).map_err(|_| {
                    BunsenError::invalid_resource(format!("{t} is not a Voxtral token id"))
                })
            })
            .collect::<BunsenResult<Vec<usize>>>()?;
        Ok(Some(self.bundle.vocab.text(&ids)?))
    }
}
