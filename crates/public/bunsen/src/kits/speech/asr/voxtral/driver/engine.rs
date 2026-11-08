//! Voxtral Realtime as the stream driver core's engine.
use alloc::sync::Arc;

use burn::{
    module::Module,
    tensor::Device,
};

use crate::{
    errors::BunsenResult,
    kits::speech::asr::{
        driver::{
            AsrEngine,
            AsrStreamContext,
            AsrStreamDriver,
            DecodeRequest,
            LogMelFeatures,
            WindowDecode,
            WordSpan,
            token_ids,
            window_frames_in,
        },
        voxtral::{
            VoxtralConstruct,
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

/// A [`VoxtralConstruct`] as an [`AsrEngine`]: what the stream driver core
/// needs of the model.
///
/// - **Features:** the construct's log-mels, streamed ([`LogMelFeatures`]), and
///   packaged per window under the front end's fixed level: no statistic of the
///   stream enters, so the engine's level is `()`, which costs nothing. A
///   window is at most the construct's policy's window of frames by default
///   ([`with_window_seconds`](Self::with_window_seconds)).
/// - **Decode:**
///   [`transcribe_features`](VoxtralConstruct::transcribe_features), one window
///   at a time: the window padded with silence as an utterance is, and decoded
///   greedily, one id per 80 ms.
/// - **Timing:** each word starts at its `[STREAMING_WORD]`'s position, less
///   the left padding and the delay, and ends where its last text id is
///   written, less the same: on the decoder's grid, 8 mel frames. Each word is
///   a segment. A window the core cut at its full length commits the words
///   before a pause in its last third
///   ([`pause_cut`](crate::kits::speech::asr::driver::pause_cut)), and the rest
///   is decoded again with the audio after it; any other window is committed
///   whole. A window with no words is skipped.
///
/// A stream's last frame is the end padding's, and goes, as the front end's
/// extractor drops it.
#[derive(Clone, Debug)]
pub struct VoxtralEngine {
    bundle: Arc<VoxtralConstruct>,
    window_frames: usize,
    device: Device,
}

impl VoxtralEngine {
    /// The engine of `bundle`, on the model's device, with windows of at most
    /// its policy's
    /// [`window_seconds`](crate::kits::speech::asr::voxtral::VoxtralPolicy::window_seconds).
    ///
    /// # Errors
    /// As [`with_window_seconds`](Self::with_window_seconds), for the
    /// policy's window.
    pub fn new(bundle: Arc<VoxtralConstruct>) -> BunsenResult<Self> {
        let device = bundle.model.devices()[0].clone();
        let seconds = bundle.policy.window_seconds;
        let engine = Self {
            bundle,
            window_frames: 0,
            device,
        };
        engine.with_window_seconds(seconds)
    }

    /// The engine with windows of at most `seconds`, rounded down to whole
    /// decoder positions.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`](crate::errors::ConstraintError) cause, if that is
    /// under one position.
    pub fn with_window_seconds(
        mut self,
        seconds: f64,
    ) -> BunsenResult<Self> {
        let front_end = &self.bundle.front_end;
        self.window_frames = window_frames_in(
            "VoxtralEngine::with_window_seconds",
            seconds,
            front_end.sample_rate(),
            front_end.hop(),
            self.bundle.model.mel_frames_per_position(),
            None,
        )?;
        Ok(self)
    }

    /// The construct.
    pub fn bundle(&self) -> &Arc<VoxtralConstruct> {
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

    /// Lays a transcript out for the core: a segment per word, a full window
    /// cut at a pause on the decoder's grid ([`WindowDecode::from_words`]).
    fn lay_out(
        &self,
        transcript: &VoxtralTranscript,
        frames: usize,
    ) -> WindowDecode<()> {
        let (words, tokens) = self.spans(transcript);
        WindowDecode::from_words(
            &words,
            tokens,
            frames,
            self.window_frames,
            self.grid_frames(),
            (),
        )
    }
}

impl AsrEngine for VoxtralEngine {
    type Carry = ();
    type Features = LogMelFeatures;
    type Level = ();
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
        let text = tokens
            .iter()
            .copied()
            .filter(|&t| !self.bundle.layout.is_control(t));
        let ids = token_ids(text, "Voxtral")?;
        Ok(Some(self.bundle.vocab.text(&ids)?))
    }
}
