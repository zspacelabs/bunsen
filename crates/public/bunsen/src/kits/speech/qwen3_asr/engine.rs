//! Qwen3-ASR as the stream driver core's engine.
use std::sync::Arc;

use burn::{
    Tensor,
    module::Module,
    tensor::Device,
};

use super::{
    Qwen3AsrBundle,
    Qwen3AsrFrontEndMeta,
    Qwen3AsrTranscript,
};
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::asr_driver::{
        AsrEngine,
        AsrStreamContext,
        AsrStreamDriver,
        DecodeRequest,
        LogMelFeatures,
        MaxLevel,
        WindowDecode,
        quiet_cut,
    },
};

/// The mel frames a full window's quiet cut must be quiet over: 90 ms.
const QUIET_SPAN_FRAMES: usize = 9;

/// The stream driver of a Qwen3-ASR model: the core's [`AsrStreamDriver`]
/// over a [`Qwen3AsrEngine`].
pub type Qwen3AsrStreamDriver = AsrStreamDriver<Qwen3AsrEngine>;

/// One Qwen3-ASR stream: the core's [`AsrStreamContext`] over a
/// [`Qwen3AsrEngine`].
pub type Qwen3AsrStreamContext = AsrStreamContext<Qwen3AsrEngine>;

/// A [`Qwen3AsrBundle`] as an [`AsrEngine`]: what the stream driver core
/// needs of the model.
///
/// - **Features:** the bundle's log-mels, streamed ([`LogMelFeatures`]), and
///   packaged per window as its front end packages an utterance: floored 8 dB
///   under the window's [`MaxLevel`], scaled, and padded with silence to the
///   front end's 30 s. A window is at most
///   [`MAX_WINDOW_SECONDS`](Qwen3AsrBundle::MAX_WINDOW_SECONDS) of frames by
///   default, and never more
///   ([`with_window_seconds`](Self::with_window_seconds)).
/// - **Decode:** [`transcribe_features`](Qwen3AsrBundle::transcribe_features),
///   one window at a time: the greedy decoder takes one prompt. A reply's
///   transcript is one segment over the window decoded; Qwen3-ASR has no
///   timestamps, so a window is committed whole, and the grid speech regions
///   snap to is one frame. A reply with no transcript (`language None`, for
///   audio without speech) skips its window.
/// - **Cuts:** a window the core cut at its full length is cut again before it
///   is decoded, at its quietest 90 ms in its last third by mean log-mel
///   ([`quiet_cut`]), and the stream moves on from there: a window that ends
///   inside a word loses or repeats it, and the model was seen to repeat
///   phrases decoding audio past a pause.
///   [`with_quiet_cut`](Self::with_quiet_cut) turns this off.
/// - **Carry:** the language, the first a committed reply names.
///
/// A stream's last frame is the end padding's, and goes, as the front end's
/// extractor drops it. The frames before it are the one-shot extractor's,
/// except the last one or two of the stream: streaming reflects at the end,
/// where the one-shot extractor pads the waveform with zeros.
#[derive(Clone, Debug)]
pub struct Qwen3AsrEngine {
    bundle: Arc<Qwen3AsrBundle>,
    window_frames: usize,
    quiet_cut: bool,
    device: Device,
}

impl Qwen3AsrEngine {
    /// The engine of `bundle`, on the model's device, with windows of at most
    /// [`MAX_WINDOW_SECONDS`](Qwen3AsrBundle::MAX_WINDOW_SECONDS), each full
    /// one cut at a quiet point.
    pub fn new(bundle: Arc<Qwen3AsrBundle>) -> Self {
        let device = bundle.model.devices()[0].clone();
        let window_frames = bundle.max_window() / bundle.front_end.hop();
        Self {
            bundle,
            window_frames,
            quiet_cut: true,
            device,
        }
    }

    /// The engine with a full window cut at its quietest point before it is
    /// decoded (`true`, the default), or decoded and committed whole.
    pub fn with_quiet_cut(
        mut self,
        on: bool,
    ) -> Self {
        self.quiet_cut = on;
        self
    }

    /// The engine with windows of at most `seconds`, rounded down to whole
    /// frames.
    ///
    /// # Panics
    /// If that is under one frame, or over
    /// [`MAX_WINDOW_SECONDS`](Qwen3AsrBundle::MAX_WINDOW_SECONDS), the most
    /// the model was trained on.
    pub fn with_window_seconds(
        mut self,
        seconds: f64,
    ) -> Self {
        let front_end = &self.bundle.front_end;
        let frames = (seconds * front_end.sample_rate() as f64 / front_end.hop() as f64) as usize;
        let most = self.bundle.max_window() / front_end.hop();
        assert!(
            frames > 0 && frames <= most,
            "a window of {seconds} s is under one frame or over {} s",
            Qwen3AsrBundle::MAX_WINDOW_SECONDS
        );
        self.window_frames = frames;
        self
    }

    /// The bundle.
    pub fn bundle(&self) -> &Arc<Qwen3AsrBundle> {
        &self.bundle
    }

    /// The frames a window is padded out to: the front end's shortest
    /// window.
    fn padded_frames(&self) -> usize {
        let front_end = &self.bundle.front_end;
        front_end.min_samples() / front_end.hop()
    }

    /// Floors a window at the range below its level, scales it, and pads it
    /// with the floor, which is what silence floors to.
    ///
    /// # Arguments
    /// * `window` - `[1, frames, n_mels]` log-mels.
    ///
    /// # Returns
    /// `[n_mels, max(frames, padded_frames)]`.
    fn package(
        &self,
        window: Tensor<3>,
        level: &MaxLevel,
    ) -> Tensor<2> {
        let whisper = &self.bundle.front_end.config.whisper;
        let [_, frames, n_mels] = window.dims();
        let width = frames.max(self.padded_frames());
        let window = if frames < width {
            let floor = level
                .max
                .clone()
                .sub_scalar(whisper.range_clamp_db)
                .reshape([1, 1, 1])
                .expand([1, width - frames, n_mels]);
            Tensor::cat(vec![window, floor], 1)
        } else {
            window
        };
        whisper
            .package_window(window, level.max.clone())
            .reshape([n_mels, width])
    }

    /// Lays a reply out for the core: its transcript, one segment over the
    /// window, or a skip when it has none.
    fn lay_out(
        &self,
        reply: Qwen3AsrTranscript,
        frames: usize,
    ) -> WindowDecode<Qwen3AsrCarry> {
        let (_, text) = self
            .bundle
            .prompt
            .split_reply(&reply.tokens, &self.bundle.decoder.eos);
        let tokens: Vec<i64> = text.iter().map(|&t| t as i64).collect();
        let carry = Qwen3AsrCarry {
            language: reply.language,
        };
        if tokens.is_empty() {
            return WindowDecode {
                tokens,
                skip: true,
                segments: Vec::new(),
                advance: frames,
                tail: None,
                carry,
            };
        }
        WindowDecode::whole(frames, tokens, carry)
    }
}

/// What a Qwen3-ASR stream carries from one window to the next.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Qwen3AsrStreamState {
    /// The language of the stream: the first a committed reply names.
    pub language: Option<String>,
}

/// What a Qwen3-ASR decode hands back to its commit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen3AsrCarry {
    /// The language the reply names, if any.
    pub language: Option<String>,
}

impl AsrEngine for Qwen3AsrEngine {
    type Carry = Qwen3AsrCarry;
    type Features = LogMelFeatures;
    type Level = MaxLevel;
    type StreamState = Qwen3AsrStreamState;

    fn sample_rate(&self) -> usize {
        self.bundle.front_end.sample_rate()
    }

    fn hop(&self) -> usize {
        self.bundle.front_end.hop()
    }

    fn window_frames(&self) -> usize {
        self.window_frames
    }

    fn grid_frames(&self) -> usize {
        1
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

    fn new_stream_state(&self) -> Qwen3AsrStreamState {
        Qwen3AsrStreamState::default()
    }

    fn decode(
        &self,
        requests: Vec<DecodeRequest<'_, Self>>,
    ) -> BunsenResult<Vec<WindowDecode<Qwen3AsrCarry>>> {
        requests
            .into_iter()
            .map(|request| {
                let mut window = request.window;
                let mut frames = window.dims()[1];
                if self.quiet_cut && frames >= self.window_frames {
                    let loudness: Vec<f32> = window
                        .clone()
                        .mean_dim(2)
                        .reshape([frames])
                        .into_data()
                        .convert::<f32>()
                        .try_to_vec()
                        .expect("f32 data");
                    frames = quiet_cut(&loudness, QUIET_SPAN_FRAMES).max(1);
                    window = window.slice_dim(1, 0..frames);
                }
                let features = self.package(window, &request.level);
                let reply = self.bundle.transcribe_features(features, frames)?;
                Ok(self.lay_out(reply, frames))
            })
            .collect()
    }

    fn commit(
        &self,
        state: &mut Qwen3AsrStreamState,
        _transcript: &[i64],
        carry: Qwen3AsrCarry,
    ) {
        if state.language.is_none() {
            state.language = carry.language;
        }
    }

    fn text(
        &self,
        tokens: &[i64],
    ) -> BunsenResult<Option<String>> {
        let ids = tokens
            .iter()
            .map(|&t| {
                usize::try_from(t).map_err(|_| {
                    BunsenError::invalid_resource(format!("{t} is not a Qwen3-ASR token id"))
                })
            })
            .collect::<BunsenResult<Vec<usize>>>()?;
        Ok(Some(self.bundle.vocab.text(&ids)?))
    }
}

impl AsrStreamContext<Qwen3AsrEngine> {
    /// The language of the stream, once a committed reply names one.
    pub fn language(&self) -> Option<&str> {
        self.engine_state().language.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::{
            gpts::qwen3::Qwen3Config,
            speech::{
                asr_driver::{
                    EmissionPolicy,
                    LevelPolicy,
                    LevelStatistic,
                    StreamClock,
                    StreamingFeatures,
                    TimedTokens,
                    TranscriptEvent,
                },
                qwen3_asr::{
                    Qwen3AsrFrontEndConfig,
                    Qwen3AsrGreedyDecoderConfig,
                    Qwen3AsrPromptConfig,
                    blocks::{
                        Qwen3Asr,
                        Qwen3AsrAudioEncoderConfig,
                        Qwen3AsrConfig,
                    },
                },
                testing::ToneConfig,
            },
            tokens::ByteLevelVocabulary,
        },
        support::testing::performance_device,
    };

    /// The toy's token count.
    const VOCAB: usize = 50;

    /// An untrained toy bundle: a 16-mel encoder, a 50-token text model and
    /// vocabulary, a prompt of small ids.
    fn toy_bundle(
        text_token: usize,
        eos: Vec<usize>,
        device: &Device,
    ) -> Qwen3AsrBundle {
        let config = Qwen3AsrConfig::new(
            Qwen3AsrAudioEncoderConfig::new(16, 32, 4, 64, 2, 8).with_n_window_infer(200),
            Qwen3Config::new(VOCAB, 24, 2, 4, 2, 8, 48),
        )
        .with_audio_token(7);
        let model: Qwen3Asr = config.init(device);
        let entries: Vec<String> = (0..VOCAB).map(|i| format!("\"t{i}\": {i}")).collect();
        let vocab =
            ByteLevelVocabulary::from_vocab_json(&format!("{{{}}}", entries.join(", ")), None)
                .unwrap();
        Qwen3AsrBundle {
            model,
            front_end: Qwen3AsrFrontEndConfig::new().with_n_mels(16).init(device),
            vocab,
            prompt: Qwen3AsrPromptConfig::new()
                .with_prefix(vec![1, 2])
                .with_suffix(vec![3])
                .with_text_token(text_token),
            decoder: Qwen3AsrGreedyDecoderConfig::new()
                .with_max_new_tokens(3)
                .with_eos(eos),
        }
    }

    /// A window is set in seconds, rounded down to whole frames, and never
    /// past the model's range.
    #[test]
    fn test_window_seconds() {
        let device = performance_device();
        let engine = Qwen3AsrEngine::new(Arc::new(toy_bundle(4, vec![5], &device)));
        assert_eq!(engine.window_frames(), 3000);
        assert_eq!(
            engine.clone().with_window_seconds(12.345).window_frames(),
            1234
        );
        assert_eq!(engine.with_window_seconds(30.0).window_frames(), 3000);
    }

    #[test]
    #[should_panic(expected = "over 30 s")]
    fn test_window_past_the_range_panics() {
        let device = performance_device();
        Qwen3AsrEngine::new(Arc::new(toy_bundle(4, vec![5], &device))).with_window_seconds(30.5);
    }

    /// A streamed window, packaged, is the front end's one-shot features,
    /// padding included, but for the frames either side of the stream's
    /// end, where streaming reflects and the one-shot extractor pads with
    /// zeros.
    #[test]
    fn test_packaging_matches_the_one_shot_front_end() {
        let device = performance_device();
        let engine = Qwen3AsrEngine::new(Arc::new(toy_bundle(4, vec![5], &device)));
        assert_eq!(engine.window_frames(), 3000);
        let samples = ToneConfig::Duration {
            seconds: 2.0,
            hz: 16_000.0,
        }
        .build();

        let (one_shot, valid) = engine
            .bundle()
            .front_end
            .forward(&samples, &device)
            .unwrap();
        assert_eq!(one_shot.dims(), [16, 3000]);

        let mut features = engine.new_features().unwrap();
        let mut frames: Vec<Tensor<3>> = features.push(&samples).unwrap().into_iter().collect();
        frames.extend(features.finish().unwrap());
        let frames = Tensor::cat(frames, 1);
        let n = frames.dims()[1] - engine.end_frames_dropped();
        let window = frames.slice_dim(1, 0..n);
        assert_eq!(n, valid);

        let level = MaxLevel::of(window.clone());
        let packaged = engine.package(window, &level);
        assert_eq!(packaged.dims(), one_shot.dims());

        for range in [0..valid - 2, valid + 2..3000] {
            packaged
                .clone()
                .slice_dim(1, range.clone())
                .to_data()
                .assert_approx_eq::<f32>(
                    &one_shot.clone().slice_dim(1, range).to_data(),
                    burn::tensor::Tolerance::absolute(1e-4),
                );
        }
    }

    /// A reply's transcript is one segment over the window; a reply without
    /// one skips it; the stream keeps the first language a commit names.
    #[test]
    fn test_lay_out_and_commit() {
        let device = performance_device();
        let engine = Qwen3AsrEngine::new(Arc::new(toy_bundle(4, vec![5], &device)));
        let reply = |tokens: Vec<usize>, language: Option<&str>| Qwen3AsrTranscript {
            tokens,
            language: language.map(String::from),
            text: String::new(),
        };

        let decode = engine.lay_out(reply(vec![10, 4, 12, 13, 5, 9], Some("English")), 100);
        assert!(!decode.skip);
        assert_eq!(decode.tokens, [12, 13]);
        assert_eq!(
            decode.segments,
            [TimedTokens {
                start: 0,
                end: 100,
                tokens: vec![12, 13],
            }]
        );
        assert_eq!((decode.advance, decode.tail), (100, None));

        let silent = engine.lay_out(reply(vec![10, 4, 5], None), 100);
        assert!(silent.skip);
        assert!(silent.segments.is_empty());
        assert_eq!(silent.advance, 100);

        let mut state = engine.new_stream_state();
        engine.commit(&mut state, &[], silent.carry);
        assert_eq!(state.language, None);
        engine.commit(&mut state, &[12, 13], decode.carry);
        assert_eq!(state.language.as_deref(), Some("English"));
        engine.commit(
            &mut state,
            &[12, 13],
            Qwen3AsrCarry {
                language: Some("German".into()),
            },
        );
        assert_eq!(state.language.as_deref(), Some("English"));

        assert_eq!(engine.text(&[12, 13]).unwrap().as_deref(), Some("t12t13"));
        assert!(engine.text(&[-1]).is_err());
    }

    /// Streamed in pieces under the offline policy, a clip shorter than a
    /// window is one committed segment over all of it.
    #[test]
    fn test_streams_through_the_core() {
        let device = performance_device();
        // No end token, and a text token the model cannot produce: every
        // reply is three tokens of transcript.
        let engine = Qwen3AsrEngine::new(Arc::new(toy_bundle(VOCAB, Vec::new(), &device)));
        let driver = Qwen3AsrStreamDriver::new(engine, EmissionPolicy::offline()).unwrap();
        let mut stream = driver
            .new_context(StreamClock::uniform(16_000), LevelPolicy::PerWindow)
            .unwrap();

        let samples = ToneConfig::Duration {
            seconds: 2.5,
            hz: 16_000.0,
        }
        .build();
        let mut events = Vec::new();
        for piece in samples.chunks(1_000) {
            events.extend(stream.write_read(piece).unwrap());
        }
        events.extend(stream.end_read().unwrap());

        let [TranscriptEvent::Committed(segment)] = events.as_slice() else {
            panic!("one committed segment, got {events:?}");
        };
        assert_eq!(segment.tokens.len(), 3);
        assert_eq!(segment.start, 0.0);
        assert!((segment.end - 2.5).abs() < 0.011, "{}", segment.end);
        assert_eq!(segment.text, driver.engine().text(&segment.tokens).unwrap());
        assert_eq!(stream.language(), None);
    }

    /// A full window is decoded up to its quietest stretch in its last
    /// third, and the stream moves on from there; turned off, it is decoded
    /// whole.
    #[test]
    fn test_full_windows_are_cut_quiet() {
        let device = performance_device();
        // A tone, silent from 0.8 s to 0.9 s, then sounding again to 1.6 s.
        let mut samples = ToneConfig::Duration {
            seconds: 1.6,
            hz: 16_000.0,
        }
        .build();
        samples[12_800..14_400].fill(0.0);
        let commits = |engine: Qwen3AsrEngine| {
            let driver = Qwen3AsrStreamDriver::new(engine, EmissionPolicy::offline()).unwrap();
            let mut stream = driver
                .new_context(StreamClock::uniform(16_000), LevelPolicy::PerWindow)
                .unwrap();
            let mut events = Vec::new();
            for piece in samples.chunks(1_000) {
                events.extend(stream.write_read(piece).unwrap());
            }
            events.extend(stream.end_read().unwrap());
            events
                .into_iter()
                .map(|e| {
                    let TranscriptEvent::Committed(segment) = e else {
                        panic!("the offline policy only commits: {e:?}");
                    };
                    (segment.start, segment.end)
                })
                .collect::<Vec<_>>()
        };
        let engine = Qwen3AsrEngine::new(Arc::new(toy_bundle(VOCAB, Vec::new(), &device)))
            .with_window_seconds(1.0);

        let cut = commits(engine.clone());
        assert!(cut.len() >= 2, "{cut:?}");
        assert!((0.8..=0.9).contains(&cut[0].1), "{cut:?}");
        assert_eq!(cut[1].0, cut[0].1);

        let whole = commits(engine.with_quiet_cut(false));
        assert!((whole[0].1 - 1.0).abs() < 1e-9, "{whole:?}");
    }
}
