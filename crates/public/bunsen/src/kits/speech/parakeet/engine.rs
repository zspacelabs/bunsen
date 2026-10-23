//! Parakeet as the stream driver core's engine.
use std::sync::Arc;

use burn::{
    Tensor,
    module::Module,
    tensor::Device,
};

use super::{
    ParakeetBundle,
    ParakeetFeatures,
    ParakeetFrontEnd,
    ParakeetFrontEndMeta,
    ParakeetTdtMeta,
    TdtGreedyDecoderConfig,
    TdtHypothesis,
    WORD_MARK,
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
        MomentsLevel,
        TimedTokens,
        WindowDecode,
        WordSpan,
        pause_cut,
        word_segments,
    },
};

/// The stream driver of a Parakeet model: the core's [`AsrStreamDriver`]
/// over a [`ParakeetEngine`].
pub type ParakeetStreamDriver = AsrStreamDriver<ParakeetEngine>;

/// One Parakeet stream: the core's [`AsrStreamContext`] over a
/// [`ParakeetEngine`].
pub type ParakeetStreamContext = AsrStreamContext<ParakeetEngine>;

/// Normalizes a window of [`ParakeetFeatures`] frames by its level, per mel
/// channel, as `front_end` normalizes an item.
///
/// # Arguments
/// * `window` - `[1, frames, n_mels]`.
///
/// # Returns
/// `[n_mels, frames]`.
fn normalize(
    front_end: &ParakeetFrontEnd,
    window: Tensor<3>,
    level: &MomentsLevel,
) -> Tensor<2> {
    let [_, frames, n_mels] = window.dims();
    let std = level
        .sample_variance()
        .sqrt()
        .add_scalar(front_end.config.norm_epsilon);
    ((window - level.mean.clone().unsqueeze_dim(1)) / std.unsqueeze_dim(1))
        .reshape([frames, n_mels])
        .swap_dims(0, 1)
}

/// A [`ParakeetBundle`] as an [`AsrEngine`]: what the stream driver core
/// needs of the model.
///
/// - **Features:** [`ParakeetFeatures`], each window normalized per mel channel
///   by its [`MomentsLevel`] as the front end normalizes an item: `(x - mean) /
///   (std + 1e-5)`, with the unbiased standard deviation. Under
///   [`PerWindow`](crate::kits::speech::asr_driver::LevelPolicy::PerWindow) a
///   window that is the whole clip is the one-shot extractor's features. A
///   window is at most [`window_frames`](AsrEngine::window_frames), 60 s by
///   default ([`with_window_seconds`](Self::with_window_seconds)).
/// - **Decode:** the encoder and the greedy transducer decode, batched across
///   requests. The tokens carry their encoder frames, and each word is a
///   segment, timed from the frame its first token is emitted on. A window the
///   core cut at its full length commits the words before a pause in its last
///   third, and the rest is decoded again with the audio after it. Any other
///   window is committed whole.
///
/// A stream's last frame is the one centred past its end, which the
/// extractor counts as padding, and goes.
#[derive(Clone, Debug)]
pub struct ParakeetEngine {
    bundle: Arc<ParakeetBundle>,
    window_frames: usize,
    device: Device,
}

impl ParakeetEngine {
    /// The longest window by default, in seconds.
    pub const DEFAULT_WINDOW_SECONDS: f64 = 60.0;

    /// The engine of `bundle`, on the model's device, with windows of at most
    /// [`DEFAULT_WINDOW_SECONDS`](Self::DEFAULT_WINDOW_SECONDS).
    pub fn new(bundle: Arc<ParakeetBundle>) -> Self {
        let device = bundle.model.devices()[0].clone();
        let engine = Self {
            bundle,
            window_frames: 0,
            device,
        };
        engine.with_window_seconds(Self::DEFAULT_WINDOW_SECONDS)
    }

    /// The engine with windows of at most `seconds`, rounded down to whole
    /// encoder frames.
    ///
    /// # Panics
    /// If that is under one encoder frame.
    pub fn with_window_seconds(
        mut self,
        seconds: f64,
    ) -> Self {
        let front_end = &self.bundle.front_end;
        let subsampling = self.bundle.model.subsampling_factor();
        let frames = (seconds * front_end.sample_rate() as f64 / front_end.hop() as f64) as usize;
        let frames = frames / subsampling * subsampling;
        assert!(
            frames > 0,
            "a window of {seconds} s is under one encoder frame"
        );
        self.window_frames = frames;
        self
    }

    /// The bundle.
    pub fn bundle(&self) -> &Arc<ParakeetBundle> {
        &self.bundle
    }

    /// The hypothesis' words, in frames of the window: a word opens on a
    /// piece with the word mark (tokens before the first such piece are a
    /// word of their own), starts on the encoder frame its first token is
    /// emitted on, and ends where its last token ends.
    fn spans(
        &self,
        hypothesis: &TdtHypothesis,
    ) -> Vec<WordSpan> {
        let subsampling = self.bundle.model.subsampling_factor();
        let vocab = &self.bundle.vocab;
        let h = hypothesis;
        let opens: Vec<usize> = (0..h.tokens.len())
            .filter(|&k| {
                k == 0
                    || vocab
                        .piece(h.tokens[k])
                        .is_some_and(|p| p.starts_with(WORD_MARK))
            })
            .collect();
        opens
            .iter()
            .enumerate()
            .map(|(i, &first)| {
                let last = opens.get(i + 1).map_or(h.tokens.len(), |&next| next) - 1;
                WordSpan {
                    first_token: first,
                    start: h.frames[first] * subsampling,
                    end: (h.frames[last] + h.durations[last].max(1)) * subsampling,
                }
            })
            .collect()
    }

    /// Lays a hypothesis out for the core: a segment per word.
    ///
    /// A window of `window_frames` was cut by length: it commits the words
    /// before its [`pause_cut`], on the encoder grid, and what follows is the
    /// tail. Any other window, or one without a cut, commits every word. A
    /// window with no tokens is skipped.
    fn lay_out(
        &self,
        hypothesis: TdtHypothesis,
        frames: usize,
    ) -> WindowDecode<()> {
        let tokens: Vec<i64> = hypothesis.tokens.iter().map(|&t| t as i64).collect();
        let words = self.spans(&hypothesis);
        let grid = self.bundle.model.subsampling_factor();
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

impl AsrEngine for ParakeetEngine {
    type Carry = ();
    type Features = ParakeetFeatures;
    type Level = MomentsLevel;
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

    /// The encoder frame: when a token can be emitted.
    fn grid_frames(&self) -> usize {
        self.bundle.model.subsampling_factor()
    }

    /// The frame centred past the end, which the extractor counts as
    /// padding.
    fn end_frames_dropped(&self) -> usize {
        1
    }

    fn new_features(&self) -> BunsenResult<ParakeetFeatures> {
        Ok(ParakeetFeatures::new(
            self.bundle.front_end.clone(),
            self.device.clone(),
        ))
    }

    fn new_stream_state(&self) {}

    /// One batch: the windows padded to the longest, each encoded and decoded
    /// over its own frames.
    fn decode(
        &self,
        requests: Vec<DecodeRequest<'_, Self>>,
    ) -> BunsenResult<Vec<WindowDecode<()>>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let lengths: Vec<usize> = requests.iter().map(|r| r.window.dims()[1]).collect();
        let longest = lengths.iter().copied().max().unwrap_or(0);
        let rows: Vec<Tensor<3>> = requests
            .into_iter()
            .map(|r| {
                let frames = r.window.dims()[1];
                let features = normalize(&self.bundle.front_end, r.window, &r.level);
                let [n_mels, _] = features.dims();
                let features = if frames < longest {
                    Tensor::cat(
                        vec![
                            features,
                            Tensor::zeros([n_mels, longest - frames], &self.device),
                        ],
                        1,
                    )
                } else {
                    features
                };
                features.unsqueeze_dim(0)
            })
            .collect();
        let model = &self.bundle.model;
        let (encoded, encoded_lengths) = model.encode(Tensor::cat(rows, 0), Some(&lengths));
        let hypotheses =
            TdtGreedyDecoderConfig::new()
                .init()
                .decode(model, encoded, &encoded_lengths);
        Ok(hypotheses
            .into_iter()
            .zip(lengths)
            .map(|(hypothesis, frames)| self.lay_out(hypothesis, frames))
            .collect())
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
            .map(|&t| {
                usize::try_from(t).map_err(|_| {
                    BunsenError::invalid_resource(format!("{t} is not a Parakeet token id"))
                })
            })
            .collect::<BunsenResult<Vec<usize>>>()?;
        Ok(Some(self.bundle.vocab.transcript(&ids)?))
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::TensorData,
        tensor::Tolerance,
    };

    use super::*;
    use crate::{
        blocks::{
            signals::DwStridingSubsamplingConfig,
            transformers::encoders::conformer::{
                ConformerBlockConfig,
                ConformerEncoderConfig,
            },
        },
        burner::module::ModuleInit,
        kits::speech::{
            asr_driver::{
                EmissionPolicy,
                LevelPolicy,
                LevelStatistic,
                StreamClock,
                StreamingFeatures,
                TranscriptEvent,
            },
            parakeet::{
                ParakeetFrontEndConfig,
                ParakeetTdt,
                ParakeetTdtConfig,
                ParakeetVocabulary,
            },
            testing::ToneConfig,
        },
        support::testing::performance_device,
    };

    /// An untrained toy bundle: a 24-mel encoder subsampling by 8, and an
    /// 11-token vocabulary, the blank last.
    fn toy_bundle(device: &Device) -> ParakeetBundle {
        let config = ParakeetTdtConfig::new(
            ConformerEncoderConfig::new(
                DwStridingSubsamplingConfig::new(24, 4, 16),
                ConformerBlockConfig::new(16, 2, 32, 5)
                    .with_bias(false)
                    .with_conv_bias(false),
                2,
            ),
            11,
            12,
        );
        let model: ParakeetTdt = config.init(device);
        let pieces = [
            "<unk>",
            "\u{2581}the",
            "\u{2581}mo",
            "on",
            ".",
            "\u{2581}why",
            "\u{2581}we",
            "\u{2581}go",
            "s",
            "t",
            "<blk>",
        ];
        ParakeetBundle {
            model,
            front_end: ParakeetFrontEndConfig::new().with_n_mels(24).init(device),
            vocab: ParakeetVocabulary::new(pieces.map(String::from).to_vec()),
        }
    }

    /// Streamed in pieces of any size and normalized by its window's level,
    /// a clip is the one-shot extractor's features, every valid frame.
    #[test]
    fn test_streamed_features_are_the_front_ends() {
        let device = performance_device();
        let front_end: ParakeetFrontEnd = ParakeetFrontEndConfig::new().init(&device);
        let samples = ToneConfig::Samples(20_850).build();
        let wave = Tensor::<2>::from_data(
            TensorData::new(samples.clone(), [1, samples.len()]),
            &device,
        );
        let (one_shot, valid) = front_end.forward(wave, None);
        let valid = valid[0];
        assert_eq!(valid, 20_850 / 160);
        let one_shot = one_shot.slice_dim(2, 0..valid).reshape([128, valid]);

        for piece in [1, 160, 999, samples.len()] {
            let mut features = ParakeetFeatures::new(front_end.clone(), device.clone());
            let mut frames = Vec::new();
            for chunk in samples.chunks(piece) {
                frames.extend(features.push(chunk).unwrap());
            }
            frames.extend(features.finish().unwrap());
            assert!(features.finish().unwrap().is_none());
            let frames = Tensor::cat(frames, 1);
            assert_eq!(frames.dims(), [1, valid + 1, 128], "piece {piece}");

            let window = frames.slice_dim(1, 0..valid);
            let level = MomentsLevel::of(window.clone());
            normalize(&front_end, window, &level)
                .to_data()
                .assert_approx_eq::<f32>(&one_shot.to_data(), Tolerance::relative(0.01));
        }

        let mut empty = ParakeetFeatures::new(front_end, device);
        assert!(empty.push(&[]).unwrap().is_none());
        assert!(empty.finish().unwrap().is_none());
    }

    /// A window cut at its full length commits up to the widest pause
    /// between words in its last third, or else up to its last word, and the
    /// rest is the tail; any other window, or one word, is one segment.
    #[test]
    fn test_lay_out() {
        let device = performance_device();
        let engine = ParakeetEngine::new(Arc::new(toy_bundle(&device))).with_window_seconds(2.0);
        assert_eq!(engine.window_frames(), 200);
        assert_eq!(engine.grid_frames(), 8);
        let full = engine.window_frames();
        let hypothesis = |tokens: Vec<usize>, frames: Vec<usize>| TdtHypothesis {
            durations: vec![1; tokens.len()],
            tokens,
            frames,
        };

        // "the moon why we go", in encoder frames of 8: the last third opens
        // at frame 133 / 8. "why" follows the widest pause there, frames 7 to
        // 18, so the cut is at 12. Each word before it is a segment, from
        // its first token's frame to the end of its last.
        let spread = hypothesis(vec![1, 2, 3, 5, 6, 7], vec![2, 5, 6, 18, 20, 23]);
        let decode = engine.lay_out(spread, full);
        assert_eq!(decode.tokens, [1, 2, 3, 5, 6, 7]);
        assert_eq!(
            decode.segments,
            [
                TimedTokens {
                    start: 16,
                    end: 24,
                    tokens: vec![1],
                },
                TimedTokens {
                    start: 40,
                    end: 56,
                    tokens: vec![2, 3],
                },
            ]
        );
        assert_eq!(
            decode.tail,
            Some(TimedTokens {
                start: 96,
                end: full,
                tokens: vec![5, 6, 7],
            })
        );
        assert_eq!(decode.advance, 96);

        // "the moon. why we gos", all early: before the last word, "we",
        // where "why" ends at frame 6.
        let words = hypothesis(vec![1, 2, 3, 4, 5, 6, 8], vec![0, 2, 3, 3, 5, 7, 8]);
        let decode = engine.lay_out(words.clone(), full);
        let spans = |d: &WindowDecode<()>| {
            d.segments
                .iter()
                .map(|s| (s.start, s.end, s.tokens.clone()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            spans(&decode),
            [(0, 8, vec![1]), (16, 32, vec![2, 3, 4]), (40, 48, vec![5]),]
        );
        assert_eq!(decode.tail.as_ref().unwrap().tokens, [6, 8]);
        assert_eq!(decode.advance, 48);

        // Uncut, every word commits; the last ends a frame past its last
        // token.
        let short = engine.lay_out(words, full - 1);
        assert_eq!(short.segments.len(), 4);
        assert_eq!(short.segments[3].tokens, [6, 8]);
        assert_eq!((short.segments[3].start, short.segments[3].end), (56, 72));
        assert_eq!((short.advance, short.tail), (full - 1, None));

        // A word whose last token ends past the next word's emission ends
        // there.
        let mut overlap = hypothesis(vec![1, 5], vec![2, 3]);
        overlap.durations = vec![4, 1];
        let decode = engine.lay_out(overlap, full - 1);
        assert_eq!(spans(&decode)[0], (16, 24, vec![1]));

        let one_word = engine.lay_out(hypothesis(vec![2, 3], vec![4, 5]), full);
        assert_eq!(spans(&one_word), [(32, 48, vec![2, 3])]);
        assert_eq!((one_word.advance, one_word.tail), (full, None));

        let silent = engine.lay_out(hypothesis(vec![], vec![]), full - 1);
        assert!(silent.skip && silent.segments.is_empty());

        assert_eq!(engine.text(&[5, 6]).unwrap().as_deref(), Some("why we"));
        assert!(engine.text(&[-1]).is_err());
    }

    /// Streamed in pieces, a clip longer than a window commits segments in
    /// order, over the clip, each spelling its tokens.
    #[test]
    fn test_streams_through_the_core() {
        let device = performance_device();
        let engine = ParakeetEngine::new(Arc::new(toy_bundle(&device))).with_window_seconds(1.0);
        let driver = ParakeetStreamDriver::new(engine, EmissionPolicy::offline()).unwrap();
        let mut stream = driver
            .new_context(StreamClock::uniform(16_000), LevelPolicy::PerWindow)
            .unwrap();
        let mut events = Vec::new();
        for piece in ToneConfig::Samples(40_000).build().chunks(1_600) {
            events.extend(stream.write_read(piece).unwrap());
        }
        events.extend(stream.end_read().unwrap());
        assert!(stream.is_finished());

        let mut end = 0.0;
        for event in &events {
            let TranscriptEvent::Committed(segment) = event else {
                panic!("the offline policy only commits: {event:?}");
            };
            assert!(segment.start >= end && segment.end >= segment.start);
            assert!(segment.end <= 2.5 + 1e-9);
            assert_eq!(segment.text, driver.engine().text(&segment.tokens).unwrap());
            end = segment.end;
        }
    }
}
