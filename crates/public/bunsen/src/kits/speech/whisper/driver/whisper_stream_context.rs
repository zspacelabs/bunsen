//! # Whisper as the core's engine, and its stream context.

use burn::{
    Tensor,
    prelude::TensorData,
    tensor::Device,
};

use crate::{
    errors::{
        BunsenResult,
        ResultContext,
    },
    kits::speech::{
        asr_driver::{
            AsrEngine,
            AsrStreamContext,
            DecodeRequest,
            MaxLevel,
            StreamingFeatures,
            TimedTokens,
            WindowDecode,
        },
        whisper::{
            blocks::WhisperMeta,
            decode::{
                DecodeConfig,
                DecodedTokens,
                WhisperFallbackConfig,
                decode_with_fallback,
            },
            driver::{
                WhisperEngine,
                support::segments::split_window,
            },
        },
    },
    ops::signal::perceptive_audio::{
        PerceptiveAudioConversionContext,
        PerceptiveAudioConverter,
        PerceptiveAudioConverterMeta,
    },
};

/// One Whisper stream: the core's [`AsrStreamContext`] over a
/// [`WhisperEngine`].
///
/// Opened by
/// [`new_context`](crate::kits::speech::asr_driver::AsrStreamDriver::new_context)
/// on a [`WhisperStreamDriver`](super::WhisperStreamDriver). Samples go in
/// through [`write_read`](AsrStreamContext::write_read), or
/// [`write`](AsrStreamContext::write) and [`read`](AsrStreamContext::read)
/// apart, and come out as
/// [`TranscriptEvent`](crate::kits::speech::asr_driver::TranscriptEvent)s;
/// [`end_read`](AsrStreamContext::end_read) ends the stream and decodes what
/// is left. The [`driver`](super) module docs walk through the fold, the
/// emission rules, the clock and the voice-activity regions.
pub type WhisperStreamContext = AsrStreamContext<WhisperEngine>;

impl AsrStreamContext<WhisperEngine> {
    /// The language the stream's prompt names, once known: configured, or
    /// detected by the stream's first commit. A draft never sets it.
    pub fn language(&self) -> Option<&str> {
        self.engine_state().language.as_deref()
    }

    /// The prompt the stream's next window decodes under.
    pub fn prompt_now(&self) -> Vec<i64> {
        let engine = self.driver().engine();
        engine.prompt_under(self.engine_state(), self.transcript(), self.language())
    }
}

/// What a Whisper stream carries from one window to the next.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WhisperStreamState {
    /// The language the prompt names: configured, or detected from the
    /// first window committed; `None` until then on a detecting driver.
    pub language: Option<String>,

    /// Where the prompt carry starts in the transcript: upstream's
    /// `prompt_reset_since`, moved past a decode that needed a high
    /// temperature so a failure does not feed the next window.
    pub prompt_reset: usize,
}

/// What a Whisper decode hands back to its commit.
#[derive(Clone, Debug, PartialEq)]
pub struct WhisperCarry {
    /// The temperature the decode was accepted at.
    pub temperature: f64,

    /// The language the decode detected, when the stream had none.
    pub detected: Option<String>,
}

/// Whisper's streaming log-mel front end: whole hops of samples through the
/// mel converter's context.
///
/// The first chunk must be long enough to produce a frame under the start
/// padding, so it waits for a window's worth of samples; after that a hop at
/// a time is enough. Finishing pads whatever remains with silence up to that
/// minimum, as upstream pads a clip; a stream that never received a sample
/// finishes with nothing.
#[derive(Clone, Debug)]
pub struct WhisperFeatures {
    converter: PerceptiveAudioConverter,

    /// The converter's streaming state; `None` once finished.
    context: Option<PerceptiveAudioConversionContext>,

    /// Samples not yet a whole hop.
    staging: Vec<f32>,

    device: Device,
}

impl WhisperFeatures {
    /// Moves whole hops from staging into the converter.
    fn drain(
        &mut self,
        flushing: bool,
    ) -> BunsenResult<Option<Tensor<3>>> {
        let (hop, n_fft) = (self.converter.hop(), self.converter.n_fft());
        let first = self
            .context
            .as_ref()
            .is_some_and(|ctx| ctx.carry().is_none());
        let minimum = if first {
            n_fft.div_ceil(hop) * hop
        } else {
            hop
        };

        let mut whole = self.staging.len() / hop * hop;
        if flushing {
            if self.staging.is_empty() {
                return Ok(None);
            }
            whole = self.staging.len().div_ceil(hop).max(minimum / hop) * hop;
            self.staging.resize(whole, 0.0);
        } else if whole < minimum {
            return Ok(None);
        }

        // `from_data` converts to the device's float dtype, so the samples
        // go up as they are.
        let chunk: Vec<f32> = self.staging.drain(..whole).collect();
        let waves: Tensor<2> = Tensor::from_data(TensorData::new(chunk, [1, whole]), &self.device);

        let ctx = self
            .context
            .take()
            .expect("the front end is open until the input ends");
        // The context built the chunk to the front end's grid, so a refusal
        // is the context's own fault.
        let (frames, ctx) = ctx.transform(waves).as_internal()?;
        self.context = Some(ctx);
        Ok(Some(frames))
    }
}

impl StreamingFeatures for WhisperFeatures {
    fn push(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<Option<Tensor<3>>> {
        self.staging.extend_from_slice(samples);
        self.drain(false)
    }

    fn finish(&mut self) -> BunsenResult<Option<Tensor<3>>> {
        if self.context.is_none() {
            return Ok(None);
        }
        let drained = self.drain(true)?;
        let tail = self.context.take().and_then(|mel| mel.finish());
        Ok(match (drained, tail) {
            (Some(a), Some(b)) => Some(Tensor::cat(vec![a, b], 1)),
            (a, b) => a.or(b),
        })
    }
}

/// One request, packaged: what it decodes under, and its window.
struct Packaged {
    /// The prompt it decodes under; what it is batched by.
    prompt: Vec<i64>,
    /// `[1, n_mels, width]`.
    window: Tensor<3>,
    /// The language detected for it, when its stream had none.
    detected: Option<String>,
    /// The window's frames.
    count: usize,
}

impl WhisperEngine {
    /// Floors a window at the range below its level and pads it out to the
    /// model's width: `[1, n_mels, width]`.
    fn package_padded(
        &self,
        window: Tensor<3>,
        level: &MaxLevel,
    ) -> Tensor<3> {
        let packaged = self.front_end().package_window(window, level.max.clone());

        let width = self.window_frames();
        let [_, n_mels, have] = packaged.dims();
        if have < width {
            let pad = Tensor::zeros([1, n_mels, width - have], &packaged.device());
            Tensor::cat(vec![packaged, pad], 2)
        } else {
            packaged
        }
    }

    /// The prompt a window decodes under: the sot sequence naming
    /// `language`, preceded by the transcript's tail after
    /// `<|startofprev|>` when carrying is on.
    pub(super) fn prompt_under(
        &self,
        state: &WhisperStreamState,
        transcript: &[i64],
        language: Option<&str>,
    ) -> Vec<i64> {
        let prompt = self
            .sot_sequence(language)
            .expect("the language is known before a prompt is built");
        let carried = &transcript[state.prompt_reset.min(transcript.len())..];
        if !self.config.condition_on_previous_text || carried.is_empty() {
            return prompt;
        }

        // Upstream keeps `n_text_ctx / 2 - 1` tokens of context.
        let keep = self.whisper_model().max_text_ctx() / 2 - 1;
        let tail = &carried[carried.len().saturating_sub(keep)..];

        let mut out = Vec::with_capacity(1 + tail.len() + prompt.len());
        out.push(self.token_layout().ids().sot_prev);
        out.extend_from_slice(tail);
        out.extend_from_slice(&prompt);
        out
    }

    /// The language a packaged window detects, when the stream has none and
    /// the driver leaves it to be detected: upstream's `detect_language`, on
    /// the features the decode will use.
    fn undetected_language(
        &self,
        state: &WhisperStreamState,
        packaged: &Tensor<3>,
    ) -> Option<String> {
        if state.language.is_some() || !self.detects_language() {
            return None;
        }
        let model = self.whisper_model();
        let xa = model.forward_encoder(packaged.clone());
        let token = model.detect_language(xa, self.token_layout().ids())[0];
        let code = self
            .token_layout()
            .language_code(token)
            .expect("detection picks a language token");
        Some(code.to_string())
    }

    /// Packages a request, and finds the prompt it decodes under.
    ///
    /// A request on a stream with no language yet detects one from its own
    /// window and decodes under it. A commit's detection reaches the stream
    /// through its carry; a draft's is dropped, so drafting cannot change
    /// the language the stream commits under.
    fn package(
        &self,
        request: &DecodeRequest<'_, Self>,
    ) -> Packaged {
        let count = request.window.dims()[1];
        let window = self.package_padded(request.window.clone(), &request.level);
        let detected = self.undetected_language(request.state, &window);
        let language = detected.as_deref().or(request.state.language.as_deref());
        Packaged {
            prompt: self.prompt_under(request.state, request.transcript, language),
            window,
            detected,
            count,
        }
    }

    /// Decodes a packaged window up the ladder, encoding it only if a rung
    /// past `first` is needed.
    ///
    /// # Arguments
    /// * `base` - the decode config at temperature zero.
    /// * `window` - `[1, n_mels, width]`, packaged.
    /// * `first` - the first rung's result when the caller has it.
    fn ladder(
        &self,
        base: &DecodeConfig,
        window: Tensor<3>,
        first: Option<DecodedTokens>,
    ) -> DecodedTokens {
        let model = self.whisper_model();
        let mut xa: Option<Tensor<3>> = None;
        decode_with_fallback(
            &self.config.fallback,
            base,
            first,
            |config| {
                let xa = xa
                    .get_or_insert_with(|| model.forward_encoder(window.clone()))
                    .clone();
                model
                    .decode_features_full(xa, config, &self.filters)
                    .pop()
                    .expect("one row in, one row out")
            },
            |ids| self.text(ids).ok().flatten(),
        )
    }

    /// Lays a decode out for the core.
    ///
    /// A window the fallback policy calls silence (its no-speech probability
    /// over the threshold, its log probability poor) is skipped whole.
    /// Without timestamps the window is one segment. With them the decode is
    /// split on its timestamps ([`split_window`]): the stream advances to the
    /// last closed timestamp, and the unfinished tail opens at its first
    /// timestamp.
    fn lay_out(
        &self,
        decoded: DecodedTokens,
        count: usize,
        detected: Option<String>,
    ) -> WindowDecode<WhisperCarry> {
        let carry = WhisperCarry {
            temperature: decoded.temperature,
            detected,
        };
        let skip = self
            .config
            .fallback
            .should_skip(decoded.no_speech_prob.map(f64::from), decoded.avg_logprob());
        let tokens = decoded.tokens;

        if skip {
            return WindowDecode {
                tokens,
                skip: true,
                segments: Vec::new(),
                advance: count,
                tail: None,
                carry,
            };
        }
        if !self.config.timestamps {
            return WindowDecode::whole(count, tokens, carry);
        }

        let ids = self.token_layout().ids();
        let fpt = self.frames_per_timestamp();
        let split = split_window(&tokens, ids, count, fpt);
        let tail = (!split.tail.is_empty()).then(|| TimedTokens {
            start: (split.tail[0] - ids.timestamp_begin) as usize * fpt,
            end: count,
            tokens: split.tail,
        });
        WindowDecode {
            segments: split
                .segments
                .into_iter()
                .map(|segment| TimedTokens {
                    start: segment.start,
                    end: segment.end,
                    tokens: segment.tokens,
                })
                .collect(),
            tokens,
            skip: false,
            advance: split.advance,
            tail,
            carry,
        }
    }
}

impl AsrEngine for WhisperEngine {
    type Carry = WhisperCarry;
    type Features = WhisperFeatures;
    type Level = MaxLevel;
    type StreamState = WhisperStreamState;

    fn sample_rate(&self) -> usize {
        self.bundle.model.sample_rate()
    }

    fn hop(&self) -> usize {
        self.audio_converter.hop()
    }

    fn window_frames(&self) -> usize {
        WhisperEngine::window_frames(self)
    }

    fn grid_frames(&self) -> usize {
        self.frames_per_timestamp()
    }

    /// Whisper's `stft[..., :-1]`: the stream's last frame is the end
    /// padding's, and goes.
    fn end_frames_dropped(&self) -> usize {
        1
    }

    fn new_features(&self) -> BunsenResult<WhisperFeatures> {
        Ok(WhisperFeatures {
            converter: self.audio_converter.clone(),
            context: Some(self.audio_converter.new_context(1)),
            staging: Vec::new(),
            device: self.devices()[0].clone(),
        })
    }

    fn new_stream_state(&self) -> WhisperStreamState {
        WhisperStreamState {
            language: self.config.language.clone(),
            prompt_reset: 0,
        }
    }

    /// Windows that decode under the same prompt share one batched first
    /// rung; a window that needs a rung above it climbs the rest of the
    /// ladder alone. A lone window decodes on its own, encoding only when
    /// its first rung needs it.
    fn decode(
        &self,
        requests: Vec<DecodeRequest<'_, Self>>,
    ) -> BunsenResult<Vec<WindowDecode<WhisperCarry>>> {
        let packaged: Vec<Packaged> = requests.iter().map(|r| self.package(r)).collect();

        if let [one] = packaged.as_slice() {
            let config = self.decode_config(one.prompt.clone());
            let decoded = self.ladder(&config, one.window.clone(), None);
            return Ok(vec![self.lay_out(decoded, one.count, one.detected.clone())]);
        }

        // Group by prompt, in order of first appearance.
        let mut groups: Vec<(Vec<i64>, Vec<usize>)> = Vec::new();
        for (k, item) in packaged.iter().enumerate() {
            match groups.iter_mut().find(|(p, _)| *p == item.prompt) {
                Some((_, members)) => members.push(k),
                None => groups.push((item.prompt.clone(), vec![k])),
            }
        }

        let mut out: Vec<Option<WindowDecode<WhisperCarry>>> = vec![None; packaged.len()];
        for (prompt, members) in groups {
            let batch = Tensor::cat(
                members
                    .iter()
                    .map(|&k| packaged[k].window.clone())
                    .collect(),
                0,
            );
            let config = self.decode_config(prompt);
            let first = self
                .whisper_model()
                .decode_windows_full(batch, &config, &self.filters);
            for (row, k) in members.into_iter().enumerate() {
                let item = &packaged[k];
                let decoded = self.ladder(&config, item.window.clone(), Some(first[row].clone()));
                out[k] = Some(self.lay_out(decoded, item.count, item.detected.clone()));
            }
        }
        Ok(out
            .into_iter()
            .map(|d| d.expect("every request decoded"))
            .collect())
    }

    /// Fixes a detected language, and resets the prompt carry after a decode
    /// that needed a temperature above 0.5 (as upstream does, so a failure
    /// does not feed the next window), or always when carrying is off.
    fn commit(
        &self,
        state: &mut WhisperStreamState,
        transcript: &[i64],
        carry: WhisperCarry,
    ) {
        if state.language.is_none() {
            state.language = carry.detected;
        }
        if !self.config.condition_on_previous_text
            || WhisperFallbackConfig::resets_prompt(carry.temperature)
        {
            state.prompt_reset = transcript.len();
        }
    }

    fn text(
        &self,
        tokens: &[i64],
    ) -> BunsenResult<Option<String>> {
        match &self.detokenizer {
            Some(detokenizer) => Ok(Some(
                detokenizer.detokenize(&self.token_layout().text_ids(tokens))?,
            )),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use burn::tensor::Device;
    use serial_test::serial;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        errors::{
            BunsenErrorKind,
            ConstraintError,
            LookupError,
            testing::ErrorMatcher,
        },
        kits::{
            speech::{
                asr_driver::{
                    CommitRule,
                    DecodeTriggers,
                    EmissionPolicy,
                    LevelPolicy,
                    LevelStatistic,
                    MaxLevel,
                    StreamClock,
                    TranscriptEvent,
                    advance_ready,
                },
                whisper::{
                    Whisper,
                    blocks::WhisperApiConfig,
                    decode::GreedyDecodeConfig,
                    driver::{
                        WhisperSpecialIds,
                        WhisperStreamDriver,
                        WhisperTokenLayout,
                        support::drop_last_frame,
                        whisper_stream_driver::WhisperStreamDriverConfig,
                    },
                    logit_filters::LogitFilter,
                },
            },
            tokens::Detokenizer,
        },
        support::testing::{
            DeviceMemoryGuard,
            performance_device,
        },
    };

    /// A layout small enough for a tiny model: 5 base ranks, 1 language.
    fn tiny_layout() -> WhisperSpecialIds {
        WhisperSpecialIds::new(5, 1).unwrap()
    }

    /// A tiny model whose vocabulary fits the tiny layout and whose window
    /// is 16 frames (2560 samples), so a short clip has several windows.
    /// Seeded, so a run is the same run on the same backend.
    fn tiny_model_on(device: &Device) -> Whisper {
        device.seed(7);
        WhisperApiConfig::new(
            /* n_mels */ 8,
            /* vocab_size */ tiny_layout().n_vocab(),
            /* d_model */ 64,
            /* max_audio_ctx */ 16,
            /* n_encoder_layers */ 1,
            /* max_text_ctx */ 16,
            /* n_decoder_layers */ 1,
        )
        .init(device)
    }

    fn config(carry: bool) -> WhisperStreamDriverConfig {
        WhisperStreamDriverConfig::new()
            .with_language(Some("en".to_string()))
            .with_max_tokens(4)
            .with_condition_on_previous_text(carry)
    }

    fn driver(
        device: &Device,
        carry: bool,
    ) -> WhisperStreamDriver {
        config(carry)
            .init_with_layout(
                tiny_model_on(device),
                WhisperTokenLayout::new(tiny_layout()),
                device,
            )
            .unwrap()
    }

    /// The rate the tiny model, and every clip here, is at.
    const RATE: usize = 16_000;

    fn clock() -> StreamClock {
        StreamClock::uniform(RATE)
    }

    /// A deterministic 1.05 s clip: a tone under a bell-shaped envelope
    /// peaking mid-clip, a rising chirp, and a little noise. The loudest
    /// frames are in the middle on purpose, so the global maximum is never
    /// in the trailing frame that packaging drops.
    fn clip() -> Vec<f32> {
        clip_seeded(0.5, 440.0)
    }

    fn clip_seeded(
        peak_at: f32,
        tone_hz: f32,
    ) -> Vec<f32> {
        let n = 16_800;
        (0..n)
            .map(|i| {
                let t = i as f32 / RATE as f32;
                let envelope = (-(t - peak_at).powi(2) * 20.0).exp();
                let tone = 0.6 * envelope * (2.0 * std::f32::consts::PI * tone_hz * t).sin();
                let chirp =
                    0.2 * (2.0 * std::f32::consts::PI * (3_100.0 * t + 800.0 * t * t)).sin();
                let noise = 0.05 * (((i * 7919) % 1000) as f32 / 500.0 - 1.0);
                tone + chirp + noise
            })
            .collect()
    }

    /// Chunk sizes from a seeded generator: 1 to 1200 samples, so most are
    /// not hop multiples and staging is exercised.
    fn random_sizes(
        seed: usize,
        total: usize,
    ) -> Vec<usize> {
        let mut state = seed;
        let mut sizes = Vec::new();
        let mut left = total;
        while left > 0 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let size = ((state >> 33) as usize % 1200 + 1).min(left);
            sizes.push(size);
            left -= size;
        }
        sizes
    }

    fn push_in_pieces(
        ctx: &mut WhisperStreamContext,
        audio: &[f32],
        sizes: &[usize],
    ) -> Vec<TranscriptEvent> {
        let mut out = Vec::new();
        let mut at = 0;
        for &size in sizes {
            out.extend(ctx.write_read(&audio[at..at + size]).unwrap());
            at += size;
        }
        assert_eq!(at, audio.len());
        out.extend(ctx.end_read().unwrap());
        out
    }

    fn tokens_of(emissions: &[TranscriptEvent]) -> Vec<Vec<i64>> {
        emissions
            .iter()
            .map(|e| e.segment().tokens.clone())
            .collect()
    }

    /// The whole clip through the front end in one call, joined with its
    /// tail: what `package_mels` and `decode_chunked` start from.
    fn joined_mels(
        driver: &WhisperStreamDriver,
        audio: &[f32],
        device: &Device,
    ) -> Tensor<3> {
        let n = audio.len();
        let waves: Tensor<2> = Tensor::from_data(
            TensorData::new(
                audio.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                [1, n],
            ),
            device,
        );
        let (mels, ctx) = driver
            .audio_converter()
            .new_context(1)
            .transform(waves)
            .unwrap();
        match ctx.finish() {
            Some(tail) => Tensor::cat(vec![mels, tail], 1),
            None => mels,
        }
    }

    fn greedy_config(driver: &WhisperStreamDriver) -> GreedyDecodeConfig {
        GreedyDecodeConfig::new(driver.prompt().to_vec(), driver.token_layout().ids().eot)
            .with_max_tokens(driver.config().max_tokens)
    }

    /// **I5, against the one-shot path.** One push of the whole clip, with
    /// the running-maximum policy fed everything before the first decode,
    /// is `decode_chunked` over `package_mels` &mdash; window for window,
    /// id for id.
    #[test]
    #[serial]
    fn test_single_push_matches_decode_chunked() {
        let device = performance_device();
        let driver = driver(&device, false);
        let audio = clip();

        let mut ctx = driver.new_context(clock(), LevelPolicy::Running).unwrap();
        let mut emissions = ctx.write_read(&audio).unwrap();
        emissions.extend(ctx.end_read().unwrap());

        let expected = driver.whisper_model().decode_chunked(
            driver
                .front_end()
                .package_mels(joined_mels(&driver, &audio, &device)),
            &greedy_config(&driver),
        );

        assert_eq!(expected.len(), 7, "6 full windows and a remainder");
        assert_eq!(tokens_of(&emissions), expected);
        assert!(emissions.iter().all(TranscriptEvent::is_committed));
        assert_eq!(
            ctx.transcript(),
            expected.concat(),
            "the transcript is the committed ids, in order",
        );
    }

    /// The chunking half of I5, stated as the equality caveat says it must
    /// be: the windows two streams decoded agree within tolerance always,
    /// their segments sit at the same times, and their ids agree whenever
    /// the windows are bit-identical.
    ///
    /// The front end's chunk-invariance is approximate on some backends (a
    /// matmul's accumulation order changes with the chunk), and an untrained
    /// model turns a last-digit difference into a flipped argmax. A trained
    /// model on speech does not; that is the validation crate's gate.
    fn assert_same_stream(
        a: &[TranscriptEvent],
        b: &[TranscriptEvent],
        label: &str,
    ) {
        assert_eq!(a.len(), b.len(), "{label}: emission count");

        for (x, y) in a.iter().zip(b) {
            assert_eq!(x, y);
            assert_eq!(x.segment().start, y.segment().start, "{label}: start");
            assert_eq!(x.segment().end, y.segment().end, "{label}: end");
        }
    }

    /// **I5, against chunking.** Random-sized pushes, most of them not hop
    /// multiples, decode the windows one push decodes &mdash; and one push
    /// decodes exactly what packaging each window by hand gives.
    #[test]
    #[serial]
    fn test_random_pushes_match_single_push() {
        let device = performance_device();
        let driver = driver(&device, false);
        let audio = clip();

        let mut whole = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        let mut expected = whole.write_read(&audio).unwrap();
        expected.extend(whole.end_read().unwrap());

        for seed in [1, 2, 3] {
            let mut ctx = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
            let got = push_in_pieces(&mut ctx, &audio, &random_sizes(seed, audio.len()));
            assert_same_stream(&got, &expected, &format!("seed {seed}"));
        }

        // The per-window one-shot path, by hand.
        let width = driver.window_frames();
        let trimmed = drop_last_frame(joined_mels(&driver, &audio, &device));
        let frames = trimmed.dims()[1];
        let config = greedy_config(&driver);
        let mut by_hand = Vec::new();
        let mut at = 0;
        while at < frames {
            let count = width.min(frames - at);
            let window = trimmed
                .clone()
                .slice_dim(1, at as isize..(at + count) as isize);
            let mut packaged = driver
                .front_end()
                .package_window(window.clone(), MaxLevel::of(window.clone()).max);
            if count < width {
                let pad = Tensor::zeros([1, packaged.dims()[1], width - count], &device);
                packaged = Tensor::cat(vec![packaged, pad], 2);
            }
            by_hand.push(driver.whisper_model().decode_window(packaged, &config));
            at += count;
        }
        assert_eq!(tokens_of(&expected), by_hand);
    }

    /// Segments sit on the clock: a window that starts at frame `f` starts
    /// at `f * hop / rate`, and the flushed remainder ends where the audio
    /// does. `push_at` moves all of it by anchoring.
    #[test]
    #[serial]
    fn test_segments_sit_on_the_clock() {
        let device = performance_device();
        let driver: WhisperStreamDriver = driver(&device, false);
        let audio = clip();
        let hop = driver.audio_converter().hop() as f64;
        let width = driver.window_frames();

        let mut ctx = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        let mut emissions = ctx.anchor_write_read(100.0, &audio).unwrap();
        emissions.extend(ctx.end_read().unwrap());

        let window_seconds = width as f64 * hop / driver.sample_rate() as f64;
        for (w, e) in emissions.iter().enumerate() {
            let segment = e.segment();
            assert!((segment.start - (100.0 + w as f64 * window_seconds)).abs() < 1e-9);
            if w + 1 < emissions.len() {
                assert!((segment.end - segment.start - window_seconds).abs() < 1e-9);
            }
        }

        // 16800 samples -> 104 frames streamed, +2 from finish, -1 trimmed:
        // 105, one per hop, and the remainder ends there.
        let last = emissions.last().unwrap().segment();
        assert_eq!(ctx.seek(), 105);
        assert!((last.end - (100.0 + 105.0 * hop / driver.sample_rate() as f64)).abs() < 1e-9);
        assert_eq!(ctx.pending_frames(), 0);
    }

    /// With carrying on, a window is prompted with the transcript's tail
    /// after `<|startofprev|>`, exactly as decoding it by hand with that
    /// prompt would.
    #[test]
    #[serial]
    fn test_prompt_carry() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let driver: WhisperStreamDriver = driver(&device, true);
        let audio = clip();
        let width = driver.window_frames();

        let mut ctx = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        let mut emissions = ctx.write_read(&audio).unwrap();
        emissions.extend(ctx.end_read().unwrap());
        let got = tokens_of(&emissions);

        // By hand: the first window with the bare prompt, the second with
        // the carry built from the first's ids.
        let trimmed = drop_last_frame(joined_mels(&driver, &audio, &device));
        let window_at = |w: usize| {
            let window = trimmed
                .clone()
                .slice_dim(1, (w * width) as isize..((w + 1) * width) as isize);
            driver
                .front_end()
                .package_window(window.clone(), MaxLevel::of(window).max)
        };
        let ids = driver.token_layout().ids();
        let first = driver
            .whisper_model()
            .decode_window(window_at(0), &greedy_config(&driver));
        assert_eq!(got[0], first);

        let keep = driver.whisper_model().max_text_ctx() / 2 - 1;
        let mut prompt = vec![ids.sot_prev];
        prompt.extend_from_slice(&first[first.len().saturating_sub(keep)..]);
        prompt.extend_from_slice(driver.prompt());
        let carried =
            GreedyDecodeConfig::new(prompt, ids.eot).with_max_tokens(driver.config().max_tokens);
        let second = driver.whisper_model().decode_window(window_at(1), &carried);
        assert_eq!(got[1], second);

        // Without carrying, the same audio prompts every window the same
        // way, so this is only a real test if the two differ somewhere.
        let bare = driver
            .whisper_model()
            .decode_window(window_at(1), &greedy_config(&driver));
        assert!(
            !first.is_empty() || second != bare,
            "the tiny model emitted nothing to carry",
        );
    }

    /// A detokenizer on the driver puts text on every segment.
    #[test]
    #[serial]
    fn test_text_through_the_detokenizer() {
        #[derive(Debug)]
        struct Numbers;
        impl Detokenizer for Numbers {
            fn detokenize(
                &self,
                ids: &[i64],
            ) -> BunsenResult<String> {
                Ok(ids.iter().map(i64::to_string).collect::<Vec<_>>().join(" "))
            }
        }

        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let driver: WhisperStreamDriver =
            driver(&device, false).with_detokenizer(Arc::new(Numbers));
        let audio = clip();

        let mut ctx = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        let mut emissions = ctx.write_read(&audio).unwrap();
        emissions.extend(ctx.end_read().unwrap());

        for e in &emissions {
            let segment = e.segment();
            let text_ids = driver.token_layout().text_ids(&segment.tokens);
            assert_eq!(
                segment.text.as_deref(),
                Some(Numbers.detokenize(&text_ids).unwrap().as_str())
            );
        }
    }

    /// The stream's lifecycle edges: nothing pushed, a stream shorter than a
    /// hop, flush twice, push after flush, feed and advance apart.
    #[test]
    #[serial]
    fn test_lifecycle_edges() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let driver: WhisperStreamDriver = driver(&device, false);

        let mut empty = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        assert!(empty.end_read().unwrap().is_empty());
        assert!(empty.end_read().unwrap().is_empty(), "flush is idempotent");
        // No pushing after flush, nor feeding.
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("the stream has ended")
            .assert_err(&empty.write_read(&[0.0; 16]));
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("the stream has ended")
            .assert_err(&empty.write(&[0.0; 16]));
        assert!(empty.is_finished());
        assert!(!empty.is_speaking());
        assert_eq!(empty.regions_pending(), 0);

        // Shorter than a hop: padded with silence to something decodable.
        let mut short = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        assert!(short.write_read(&[0.1; 50]).unwrap().is_empty());
        let emissions = short.end_read().unwrap();
        assert_eq!(emissions.len(), 1);
        assert!(short.pending_frames() == 0);

        // Feed then advance is push; end_input then advance is flush.
        let audio = clip();
        let mut split = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        split.write(&audio).unwrap();
        assert_eq!(split.transcript().len(), 0, "feeding decodes nothing");
        let mut got = split.read().unwrap();
        split.end_input().unwrap();
        got.extend(split.read().unwrap());

        let mut pushed = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        let mut expected = pushed.write_read(&audio).unwrap();
        expected.extend(pushed.end_read().unwrap());
        assert_eq!(got, expected);
    }

    /// **Batching is scheduling, nothing else.** Two streams advanced
    /// together commit exactly what each commits alone: same windows, same
    /// times, same ids &mdash; and a third stream with no audio is left
    /// alone.
    #[test]
    #[serial]
    fn test_advance_ready_matches_solo() {
        let device = performance_device();
        let driver = driver(&device, false).with_logit_filters(decisive());
        let clips = [clip_seeded(0.5, 440.0), clip_seeded(0.3, 660.0)];

        let mut solo = Vec::new();
        for audio in &clips {
            let mut ctx = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
            let mut emissions = ctx.write_read(audio).unwrap();
            emissions.extend(ctx.end_read().unwrap());
            solo.push(emissions);
        }
        if tokens_of(&solo[0]) == tokens_of(&solo[1]) {
            eprintln!(
                "the seeded tiny model says the same of both clips; scheduling is still pinned"
            );
        }

        let mut contexts: Vec<WhisperStreamContext> = (0..3)
            .map(|_| driver.new_context(clock(), LevelPolicy::PerWindow).unwrap())
            .collect();
        // The two clips go in whole, so the front end's arithmetic is the
        // solo run's; what differs is only that six windows per stream are
        // decoded twelve at a time. The third stream never gets audio.
        for (ctx, audio) in contexts.iter_mut().zip(&clips) {
            ctx.write(audio).unwrap();
        }
        let mut batched = advance_ready(&driver, &mut contexts).unwrap();
        assert_eq!(batched[0].len(), 6, "six full windows before the end");
        for ctx in contexts.iter_mut().take(2) {
            ctx.end_input().unwrap();
        }
        let tail = advance_ready(&driver, &mut contexts).unwrap();
        for (all, more) in batched.iter_mut().zip(tail) {
            all.extend(more);
        }

        assert_eq!(batched[0], solo[0]);
        assert_eq!(batched[1], solo[1]);
        assert!(batched[2].is_empty());
        assert_eq!(contexts[0].seek(), 105);
    }

    /// Batching is by prompt: a stream that has already committed a window
    /// carries a different prompt from one that has not, so the two form
    /// two groups in one pass, and each decodes as it would alone. The
    /// decode is dictated from the prompt, so a window batched under the
    /// wrong prompt would show.
    #[test]
    #[serial]
    fn test_advance_ready_with_prompt_carry() {
        let device = performance_device();
        let driver = driver(&device, true).with_logit_filters(decisive());
        let audio = clip();
        // 4000 samples: one 16-frame window and change.
        let first = RATE / 4;

        // Solo: A decodes its first window early, then the rest; B all at
        // once.
        let mut a = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        let mut solo_a = a.write_read(&audio[..first]).unwrap();
        assert_eq!(solo_a.len(), 1, "one window committed early");
        solo_a.extend(a.write_read(&audio[first..]).unwrap());
        solo_a.extend(a.end_read().unwrap());
        let mut b = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        let mut solo_b = b.write_read(&audio).unwrap();
        solo_b.extend(b.end_read().unwrap());
        assert_ne!(
            solo_a[0].segment().tokens,
            solo_a[1].segment().tokens,
            "the carried prompt changes what is decoded"
        );

        // Batched: A has committed its first window, B nothing; fed the
        // rest, they advance together under two prompts.
        let mut ctx_a = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        let mut batched_a = ctx_a.write_read(&audio[..first]).unwrap();
        ctx_a.write(&audio[first..]).unwrap();
        ctx_a.end_input().unwrap();
        let mut ctx_b = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        ctx_b.write(&audio).unwrap();
        ctx_b.end_input().unwrap();
        let mut contexts = vec![ctx_a, ctx_b];
        let out = advance_ready(&driver, &mut contexts).unwrap();
        batched_a.extend(out[0].clone());
        assert_eq!(batched_a, solo_a);
        assert_eq!(out[1], solo_b);
    }

    /// Dictates the decode: whatever the untrained model thinks (its
    /// near-ties flip between runs and backends), the next token is the
    /// scripted one for its position in the window. The script obeys the
    /// timestamp grammar, so the rules, applied after it, pass it through.
    #[derive(Debug)]
    struct Script(Vec<i64>);

    impl LogitFilter for Script {
        fn apply(
            &self,
            logits: Tensor<2>,
            tokens: &[Vec<i64>],
            prompt_len: usize,
        ) -> Tensor<2> {
            let [rows, vocab] = logits.dims();
            let mut data = vec![f32::NEG_INFINITY; rows * vocab];
            for (row, t) in tokens.iter().enumerate() {
                let at = (t.len() - prompt_len).min(self.0.len() - 1);
                data[row * vocab + self.0[at] as usize] = 0.0;
            }
            Tensor::from_data(
                burn::tensor::TensorData::new(data, [rows, vocab]),
                &logits.device(),
            )
        }
    }

    /// Dictates the decode from the prompt: at sampled position `at` the
    /// token is `(prompt_len + at) % 4 + 1`, so streams under different
    /// prompts decode differently, and a window batched under the wrong
    /// prompt would show. Independent of the model, whose forward on this
    /// backend varies from call to call by more than a near-tie.
    #[derive(Debug)]
    struct Echo;

    impl LogitFilter for Echo {
        fn apply(
            &self,
            logits: Tensor<2>,
            tokens: &[Vec<i64>],
            prompt_len: usize,
        ) -> Tensor<2> {
            let [rows, vocab] = logits.dims();
            let mut data = vec![f32::NEG_INFINITY; rows * vocab];
            for (row, t) in tokens.iter().enumerate() {
                let at = t.len() - prompt_len;
                data[row * vocab + (prompt_len + at) % 4 + 1] = 0.0;
            }
            Tensor::from_data(
                burn::tensor::TensorData::new(data, [rows, vocab]),
                &logits.device(),
            )
        }
    }

    /// The filters that make a plumbing test's decode a function of its
    /// prompt alone.
    fn decisive() -> Vec<Arc<dyn LogitFilter>> {
        vec![Arc::new(Echo)]
    }

    /// Each window's script: two closed segments and a reopened tail,
    /// `<|1|> a <|3|><|3|> b <|5|><|5|> c`, at the cap of 8 tokens. The
    /// first timestamp is within the cap of index 2, so the cap is a real
    /// constraint that the script satisfies; the seek advances 10 frames
    /// of 16 per decode, so windows overlap.
    fn script() -> Vec<i64> {
        let tb = tiny_layout().timestamp_begin;
        vec![tb + 1, 1, tb + 3, tb + 3, 2, tb + 5, tb + 5, 3]
    }

    /// A driver that emits timestamps under the script. No prompt carry:
    /// the tiny model's text context is 16, and a carried prompt would
    /// leave the script no room.
    fn timestamped(
        device: &Device,
        commit: CommitRule,
    ) -> WhisperStreamDriver {
        let scripted: Arc<dyn LogitFilter> = Arc::new(Script(script()));
        config(false)
            .with_max_tokens(8)
            .with_timestamps(true)
            .with_max_initial_timestamp(Some(0.04))
            .with_emission(EmissionPolicy::new(DecodeTriggers::new(), commit))
            .init_with_layout(
                tiny_model_on(device),
                WhisperTokenLayout::new(tiny_layout()),
                device,
            )
            .unwrap()
            .with_logit_filters(vec![scripted])
    }

    /// Pushes a clip through a fresh context and flushes it.
    fn run_clip(
        driver: &WhisperStreamDriver,
        audio: &[f32],
        at: Option<f64>,
    ) -> Vec<TranscriptEvent> {
        let mut ctx = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        let mut emissions = match at {
            Some(time) => ctx.anchor_write_read(time, audio).unwrap(),
            None => ctx.write_read(audio).unwrap(),
        };
        emissions.extend(ctx.end_read().unwrap());
        assert_eq!(ctx.pending_frames(), 0, "a flush consumes everything");
        assert_eq!(ctx.seek(), 105);
        let committed: Vec<i64> = emissions
            .iter()
            .filter(|e| e.is_committed())
            .flat_map(|e| e.segment().tokens.clone())
            .collect();
        assert_eq!(
            ctx.transcript(),
            &committed[..],
            "the transcript is what was committed"
        );
        emissions
    }

    /// With timestamps on, a window's decode is split on its consecutive
    /// timestamps into segments on the clock, and the seek advances to the
    /// last closed timestamp rather than a whole window: 105 frames at 10
    /// per decode is 11 decodes, 22 segments, each 4 frames long.
    #[test]
    #[serial]
    fn test_timestamps_split_windows_into_segments() {
        let device = performance_device();
        let driver = timestamped(&device, CommitRule::Complete);
        let tb = tiny_layout().timestamp_begin;
        let hop = driver.audio_converter().hop() as f64;
        let frame = |f: f64| f * hop / driver.sample_rate() as f64;
        assert_eq!(driver.frames_per_timestamp(), 2);
        assert_eq!(
            driver.filters().len(),
            2,
            "the test's filter, then the rules appended by the driver"
        );

        let emissions = run_clip(&driver, &clip(), None);
        assert_eq!(emissions.len(), 22);

        for (k, e) in emissions.iter().enumerate() {
            assert!(e.is_committed(), "Complete never drafts");
            let s = e.segment();
            let (window, which) = (k / 2, k % 2);
            let seek = (window * 10) as f64;
            let expected = if which == 0 {
                (vec![tb + 1, 1, tb + 3], seek + 2.0, seek + 6.0)
            } else {
                (vec![tb + 3, 2, tb + 5], seek + 6.0, seek + 10.0)
            };
            assert_eq!(s.tokens, expected.0, "segment {k}");
            assert!(
                (s.start - frame(expected.1)).abs() < 1e-9,
                "segment {k}: {s:?}"
            );
            assert!(
                (s.end - frame(expected.2)).abs() < 1e-9,
                "segment {k}: {s:?}"
            );
        }
    }

    /// **I8.** The same clip on a clock anchored at 100 s yields the same
    /// segments, shifted by 100 s.
    #[test]
    #[serial]
    fn test_segment_times_are_invariant_to_clock_origin() {
        let device = performance_device();
        let driver = timestamped(&device, CommitRule::Complete);
        let audio = clip();

        let base = run_clip(&driver, &audio, None);
        let shifted = run_clip(&driver, &audio, Some(100.0));
        assert_eq!(base.len(), shifted.len());
        for (a, b) in base.iter().zip(&shifted) {
            assert_eq!(a.segment().tokens, b.segment().tokens);
            assert!((b.segment().start - a.segment().start - 100.0).abs() < 1e-9);
            assert!((b.segment().end - a.segment().end - 100.0).abs() < 1e-9);
        }
    }

    /// Under `LastTimestamp` the unfinished tail of a window is emitted as
    /// a draft opening on its timestamp, and what is committed is exactly
    /// what `Complete` commits.
    #[test]
    #[serial]
    fn test_last_timestamp_drafts_the_tail() {
        let device = performance_device();
        let audio = clip();
        let complete = run_clip(&timestamped(&device, CommitRule::Complete), &audio, None);
        let last = run_clip(
            &timestamped(&device, CommitRule::LastTimestamp),
            &audio,
            None,
        );

        let committed: Vec<&TranscriptEvent> = last.iter().filter(|e| e.is_committed()).collect();
        assert_eq!(committed.len(), complete.len());
        for (a, b) in committed.iter().zip(&complete) {
            assert_eq!(a.segment(), b.segment());
        }

        let tb = tiny_layout().timestamp_begin;
        let drafts: Vec<&TranscriptEvent> = last.iter().filter(|e| !e.is_committed()).collect();
        // One per decode, except the last: its 5 frames end before the
        // tail's timestamp at frame 10, so there is nothing to draft yet.
        assert_eq!(drafts.len(), 10);
        for draft in drafts {
            let s = draft.segment();
            assert_eq!(s.tokens, vec![tb + 5, 3], "the reopened tail");
            assert!(s.start <= s.end);
        }
    }

    /// Degenerate at temperature zero, sound once sampling has broken the
    /// loop: while the sampled history is all `1`s (or empty) the logits
    /// are nearly flat with `1` a hair ahead, so the argmax loops on `1` at
    /// a log probability near `-log(vocab)`; once any other token appears,
    /// which only sampling can bring, the logits peak on `2`.
    #[derive(Debug)]
    struct Degenerate;

    impl LogitFilter for Degenerate {
        fn apply(
            &self,
            logits: Tensor<2>,
            tokens: &[Vec<i64>],
            prompt_len: usize,
        ) -> Tensor<2> {
            let [rows, vocab] = logits.dims();
            let mut data = vec![-0.01f32; rows * vocab];
            for (row, t) in tokens.iter().enumerate() {
                let at = row * vocab;
                if t[prompt_len..].iter().all(|&x| x == 1) {
                    data[at + 1] = 0.0;
                } else {
                    data[at..at + vocab].fill(0.0);
                    data[at + 2] = 30.0;
                }
            }
            Tensor::from_data(
                burn::tensor::TensorData::new(data, [rows, vocab]),
                &logits.device(),
            )
        }
    }

    /// A carrying driver under the degenerate filter and a fallback
    /// policy.
    fn degenerate_driver(
        device: &Device,
        fallback: WhisperFallbackConfig,
    ) -> WhisperStreamDriver {
        let filter: Arc<dyn LogitFilter> = Arc::new(Degenerate);
        config(true)
            .with_max_tokens(8)
            .with_fallback(fallback)
            .init_with_layout(
                tiny_model_on(device),
                WhisperTokenLayout::new(tiny_layout()),
                device,
            )
            .unwrap()
            .with_logit_filters(vec![filter])
    }

    /// The ladder: at temperature zero the decode loops and fails the log
    /// probability threshold; without a ladder that is what is emitted.
    /// With upstream's ladder, sampling at the first rung above zero breaks
    /// the loop and the decode recovers, below the prompt-reset line, so
    /// the carry stands. A ladder whose only rung above zero is past that
    /// line recovers too, and resets the carry.
    #[test]
    #[serial]
    fn test_fallback_ladder_recovers() {
        let device = performance_device();
        let audio = clip();
        let flat = degenerate_driver(&device, WhisperFallbackConfig::new());
        let mut ctx = flat.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        let mut looped = ctx.write_read(&audio).unwrap();
        looped.extend(ctx.end_read().unwrap());
        assert!(!looped.is_empty());
        // Eight 1s in the first window; fewer after, once the carried
        // prompt has taken its share of the tiny model's 16-token context.
        assert!(
            looped
                .iter()
                .all(|e| !e.segment().tokens.is_empty()
                    && e.segment().tokens.iter().all(|&t| t == 1)),
            "no ladder: the loop is what there is: {looped:?}"
        );
        let carry_len = ctx.prompt_now().len();
        assert!(carry_len > flat.prompt().len(), "the carry stands");

        let climbing = degenerate_driver(&device, WhisperFallbackConfig::upstream());
        let mut ctx = climbing
            .new_context(clock(), LevelPolicy::PerWindow)
            .unwrap();
        let mut recovered = ctx.write_read(&audio).unwrap();
        recovered.extend(ctx.end_read().unwrap());
        assert_eq!(recovered.len(), looped.len());
        for e in &recovered {
            let t = &e.segment().tokens;
            assert!(t.contains(&2), "sampling broke the loop: {t:?}");
        }
        assert!(
            ctx.prompt_now().len() > climbing.prompt().len(),
            "recovered below 0.5: the carry stands"
        );

        let hot = degenerate_driver(
            &device,
            WhisperFallbackConfig::new().with_temperatures(vec![0.0, 0.8]),
        );
        let mut ctx = hot.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        let mut reset = ctx.write_read(&audio).unwrap();
        reset.extend(ctx.end_read().unwrap());
        assert!(reset.iter().all(|e| e.segment().tokens.contains(&2)));
        assert_eq!(
            ctx.prompt_now(),
            hot.prompt(),
            "past 0.5: the carry was reset"
        );
    }

    /// The silence rule: a window whose no-speech probability is over the
    /// threshold and whose log probability is poor is skipped whole, the
    /// seek moving past it with nothing emitted; a good enough log
    /// probability keeps it.
    #[test]
    #[serial]
    fn test_no_speech_skips_a_window() {
        let device = performance_device();
        let audio = clip();

        let skipping = degenerate_driver(
            &device,
            WhisperFallbackConfig::new().with_no_speech_threshold(Some(0.0)),
        );
        let mut ctx = skipping
            .new_context(clock(), LevelPolicy::PerWindow)
            .unwrap();
        let mut emissions = ctx.write_read(&audio).unwrap();
        emissions.extend(ctx.end_read().unwrap());
        assert!(
            emissions.is_empty(),
            "every window is silence: {emissions:?}"
        );
        assert_eq!(ctx.seek(), 105, "the seek still moved past them");
        assert!(ctx.transcript().is_empty());

        let kept = degenerate_driver(
            &device,
            WhisperFallbackConfig::new()
                .with_no_speech_threshold(Some(0.0))
                .with_logprob_threshold(Some(-1e9)),
        );
        let mut ctx = kept.new_context(clock(), LevelPolicy::PerWindow).unwrap();
        let mut emissions = ctx.write_read(&audio).unwrap();
        emissions.extend(ctx.end_read().unwrap());
        assert!(
            !emissions.is_empty(),
            "a good enough log probability keeps a window"
        );
    }

    /// A multilingual driver given no language detects it from the first
    /// window: on a layout with one language, that one, and the stream then
    /// decodes exactly as one configured with it. Solo and batched agree.
    #[test]
    #[serial]
    fn test_language_is_detected_per_stream() {
        let device = performance_device();
        let detecting = WhisperStreamDriverConfig::new()
            .with_max_tokens(4)
            .init_with_layout(
                tiny_model_on(&device),
                WhisperTokenLayout::new(tiny_layout()),
                &device,
            )
            .unwrap()
            .with_logit_filters(decisive());
        assert!(detecting.detects_language());
        assert!(detecting.prompt().is_empty());
        let only = crate::kits::speech::whisper::blocks::LANGUAGES[0];
        assert_eq!(only, "en");

        let audio = clip();
        let mut solo = detecting
            .new_context(clock(), LevelPolicy::PerWindow)
            .unwrap();
        assert_eq!(solo.language(), None);
        let mut expected = solo.write_read(&audio).unwrap();
        expected.extend(solo.end_read().unwrap());
        assert_eq!(solo.language(), Some(only));

        // The same as configuring it.
        let configured = driver(&device, true).with_logit_filters(decisive());
        assert_eq!(run_clip(&configured, &audio, None), expected);

        // Batched: fed, then advanced together.
        let mut batch = vec![
            detecting
                .new_context(clock(), LevelPolicy::PerWindow)
                .unwrap(),
            detecting
                .new_context(clock(), LevelPolicy::PerWindow)
                .unwrap(),
        ];
        for ctx in &mut batch {
            ctx.write(&audio).unwrap();
            ctx.end_input().unwrap();
        }
        let out = advance_ready(&detecting, &mut batch).unwrap();
        assert_eq!(out[0], expected);
        assert_eq!(out[1], expected);
        assert_eq!(batch[1].language(), Some(only));
    }

    /// The driver reports the model's rate, and the grid derived from it.
    #[test]
    fn test_driver_reports_the_models_rate() {
        let device = performance_device();
        let driver = WhisperStreamDriverConfig::new()
            .init_with_layout(
                tiny_model_on(&device),
                WhisperTokenLayout::new(tiny_layout()),
                &device,
            )
            .unwrap();
        assert_eq!(driver.sample_rate(), 16_000);
        assert_eq!(driver.audio_converter().hop(), 160);
        assert_eq!(driver.frames_per_timestamp(), 2);
        assert_eq!(driver.encoder_grid(), 320);
        assert_eq!(driver.interval_samples(), None);
    }

    /// `CommitRule::Agreement` is not implemented, so the driver refuses it
    /// at construction, as unsupported.
    #[test]
    fn test_init_refuses_agreement() {
        let device = performance_device();
        let refused = config(false)
            .with_emission(EmissionPolicy::new(
                DecodeTriggers::new(),
                CommitRule::Agreement { runs: 2 },
            ))
            .init_with_layout(
                tiny_model_on(&device),
                WhisperTokenLayout::new(tiny_layout()),
                &device,
            );
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("not implemented")
            .assert_err(&refused);
    }

    /// The `interval` trigger drafts only while the voice-activity gate says
    /// speech is in progress, and the gate runs only under `endpoint`:
    /// without it no draft would ever be made, so the driver refuses the
    /// pairing at construction.
    #[test]
    fn test_init_refuses_interval_without_endpoint() {
        let device = performance_device();
        let refused = config(false)
            .with_emission(EmissionPolicy::new(
                DecodeTriggers::new().with_interval(Some(std::time::Duration::from_millis(50))),
                CommitRule::LastTimestamp,
            ))
            .init_with_layout(
                tiny_model_on(&device),
                WhisperTokenLayout::new(tiny_layout()),
                &device,
            );
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("endpoint")
            .assert_err(&refused);
    }

    /// The configuration refuses a broken rule and an unknown language, and a
    /// stream refuses a missing voice-activity model and a clock at the
    /// wrong rate, each with the kind it is.
    #[test]
    #[serial]
    fn test_init_and_new_context_refuse_bad_settings() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let policy = WhisperTokenLayout::new(tiny_layout());
        let base = WhisperStreamDriverConfig::new().with_language(Some("en".to_string()));

        assert!(
            base.init_with_layout(tiny_model_on(&device), policy.clone(), &device)
                .is_ok()
        );
        // Multilingual without a language detects it per stream.
        assert!(
            WhisperStreamDriverConfig::new()
                .init_with_layout(tiny_model_on(&device), policy.clone(), &device)
                .unwrap()
                .detects_language()
        );
        assert!(
            base.clone()
                .with_timestamps(true)
                .init_with_layout(tiny_model_on(&device), policy.clone(), &device)
                .is_ok()
        );
        assert!(
            base.clone()
                .with_emission(EmissionPolicy::responsive())
                .init_with_layout(tiny_model_on(&device), policy.clone(), &device)
                .is_ok(),
            "responsive is the third deployment target",
        );
        // An interval of zero.
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .has_cause::<ConstraintError>()
            .assert_err(
                &base
                    .clone()
                    .with_emission(EmissionPolicy::new(
                        DecodeTriggers::new()
                            .with_endpoint(true)
                            .with_interval(Some(std::time::Duration::ZERO)),
                        CommitRule::Complete,
                    ))
                    .init_with_layout(tiny_model_on(&device), policy.clone(), &device),
            );
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .has_cause::<LookupError>()
            .assert_err(
                &base
                    .clone()
                    .with_language(Some("xx".to_string()))
                    .init_with_layout(tiny_model_on(&device), policy.clone(), &device),
            );

        // Conservative is constructible, but a stream under it needs a VAD.
        let conservative = base
            .clone()
            .with_emission(EmissionPolicy::conservative())
            .init_with_layout(tiny_model_on(&device), policy, &device)
            .unwrap();
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("voice-activity model")
            .assert_err(&conservative.new_context(clock(), LevelPolicy::PerWindow));

        // A clock at the wrong rate is refused at the stream, not later.
        let driver: WhisperStreamDriver = driver(&device, false);
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .has_cause::<ConstraintError>()
            .assert_err(&driver.new_context(StreamClock::uniform(8_000), LevelPolicy::PerWindow));
    }

    /// Against the real voice-activity model: regions become segments in
    /// the parent stream's frame of reference, and silence is never
    /// decoded.
    #[cfg(feature = "silero-weights")]
    mod voice_activity {
        use burn::module::Module;

        use super::*;
        use crate::{
            burner::{
                module::DTypeMapper,
                tensor::device_float_dtype,
            },
            kits::speech::{
                asr_driver::{
                    TranscriptSegment,
                    VoiceActivityFilterConfig,
                },
                silero_vad::SileroVad,
            },
            support::{
                audio::load_audio_mono_sr,
                testing::cpu_device,
            },
        };

        /// Silero and the tiny model on the CPU backend, so the golden
        /// regions from the gate's own test hold here too.

        fn speech() -> Vec<f32> {
            let path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/testdata/audio/jfk_moon_4s.mp3"
            );
            load_audio_mono_sr(path, RATE).unwrap()
        }

        fn conservative_driver(device: &Device) -> WhisperStreamDriver {
            let vad = SileroVad::load_16khz_pretrained(device).unwrap();
            config(false)
                .with_emission(EmissionPolicy::conservative())
                .init_with_layout(
                    tiny_model_on(device),
                    WhisperTokenLayout::new(tiny_layout()),
                    device,
                )
                .unwrap()
                .with_vad(
                    Arc::new(vad),
                    VoiceActivityFilterConfig::fast_whisper_burn(),
                )
                .unwrap()
        }

        /// A model or a filter at another rate, or a filter with the wrong
        /// chunk, is refused at attach time; the matching pair is not.
        #[test]
        #[serial]
        fn test_with_vad_refuses_a_mismatch() {
            let device = cpu_device();
            let driver = || {
                config(false)
                    .with_emission(EmissionPolicy::conservative())
                    .init_with_layout(
                        tiny_model_on(&device),
                        WhisperTokenLayout::new(tiny_layout()),
                        &device,
                    )
                    .unwrap()
            };
            let filter = VoiceActivityFilterConfig::fast_whisper_burn;

            let mismatch =
                ErrorMatcher::kind(BunsenErrorKind::Illegal).has_cause::<ConstraintError>();
            // The rate check reads the model's geometry, not its weights.
            let eight: SileroVad = crate::burner::module::ModuleInit::init(
                &crate::kits::speech::silero_vad::SileroVadSignalConfig::standard_8khz(),
                &device,
            );
            mismatch.assert_err(&driver().with_vad(Arc::new(eight), filter()));

            let vad = SileroVad::load_16khz_pretrained(&device).unwrap();
            mismatch.assert_err(
                &driver().with_vad(Arc::new(vad.clone()), filter().with_sample_rate(8_000)),
            );
            mismatch.assert_err(
                &driver().with_vad(Arc::new(vad.clone()), filter().with_samples_per_chunk(256)),
            );

            let attached = driver().with_vad(Arc::new(vad), filter()).unwrap();
            assert_eq!(
                attached.va_filter_config().map(|f| f.samples_per_chunk),
                Some(512)
            );
        }

        /// The gate's golden regions, padded and snapped outward, are
        /// exactly what gets decoded: the segments tile each region, a
        /// window at a time, nothing lands in the pause between them, and
        /// their times are the parent clock's, anchor and all. Pushed in
        /// random pieces, so the VAD's own staging is exercised.
        #[test]
        #[serial]
        fn test_regions_become_segments_on_the_parent_clock() {
            let device = cpu_device();
            let vad = SileroVad::load_16khz_pretrained(&device).unwrap();
            let regions_only = EmissionPolicy::new(
                DecodeTriggers::new()
                    .with_window_full(false)
                    .with_endpoint(true),
                CommitRule::LastTimestamp,
            );
            let driver = config(false)
                .with_emission(regions_only)
                .init_with_layout(
                    tiny_model_on(&device),
                    WhisperTokenLayout::new(tiny_layout()),
                    &device,
                )
                .unwrap()
                .with_vad(
                    Arc::new(vad),
                    VoiceActivityFilterConfig::fast_whisper_burn(),
                )
                .unwrap();
            let audio = speech();
            assert_eq!(audio.len(), 64_000);

            let mut ctx = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
            let sizes = random_sizes(11, audio.len());
            let mut emissions = Vec::new();
            let mut at = 0;
            for (k, &size) in sizes.iter().enumerate() {
                let piece = &audio[at..at + size];
                emissions.extend(if k == 0 {
                    ctx.anchor_write_read(100.0, piece).unwrap()
                } else {
                    ctx.write_read(piece).unwrap()
                });
                at += size;
            }
            emissions.extend(ctx.end_read().unwrap());

            // (0, 16864) and (29728, 57312) from the gate's golden test,
            // snapped outward onto the 320-sample grid.
            let expected = [(0usize, 16_960usize), (29_440, 57_600)];
            let window = driver.window_frames() as f64 * driver.audio_converter().hop() as f64
                / driver.sample_rate() as f64;
            let close = |a: f64, b: f64| (a - b).abs() < 1e-9;

            let mut segments = emissions.iter().map(TranscriptEvent::segment).peekable();
            for (start, end) in expected {
                let (start, end) = (
                    100.0 + start as f64 / RATE as f64,
                    100.0 + end as f64 / RATE as f64,
                );
                let mut at = start;
                while let Some(segment) = segments.next_if(|s| s.start < end) {
                    assert!(close(segment.start, at), "gap before {segment:?}");
                    assert!(segment.end - segment.start <= window + 1e-9);
                    assert!(segment.end <= end + 1e-9, "past the region: {segment:?}");
                    at = segment.end;
                }
                assert!(
                    close(at, end),
                    "region not covered to {end}: stopped at {at}\n{emissions:#?}"
                );
            }
            assert!(segments.next().is_none(), "segments outside every region");
            assert!(emissions.iter().all(TranscriptEvent::is_committed));
            assert_eq!(ctx.regions_pending(), 0);
            assert!(!ctx.is_speaking());
            assert_eq!(
                emissions.len(),
                (106_usize).div_ceil(16) + (176_usize).div_ceil(16),
                "a window at a time over 106 and 176 frames",
            );
        }

        /// Under `conservative()` the full-window trigger runs too: a
        /// window with speech in progress decodes as it fills, so the window
        /// that opens a region may begin a little before it; but nothing is
        /// ever decoded from the pause, and both regions are covered.
        #[test]
        #[serial]
        fn test_conservative_skips_silence() {
            let device = cpu_device();
            let driver = conservative_driver(&device);
            let audio = speech();

            let mut ctx = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
            let mut emissions = ctx.write_read(&audio).unwrap();
            emissions.extend(ctx.end_read().unwrap());

            let window = driver.window_frames() as f64 * driver.audio_converter().hop() as f64
                / driver.sample_rate() as f64;
            let (gap_start, gap_end) = (16_960.0 / RATE as f64, 29_440.0 / RATE as f64);
            let mut covered_to = 0.0_f64;
            for e in &emissions {
                let s = e.segment();
                assert!(
                    !(s.start >= gap_start && s.end <= gap_end),
                    "a window of silence was decoded: {s:?}",
                );
                assert!(s.start >= covered_to - 1e-9, "out of order: {s:?}");
                if s.start > gap_start + 1e-9 {
                    assert!(
                        s.start >= gap_end - window - 1e-9,
                        "started deep in the pause: {s:?}"
                    );
                }
                covered_to = s.end;
            }
            assert!(
                (covered_to - 57_600.0 / RATE as f64).abs() < 1e-9,
                "covered to {covered_to}\n{emissions:#?}"
            );
            assert_eq!(ctx.regions_pending(), 0);
        }

        /// A responsive driver: conservative plus a draft every 50 ms of
        /// media time while speech is in progress &mdash; shorter than
        /// `responsive()`'s 600 ms because the tiny model's window is 0.16 s
        /// of audio, and a full window commits before a longer interval
        /// could draft. Its decode is scripted so the pin is about
        /// scheduling, not the untrained model.
        fn responsive_driver(device: &Device) -> WhisperStreamDriver {
            let vad = SileroVad::load_16khz_pretrained(device).unwrap();
            let scripted: Arc<dyn LogitFilter> = Arc::new(Script(vec![3, 1, 4, 1]));
            let policy = EmissionPolicy::new(
                DecodeTriggers::new()
                    .with_endpoint(true)
                    .with_interval(Some(std::time::Duration::from_millis(50))),
                CommitRule::LastTimestamp,
            );
            config(false)
                .with_emission(policy)
                .init_with_layout(
                    tiny_model_on(device),
                    WhisperTokenLayout::new(tiny_layout()),
                    device,
                )
                .unwrap()
                .with_vad(
                    Arc::new(vad),
                    VoiceActivityFilterConfig::fast_whisper_burn(),
                )
                .unwrap()
                .with_logit_filters(vec![scripted])
        }

        /// The clip pushed 100 ms at a time, then flushed.
        fn in_pieces(ctx: &mut WhisperStreamContext) -> Vec<TranscriptEvent> {
            let audio = speech();
            let mut out = Vec::new();
            for piece in audio.chunks(RATE / 10) {
                out.extend(ctx.write_read(piece).unwrap());
            }
            out.extend(ctx.end_read().unwrap());
            out
        }

        /// **I9.** Adding the interval trigger cannot change the transcript:
        /// `responsive()` commits exactly what `conservative()` commits, and
        /// drafts besides: each covering the audio since the last commit,
        /// paced at least an interval apart, none after the last commit.
        #[test]
        #[serial]
        fn test_responsive_commits_what_conservative_commits() {
            let device = cpu_device();
            let scripted: Arc<dyn LogitFilter> = Arc::new(Script(vec![3, 1, 4, 1]));
            let conservative = conservative_driver(&device).with_logit_filters(vec![scripted]);
            let responsive = responsive_driver(&device);

            let mut ctx = conservative
                .new_context(clock(), LevelPolicy::PerWindow)
                .unwrap();
            let quiet = in_pieces(&mut ctx);
            assert!(
                quiet.iter().all(|e| e.is_committed()),
                "conservative never drafts"
            );

            let mut ctx = responsive
                .new_context(clock(), LevelPolicy::PerWindow)
                .unwrap();
            let chatty = in_pieces(&mut ctx);
            let committed: Vec<&TranscriptEvent> =
                chatty.iter().filter(|e| e.is_committed()).collect();
            assert_eq!(committed.len(), quiet.len());
            for (a, b) in committed.iter().zip(&quiet) {
                assert_eq!(a.segment(), b.segment());
            }

            let drafts: Vec<&TranscriptSegment> = chatty
                .iter()
                .filter(|e| !e.is_committed())
                .map(|e| e.segment())
                .collect();
            assert!(drafts.len() >= 3, "{} drafts", drafts.len());
            let interval = 0.05;
            let mut last_end = f64::NEG_INFINITY;
            let mut last_commit_end = 0.0;
            let mut seen_commits = 0;
            for e in &chatty {
                let s = e.segment();
                if e.is_committed() {
                    last_commit_end = s.end;
                    seen_commits += 1;
                    continue;
                }
                assert!(
                    s.start >= last_commit_end - 1e-9,
                    "a draft starts at the seek: {s:?}"
                );
                assert!(s.end > s.start, "{s:?}");
                assert!(
                    s.end - last_end >= interval - 1e-9,
                    "paced: {s:?} after {last_end}"
                );
                assert_eq!(s.tokens, vec![3, 1, 4, 1]);
                last_end = s.end;
            }
            assert!(seen_commits > 0);
            assert!(
                chatty.last().unwrap().is_committed(),
                "the flush commits everything; nothing is left to draft"
            );
        }

        /// Batched, a responsive stream drafts and commits exactly as it
        /// does alone: fed a piece at a time and advanced after each.
        #[test]
        #[serial]
        fn test_advance_ready_drafts_like_solo() {
            let device = cpu_device();
            let driver = responsive_driver(&device);
            let audio = speech();

            let mut solo = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
            let mut batch = vec![driver.new_context(clock(), LevelPolicy::PerWindow).unwrap()];
            let (mut expected, mut got) = (Vec::new(), Vec::new());
            for piece in audio.chunks(RATE / 10) {
                expected.extend(solo.write_read(piece).unwrap());
                batch[0].write(piece).unwrap();
                got.extend(advance_ready(&driver, &mut batch).unwrap().remove(0));
            }
            expected.extend(solo.end_read().unwrap());
            batch[0].end_input().unwrap();
            got.extend(advance_ready(&driver, &mut batch).unwrap().remove(0));
            assert_eq!(got, expected);
            assert!(
                expected.iter().any(|e| !e.is_committed()),
                "there were drafts"
            );
        }

        /// Dictates `1 + i` at every step, `i` the index of the language the
        /// prompt names (`0` without one), so each decode shows the language
        /// it ran under.
        #[derive(Debug)]
        struct ByLanguage(WhisperSpecialIds);

        impl LogitFilter for ByLanguage {
            fn apply(
                &self,
                logits: Tensor<2>,
                tokens: &[Vec<i64>],
                prompt_len: usize,
            ) -> Tensor<2> {
                let [rows, vocab] = logits.dims();
                let mut data = vec![f32::NEG_INFINITY; rows * vocab];
                let languages =
                    self.0.language_begin..self.0.language_begin + self.0.num_languages as i64;
                for (row, t) in tokens.iter().enumerate() {
                    let id = t[..prompt_len]
                        .iter()
                        .find(|id| languages.contains(id))
                        .map_or(0, |id| id - languages.start + 1);
                    data[row * vocab + id as usize] = 0.0;
                }
                Tensor::from_data(TensorData::new(data, [rows, vocab]), &logits.device())
            }
        }

        /// A driver that detects the language, on a two-language layout,
        /// whose detection turns on the audio. The untrained model's barely
        /// does: its encoder output is mostly positional embedding, and its
        /// first decoder step mostly the token's own. So the encoder's
        /// positional embedding is zeroed and the decoder's cross-attention
        /// output scaled up 300 times. Seeded so that the first draft's
        /// frames and the first committed window's detect different
        /// languages.
        fn detecting_driver(
            device: &Device,
            policy: EmissionPolicy,
        ) -> WhisperStreamDriver {
            let ids = WhisperSpecialIds::new(5, 2).unwrap();
            device.seed(1);
            // Materialized now, so that every call builds the same weights.
            let mut model: Whisper = WhisperApiConfig::new(8, ids.n_vocab(), 64, 16, 1, 16, 1)
                .init(device)
                .map(&mut DTypeMapper::new(device_float_dtype(&device)));
            let positions = &mut model.encoder.positional_embedding;
            *positions = positions.clone().map(|p| p.zeros_like());
            let cross = &mut model.decoder.blocks[0].cross_attn.output;
            cross.weight = cross.weight.clone().map(|w| w * 300.0);

            let vad = SileroVad::load_16khz_pretrained(device).unwrap();
            let scripted: Arc<dyn LogitFilter> = Arc::new(ByLanguage(ids));
            WhisperStreamDriverConfig::new()
                .with_max_tokens(4)
                .with_condition_on_previous_text(false)
                .with_emission(policy)
                .init_with_layout(model, WhisperTokenLayout::new(ids), device)
                .unwrap()
                .with_vad(
                    Arc::new(vad),
                    VoiceActivityFilterConfig::fast_whisper_burn(),
                )
                .unwrap()
                .with_logit_filters(vec![scripted])
        }

        /// **Drafts and language detection.** Adding drafts cannot change the
        /// language a stream commits under, so it cannot change what is
        /// committed: until a commit has detected the language, a draft
        /// detects one for its own decode and keeps nothing. Here the first
        /// draft's frames detect another language than the first committed
        /// window's, so a draft that fixed the stream's language would show.
        /// Batched, the same.
        #[test]
        #[serial]
        fn test_drafts_never_fix_the_language() {
            let device = cpu_device();

            let conservative = detecting_driver(&device, EmissionPolicy::conservative());
            assert!(conservative.detects_language());
            let mut ctx = conservative
                .new_context(clock(), LevelPolicy::PerWindow)
                .unwrap();
            let quiet = in_pieces(&mut ctx);
            let language = ctx.language().map(str::to_string);
            assert!(language.is_some());

            let responsive = EmissionPolicy::new(
                DecodeTriggers::new()
                    .with_endpoint(true)
                    .with_interval(Some(std::time::Duration::from_millis(50))),
                CommitRule::LastTimestamp,
            );
            let driver = detecting_driver(&device, responsive);
            let mut ctx = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
            let mut chatty = Vec::new();
            let mut fixed_by_a_draft = false;
            for piece in speech().chunks(RATE / 10) {
                chatty.extend(ctx.write_read(piece).unwrap());
                if !chatty.iter().any(TranscriptEvent::is_committed) {
                    fixed_by_a_draft |= ctx.language().is_some();
                }
            }
            chatty.extend(ctx.end_read().unwrap());

            let committed: Vec<TranscriptEvent> = chatty
                .iter()
                .filter(|e| e.is_committed())
                .cloned()
                .collect();
            assert_eq!(
                committed, quiet,
                "responsive commits what conservative commits"
            );
            assert_eq!(ctx.language(), language.as_deref());
            assert!(!fixed_by_a_draft, "only a commit fixes the language");

            // What makes this a test: a draft comes first, under another
            // language than the first commit's.
            assert!(!chatty[0].is_committed(), "a draft comes first");
            assert_ne!(
                chatty[0].segment().tokens,
                quiet[0].segment().tokens,
                "the first draft's frames detect another language"
            );

            let mut batch = vec![driver.new_context(clock(), LevelPolicy::PerWindow).unwrap()];
            let mut batched = Vec::new();
            for piece in speech().chunks(RATE / 10) {
                batch[0].write(piece).unwrap();
                batched.extend(advance_ready(&driver, &mut batch).unwrap().remove(0));
            }
            batch[0].end_input().unwrap();
            batched.extend(advance_ready(&driver, &mut batch).unwrap().remove(0));
            assert_eq!(batched, chatty);
        }

        /// Under the offline policy an attached VAD is simply unused: the
        /// clip decodes as one window at the end, silence and all.
        #[test]
        #[serial]
        fn test_offline_ignores_the_vad() {
            let device = cpu_device();
            let vad = SileroVad::load_16khz_pretrained(&device).unwrap();
            let driver = config(false)
                .init_with_layout(
                    tiny_model_on(&device),
                    WhisperTokenLayout::new(tiny_layout()),
                    &device,
                )
                .unwrap()
                .with_vad(
                    Arc::new(vad),
                    VoiceActivityFilterConfig::fast_whisper_burn(),
                )
                .unwrap();
            let audio = speech();

            let mut ctx = driver.new_context(clock(), LevelPolicy::PerWindow).unwrap();
            let mut emissions = ctx.write_read(&audio).unwrap();
            emissions.extend(ctx.end_read().unwrap());

            // 400 frames of 16 per window: 25 windows, no remainder.
            assert_eq!(emissions.len(), 25);
            assert_eq!(ctx.regions_pending(), 0);
        }
    }
}
