//! # The stream context: one transcription in progress.
//!
//! [`push`](WhisperStreamContext::push) is a fold: append to staging, drain
//! whole hops into the mel context, append the frames to the ring, offer them
//! to the clamp policy, then run whatever the emission policy says is due.
//! The hop-alignment requirement never reaches the caller &mdash; staging
//! makes the API honest against a sound card that hands over 480- or
//! 1024-sample buffers.
//!
//! Nothing tensor-shaped crosses a window boundary. Continuity is three
//! host-side values: the seek pointer, the prompt carry, and the clock. The
//! ring holds every frame from `seek` onward and nothing before it.
//!
//! ## What a decode may touch
//!
//! Anything a provisional decode reaches takes `&self`: packaging asks the
//! clamp policy for a reference through `&self`, the model is pure, and the
//! prompt is read. That is what keeps drafts, when they arrive, from becoming
//! a second code path &mdash; and it is checkable today, through the
//! test-only probe.

use burn::{
    Tensor,
    module::Module,
    prelude::{
        Backend,
        TensorData,
    },
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::whisper::{
        blocks::WhisperMeta,
        clamp::ClampPolicy,
        clock::TimestampHistory,
        decode::GreedyDecodeConfig,
        driver::WhisperDriver,
        emission::{
            Emission,
            Segment,
        },
        mel::package_window,
    },
    ops::signal::mels::{
        MelConversionContext,
        MelConverterMeta,
    },
};

/// One stream: the only stateful type in the driver.
///
/// Opened by [`WhisperDriver::new_context`]. Its tensor state &mdash; the
/// driver's handle, the mel carry and the frame ring &mdash; is `Module`
/// typed; everything else is host-side bookkeeping, small enough to
/// snapshot.
///
/// Not itself a `Module`, deliberately: `burn`'s derive treats every field
/// whose type mentions `B` as a module, `#[module(skip)]` or not, and
/// carries a skipped generic unchanged into the autodiff inner module, so
/// neither a boxed policy nor a policy type parameter survives it. The
/// policy stays behind its trait, boxed, and the context is `Clone + Debug`.
#[derive(Clone, Debug)]
pub struct WhisperStreamContext<B: Backend> {
    driver: WhisperDriver<B>,

    /// The mel front end's streaming state; `None` once flushed.
    mel: Option<MelConversionContext<B>>,

    /// `[1, frames, n_mels]` log-mels from `origin` onward; `None` when
    /// empty.
    frames: Option<Tensor<B, 3>>,

    /// Samples not yet a whole hop.
    staging: Vec<f32>,

    /// Samples pushed so far, staging included.
    samples_seen: u64,

    /// The stream frame index of the ring's first frame.
    origin: usize,

    /// The stream frame index the next window starts at.
    seek: usize,

    clock: TimestampHistory,

    /// Every committed id, in order.
    transcript: Vec<i64>,

    clamp: Box<dyn ClampPolicy<B>>,

    finished: bool,

    /// Every window handed to a decode, in order. Test-only: how the
    /// chunking invariant is stated at frame level.
    #[cfg(test)]
    trace: Vec<Tensor<B, 3>>,
}

impl<B: Backend> WhisperStreamContext<B> {
    pub(super) fn open(
        driver: WhisperDriver<B>,
        clock: TimestampHistory,
        clamp: Box<dyn ClampPolicy<B>>,
    ) -> Self {
        let mel = driver.mel().new_context(1);
        Self {
            driver,
            mel: Some(mel),
            frames: None,
            staging: Vec::new(),
            samples_seen: 0,
            origin: 0,
            seek: 0,
            clock,
            transcript: Vec::new(),
            clamp,
            finished: false,
            #[cfg(test)]
            trace: Vec::new(),
        }
    }

    // ---- observation -------------------------------------------------

    /// The stream's clock.
    pub fn clock(&self) -> &TimestampHistory {
        &self.clock
    }

    /// Every committed id, in order.
    pub fn transcript(&self) -> &[i64] {
        &self.transcript
    }

    /// Samples pushed so far.
    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// The stream frame index the next window starts at.
    pub fn seek(&self) -> usize {
        self.seek
    }

    /// Frames produced so far.
    pub fn frames_seen(&self) -> usize {
        self.origin + self.ring_len()
    }

    /// Frames past the seek pointer, waiting for a decode.
    pub fn pending_frames(&self) -> usize {
        self.frames_seen() - self.seek
    }

    /// Whether [`flush`](Self::flush) has run.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    fn ring_len(&self) -> usize {
        self.frames.as_ref().map_or(0, |f| f.dims()[1])
    }

    // ---- input -------------------------------------------------------

    /// Pushes samples, of any length, and returns what became final.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] after [`flush`](Self::flush).
    pub fn push(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<Vec<Emission>> {
        if self.finished {
            return Err(BunsenError::Invalid(
                "the stream was flushed; nothing more can be pushed".to_string(),
            ));
        }

        self.staging.extend_from_slice(samples);
        self.samples_seen += samples.len() as u64;
        self.drain_staging(false)?;
        self.run_due(false)
    }

    /// [`push`](Self::push), anchoring the clock: the first sample of
    /// `samples` was at media time `time`.
    ///
    /// # Errors
    /// As [`push`](Self::push) and [`TimestampHistory::anchor`].
    pub fn push_at(
        &mut self,
        samples: &[f32],
        time: f64,
    ) -> BunsenResult<Vec<Emission>> {
        self.clock.anchor(self.samples_seen, time)?;
        self.push(samples)
    }

    /// Ends the stream: flushes the front end, drops Whisper's trailing
    /// frame, and decodes whatever is left past the seek pointer.
    ///
    /// Idempotent; a second flush returns nothing.
    pub fn flush(&mut self) -> BunsenResult<Vec<Emission>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;

        self.drain_staging(true)?;

        if let Some(mel) = self.mel.take()
            && let Some(tail) = mel.finish()
        {
            self.ingest(tail);
        }

        // Whisper's `stft[..., :-1]`: the stream's last frame is the end
        // padding's, and goes.
        if let Some(frames) = self.frames.take() {
            let n = frames.dims()[1];
            self.frames = (n > 1).then(|| frames.slice_dim(1, 0..n as isize - 1));
        }

        self.run_due(true)
    }

    /// Moves whole hops from staging into the front end.
    ///
    /// The first chunk must be long enough to produce a frame under the
    /// start padding, so it waits for a window's worth of samples; after
    /// that a hop at a time is enough. Flushing pads whatever remains with
    /// silence up to that minimum, as upstream pads a clip; a stream that
    /// never received a sample flushes to nothing.
    fn drain_staging(
        &mut self,
        flushing: bool,
    ) -> BunsenResult<()> {
        let mel = self.driver.mel();
        let (hop, n_fft) = (mel.hop(), mel.n_fft());
        let first = self.mel.as_ref().is_some_and(|ctx| ctx.carry().is_none());
        let minimum = if first {
            n_fft.div_ceil(hop) * hop
        } else {
            hop
        };

        let mut whole = self.staging.len() / hop * hop;
        if flushing {
            if self.staging.is_empty() {
                return Ok(());
            }
            whole = self.staging.len().div_ceil(hop).max(minimum / hop) * hop;
            self.staging.resize(whole, 0.0);
        } else if whole < minimum {
            return Ok(());
        }
        if whole == 0 {
            return Ok(());
        }

        let chunk: Vec<f64> = self.staging.drain(..whole).map(f64::from).collect();
        let device = self.driver.devices()[0].clone();
        let waves: Tensor<B, 2> = Tensor::from_data(TensorData::new(chunk, [1, whole]), &device);

        let ctx = self.mel.take().expect("the front end is open until flush");
        let (frames, ctx) = ctx.transform(waves)?;
        self.mel = Some(ctx);
        self.ingest(frames);
        Ok(())
    }

    /// Appends frames to the ring, offering them to the clamp policy first.
    fn ingest(
        &mut self,
        new: Tensor<B, 3>,
    ) {
        self.clamp.observe(&new);
        self.frames = Some(match self.frames.take() {
            Some(ring) => Tensor::cat(vec![ring, new], 1),
            None => new,
        });
    }

    // ---- decoding ----------------------------------------------------

    /// Decodes every full window past the seek pointer, and on a flush the
    /// remainder too.
    fn run_due(
        &mut self,
        flushing: bool,
    ) -> BunsenResult<Vec<Emission>> {
        let width = self.driver.window_frames();
        let mut out = Vec::new();

        while self.pending_frames() >= width {
            let window = self.pending_window(width);
            #[cfg(test)]
            self.trace.push(window.clone());
            let tokens = self.decode_frames(window);
            out.push(self.commit(tokens, width)?);
        }

        if flushing && self.pending_frames() > 0 {
            let count = self.pending_frames();
            let window = self.pending_window(count);
            #[cfg(test)]
            self.trace.push(window.clone());
            let tokens = self.decode_frames(window);
            out.push(self.commit(tokens, count)?);
        }

        Ok(out)
    }

    /// The `count` frames at the seek pointer: `[1, count, n_mels]`.
    fn pending_window(
        &self,
        count: usize,
    ) -> Tensor<B, 3> {
        let ring = self.frames.as_ref().expect("pending frames exist");
        let start = (self.seek - self.origin) as isize;
        ring.clone().slice_dim(1, start..start + count as isize)
    }

    /// Packages a window against the clamp policy's reference, pads it out
    /// to the model's width, and decodes it. Takes `&self`: with
    /// [`pending_window`](Self::pending_window) this is the whole of a
    /// provisional decode.
    fn decode_frames(
        &self,
        window: Tensor<B, 3>,
    ) -> Vec<i64> {
        let reference = self.clamp.reference(&window);
        let packaged = package_window(window, reference);

        let width = self.driver.window_frames();
        let [_, n_mels, have] = packaged.dims();
        let padded = if have < width {
            let pad = Tensor::zeros([1, n_mels, width - have], &packaged.device());
            Tensor::cat(vec![packaged, pad], 2)
        } else {
            packaged
        };

        let config = GreedyDecodeConfig::new(self.prompt_now(), self.driver.policy().ids().eot)
            .with_max_tokens(self.driver.max_tokens());
        self.driver.model().decode_window(padded, &config)
    }

    /// The prompt for the next window: the sot sequence, preceded by the
    /// transcript's tail after `<|startofprev|>` when carrying is on.
    fn prompt_now(&self) -> Vec<i64> {
        let prompt = self.driver.prompt();
        if !self.driver.carries_prompt() || self.transcript.is_empty() {
            return prompt.to_vec();
        }

        // Upstream keeps `n_text_ctx / 2 - 1` tokens of context.
        let keep = self.driver.model().max_text_ctx() / 2 - 1;
        let tail = &self.transcript[self.transcript.len().saturating_sub(keep)..];

        let mut carried = Vec::with_capacity(1 + tail.len() + prompt.len());
        carried.push(self.driver.policy().ids().sot_prev);
        carried.extend_from_slice(tail);
        carried.extend_from_slice(prompt);
        carried
    }

    /// Commits a decode of `count` frames at the seek pointer: records the
    /// ids, advances the seek pointer, drops the consumed frames, and places
    /// the segment on the clock.
    fn commit(
        &mut self,
        tokens: Vec<i64>,
        count: usize,
    ) -> BunsenResult<Emission> {
        let hop = self.driver.mel().hop() as u64;
        let start = self.clock.time_at(self.seek as u64 * hop);
        let end = self.clock.time_at((self.seek + count) as u64 * hop);

        let text = match self.driver.detokenizer() {
            Some(detokenizer) => {
                Some(detokenizer.detokenize(&self.driver.policy().text_ids(&tokens))?)
            }
            None => None,
        };

        self.transcript.extend_from_slice(&tokens);
        self.seek += count;

        // Retain from the seek pointer onward, nothing before it.
        let drop = (self.seek - self.origin) as isize;
        self.frames = self.frames.take().and_then(|ring| {
            let n = ring.dims()[1] as isize;
            (drop < n).then(|| ring.slice_dim(1, drop..n))
        });
        self.origin = self.seek;

        Ok(Emission::Committed(Segment {
            start,
            end,
            tokens,
            text,
        }))
    }

    // ---- test probes -------------------------------------------------

    /// A provisional decode of everything past the seek pointer, without
    /// touching the context: what a draft will be, once drafts exist.
    #[cfg(test)]
    pub(crate) fn probe_decode(&self) -> Option<Vec<i64>> {
        let count = self.pending_frames();
        (count > 0).then(|| self.decode_frames(self.pending_window(count)))
    }

    /// Everything the context holds, as one comparable string.
    #[cfg(test)]
    pub(crate) fn fingerprint(&self) -> String {
        let data = |t: &Tensor<B, 3>| t.to_data().convert::<f32>().to_vec::<f32>().unwrap();
        let ring = self.frames.as_ref().map(data);
        let carry = self
            .mel
            .as_ref()
            .and_then(|m| m.carry())
            .map(|c| c.to_data().convert::<f32>().to_vec::<f32>().unwrap());
        let reference = self.frames.as_ref().map(|f| {
            self.clamp
                .reference(f)
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap()
        });

        format!(
            "{:?}|{}|{}|{}|{:?}|{:?}|{}|{ring:?}|{carry:?}|{reference:?}",
            self.staging,
            self.samples_seen,
            self.origin,
            self.seek,
            self.clock,
            self.transcript,
            self.finished,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use burn::tensor::{
        Tolerance,
        backend::BackendTypes,
    };
    use serial_test::serial;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        errors::BunsenResult,
        kits::{
            speech::whisper::{
                Whisper,
                blocks::WhisperApiConfig,
                clamp::{
                    MaxSeen,
                    PerWindow,
                },
                driver::{
                    SAMPLE_RATE,
                    WhisperDriverConfig,
                },
                emission::EmissionPolicy,
                mel::{
                    package_mels,
                    trim_stream_tail,
                },
                tokens::{
                    TokenPolicy,
                    WhisperSpecialIds,
                },
            },
            tokens::Detokenizer,
        },
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;
    type F = <B as BackendTypes>::FloatElem;
    type Device = burn::prelude::Device<B>;

    /// A layout small enough for a tiny model: 5 base ranks, 1 language.
    fn tiny_layout() -> WhisperSpecialIds {
        WhisperSpecialIds::new(5, 1).unwrap()
    }

    /// A tiny model whose vocabulary fits the tiny layout and whose window
    /// is 16 frames (2560 samples), so a short clip has several windows.
    /// Seeded, so a run is the same run on the same backend.
    fn tiny_model(device: &Device) -> Whisper<B> {
        B::seed(device, 7);
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

    fn driver(
        device: &Device,
        carry: bool,
    ) -> WhisperDriver<B> {
        WhisperDriverConfig::new()
            .with_language(Some("en".to_string()))
            .with_max_tokens(4)
            .with_condition_on_previous_text(carry)
            .init_with_policy(tiny_model(device), TokenPolicy::new(tiny_layout()), device)
            .unwrap()
    }

    fn clock() -> TimestampHistory {
        TimestampHistory::uniform(SAMPLE_RATE)
    }

    /// A deterministic 1.05 s clip: a tone under a bell-shaped envelope
    /// peaking mid-clip, a rising chirp, and a little noise. The loudest
    /// frames are in the middle on purpose, so the global maximum is never
    /// in the trailing frame that packaging drops.
    fn clip() -> Vec<f32> {
        let n = 16_800;
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                let envelope = (-(t - 0.5).powi(2) * 20.0).exp();
                let tone = 0.6 * envelope * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
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
        seed: u64,
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
        ctx: &mut WhisperStreamContext<B>,
        audio: &[f32],
        sizes: &[usize],
    ) -> Vec<Emission> {
        let mut out = Vec::new();
        let mut at = 0;
        for &size in sizes {
            out.extend(ctx.push(&audio[at..at + size]).unwrap());
            at += size;
        }
        assert_eq!(at, audio.len());
        out.extend(ctx.flush().unwrap());
        out
    }

    fn tokens_of(emissions: &[Emission]) -> Vec<Vec<i64>> {
        emissions
            .iter()
            .map(|e| e.segment().tokens.clone())
            .collect()
    }

    /// The whole clip through the front end in one call, joined with its
    /// tail: what `package_mels` and `decode_chunked` start from.
    fn joined_mels(
        driver: &WhisperDriver<B>,
        audio: &[f32],
        device: &Device,
    ) -> Tensor<B, 3> {
        let n = audio.len();
        let waves: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(
                audio.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                [1, n],
            ),
            device,
        );
        let (mels, ctx) = driver.mel().new_context(1).transform(waves).unwrap();
        match ctx.finish() {
            Some(tail) => Tensor::cat(vec![mels, tail], 1),
            None => mels,
        }
    }

    fn greedy_config(driver: &WhisperDriver<B>) -> GreedyDecodeConfig {
        GreedyDecodeConfig::new(driver.prompt().to_vec(), driver.policy().ids().eot)
            .with_max_tokens(driver.max_tokens())
    }

    /// **I5, against the one-shot path.** One push of the whole clip, with
    /// the running-maximum policy fed everything before the first decode,
    /// is `decode_chunked` over `package_mels` &mdash; window for window,
    /// id for id.
    #[test]
    #[serial]
    fn test_single_push_matches_decode_chunked() {
        let device = Device::default();
        let driver = driver(&device, false);
        let audio = clip();

        let mut ctx = driver.new_context(clock(), MaxSeen::new()).unwrap();
        let mut emissions = ctx.push(&audio).unwrap();
        emissions.extend(ctx.flush().unwrap());

        let expected = driver.model().decode_chunked(
            package_mels(joined_mels(&driver, &audio, &device)),
            &greedy_config(&driver),
        );

        assert_eq!(expected.len(), 7, "6 full windows and a remainder");
        assert_eq!(tokens_of(&emissions), expected);
        assert!(emissions.iter().all(Emission::is_committed));
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
        a: (&WhisperStreamContext<B>, &[Emission]),
        b: (&WhisperStreamContext<B>, &[Emission]),
        label: &str,
    ) {
        let (ctx_a, got) = a;
        let (ctx_b, expected) = b;
        assert_eq!(
            ctx_a.trace.len(),
            ctx_b.trace.len(),
            "{label}: window count"
        );
        assert_eq!(got.len(), expected.len(), "{label}: emission count");

        let mut identical = true;
        for (w, (x, y)) in ctx_a.trace.iter().zip(&ctx_b.trace).enumerate() {
            let (x, y) = (x.to_data(), y.to_data());
            assert_eq!(x.shape, y.shape, "{label}: window {w} shape");
            x.assert_approx_eq::<F>(&y, Tolerance::rel_abs(1e-5, 1e-6));
            identical &= x == y;
        }

        for (g, e) in got.iter().zip(expected) {
            assert_eq!(g.segment().start, e.segment().start, "{label}: start");
            assert_eq!(g.segment().end, e.segment().end, "{label}: end");
        }

        if identical {
            assert_eq!(
                got, expected,
                "{label}: identical windows must decode identically"
            );
        } else {
            eprintln!("{label}: windows differ within tolerance; ids not compared");
        }
    }

    /// **I5, against chunking.** Random-sized pushes, most of them not hop
    /// multiples, decode the windows one push decodes &mdash; and one push
    /// decodes exactly what packaging each window by hand gives.
    #[test]
    #[serial]
    fn test_random_pushes_match_single_push() {
        let device = Device::default();
        let driver = driver(&device, false);
        let audio = clip();

        let mut whole = driver.new_context(clock(), PerWindow).unwrap();
        let mut expected = whole.push(&audio).unwrap();
        expected.extend(whole.flush().unwrap());

        for seed in [1, 2, 3] {
            let mut ctx = driver.new_context(clock(), PerWindow).unwrap();
            let got = push_in_pieces(&mut ctx, &audio, &random_sizes(seed, audio.len()));
            assert_same_stream((&ctx, &got), (&whole, &expected), &format!("seed {seed}"));
        }

        // A dynamic policy is the same policy, and the same chunking is the
        // same arithmetic, so this one is exact.
        let boxed: Box<dyn ClampPolicy<B>> = Box::new(PerWindow);
        let mut dynamic = driver.new_context(clock(), boxed).unwrap();
        let mut got = dynamic.push(&audio).unwrap();
        got.extend(dynamic.flush().unwrap());
        assert_eq!(got, expected);

        // The per-window one-shot path, by hand.
        let width = driver.window_frames();
        let trimmed = trim_stream_tail(joined_mels(&driver, &audio, &device));
        let frames = trimmed.dims()[1];
        let config = greedy_config(&driver);
        let mut by_hand = Vec::new();
        let mut at = 0;
        while at < frames {
            let count = width.min(frames - at);
            let window = trimmed
                .clone()
                .slice_dim(1, at as isize..(at + count) as isize);
            let mut packaged = package_window(window.clone(), PerWindow.reference(&window));
            if count < width {
                let pad = Tensor::zeros([1, packaged.dims()[1], width - count], &device);
                packaged = Tensor::cat(vec![packaged, pad], 2);
            }
            by_hand.push(driver.model().decode_window(packaged, &config));
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
        let device = Device::default();
        let driver = driver(&device, false);
        let audio = clip();
        let hop = driver.mel().hop() as f64;
        let width = driver.window_frames();

        let mut ctx = driver.new_context(clock(), PerWindow).unwrap();
        let mut emissions = ctx.push_at(&audio, 100.0).unwrap();
        emissions.extend(ctx.flush().unwrap());

        let window_seconds = width as f64 * hop / SAMPLE_RATE as f64;
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
        assert!((last.end - (100.0 + 105.0 * hop / SAMPLE_RATE as f64)).abs() < 1e-9);
        assert_eq!(ctx.pending_frames(), 0);
    }

    /// **I6.** A provisional decode leaves the context byte-identical, and
    /// says the same thing twice.
    #[test]
    #[serial]
    fn test_probe_decode_leaves_the_context_untouched() {
        let device = Device::default();
        let driver = driver(&device, false);
        let audio = clip();

        let mut ctx = driver.new_context(clock(), MaxSeen::new()).unwrap();
        // Enough for one committed window and a partial second one.
        let committed = ctx.push(&audio[..4_000]).unwrap();
        assert_eq!(committed.len(), 1);
        assert!(ctx.pending_frames() > 0);

        let before = ctx.fingerprint();
        let first = ctx.probe_decode().expect("frames are pending");
        let second = ctx.probe_decode().expect("frames are pending");
        assert_eq!(ctx.fingerprint(), before);
        assert_eq!(first, second);
        assert_eq!(ctx.transcript(), committed[0].segment().tokens);
    }

    /// With carrying on, a window is prompted with the transcript's tail
    /// after `<|startofprev|>`, exactly as decoding it by hand with that
    /// prompt would.
    #[test]
    #[serial]
    fn test_prompt_carry() {
        let device = Device::default();
        let driver = driver(&device, true);
        let audio = clip();
        let width = driver.window_frames();

        let mut ctx = driver.new_context(clock(), PerWindow).unwrap();
        let mut emissions = ctx.push(&audio).unwrap();
        emissions.extend(ctx.flush().unwrap());
        let got = tokens_of(&emissions);

        // By hand: the first window with the bare prompt, the second with
        // the carry built from the first's ids.
        let trimmed = trim_stream_tail(joined_mels(&driver, &audio, &device));
        let window_at = |w: usize| {
            let window = trimmed
                .clone()
                .slice_dim(1, (w * width) as isize..((w + 1) * width) as isize);
            package_window(window.clone(), PerWindow.reference(&window))
        };
        let ids = driver.policy().ids();
        let first = driver
            .model()
            .decode_window(window_at(0), &greedy_config(&driver));
        assert_eq!(got[0], first);

        let keep = driver.model().max_text_ctx() / 2 - 1;
        let mut prompt = vec![ids.sot_prev];
        prompt.extend_from_slice(&first[first.len().saturating_sub(keep)..]);
        prompt.extend_from_slice(driver.prompt());
        let carried = GreedyDecodeConfig::new(prompt, ids.eot).with_max_tokens(driver.max_tokens());
        let second = driver.model().decode_window(window_at(1), &carried);
        assert_eq!(got[1], second);

        // Without carrying, the same audio prompts every window the same
        // way, so this is only a real test if the two differ somewhere.
        let bare = driver
            .model()
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

        let device = Device::default();
        let driver = driver(&device, false).with_detokenizer(Arc::new(Numbers));
        let audio = clip();

        let mut ctx = driver.new_context(clock(), PerWindow).unwrap();
        let mut emissions = ctx.push(&audio).unwrap();
        emissions.extend(ctx.flush().unwrap());

        for e in &emissions {
            let segment = e.segment();
            let text_ids = driver.policy().text_ids(&segment.tokens);
            assert_eq!(
                segment.text.as_deref(),
                Some(Numbers.detokenize(&text_ids).unwrap().as_str())
            );
        }
    }

    /// The stream's lifecycle edges: nothing pushed, a stream shorter than a
    /// hop, flush twice, push after flush.
    #[test]
    #[serial]
    fn test_lifecycle_edges() {
        let device = Device::default();
        let driver = driver(&device, false);

        let mut empty = driver.new_context(clock(), PerWindow).unwrap();
        assert!(empty.flush().unwrap().is_empty());
        assert!(empty.flush().unwrap().is_empty(), "flush is idempotent");
        assert!(empty.push(&[0.0; 16]).is_err(), "no pushing after flush");
        assert!(empty.is_finished());

        // Shorter than a hop: padded with silence to something decodable.
        let mut short = driver.new_context(clock(), PerWindow).unwrap();
        assert!(short.push(&[0.1; 50]).unwrap().is_empty());
        let emissions = short.flush().unwrap();
        assert_eq!(emissions.len(), 1);
        assert!(short.pending_frames() == 0);
    }

    /// The configuration refuses what this slice cannot do, with a reason,
    /// and refuses a mismatched language.
    #[test]
    fn test_init_refuses_the_unsupported() {
        let device = Device::default();
        let policy = TokenPolicy::new(tiny_layout());
        let base = WhisperDriverConfig::new().with_language(Some("en".to_string()));

        assert!(
            base.init_with_policy(tiny_model(&device), policy, &device)
                .is_ok()
        );
        assert!(
            WhisperDriverConfig::new()
                .init_with_policy(tiny_model(&device), policy, &device)
                .is_err(),
            "multilingual without a language",
        );
        assert!(
            base.clone()
                .with_timestamps(true)
                .init_with_policy(tiny_model(&device), policy, &device)
                .is_err()
        );
        assert!(
            base.clone()
                .with_emission(EmissionPolicy::conservative())
                .init_with_policy(tiny_model(&device), policy, &device)
                .is_err()
        );
        assert!(
            base.clone()
                .with_emission(EmissionPolicy::responsive())
                .init_with_policy(tiny_model(&device), policy, &device)
                .is_err()
        );
        assert!(
            base.clone()
                .with_language(Some("xx".to_string()))
                .init_with_policy(tiny_model(&device), policy, &device)
                .is_err()
        );

        // A clock at the wrong rate is refused at the stream, not later.
        let driver = driver(&device, false);
        assert!(
            driver
                .new_context(TimestampHistory::uniform(8_000), PerWindow)
                .is_err()
        );
    }
}
