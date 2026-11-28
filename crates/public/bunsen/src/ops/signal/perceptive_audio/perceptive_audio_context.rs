//! Streaming waveform to log-mel conversion.

use burn::{
    Tensor,
    config::Config,
    module::Module,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    ops::signal::perceptive_audio::{
        PaddingMode,
        PerceptiveAudioConverter,
        PerceptiveAudioConverterMeta,
    },
};

/// Where a stream sits in its lifecycle.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum StreamPhase {
    /// Nothing consumed yet; the next chunk receives the start padding.
    Start,

    /// At least one chunk consumed; the next chunk is prefixed by the carry.
    Running,
}

/// One stream through a [`PerceptiveAudioConverter`]: the samples a future
/// frame still needs, and where the stream sits in its lifecycle.
///
/// Built by [`PerceptiveAudioConverter::new_stream_context`], which fixes the
/// batch size; each batch row is an independent stream.
/// [`transform_stream`](PerceptiveAudioConverter::transform_stream) feeds it
/// hop-aligned chunks, each producing exactly the frames it would have
/// produced in one call; [`finish_stream`](PerceptiveAudioConverter::finish_stream)
/// flushes the end padding, once.
///
/// A `Module` over bare tensors: the carry rides `to_device` but is neither
/// recorded nor visited.
///
/// # Chunking
///
/// Every stage is elementwise or frame-local, so transforming a signal in
/// chunks and concatenating gives the same result as transforming it whole.
/// Dynamic-range packaging that reduces over a whole clip — see
/// [`RangeClamp`](crate::ops::signal::perceptive_audio::RangeClamp) — is
/// deliberately not part of this pipeline; apply it once to the joined result.
#[derive(Module, Debug)]
pub struct PerceptiveAudioConverterStreamContext {
    /// The samples a future frame still needs: `[batch, carry_len]`.
    ///
    /// `None` before the first chunk. In steady state `carry_len` is
    /// invariant — see
    /// [`transform_stream`](PerceptiveAudioConverter::transform_stream).
    carry: Option<Tensor<2>>,

    #[module(skip)]
    batch_size: usize,

    #[module(skip)]
    phase: StreamPhase,

    #[module(skip)]
    finished: bool,
}

impl PerceptiveAudioConverterStreamContext {
    /// The number of independent streams; fixed at construction.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Where this stream sits in its lifecycle.
    pub fn phase(&self) -> StreamPhase {
        self.phase
    }

    /// The carried samples, if any: `[batch, carry_len]`.
    pub fn carry(&self) -> Option<&Tensor<2>> {
        self.carry.as_ref()
    }

    /// Drops the carry and returns to [`StreamPhase::Start`], unfinished.
    pub fn reset(&mut self) {
        self.carry = None;
        self.phase = StreamPhase::Start;
        self.finished = false;
    }
}

impl PerceptiveAudioConverter {
    /// A stream of `batch_size` independent rows through these constants.
    ///
    /// # Panics
    /// If `batch_size` is zero.
    pub fn new_stream_context(
        &self,
        batch_size: usize,
    ) -> PerceptiveAudioConverterStreamContext {
        assert_ne!(
            batch_size, 0,
            "PerceptiveAudioConverter batch_size must be non-zero"
        );
        PerceptiveAudioConverterStreamContext {
            carry: None,
            batch_size,
            phase: StreamPhase::Start,
            finished: false,
        }
    }

    /// Converts the stream's next hop-aligned chunk of waveform into
    /// log-mels: the carry or the start padding prepended, then
    /// [`forward`](Self::forward)'s stages.
    ///
    /// # Frame count
    ///
    /// In steady state this yields exactly `samples / hop` frames, and the
    /// carry length is invariant. That holds because the carry is
    /// `n_fft - hop + r` where `r = (start_pad - n_fft) mod hop` depends only
    /// on the padding and the hop, not on how the signal was chunked.
    ///
    /// The first call is the exception: it also absorbs the start padding, so
    /// it yields `(start_pad + samples - n_fft) / hop + 1`. For the default
    /// geometry over 30 s that is 2999 frames with a 360-sample carry, and
    /// [`finish_stream`](Self::finish_stream) contributes the remaining 2 —
    /// 3001 in total, matching `librosa` with `center=True`.
    ///
    /// # Arguments
    /// * `waves`: `[batch, samples]`, with `samples` a non-zero multiple of
    ///   `hop`.
    /// * `ctx`: the stream, advanced past `waves`.
    ///
    /// # Returns
    /// `[batch, frames, n_mels]` log-mels.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if the stream is
    /// finished, if the batch size does not match, if `samples` is zero or
    /// not a multiple of `hop`, or if the first chunk is too short to produce
    /// a frame.
    pub fn transform_stream(
        &self,
        waves: Tensor<2>,
        ctx: &mut PerceptiveAudioConverterStreamContext,
    ) -> BunsenResult<Tensor<3>> {
        let x = self.extend_stream(waves, ctx)?;
        let mels = self.forward(x);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "frames", "n_mels"],
            &mels.dims(),
            &[("batch", ctx.batch_size()), ("n_mels", self.n_mels())]
        );
        Ok(mels)
    }

    /// Flushes the stream's carry through the end padding, if any is
    /// configured; the stream is then finished.
    ///
    /// Returns `None` when
    /// [`end_padding`](PerceptiveAudioConverterMeta::end_padding) is
    /// [`PaddingMode::None`], when nothing has been transformed yet, or when
    /// the stream was finished already.
    ///
    /// # Returns
    /// `[batch, frames, n_mels]` tail frames.
    pub fn finish_stream(
        &self,
        ctx: &mut PerceptiveAudioConverterStreamContext,
    ) -> Option<Tensor<3>> {
        if ctx.finished {
            return None;
        }
        ctx.finished = true;
        let carry = ctx.carry.take()?;

        let pad = self.end_padding().pad_len(self.n_fft());
        if pad == 0 {
            return None;
        }

        let [len] = crate::contracts::unpack_shape_contract!(
            ["batch", "len"],
            &carry.dims(),
            &["len"],
            &[("batch", ctx.batch_size)]
        );

        let tail = match self.end_padding() {
            PaddingMode::None => unreachable!("pad_len is zero for None"),

            PaddingMode::Zero => Tensor::cat(
                vec![
                    carry.clone(),
                    Tensor::zeros([ctx.batch_size, pad], &carry.device()),
                ],
                1,
            ),

            // Mirror about the final sample: `numpy`'s "reflect" right-pad of
            // `p` takes `x[-2] .. x[-p-1]`, so slice the `p` samples ending
            // one short of the end and reverse them.
            //
            // This reads only the last `p + 1` samples, and the carry is
            // always longer than that, so reflecting the carry is identical to
            // reflecting the whole signal — which is what makes the streaming
            // result match a one-shot `center=True` run.
            PaddingMode::Reflect => {
                debug_assert!(
                    len > pad,
                    "carry ({len}) must exceed the reflect pad ({pad})",
                );
                let mirror = carry
                    .clone()
                    .slice_dim(1, (len - 1 - pad) as isize..(len - 1) as isize)
                    .flip([1]);
                Tensor::cat(vec![carry, mirror], 1)
            }
        };

        let [tail_len] = crate::contracts::unpack_shape_contract!(
            ["batch", "len"],
            &tail.dims(),
            &["len"],
            &[("batch", ctx.batch_size)]
        );
        if self.frame_count(tail_len) == 0 {
            return None;
        }

        let mels = self.forward(tail);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "frames", "n_mels"],
            &mels.dims(),
            &[("batch", ctx.batch_size), ("n_mels", self.n_mels())]
        );
        Some(mels)
    }

    /// Prepends the start padding or the carry to the stream's next chunk,
    /// and takes the new carry off the tail: the one stage that touches
    /// state. The carry holds **raw** samples, so a sample-domain filter (when
    /// one lands) sees unfiltered history.
    ///
    /// `[batch, samples]` -> `[batch, extended]`.
    pub(crate) fn extend_stream(
        &self,
        waves: Tensor<2>,
        ctx: &mut PerceptiveAudioConverterStreamContext,
    ) -> BunsenResult<Tensor<2>> {
        if ctx.finished {
            return Err(BunsenError::illegal(
                "PerceptiveAudioConverter: a push to a finished stream",
            ));
        }
        let [batch, samples] =
            crate::contracts::unpack_shape_contract!(["batch", "samples"], &waves.dims());

        if batch != ctx.batch_size {
            return Err(BunsenError::illegal(format!(
                "PerceptiveAudioConverter: waves batch ({batch}) must \
                 match the context batch size ({})",
                ctx.batch_size,
            )));
        }

        let hop = self.hop();
        if samples == 0 || samples % hop != 0 {
            return Err(BunsenError::illegal(format!(
                "PerceptiveAudioConverter: waves samples ({samples}) \
                 must be a non-zero multiple of hop ({hop})",
            )));
        }

        let extended = match ctx.phase {
            StreamPhase::Running => match ctx.carry.take() {
                Some(carry) => Tensor::cat(vec![carry, waves], 1),
                None => waves,
            },

            StreamPhase::Start => {
                let pad = self.start_padding().pad_len(self.n_fft());

                match self.start_padding() {
                    PaddingMode::None => waves,

                    PaddingMode::Zero => {
                        Tensor::cat(vec![Tensor::zeros([batch, pad], &waves.device()), waves], 1)
                    }

                    // Mirror about the first sample: `numpy`'s "reflect"
                    // left-pad of `p` takes `x[p] .. x[1]`, so slice the `p`
                    // samples starting at index 1 and reverse them. That reads
                    // `p + 1` samples, hence `min_first_chunk`.
                    PaddingMode::Reflect => {
                        if samples < self.min_first_chunk() {
                            return Err(BunsenError::illegal(format!(
                                "PerceptiveAudioConverter: reflect start \
                                 padding needs at least {} samples in the first \
                                 chunk, got {samples}",
                                self.min_first_chunk(),
                            )));
                        }
                        let mirror = waves.clone().slice_dim(1, 1..(pad + 1) as isize).flip([1]);
                        Tensor::cat(vec![mirror, waves], 1)
                    }
                }
            }
        };

        let [extended_len] = crate::contracts::unpack_shape_contract!(
            ["batch", "extended"],
            &extended.dims(),
            &["extended"],
            &[("batch", batch)]
        );
        let frames = self.frame_count(extended_len);
        if frames == 0 {
            return Err(BunsenError::illegal(format!(
                "PerceptiveAudioConverter: first chunk of {samples} \
                 samples is too short to fill a frame; with {:?} start padding \
                 it needs at least {} samples",
                self.start_padding(),
                self.n_fft() - self.start_padding().pad_len(self.n_fft()),
            )));
        }

        // Everything past the last frame's start belongs to the next call.
        // Deriving it from the frame count keeps this correct on any geometry,
        // rather than depending on a closed form for the carry length.
        let consumed = frames * hop;
        ctx.carry = Some(extended.clone().slice_dim(1, consumed as isize..));
        ctx.phase = StreamPhase::Running;

        Ok(extended)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        errors::{
            WithOkOrPanic,
            testing::ErrorMatcher,
        },
        ops::signal::perceptive_audio::PerceptiveAudioConverterConfig,
        support::testing::{
            DeviceMemoryGuard,
            assert_close_to_vec,
            assert_tensor_close_to_vec,
            assert_tensors_close,
            performance_device,
            tensor_to_vec_as,
        },
    };

    /// Builds a `[batch, samples]` tensor from a row-major host buffer.
    fn from_rows(
        rows: &[Vec<f64>],
        device: &burn::prelude::Device,
    ) -> Tensor<2> {
        let samples = rows[0].len();
        Tensor::from_data(
            burn::prelude::TensorData::new(rows.concat(), [rows.len(), samples]),
            device,
        )
    }

    /// A deterministic sample in `[-1, 1]`.
    fn sample(
        row: usize,
        i: usize,
    ) -> f64 {
        (((row * 7919 + i * 104729) % 2003) as f64 / 1001.0) - 1.0
    }

    fn rows(
        batch: usize,
        samples: usize,
    ) -> Vec<Vec<f64>> {
        (0..batch)
            .map(|r| (0..samples).map(|i| sample(r, i)).collect())
            .collect()
    }

    #[test]
    #[serial_test::serial]
    fn test_context_meta_and_lifecycle() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let opts = PerceptiveAudioConverterConfig::default();
        let conv: PerceptiveAudioConverter = opts.try_init(&device).ok_or_panic();

        let mut ctx = conv.new_stream_context(2);

        assert_eq!(ctx.batch_size(), 2);
        assert_eq!(ctx.phase(), StreamPhase::Start);
        assert!(ctx.carry().is_none());

        let x = from_rows(&rows(2, 1600), &device);
        conv.transform_stream(x, &mut ctx).unwrap();

        assert_eq!(ctx.phase(), StreamPhase::Running);
        assert!(ctx.carry().is_some());

        ctx.reset();
        assert_eq!(ctx.phase(), StreamPhase::Start);
        assert!(ctx.carry().is_none());
    }

    /// The frame accounting documented under
    /// [`transform_stream`](PerceptiveAudioConverter::transform_stream)'s `# Frame
    /// count`, asserted against the real thing.
    #[test]
    #[serial_test::serial]
    fn test_frame_accounting_over_a_30s_window() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let opts = PerceptiveAudioConverterConfig::default();
        let conv: PerceptiveAudioConverter = opts.try_init(&device).ok_or_panic();

        // 30 s at the default 16 kHz sample rate.
        let samples = 480_000;
        let x = from_rows(&rows(1, samples), &device);

        let mut ctx = conv.new_stream_context(1);
        let mels = conv.transform_stream(x, &mut ctx).unwrap();

        // Start padding is 200, which is not a hop multiple, so the first call
        // is short of `samples / hop` and leaves a 360-sample carry.
        assert_eq!(mels.dims(), [1, 2999, opts.n_mels]);
        assert_eq!(ctx.carry().unwrap().dims(), [1, 360]);

        let tail = conv
            .finish_stream(&mut ctx)
            .expect("reflect end padding yields a tail");
        assert_eq!(tail.dims(), [1, 2, opts.n_mels]);

        // 2999 + 2 == 3001, which is what `librosa` gives with center=True.
        assert_eq!(2999 + 2, 3001);
    }

    /// In steady state each chunk yields exactly `samples / hop` frames and
    /// the carry length does not drift.
    #[test]
    #[serial_test::serial]
    fn test_running_frame_count_is_invariant() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let conv: PerceptiveAudioConverter = PerceptiveAudioConverterConfig::default()
            .try_init(&device)
            .ok_or_panic();
        let hop = conv.hop();

        let mut ctx = conv.new_stream_context(2);
        let mut carry_len = None;

        for (step, chunk) in [3200, 1600, 4800, 1600].into_iter().enumerate() {
            let x = from_rows(&rows(2, chunk), &device);
            let mels = conv.transform_stream(x, &mut ctx).unwrap();

            if step > 0 {
                assert_eq!(
                    mels.dims()[1],
                    chunk / hop,
                    "step {step}: running frame count should be samples / hop",
                );
            }

            let len = ctx.carry().unwrap().dims()[1];
            match carry_len {
                None => carry_len = Some(len),
                Some(prev) => assert_eq!(prev, len, "carry length drifted at step {step}"),
            }
        }

        assert_eq!(carry_len, Some(360));
    }

    /// **The milestone.** Splitting a signal into chunks must give bit-equal
    /// output to running it whole.
    #[test]
    #[serial_test::serial]
    fn test_chunked_transform_is_a_homomorphism() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let conv: PerceptiveAudioConverter = PerceptiveAudioConverterConfig::default()
            .try_init(&device)
            .ok_or_panic();

        let (batch, total) = (3, 32_000);
        let host = rows(batch, total);

        // Whole-signal reference, plus its tail.
        let whole = from_rows(&host, &device);
        let mut whole_ctx = conv.new_stream_context(batch);
        let whole_mels = conv.transform_stream(whole, &mut whole_ctx).unwrap();
        let whole_tail = conv.finish_stream(&mut whole_ctx).unwrap();

        // Several hop-aligned splits, including a ragged one.
        for split in [
            vec![32_000],
            vec![16_000, 16_000],
            vec![320, 31_680],
            vec![6_400, 3_200, 12_800, 9_600],
        ] {
            assert_eq!(split.iter().sum::<usize>(), total);

            let mut ctx = conv.new_stream_context(batch);
            let mut pieces = Vec::new();
            let mut at = 0;

            for n in &split {
                let chunk: Vec<Vec<f64>> = host.iter().map(|r| r[at..at + n].to_vec()).collect();
                let mels = conv
                    .transform_stream(from_rows(&chunk, &device), &mut ctx)
                    .unwrap();
                pieces.push(mels);
                at += n;
            }

            let joined: Tensor<3> = Tensor::cat(pieces, 1);
            assert_eq!(
                joined.dims(),
                whole_mels.dims(),
                "split {split:?} produced a different frame count",
            );

            // Same backend, same dtype, same arithmetic — but the chunks are
            // concatenated differently, so allow a small float tolerance.
            assert_tensors_close(&joined, &whole_mels, Tolerance::absolute(1e-4));

            let tail = conv.finish_stream(&mut ctx).unwrap();
            assert_tensors_close(&tail, &whole_tail, Tolerance::absolute(1e-4));
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_batch_rows_are_independent() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let conv: PerceptiveAudioConverter = PerceptiveAudioConverterConfig::default()
            .try_init(&device)
            .ok_or_panic();

        let (batch, samples) = (3, 8_000);
        let host = rows(batch, samples);

        let together = conv
            .transform_stream(
                from_rows(&host, &device),
                &mut conv.new_stream_context(batch),
            )
            .unwrap();

        let dims = together.dims();
        let per_row = dims[1] * dims[2];
        let together = tensor_to_vec_as::<f64, _>(&together);

        for row in 0..batch {
            let alone = conv
                .transform_stream(
                    from_rows(&host[row..row + 1], &device),
                    &mut conv.new_stream_context(1),
                )
                .unwrap();

            assert_close_to_vec(
                &tensor_to_vec_as::<f64, _>(&alone),
                &together[row * per_row..(row + 1) * per_row],
                1e-4,
            );
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_transform_rejects_bad_input() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let conv: PerceptiveAudioConverter = PerceptiveAudioConverterConfig::default()
            .try_init(&device)
            .ok_or_panic();

        // Not a hop multiple.
        let mut ctx = conv.new_stream_context(2);
        let bad = from_rows(&rows(2, 1601), &device);
        ErrorMatcher::illegal()
            .message_contains("multiple of hop")
            .assert_err(&conv.transform_stream(bad, &mut ctx));

        // Wrong batch.
        let mut ctx = conv.new_stream_context(2);
        let bad = from_rows(&rows(3, 1600), &device);
        ErrorMatcher::illegal()
            .message_contains("batch (3)")
            .assert_err(&conv.transform_stream(bad, &mut ctx));

        // Too short to reflect: `min_first_chunk` is 201, and 160 is the
        // largest hop-aligned chunk below it.
        let mut ctx = conv.new_stream_context(1);
        let bad = from_rows(&rows(1, 160), &device);
        ErrorMatcher::illegal()
            .message_contains("reflect start padding")
            .assert_err(&conv.transform_stream(bad, &mut ctx));
    }

    #[test]
    #[serial_test::serial]
    fn test_finish_respects_end_padding() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        // No end padding: nothing to flush.
        let unpadded: PerceptiveAudioConverter = PerceptiveAudioConverterConfig::default()
            .with_end_padding(PaddingMode::None)
            .try_init(&device)
            .ok_or_panic();
        let mut ctx = unpadded.new_stream_context(1);
        unpadded
            .transform_stream(from_rows(&rows(1, 3200), &device), &mut ctx)
            .unwrap();
        assert!(unpadded.finish_stream(&mut ctx).is_none());

        // Nothing transformed yet: nothing to flush either.
        assert!(
            unpadded
                .finish_stream(&mut unpadded.new_stream_context(1))
                .is_none()
        );

        // Zero end padding produces the same frame count as reflect; only the
        // values differ.
        let zero: PerceptiveAudioConverter = PerceptiveAudioConverterConfig::default()
            .with_end_padding(PaddingMode::Zero)
            .try_init(&device)
            .ok_or_panic();
        let mut ctx = zero.new_stream_context(1);
        zero.transform_stream(from_rows(&rows(1, 3200), &device), &mut ctx)
            .unwrap();
        assert_eq!(zero.finish_stream(&mut ctx).unwrap().dims()[1], 2);
        // A stream finishes once.
        assert!(zero.finish_stream(&mut ctx).is_none());
    }

    /// The stream's one stateful stage, then the stateless stages, is the
    /// stream's transform.
    #[test]
    #[serial_test::serial]
    fn test_extend_then_forward_is_the_transform() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let conv: PerceptiveAudioConverter = PerceptiveAudioConverterConfig::default()
            .try_init(&device)
            .ok_or_panic();

        let x = from_rows(&rows(2, 3200), &device);

        let mut whole = conv.new_stream_context(2);
        let transformed = conv.transform_stream(x.clone(), &mut whole).unwrap();

        let mut staged = conv.new_stream_context(2);
        let extended = conv.extend_stream(x, &mut staged).unwrap();
        let staged = conv.forward(extended);

        assert_tensors_close(&staged, &transformed, Tolerance::default());
    }

    /// `extend_stream` in isolation: the carry contents must be exactly the
    /// tail of the extended signal, and the extension must start with the
    /// mirrored prefix.
    #[test]
    #[serial_test::serial]
    fn test_extend_stage_carry_contents() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let conv: PerceptiveAudioConverter = PerceptiveAudioConverterConfig::default()
            .try_init(&device)
            .ok_or_panic();
        let (n_fft, hop) = (conv.n_fft(), conv.hop());
        let pad = n_fft / 2;

        let host = rows(1, 1600);
        let x = from_rows(&host, &device);

        let mut ctx = conv.new_stream_context(1);
        let ext = conv.extend_stream(x, &mut ctx).unwrap();
        let ext_host = tensor_to_vec_as::<f64, _>(&ext);

        assert_eq!(ext_host.len(), pad + 1600);

        // The first `pad` samples mirror x[1..=pad], reversed.
        let expected_prefix: Vec<f64> = (1..=pad).rev().map(|i| host[0][i]).collect();
        assert_close_to_vec(&ext_host[..pad], &expected_prefix, 1e-6);

        // ...and the rest is the signal itself.
        assert_close_to_vec(&ext_host[pad..], &host[0], 1e-6);

        // The carry is everything past the last frame's start.
        let frames = conv.frame_count(ext_host.len());
        let consumed = frames * hop;
        assert_tensor_close_to_vec(
            &ctx.carry().unwrap(),
            &ext_host[consumed..],
            Tolerance::default(),
        );
    }
}
