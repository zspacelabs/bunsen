//! # Streaming waveform to log-mel conversion.
//!
//! [`MelConversionContext`] binds a carried sample queue to a
//! [`MelConverter`], so a signal can be fed in hop-aligned chunks and produce
//! exactly the frames it would have produced in one call.
//!
//! Each [`transform`](MelConversionContext::transform) is a fold over a fixed
//! stack of `t_stage_*` methods, each shaped `(self, a) -> (b, Self)`. Only
//! two of them touch state, which is the part worth reviewing; the rest
//! delegate to the stateless [`MelConverter`] stages.

use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::Backend,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    ops::signal::mels::{
        MelConverter,
        MelConverterMeta,
        PaddingMode,
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

/// Streaming state for waveform to log-mel conversion.
///
/// Built by [`MelConverter::new_context`], which fixes the batch size. Each
/// batch row is an independent stream.
///
/// [`transform`](Self::transform) takes `self` and hands it back, rather than
/// borrowing mutably, because that is what lets the pipeline be written as a
/// fold over the `t_stage_*` stack. [`finish`](Self::finish) consumes the
/// context, so transforming after finishing is a type error.
///
/// Like [`MelConverter`], this is a `Module` over bare tensors — the carry
/// rides `to_device` but is neither recorded nor visited.
///
/// # Chunking and per-call reductions
///
/// [`RangeClamp::PerCall`](crate::ops::signal::mels::RangeClamp::PerCall)
/// reduces over one call's frames, so a streamed run clamps each chunk
/// against its own maximum: chunking is not transparent while it is enabled.
/// Stream with the clamp and the affine **off**, then apply them once to the
/// joined result.
#[derive(Module, Debug)]
pub struct MelConversionContext<B: Backend> {
    /// The analysis constants; shared, never mutated.
    converter: MelConverter<B>,

    /// The samples a future frame still needs: `[batch, carry_len]`.
    ///
    /// `None` before the first chunk. In steady state `carry_len` is
    /// invariant — see [`transform`](Self::transform).
    carry: Option<Tensor<B, 2>>,

    #[module(skip)]
    batch_size: usize,

    #[module(skip)]
    phase: StreamPhase,
}

impl<B: Backend> MelConverterMeta for MelConversionContext<B> {
    fn sample_rate(&self) -> usize {
        self.converter.sample_rate()
    }

    fn n_fft(&self) -> usize {
        self.converter.n_fft()
    }

    fn hop(&self) -> usize {
        self.converter.hop()
    }

    fn n_mels(&self) -> usize {
        self.converter.n_mels()
    }

    fn pad_to_pow2(&self) -> bool {
        self.converter.pad_to_pow2()
    }

    fn start_padding(&self) -> PaddingMode {
        self.converter.start_padding()
    }

    fn end_padding(&self) -> PaddingMode {
        self.converter.end_padding()
    }
}

impl<B: Backend> MelConversionContext<B> {
    /// The number of independent streams; fixed at construction.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Where this stream sits in its lifecycle.
    pub fn phase(&self) -> StreamPhase {
        self.phase
    }

    /// The converter supplying the analysis constants.
    pub fn converter(&self) -> &MelConverter<B> {
        &self.converter
    }

    /// The carried samples, if any: `[batch, carry_len]`.
    pub fn carry(&self) -> Option<&Tensor<B, 2>> {
        self.carry.as_ref()
    }

    /// Drops the carry and returns to [`StreamPhase::Start`].
    pub fn reset(&mut self) {
        self.carry = None;
        self.phase = StreamPhase::Start;
    }

    /// Converts one hop-aligned chunk of waveform into log-mels.
    ///
    /// A fold over the stage stack; see the `t_stage_*` methods for what each
    /// step does.
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
    /// [`finish`](Self::finish) contributes the remaining 2 — 3001 in total,
    /// matching `librosa` with `center=True`.
    ///
    /// # Arguments
    /// * `waves`: `[batch, samples]`, with `samples` a non-zero multiple of
    ///   `hop`.
    ///
    /// # Returns
    /// `[batch, frames, n_mels]` log-mels, and the advanced context.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if the batch size does not match, if `samples`
    /// is zero or not a multiple of `hop`, or if the first chunk is too short
    /// to produce a frame.
    pub fn transform(
        self,
        waves: Tensor<B, 2>,
    ) -> BunsenResult<(Tensor<B, 3>, Self)> {
        let (x, this) = self.t_stage_extend(waves)?;
        let (x, this) = this.t_stage_preproc(x);
        let (x, this) = this.t_stage_frame(x);
        let (x, this) = this.t_stage_spectrum(x);
        let (x, this) = this.t_stage_mel(x);
        let (mels, this) = this.t_stage_compress(x);

        Ok((mels, this))
    }

    /// Flushes the carry through the end padding, if any is configured.
    ///
    /// Returns `None` when [`end_padding`](MelConverterMeta::end_padding) is
    /// [`PaddingMode::None`], or when nothing has been transformed yet.
    ///
    /// Consumes the context: a stream ends once.
    ///
    /// # Returns
    /// `[batch, frames, n_mels]` tail frames.
    pub fn finish(self) -> Option<Tensor<B, 3>> {
        let carry = self.carry.clone()?;

        let pad = self.end_padding().pad_len(self.n_fft());
        if pad == 0 {
            return None;
        }

        let len = carry.dims()[1];

        let tail = match self.end_padding() {
            PaddingMode::None => unreachable!("pad_len is zero for None"),

            PaddingMode::Zero => Tensor::cat(
                vec![
                    carry.clone(),
                    Tensor::zeros([self.batch_size, pad], &carry.device()),
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

        if self.converter.frame_count(tail.dims()[1]) == 0 {
            return None;
        }

        Some(self.converter.forward(tail))
    }

    // ---- the stage stack ----------------------------------------------
    //
    // Each stage is `(self, a) -> (b, Self)`. They are `pub(crate)` so the
    // test layer can drive a single stage, or a prefix of the stack, without
    // constructing a whole valid stream — but they are not public API.

    /// Stage 1: prepends the start padding or the carry, and takes the new
    /// carry off the tail.
    ///
    /// The only fallible stage, and with
    /// [`t_stage_compress`](Self::t_stage_compress) one of only two that touch
    /// state. The carry holds **raw** samples: pre-emphasis needs unfiltered
    /// history.
    ///
    /// `[batch, samples]` -> `[batch, extended]`.
    #[doc(hidden)]
    pub(crate) fn t_stage_extend(
        mut self,
        waves: Tensor<B, 2>,
    ) -> BunsenResult<(Tensor<B, 2>, Self)> {
        let [batch, samples] = waves.dims();

        if batch != self.batch_size {
            return Err(BunsenError::Invalid(format!(
                "MelConversionContext batch ({batch}) must match the context \
                 batch size ({})",
                self.batch_size,
            )));
        }

        let hop = self.hop();
        if samples == 0 || samples % hop != 0 {
            return Err(BunsenError::Invalid(format!(
                "MelConversionContext samples ({samples}) must be a non-zero \
                 multiple of hop ({hop})",
            )));
        }

        let extended = match self.phase {
            StreamPhase::Running => match self.carry.take() {
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
                            return Err(BunsenError::Invalid(format!(
                                "MelConversionContext reflect start padding \
                                 needs at least {} samples in the first chunk, \
                                 got {samples}",
                                self.min_first_chunk(),
                            )));
                        }
                        let mirror = waves.clone().slice_dim(1, 1..(pad + 1) as isize).flip([1]);
                        Tensor::cat(vec![mirror, waves], 1)
                    }
                }
            }
        };

        let frames = self.converter.frame_count(extended.dims()[1]);
        if frames == 0 {
            return Err(BunsenError::Invalid(format!(
                "MelConversionContext first chunk of {samples} samples is too \
                 short to fill a frame; with {:?} start padding it needs at \
                 least {} samples",
                self.start_padding(),
                self.n_fft() - self.start_padding().pad_len(self.n_fft()),
            )));
        }

        // Everything past the last frame's start belongs to the next call.
        // Deriving it from the frame count keeps this correct on any geometry,
        // rather than depending on a closed form for the carry length.
        let consumed = frames * hop;
        self.carry = Some(extended.clone().slice_dim(1, consumed as isize..));
        self.phase = StreamPhase::Running;

        Ok((extended, self))
    }

    /// Stage 2: sample-domain preprocessing.
    ///
    /// Currently the identity. This is where pre-emphasis and DC removal will
    /// land; both are rejected by
    /// [`validate`](crate::ops::signal::mels::MelConverterOptions::validate)
    /// until then, so the stage cannot silently drop a configured filter.
    ///
    /// `[batch, extended]` -> `[batch, extended]`.
    #[doc(hidden)]
    pub(crate) fn t_stage_preproc(
        self,
        x: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Self) {
        (x, self)
    }

    /// Stage 3: framing and windowing.
    ///
    /// `[batch, extended]` -> `[batch, frames, n_fft]`.
    #[doc(hidden)]
    pub(crate) fn t_stage_frame(
        self,
        x: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Self) {
        let framed = self.converter.frame(x);
        (framed, self)
    }

    /// Stage 4: frames to a power or magnitude spectrum.
    ///
    /// `[batch, frames, n_fft]` -> `[batch, frames, n_bins]`.
    #[doc(hidden)]
    pub(crate) fn t_stage_spectrum(
        self,
        x: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Self) {
        let spectrum = self.converter.spectrum(x);
        (spectrum, self)
    }

    /// Stage 5: spectrum onto the mel scale.
    ///
    /// `[batch, frames, n_bins]` -> `[batch, frames, n_mels]`.
    #[doc(hidden)]
    pub(crate) fn t_stage_mel(
        self,
        x: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Self) {
        let mels = self.converter.mel(x);
        (mels, self)
    }

    /// Stage 6: mel energies to log-mels.
    ///
    /// `[batch, frames, n_mels]` -> `[batch, frames, n_mels]`.
    #[doc(hidden)]
    pub(crate) fn t_stage_compress(
        self,
        x: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Self) {
        let mels = self.converter.compress(x);
        (mels, self)
    }
}

impl<B: Backend> MelConverter<B> {
    /// Opens a streaming conversion over these constants.
    ///
    /// # Arguments
    /// * `batch_size`: the number of independent streams; must be non-zero.
    pub fn new_context(
        &self,
        batch_size: usize,
    ) -> MelConversionContext<B> {
        assert_ne!(batch_size, 0, "MelConverter batch_size must be non-zero",);

        MelConversionContext {
            converter: self.clone(),
            carry: None,
            batch_size,
            phase: StreamPhase::Start,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;
    use crate::{
        burner::{
            module::ModuleInit,
            tensor::TensorDataToVecAsExt,
        },
        errors::WithOkOrPanic,
        ops::signal::mels::{
            MelConverterOptions,
            RangeClamp,
        },
        support::testing::{
            PerformanceBackend,
            assert_close_to_vec,
            assert_tensor_close_to_vec,
            assert_tensors_close,
        },
    };

    type B = PerformanceBackend;

    /// Builds a `[batch, samples]` tensor from a row-major host buffer.
    fn from_rows<B: Backend>(
        rows: &[Vec<f64>],
        device: &burn::prelude::Device<B>,
    ) -> Tensor<B, 2> {
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

    /// Streaming is only a homomorphism when the clamp does not depend on the
    /// call, so the chunking tests disable it.
    fn streaming_options() -> MelConverterOptions {
        MelConverterOptions::default().with_range_clamp(None)
    }

    #[test]
    fn test_context_meta_and_lifecycle() {
        let device = Default::default();
        let opts = streaming_options();
        let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();

        let ctx = conv.new_context(2);

        assert_eq!(ctx.batch_size(), 2);
        assert_eq!(ctx.phase(), StreamPhase::Start);
        assert!(ctx.carry().is_none());

        // Meta reads the same through the context as through the converter.
        assert_eq!(ctx.n_fft(), conv.n_fft());
        assert_eq!(ctx.hop(), conv.hop());
        assert_eq!(ctx.n_mels(), conv.n_mels());
        assert_eq!(ctx.n_bins(), conv.n_bins());
        assert_eq!(ctx.min_first_chunk(), conv.min_first_chunk());

        let x = from_rows::<B>(&rows(2, 1600), &device);
        let (_, ctx) = ctx.transform(x).unwrap();

        assert_eq!(ctx.phase(), StreamPhase::Running);
        assert!(ctx.carry().is_some());

        let mut ctx = ctx;
        ctx.reset();
        assert_eq!(ctx.phase(), StreamPhase::Start);
        assert!(ctx.carry().is_none());
    }

    /// The frame accounting worked out in `MEL_CONVERTER_PLAN.md`, asserted
    /// against the real thing.
    #[test]
    fn test_frame_accounting_over_a_30s_window() {
        let device = Default::default();
        let opts = streaming_options();
        let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();

        // 30 s at the default 16 kHz sample rate.
        let samples = 480_000;
        let x = from_rows::<B>(&rows(1, samples), &device);

        let (mels, ctx) = conv.new_context(1).transform(x).unwrap();

        // Start padding is 200, which is not a hop multiple, so the first call
        // is short of `samples / hop` and leaves a 360-sample carry.
        assert_eq!(mels.dims(), [1, 2999, opts.n_mels]);
        assert_eq!(ctx.carry().unwrap().dims(), [1, 360]);

        let tail = ctx.finish().expect("reflect end padding yields a tail");
        assert_eq!(tail.dims(), [1, 2, opts.n_mels]);

        // 2999 + 2 == 3001, which is what `librosa` gives with center=True.
        assert_eq!(2999 + 2, 3001);
    }

    /// In steady state each chunk yields exactly `samples / hop` frames and
    /// the carry length does not drift.
    #[test]
    fn test_running_frame_count_is_invariant() {
        let device = Default::default();
        let conv: MelConverter<B> = streaming_options().try_init(&device).ok_or_panic();
        let hop = conv.hop();

        let mut ctx = conv.new_context(2);
        let mut carry_len = None;

        for (step, chunk) in [3200, 1600, 4800, 1600].into_iter().enumerate() {
            let x = from_rows::<B>(&rows(2, chunk), &device);
            let (mels, next) = ctx.transform(x).unwrap();
            ctx = next;

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
    fn test_chunked_transform_is_a_homomorphism() {
        let device = Default::default();
        let conv: MelConverter<B> = streaming_options().try_init(&device).ok_or_panic();

        let (batch, total) = (3, 32_000);
        let host = rows(batch, total);

        // Whole-signal reference, plus its tail.
        let whole = from_rows::<B>(&host, &device);
        let (whole_mels, whole_ctx) = conv.new_context(batch).transform(whole).unwrap();
        let whole_tail = whole_ctx.finish().unwrap();

        // Several hop-aligned splits, including a ragged one.
        for split in [
            vec![32_000],
            vec![16_000, 16_000],
            vec![320, 31_680],
            vec![6_400, 3_200, 12_800, 9_600],
        ] {
            assert_eq!(split.iter().sum::<usize>(), total);

            let mut ctx = conv.new_context(batch);
            let mut pieces = Vec::new();
            let mut at = 0;

            for n in &split {
                let chunk: Vec<Vec<f64>> = host.iter().map(|r| r[at..at + n].to_vec()).collect();
                let (mels, next) = ctx.transform(from_rows::<B>(&chunk, &device)).unwrap();
                ctx = next;
                pieces.push(mels);
                at += n;
            }

            let joined: Tensor<B, 3> = Tensor::cat(pieces, 1);
            assert_eq!(
                joined.dims(),
                whole_mels.dims(),
                "split {split:?} produced a different frame count",
            );

            // Same backend, same dtype, same arithmetic — but the chunks are
            // concatenated differently, so allow a small float tolerance.
            assert_tensors_close(&joined, &whole_mels, Tolerance::absolute(1e-4));

            let tail = ctx.finish().unwrap();
            assert_tensors_close(&tail, &whole_tail, Tolerance::absolute(1e-4));
        }
    }

    #[test]
    fn test_batch_rows_are_independent() {
        let device = Default::default();
        let conv: MelConverter<B> = streaming_options().try_init(&device).ok_or_panic();

        let (batch, samples) = (3, 8_000);
        let host = rows(batch, samples);

        let (together, _) = conv
            .new_context(batch)
            .transform(from_rows::<B>(&host, &device))
            .unwrap();

        let dims = together.dims();
        let per_row = dims[1] * dims[2];
        let together = together.to_data().to_vec_as::<f64>().unwrap();

        for row in 0..batch {
            let (alone, _) = conv
                .new_context(1)
                .transform(from_rows::<B>(&host[row..row + 1], &device))
                .unwrap();

            assert_close_to_vec(
                &alone.to_data().to_vec_as::<f64>().unwrap(),
                &together[row * per_row..(row + 1) * per_row],
                1e-4,
            );
        }
    }

    #[test]
    fn test_transform_rejects_bad_input() {
        let device = Default::default();
        let conv: MelConverter<B> = streaming_options().try_init(&device).ok_or_panic();

        // Not a hop multiple.
        let ctx = conv.new_context(2);
        let bad = from_rows::<B>(&rows(2, 1601), &device);
        assert!(matches!(ctx.transform(bad), Err(BunsenError::Invalid(_)),));

        // Wrong batch.
        let ctx = conv.new_context(2);
        let bad = from_rows::<B>(&rows(3, 1600), &device);
        assert!(matches!(ctx.transform(bad), Err(BunsenError::Invalid(_)),));

        // Too short to reflect: `min_first_chunk` is 201, and 160 is the
        // largest hop-aligned chunk below it.
        let ctx = conv.new_context(1);
        let bad = from_rows::<B>(&rows(1, 160), &device);
        assert!(matches!(ctx.transform(bad), Err(BunsenError::Invalid(_)),));
    }

    #[test]
    fn test_finish_respects_end_padding() {
        let device = Default::default();

        // No end padding: nothing to flush.
        let unpadded: MelConverter<B> = streaming_options()
            .with_end_padding(PaddingMode::None)
            .try_init(&device)
            .ok_or_panic();
        let (_, ctx) = unpadded
            .new_context(1)
            .transform(from_rows::<B>(&rows(1, 3200), &device))
            .unwrap();
        assert!(ctx.finish().is_none());

        // Nothing transformed yet: nothing to flush either.
        assert!(unpadded.new_context(1).finish().is_none());

        // Zero end padding produces the same frame count as reflect; only the
        // values differ.
        let zero: MelConverter<B> = streaming_options()
            .with_end_padding(PaddingMode::Zero)
            .try_init(&device)
            .ok_or_panic();
        let (_, ctx) = zero
            .new_context(1)
            .transform(from_rows::<B>(&rows(1, 3200), &device))
            .unwrap();
        assert_eq!(ctx.finish().unwrap().dims()[1], 2);
    }

    /// Driving one stage at a time must match driving the whole fold — the
    /// payoff for the stack being decomposed.
    #[test]
    fn test_stage_stack_matches_transform() {
        let device = Default::default();
        let conv: MelConverter<B> = streaming_options().try_init(&device).ok_or_panic();

        let x = from_rows::<B>(&rows(2, 3200), &device);

        let (folded, _) = conv.new_context(2).transform(x.clone()).unwrap();

        let ctx = conv.new_context(2);
        let (a, ctx) = ctx.t_stage_extend(x).unwrap();
        let (a, ctx) = ctx.t_stage_preproc(a);
        let (a, ctx) = ctx.t_stage_frame(a);
        let (a, ctx) = ctx.t_stage_spectrum(a);
        let (a, ctx) = ctx.t_stage_mel(a);
        let (staged, _) = ctx.t_stage_compress(a);

        assert_tensors_close(&staged, &folded, Tolerance::default());
    }

    /// `t_stage_extend` in isolation: the carry contents must be exactly the
    /// tail of the extended signal, and the extension must start with the
    /// mirrored prefix.
    #[test]
    fn test_extend_stage_carry_contents() {
        let device = Default::default();
        let conv: MelConverter<B> = streaming_options().try_init(&device).ok_or_panic();
        let (n_fft, hop) = (conv.n_fft(), conv.hop());
        let pad = n_fft / 2;

        let host = rows(1, 1600);
        let x = from_rows::<B>(&host, &device);

        let (ext, ctx) = conv.new_context(1).t_stage_extend(x).unwrap();
        let ext_host = ext.to_data().to_vec_as::<f64>().unwrap();

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

    /// `PerCall` is documented as not chunk-invariant; pin that so nobody
    /// "fixes" the homomorphism tests by turning it back on.
    ///
    /// The signal has to earn it: `db` is in log-units, so the default 8.0 is
    /// an 80 dB window that ordinary noise never spans — the clamp simply
    /// never fires and both chunkings agree. Making the second half silent is
    /// what separates them, and it is the realistic case: a clip zero-padded
    /// to a fixed window ends in silence, which is what the clamp exists to
    /// floor.
    #[test]
    fn test_per_call_clamp_is_not_chunk_invariant() {
        let device = Default::default();
        let conv: MelConverter<B> = MelConverterOptions::default()
            .with_range_clamp(Some(RangeClamp::PerCall { db: 8.0 }))
            .try_init(&device)
            .ok_or_panic();

        // Loud first half, digital silence second half.
        let host: Vec<Vec<f64>> = rows(1, 6400)
            .into_iter()
            .map(|r| {
                r.into_iter()
                    .enumerate()
                    .map(|(i, v)| if i < 3200 { v } else { 0.0 })
                    .collect()
            })
            .collect();

        let (whole, _) = conv
            .new_context(1)
            .transform(from_rows::<B>(&host, &device))
            .unwrap();

        let mut ctx = conv.new_context(1);
        let mut pieces = Vec::new();
        for at in [0, 3200] {
            let chunk: Vec<Vec<f64>> = host.iter().map(|r| r[at..at + 3200].to_vec()).collect();
            let (mels, next) = ctx.transform(from_rows::<B>(&chunk, &device)).unwrap();
            ctx = next;
            pieces.push(mels);
        }
        let joined: Tensor<B, 3> = Tensor::cat(pieces, 1);

        assert_eq!(joined.dims(), whole.dims());

        let (a, b) = (
            joined.to_data().to_vec_as::<f64>().unwrap(),
            whole.to_data().to_vec_as::<f64>().unwrap(),
        );
        let differs = a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-6);
        assert!(
            differs,
            "PerCall clamp unexpectedly matched across chunkings; if this \
             became true, the clamp is no longer per-call",
        );

        // The whole-signal run floors the silence against the loud maximum;
        // the chunked run floors it against the quiet chunk's own maximum, so
        // it sits lower. That direction is the whole point.
        let quietest_whole = b.iter().copied().fold(f64::INFINITY, f64::min);
        let quietest_joined = a.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            quietest_joined < quietest_whole,
            "expected the chunked run to floor lower ({quietest_joined} vs \
             {quietest_whole})",
        );
    }
}
