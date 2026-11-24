//! The codec decoder's stream.

use alloc::vec::Vec;

use burn::{
    module::Module,
    prelude::*,
    tensor::{
        DType,
        module::conv1d,
        ops::ConvOptions,
    },
};

use super::{
    Zonos3Codec,
    patch_up,
};
use crate::{
    blocks::transformers::decoders::stack::DecoderStackMeta,
    burner::module::ModuleInit,
    contracts::unpack_shape_contract,
    ops::{
        conv::{
            CausalConvStreamContext,
            CausalPadding,
        },
        signal::{
            InverseStftStreamContext,
            kernels::sinc::SincResampleStreamContext,
        },
        transformers::attention::KVCache,
    },
};

/// The positions a stage's KV cache allocates up front; its windowed layers
/// keep their own windows whatever this is.
const STREAM_CACHE_PRE_SIZE: usize = 128;

/// One stream through a [`Zonos3Codec`]: each causal layer's state, and the
/// samples its non-causal tail still waits on.
///
/// The trunk and the heads are causal: their state is the `ConvNeXt` blocks'
/// last inputs and each stage's windowed KV cache, at that stage's frame
/// rate. The tail is not: each iSTFT branch returns a sample once no later
/// frame reaches it, the centred mixer once it has seen the half-kernel
/// after it, and the resampler once it has its kernel's width. So a stream
/// lags the frames it was given by a fixed number of samples, which
/// [`finish_stream`](Zonos3Codec::finish_stream) returns; joined, the
/// stream's audio is [`forward`](Zonos3Codec::forward)'s over the whole take.
///
/// Made by [`Zonos3Codec::new_stream_context`]. The codec never owns one, so
/// one codec serves any number of streams; each batch row is one stream.
///
/// A `Module` over bare tensors: moved with `to_device`, never recorded or
/// visited.
#[derive(Module, Debug)]
pub struct Zonos3CodecStreamContext {
    /// Each semantic `ConvNeXt` block's last inputs; `None` before the first.
    pub semantic: Vec<Option<Tensor<3>>>,

    /// Each trunk stage's KV cache.
    pub stages: Vec<KVCache>,

    /// Each head `ConvNeXt` block's last inputs; `None` before the first.
    pub head: Vec<Option<Tensor<3>>>,

    /// Each iSTFT branch's stream.
    pub istfts: Vec<InverseStftStreamContext>,

    /// `[B, n_i]`: each branch's samples not yet mixed; `None` when none.
    pub branches: Vec<Option<Tensor<2>>>,

    /// The mixer's stream, as a causal convolution `(K - 1) / 2` behind the
    /// centred one.
    pub mixer: CausalConvStreamContext,

    /// The mixer's outputs still to drop: the causal convolution's first
    /// `(K - 1) / 2`, before the centred one's first.
    pub mixer_skip: usize,

    /// The resampler's stream.
    pub resampler: SincResampleStreamContext,

    /// The frames of codes decoded so far.
    pub frames: usize,

    /// The head's frames so far (the trunk's last rate).
    pub head_frames: usize,

    /// The output samples returned so far.
    pub emitted: usize,

    /// Whether [`finish_stream`](Zonos3Codec::finish_stream) has run.
    pub finished: bool,
}

impl Zonos3Codec {
    /// The mixer's half-kernel, `(K - 1) / 2`: the samples it reads past
    /// each output.
    fn mixer_half(&self) -> usize {
        self.mixer.weight.dims()[2] / 2
    }

    /// A stream through the codec, not yet started, of `batch` rows on
    /// `device`.
    pub fn new_stream_context(
        &self,
        batch: usize,
        device: &Device,
    ) -> Zonos3CodecStreamContext {
        let k = self.mixer.weight.dims()[2];
        Zonos3CodecStreamContext {
            semantic: self.semantic_blocks.iter().map(|_| None).collect(),
            stages: self
                .stages
                .iter()
                .map(|stage| {
                    stage
                        .stack
                        .kv_cache_config(STREAM_CACHE_PRE_SIZE)
                        .with_batch_size(batch)
                        .init(device)
                })
                .collect(),
            head: self.head_blocks.iter().map(|_| None).collect(),
            istfts: self
                .istfts
                .iter()
                .map(|istft| istft.new_stream_context())
                .collect(),
            branches: self.istfts.iter().map(|_| None).collect(),
            mixer: CausalConvStreamContext::new(k, 1, CausalPadding::Zeros),
            mixer_skip: self.mixer_half(),
            resampler: self.resampler.new_stream_context(),
            frames: 0,
            head_frames: 0,
            emitted: 0,
            finished: false,
        }
    }

    /// The stream's next frames of codes, to the samples now final.
    ///
    /// # Arguments
    /// - `codes`: `[B, Q, T]`, `T ≥ 1`, the take's next frames: the first
    ///   codebook's codes, then the residual codebooks'.
    /// - `ctx`: the stream, from
    ///   [`new_stream_context`](Self::new_stream_context).
    ///
    /// # Returns
    /// `[B, n]`, f32, the samples now final; `None` when none is (the
    /// stream's first frames fill its tail's lookahead).
    ///
    /// # Panics
    /// For a finished stream.
    pub fn forward_stream(
        &self,
        codes: Tensor<3, Int>,
        ctx: &mut Zonos3CodecStreamContext,
    ) -> Option<Tensor<2>> {
        assert!(!ctx.finished, "a push to a finished ZONOS3 codec stream");
        let [b, t] = unpack_shape_contract!(
            ["B", "Q", "T"],
            &codes.dims(),
            &["B", "T"],
            &[("Q", self.quantizers.len())]
        );
        let codebook = |i: usize| codes.clone().slice(s![.., i..i + 1, ..]).reshape([b, t]);

        let mut semantic = self.quantizers[0].forward(codebook(0));
        for (block, state) in self.semantic_blocks.iter().zip(&mut ctx.semantic) {
            let (y, next) = block.forward_with_state(semantic, state.take());
            *state = Some(next);
            semantic = y;
        }
        let latent = self
            .quantizers
            .iter()
            .enumerate()
            .skip(1)
            .fold(self.semantic_proj.forward(semantic), |z, (i, q)| {
                z + q.forward(codebook(i))
            });

        let mut x = latent;
        for (i, (stage, cache)) in self.stages.iter().zip(&mut ctx.stages).enumerate() {
            if i > 0 {
                x = patch_up(x);
            }
            x = stage.forward_cached(x, cache);
        }

        let frames = x.dims()[1];
        let mut h = x;
        for (block, state) in self.head_blocks.iter().zip(&mut ctx.head) {
            let (y, next) = block.forward_with_state(h, state.take());
            *state = Some(next);
            h = y;
        }
        let h = self.head_norm.forward(h).cast(DType::F32);
        for i in 0..self.istfts.len() {
            let (real, imag) = self.branch_spectra(i, h.clone());
            if let Some(samples) = self.istfts[i].istft_stream(real, imag, &mut ctx.istfts[i]) {
                append(&mut ctx.branches[i], samples);
            }
        }
        ctx.frames += t;
        ctx.head_frames += frames;
        let mixed = self.mix_ready(ctx, None);
        self.resample_ready(ctx, mixed)
    }

    /// The end of the stream: the samples its tail held back, to the whole
    /// decode's `T · hop`.
    ///
    /// # Returns
    /// `[B, n]`, f32; `None` when nothing is left, no frame was pushed, or
    /// the stream was finished already.
    pub fn finish_stream(
        &self,
        ctx: &mut Zonos3CodecStreamContext,
    ) -> Option<Tensor<2>> {
        if ctx.finished {
            return None;
        }
        ctx.finished = true;
        if ctx.frames == 0 {
            return None;
        }
        let length = ctx.head_frames * self.head_hop;
        for i in 0..self.istfts.len() {
            if let Some(samples) = self.istfts[i].finish_stream(&mut ctx.istfts[i], Some(length)) {
                append(&mut ctx.branches[i], samples);
            }
        }
        // The centred mixer's zeros after the signal.
        let half = self.mixer_half();
        let mixed = self.mix_ready(ctx, Some(half));
        let mut parts = Vec::with_capacity(2);
        parts.extend(self.resample_ready(ctx, mixed));
        if let Some(y) = self.resampler.finish_stream(&mut ctx.resampler) {
            ctx.emitted += y.dims()[1];
            parts.push(y);
        }
        let audio = (!parts.is_empty()).then(|| Tensor::cat(parts, 1))?;
        // The whole decode's cut to `T · hop`.
        let total = ctx.frames * self.hop;
        let over = ctx.emitted.saturating_sub(total);
        let n = audio.dims()[1];
        ctx.emitted -= over;
        (n > over).then(|| audio.narrow(1, 0, n - over))
    }

    /// The branches' samples that every branch has, through the mixer:
    /// `[B, n]` mixed; `flush` zeros pushed after them first (the finish's).
    fn mix_ready(
        &self,
        ctx: &mut Zonos3CodecStreamContext,
        flush: Option<usize>,
    ) -> Option<Tensor<2>> {
        let ready = ctx
            .branches
            .iter()
            .map(|b| b.as_ref().map_or(0, |b| b.dims()[1]))
            .min()
            .unwrap_or(0);
        let mut input = None;
        if ready > 0 {
            let taken: Vec<Tensor<2>> = ctx
                .branches
                .iter_mut()
                .map(|slot| take_front(slot, ready))
                .collect();
            input = Some(Tensor::stack::<3>(taken, 1));
        }
        if let Some(zeros) = flush.filter(|&z| z > 0) {
            let like = input.as_ref().map(|x: &Tensor<3>| (x.dims()[0], x.dtype()));
            let (rows, dtype) = like.unwrap_or((ctx.istfts[0].batch, DType::F32));
            let pad = Tensor::zeros(
                [rows, self.istfts.len(), zeros],
                (&self.mixer.weight.device(), dtype),
            );
            input = Some(match input {
                Some(x) => Tensor::cat(alloc::vec![x, pad], 2),
                None => pad,
            });
        }
        let window = ctx.mixer.push(input?)?;
        let options = ConvOptions {
            stride: [1],
            padding: [(0, 0)],
            dilation: [1],
            groups: 1,
        };
        let y = conv1d(window, self.mixer.weight.val(), None, options);
        let n = y.dims()[2];
        let skip = ctx.mixer_skip.min(n);
        ctx.mixer_skip -= skip;
        (n > skip).then(|| y.narrow(2, skip, n - skip).squeeze_dim::<2>(1))
    }

    /// Mixed samples through the resampler: the output samples now final.
    fn resample_ready(
        &self,
        ctx: &mut Zonos3CodecStreamContext,
        mixed: Option<Tensor<2>>,
    ) -> Option<Tensor<2>> {
        let y = self.resampler.resample_stream(mixed?, &mut ctx.resampler)?;
        ctx.emitted += y.dims()[1];
        Some(y)
    }
}

/// `samples` after what `slot` holds.
fn append(
    slot: &mut Option<Tensor<2>>,
    samples: Tensor<2>,
) {
    *slot = Some(match slot.take() {
        Some(held) => Tensor::cat(alloc::vec![held, samples], 1),
        None => samples,
    });
}

/// The first `n` samples `slot` holds, the rest left in it.
fn take_front(
    slot: &mut Option<Tensor<2>>,
    n: usize,
) -> Tensor<2> {
    let held = slot.take().expect("samples to take");
    let len = held.dims()[1];
    if len > n {
        *slot = Some(held.clone().narrow(1, n, len - n));
    }
    held.narrow(1, 0, n)
}
