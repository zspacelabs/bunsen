//! The codec decoder's streams: exact, and as the render decodes.

use alloc::vec::Vec;

use burn::{
    module::Module,
    prelude::*,
};

use super::{
    Qwen3TtsCodecDecoder,
    Qwen3TtsCodecUpsample,
    Qwen3TtsDecodeConfig,
};
use crate::{
    blocks::{
        signals::DacDecoderStreamContext,
        transformers::decoders::stack::DecoderStackMeta,
    },
    burner::module::ModuleInit,
    contracts::unpack_shape_contract,
    ops::{
        conv::{
            CausalConvStreamContext,
            CausalConvTransposeStreamContext,
        },
        transformers::attention::KVCache,
    },
};

/// The positions a codec stream's KV cache allocates up front; its windowed
/// layers keep their own windows whatever this is.
const STREAM_CACHE_PRE_SIZE: usize = 128;

/// One stream through a [`Qwen3TtsCodecUpsample`]: its transposed
/// convolution's overlap and its `ConvNeXt` block's last inputs.
///
/// A `Module` over bare tensors: moved with `to_device`, never recorded or
/// visited.
#[derive(Module, Debug)]
pub struct Qwen3TtsCodecUpsampleStreamContext {
    /// The transposed convolution's.
    pub up: CausalConvTransposeStreamContext,

    /// `[B, K - 1, L]`, the `ConvNeXt` block's last inputs
    /// ([`forward_with_state`](crate::blocks::conv::CausalConvNext1d::forward_with_state));
    /// `None` before the first.
    pub block: Option<Tensor<3>>,
}

impl Qwen3TtsCodecUpsample {
    /// A stream through the upsampler, not yet started.
    pub fn new_stream_context(&self) -> Qwen3TtsCodecUpsampleStreamContext {
        Qwen3TtsCodecUpsampleStreamContext {
            up: self.up.new_stream_context(),
            block: None,
        }
    }

    /// The stream's next chunk, `[B, L, T]` to `[B, L, r T]`; joined, the
    /// chunks' outputs are [`forward`](Self::forward)'s.
    pub fn forward_stream(
        &self,
        x: Tensor<3>,
        ctx: &mut Qwen3TtsCodecUpsampleStreamContext,
    ) -> Tensor<3> {
        let up = self.up.forward_stream(x, &mut ctx.up).swap_dims(1, 2);
        let (y, state) = self.block.forward_with_state(up, ctx.block.take());
        ctx.block = Some(state);
        y.swap_dims(1, 2)
    }
}

/// One exact stream through a [`Qwen3TtsCodecDecoder`]: each causal layer's
/// state, so that a chunk of codes' audio is final at once and, joined, the
/// chunks' audio is the whole [`forward`](Qwen3TtsCodecDecoder::forward)'s.
///
/// Made by [`Qwen3TtsCodecDecoder::new_stream_context`], driven by
/// [`forward_stream`](Qwen3TtsCodecDecoder::forward_stream). The decoder
/// never owns one, so one decoder serves any number of streams; each batch
/// row is one stream.
///
/// A `Module` over bare tensors: moved with `to_device`, never recorded or
/// visited.
#[derive(Module, Debug)]
pub struct Qwen3TtsCodecStreamContext {
    /// The pre-conv's.
    pub pre_conv: CausalConvStreamContext,

    /// The transformer's KV cache, each windowed layer keeping its window.
    pub transformer: KVCache,

    /// Each upsampler's, in order.
    pub upsample: Vec<Qwen3TtsCodecUpsampleStreamContext>,

    /// The vocoder's.
    pub vocoder: DacDecoderStreamContext,

    /// The frames of codes decoded so far.
    pub frames: usize,
}

/// One stream through a [`Qwen3TtsCodecDecoder`] as
/// [`chunked_decode`](Qwen3TtsCodecDecoder::chunked_decode) decodes a take,
/// so that, joined, its chunks' audio is the render's.
///
/// The take is cut into chunks of
/// [`chunk`](Qwen3TtsDecodeConfig::chunk) frames from its start. Each chunk
/// is an exact stream ([`Qwen3TtsCodecStreamContext`]) of its own, opened at
/// the chunk's start and first fed the frames of left context the render
/// decodes it with, their audio dropped. A chunk's frames are then decoded as
/// they come, wherever the pushes cut them.
///
/// Made by [`Qwen3TtsCodecDecoder::new_chunked_stream_context`], driven by
/// [`forward_chunked_stream`](Qwen3TtsCodecDecoder::forward_chunked_stream).
///
/// A `Module` over bare tensors: moved with `to_device`, never recorded or
/// visited.
#[derive(Module, Debug)]
pub struct Qwen3TtsChunkedStreamContext {
    /// How the take is cut.
    #[module(skip)]
    pub config: Qwen3TtsDecodeConfig,

    /// The current chunk's exact stream; `None` before the take's first
    /// frame.
    pub stream: Option<Qwen3TtsCodecStreamContext>,

    /// `[B, K, ≤ context]`, the take's last frames of codes, the next
    /// chunk's left context; `None` before the first.
    pub recent: Option<Tensor<3, Int>>,

    /// The frames of the take decoded so far.
    pub frames: usize,
}

impl Qwen3TtsCodecDecoder {
    /// An exact stream through the decoder, not yet started, of `batch`
    /// rows on `device`.
    pub fn new_stream_context(
        &self,
        batch: usize,
        device: &Device,
    ) -> Qwen3TtsCodecStreamContext {
        Qwen3TtsCodecStreamContext {
            pre_conv: self.pre_conv.new_stream_context(),
            transformer: self
                .pre_transformer
                .stack
                .kv_cache_config(STREAM_CACHE_PRE_SIZE)
                .with_batch_size(batch)
                .init(device),
            upsample: self
                .upsample
                .iter()
                .map(Qwen3TtsCodecUpsample::new_stream_context)
                .collect(),
            vocoder: self.decoder.new_stream_context(),
            frames: 0,
        }
    }

    /// The exact stream's next chunk of codes, to its audio: final at once,
    /// and, joined, the chunks' audio is [`forward`](Self::forward)'s over
    /// the whole take.
    ///
    /// # Arguments
    /// - `codes`: `[B, K, T]`, `T ≥ 1`, the take's next frames.
    /// - `ctx`: the stream, from
    ///   [`new_stream_context`](Self::new_stream_context).
    ///
    /// # Returns
    /// `[B, 1, T · samples_per_frame]`, clamped as `forward` clamps.
    pub fn forward_stream(
        &self,
        codes: Tensor<3, Int>,
        ctx: &mut Qwen3TtsCodecStreamContext,
    ) -> Tensor<3> {
        let k = self.quantizer.rvq_first.codebooks.len() + self.quantizer.rvq_rest.codebooks.len();
        let [_, t] =
            unpack_shape_contract!(["B", "K", "T"], &codes.dims(), &["B", "T"], &[("K", k)]);
        let quantized = self.quantizer.decode(codes);
        let z = self
            .pre_conv
            .forward_stream(quantized, &mut ctx.pre_conv)
            .expect("a stride-1 pre-conv completes every frame");
        let z = self
            .pre_transformer
            .forward_cached(z.swap_dims(1, 2), &mut ctx.transformer)
            .swap_dims(1, 2);
        let z = self
            .upsample
            .iter()
            .zip(&mut ctx.upsample)
            .fold(z, |z, (stage, ctx)| stage.forward_stream(z, ctx));
        let y = self.decoder.forward_stream(z, &mut ctx.vocoder);
        ctx.frames += t;
        match self.output_clamp {
            Some(c) => y.clamp(-c, c),
            None => y,
        }
    }

    /// A stream through the decoder as the render decodes a take
    /// ([`chunked_decode`](Self::chunked_decode) under `config`), not yet
    /// started.
    pub fn new_chunked_stream_context(
        &self,
        config: Qwen3TtsDecodeConfig,
    ) -> Qwen3TtsChunkedStreamContext {
        assert!(config.chunk >= 1, "a decode chunk of at least one frame");
        Qwen3TtsChunkedStreamContext {
            config,
            stream: None,
            recent: None,
            frames: 0,
        }
    }

    /// The chunked stream's next frames of codes, to their audio; joined,
    /// the pushes' audio is [`chunked_decode`](Self::chunked_decode)'s over
    /// the whole take, however the pushes cut it.
    ///
    /// # Arguments
    /// - `codes`: `[B, K, T]`, `T ≥ 1`, the take's next frames.
    /// - `ctx`: the stream, from
    ///   [`new_chunked_stream_context`](Self::new_chunked_stream_context).
    ///
    /// # Returns
    /// `[B, 1, T · samples_per_frame]`.
    pub fn forward_chunked_stream(
        &self,
        codes: Tensor<3, Int>,
        ctx: &mut Qwen3TtsChunkedStreamContext,
    ) -> Tensor<3> {
        let Qwen3TtsDecodeConfig { chunk, context } = ctx.config;
        let [b, _, t] = codes.dims();
        let device = codes.device();
        let mut parts = Vec::new();
        let mut at = 0;
        while at < t {
            let start = ctx.frames;
            if start.is_multiple_of(chunk) {
                // A chunk starts: a fresh stream, fed the render's left
                // context (upstream takes one only when a whole one fits
                // after frame 0), its audio dropped.
                let mut stream = self.new_stream_context(b, &device);
                let left = if start > context { context } else { start };
                if left > 0 {
                    let recent = ctx.recent.clone().expect("the frames before the chunk");
                    let n = recent.dims()[2];
                    let _ = self.forward_stream(recent.slice(s![.., .., n - left..n]), &mut stream);
                }
                ctx.stream = Some(stream);
            }
            let take = (chunk - start % chunk).min(t - at);
            let piece = codes.clone().slice(s![.., .., at..at + take]);
            let stream = ctx.stream.as_mut().expect("the chunk's stream");
            parts.push(self.forward_stream(piece.clone(), stream));
            if context > 0 {
                let joined = match ctx.recent.take() {
                    Some(recent) => Tensor::cat(alloc::vec![recent, piece], 2),
                    None => piece,
                };
                let n = joined.dims()[2];
                ctx.recent = Some(joined.slice(s![.., .., n.saturating_sub(context)..n]));
            }
            ctx.frames += take;
            at += take;
        }
        Tensor::cat(parts, 2)
    }
}
