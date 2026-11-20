//! A depth decoder: a frame's codebooks, one after another, from the
//! backbone's hidden state.
//!
//! An audio language model whose frame is several codebooks deep (CSM's,
//! Kyutai's DSM's, Dia2's, Qwen3-TTS's code predictor) predicts the first
//! with its backbone and the rest with a second, small decoder that runs
//! along the frame's depth: [`DepthDecoder`]. Each frame starts it afresh,
//! from rows the backbone hands it (its last hidden state, the first
//! codebook's embedding), and each codebook after that is a step through
//! its own cache: the codebook before's code embedded by that codebook's
//! own table, the next codebook's logits read by its own head.

use alloc::vec::Vec;

use burn::{
    nn::{
        Embedding,
        EmbeddingConfig,
        Linear,
        LinearConfig,
    },
    prelude::*,
    tensor::FloatDType,
};

use super::{
    causal::CausalDecoderBlockMeta,
    stack::{
        DecoderStack,
        DecoderStackMeta,
        UniformStackConfig,
    },
};
use crate::{
    blocks::transformers::embedding::RotarySet,
    burner::module::{
        CanonicalKeys,
        ModuleInit,
        Rekey,
    },
    contracts::unpack_shape_contract,
    errors::BunsenResult,
    ops::transformers::attention::{
        KVCache,
        KVCacheConfig,
    },
};

/// Common meta for [`DepthDecoder`] and [`DepthDecoderConfig`].
pub trait DepthDecoderMeta {
    /// The width of the rows it is given and of its tables: the backbone's.
    fn input_width(&self) -> usize;

    /// Its stack's width.
    fn width(&self) -> usize;

    /// The codebooks it predicts, `K`: codebooks `1..=K` of a frame whose
    /// codebook 0 the backbone predicts.
    fn codebooks(&self) -> usize;

    /// Codes per codebook.
    fn vocab(&self) -> usize;
}

/// Config for [`DepthDecoder`].
#[derive(Config, Debug)]
pub struct DepthDecoderConfig {
    /// The width of the rows it is given and of its tables: the backbone's.
    pub input_width: usize,

    /// The codebooks it predicts, `K`.
    pub codebooks: usize,

    /// Codes per codebook.
    pub vocab: usize,

    /// The stack; its layers' width is the decoder's
    /// [`width`](DepthDecoderMeta::width).
    pub stack: UniformStackConfig,

    /// The dtype the logits leave in; `None`, the model's.
    #[config(default = "Some(FloatDType::F32)")]
    pub logits_dtype: Option<FloatDType>,
}

impl DepthDecoderMeta for DepthDecoderConfig {
    fn input_width(&self) -> usize {
        self.input_width
    }

    fn width(&self) -> usize {
        self.stack.layer.n_embed()
    }

    fn codebooks(&self) -> usize {
        self.codebooks
    }

    fn vocab(&self) -> usize {
        self.vocab
    }
}

impl ModuleInit<DepthDecoder> for DepthDecoderConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<DepthDecoder> {
        let width = self.width();
        Ok(DepthDecoder {
            input_proj: (self.input_width != width)
                .then(|| LinearConfig::new(self.input_width, width).init(device)),
            stack: self.stack.try_init(device)?,
            tables: (0..self.codebooks)
                .map(|_| EmbeddingConfig::new(self.vocab, self.input_width).init(device))
                .collect(),
            heads: (0..self.codebooks)
                .map(|_| {
                    LinearConfig::new(width, self.vocab)
                        .with_bias(false)
                        .init(device)
                })
                .collect(),
            logits_dtype: self.logits_dtype,
        })
    }
}

/// A depth decoder (the [module docs](self)): a small causal decoder over
/// one frame's codebooks.
///
/// ```text
/// h   = stack(proj(context))                        (a fresh cache: the backbone's rows)
/// c_1 = choose(heads[1](h_last))
/// h   = stack(proj(tables[k - 1](c_{k-1})))         (k = 2..=K, one row a step, cached)
/// c_k = choose(heads[k](h))
/// ```
///
/// `tables[k]` embeds codebook `k`'s code, at the backbone's width (the
/// backbone sums a frame's embeddings through them too); `heads[k]` reads
/// codebook `k`'s logits. Rows enter through a projection to the stack's
/// width (`input_proj`) when the widths differ.
///
/// Built by [`DepthDecoderConfig`].
#[derive(Module, Debug)]
pub struct DepthDecoder {
    /// The projection from the backbone's width, when it is not the stack's.
    pub input_proj: Option<Linear>,

    /// The stack: its layers, final norm and rotary tables.
    pub stack: DecoderStack,

    /// Each codebook's table, `[V, D_in]`: codebooks `1..=K`, by index
    /// `k - 1`.
    pub tables: Vec<Embedding>,

    /// Each codebook's head, `[D, V]`, without a bias: codebooks `1..=K`, by
    /// index `k - 1`.
    pub heads: Vec<Linear>,

    /// The dtype the logits leave in.
    #[module(skip)]
    pub logits_dtype: Option<FloatDType>,
}

impl DepthDecoderMeta for DepthDecoder {
    fn input_width(&self) -> usize {
        self.tables[0].weight.dims()[1]
    }

    fn width(&self) -> usize {
        self.heads[0].weight.dims()[0]
    }

    fn codebooks(&self) -> usize {
        self.heads.len()
    }

    fn vocab(&self) -> usize {
        self.heads[0].weight.dims()[1]
    }
}

impl DepthDecoder {
    /// A cache for one frame: `context` rows and a step per codebook after
    /// the first.
    pub fn kv_cache_config(
        &self,
        context: usize,
    ) -> KVCacheConfig {
        self.stack.kv_cache_config(context + self.codebooks())
    }

    /// The rotary tables for positions `0..positions`.
    pub fn rotary(
        &self,
        positions: usize,
        device: &Device,
    ) -> RotarySet {
        self.stack.rotary(positions, device)
    }

    /// Codebook `k`'s codes, by its table: `[B, T]` to `[B, T, D_in]`.
    ///
    /// # Panics
    /// Unless `1 ≤ k ≤ K`.
    pub fn embed(
        &self,
        k: usize,
        codes: Tensor<2, Int>,
    ) -> Tensor<3> {
        self.tables[self.index(k)].forward(codes)
    }

    /// Rows through the projection and the stack, at positions
    /// `start..start + T`.
    ///
    /// # Arguments
    /// - `rows`: `[B, T, D_in]`.
    /// - `r_emb`: rotary tables covering `start + T` positions.
    /// - `kv_cache`: the frame's cache, at `start`; `None`, `start` is 0.
    ///
    /// # Returns
    /// `[B, T, D]`, normalized.
    pub fn forward_hidden(
        &self,
        rows: Tensor<3>,
        r_emb: &RotarySet,
        kv_cache: Option<&mut KVCache>,
    ) -> Tensor<3> {
        let rows = match &self.input_proj {
            Some(proj) => proj.forward(rows),
            None => rows,
        };
        self.stack.forward_hidden(rows, r_emb, kv_cache)
    }

    /// Codebook `k`'s logits, by its head: `[B, T, D]` to `[B, T, V]`, in
    /// `logits_dtype`.
    ///
    /// # Panics
    /// Unless `1 ≤ k ≤ K`.
    pub fn logits(
        &self,
        k: usize,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        let logits = self.heads[self.index(k)].forward(hidden);
        match self.logits_dtype {
            Some(dtype) => logits.cast(dtype),
            None => logits,
        }
    }

    /// One frame's codebooks `1..=K`, from the backbone's `context` rows, a
    /// fresh cache, each codebook's code chosen from its logits.
    ///
    /// # Arguments
    /// - `context`: `[B, P, D_in]`, `P ≥ 1`: what the backbone hands the frame
    ///   (its last hidden state, the first codebook's embedding).
    /// - `choose`: codebook `k` and its logits, `[B, V]`, to its codes, `[B]`:
    ///   a sampler's choice, or a script's.
    ///
    /// # Returns
    /// `[B, K]`, codebooks `1..=K`.
    pub fn decode(
        &self,
        context: Tensor<3>,
        mut choose: impl FnMut(usize, Tensor<2>) -> Tensor<1, Int>,
    ) -> Tensor<2, Int> {
        let [b, p] = unpack_shape_contract!(
            ["B", "P", "D_in"],
            &context.dims(),
            &["B", "P"],
            &[("D_in", self.input_width())]
        );
        assert!(p >= 1, "a depth decoder's context of at least one row");
        let k = self.codebooks();
        let device = context.device();
        let r_emb = self.rotary(p + k - 1, &device);
        let mut cache = self.kv_cache_config(p).with_batch_size(b).init(&device);
        let last = |hidden: Tensor<3>| {
            let t = hidden.dims()[1];
            hidden.slice(s![.., t - 1..t, ..])
        };
        let mut hidden = last(self.forward_hidden(context, &r_emb, Some(&mut cache)));
        let mut codes = Vec::with_capacity(k);
        for book in 1..=k {
            let logits = self.logits(book, hidden.clone()).reshape([b, self.vocab()]);
            let chosen = choose(book, logits);
            if book < k {
                let row = self.embed(book, chosen.clone().reshape([b, 1]));
                hidden = self.forward_hidden(row, &r_emb, Some(&mut cache));
            }
            codes.push(chosen.reshape([b, 1]));
        }
        Tensor::cat(codes, 1)
    }

    fn index(
        &self,
        k: usize,
    ) -> usize {
        assert!(
            (1..=self.codebooks()).contains(&k),
            "codebook {k} of a depth decoder over 1..={}",
            self.codebooks()
        );
        k - 1
    }
}

impl CanonicalKeys for DepthDecoder {
    fn canonical_rekey(&self) -> Rekey {
        Rekey::keep_rest().mount_canonical("stack.", &self.stack)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::RmsNormConfig,
        tensor::{
            Distribution,
            Tolerance,
        },
    };

    use super::*;
    use crate::{
        blocks::{
            mlp::BunsenGatedMlpConfig,
            norm::BunsenNormalizationConfig,
            transformers::{
                attention::csa::CausalSelfAttentionConfig,
                decoders::causal::CausalDecoderBlockConfig,
                embedding::RotaryEmbeddingConfig,
            },
        },
        support::testing::{
            assert_tensors_close,
            cpu_device,
            seeded_tensor,
        },
    };

    fn config(
        input_width: usize,
        width: usize,
    ) -> DepthDecoderConfig {
        let norm = BunsenNormalizationConfig::Rms(RmsNormConfig::new(0));
        let layer = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(2, 1, width)
                .with_head_dim(Some(4))
                .into(),
            BunsenGatedMlpConfig::new(width, 2 * width).into(),
        )
        .with_norm(norm.clone());
        DepthDecoderConfig::new(
            input_width,
            3,
            7,
            UniformStackConfig::new(2, layer, norm, RotaryEmbeddingConfig::new(0, 4)),
        )
    }

    /// The config and the module built from it answer the meta alike; the
    /// projection is there only when the widths differ.
    #[test]
    fn test_meta() {
        let device = cpu_device();
        for (input_width, projected) in [(10, true), (8, false)] {
            let config = config(input_width, 8);
            let model: DepthDecoder = config.init(&device);
            for meta in [&config as &dyn DepthDecoderMeta, &model] {
                assert_eq!(
                    [
                        meta.input_width(),
                        meta.width(),
                        meta.codebooks(),
                        meta.vocab()
                    ],
                    [input_width, 8, 3, 7]
                );
            }
            assert_eq!(model.input_proj.is_some(), projected);
        }
    }

    /// A frame's decode, one codebook a cached step, is the uncached
    /// forward over the context and the chosen codes' rows: each
    /// codebook's logits read at its row.
    #[test]
    fn test_decode_is_the_uncached_forward() {
        let device = cpu_device();
        let model: DepthDecoder = config(10, 8).init(&device);
        let context: Tensor<3> = seeded_tensor(1, [2, 2, 10], Distribution::Default, &device);
        let mut seen = Vec::new();
        let codes = model.decode(context.clone(), |k, logits| {
            seen.push((k, logits.clone()));
            logits.argmax(1).reshape([2])
        });
        assert_eq!(codes.dims(), [2, 3]);
        assert_eq!(
            seen.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );

        let rows = Tensor::cat(
            vec![
                context,
                model.embed(1, codes.clone().slice(s![.., 0..1])),
                model.embed(2, codes.clone().slice(s![.., 1..2])),
            ],
            1,
        );
        let r_emb = model.rotary(4, &device);
        let hidden = model.forward_hidden(rows, &r_emb, None);
        for (k, logits) in seen {
            let row = k; // codebook k's logits come from row P + k - 2, P = 2
            let expected = model
                .logits(k, hidden.clone().slice(s![.., row..row + 1, ..]))
                .reshape([2, 7]);
            assert_tensors_close(&logits, &expected, Tolerance::default());
        }
    }

    #[test]
    #[should_panic(expected = "codebook 4 of a depth decoder over 1..=3")]
    fn test_codebook_checked() {
        let device = cpu_device();
        let model: DepthDecoder = config(8, 8).init(&device);
        model.embed(4, Tensor::zeros([1, 1], &device));
    }
}
