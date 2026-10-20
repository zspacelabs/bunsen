//! # Bidirectional self-attention with rotary position embeddings.
//!
//! [`RopeSelfAttention`]: every position attends to every valid position,
//! queries and keys rotated by a [`RotaryEmbedding`] over the whole input.
//! The encoder layer of NVIDIA's `Nemotron-3-Diarization`.

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
    },
    prelude::Bool,
    tensor::{
        Device,
        activation::softmax,
    },
};

use crate::{
    blocks::transformers::embedding::{
        RotaryEmbedding,
        RotaryEmbeddingMeta,
    },
    burner::module::ModuleInit,
    contracts::unpack_shape_contract,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
};

/// Common meta for [`RopeSelfAttention`] and [`RopeSelfAttentionConfig`].
pub trait RopeSelfAttentionMeta {
    /// The model width.
    fn d_model(&self) -> usize;

    /// The head count.
    fn n_heads(&self) -> usize;

    /// Each head's width: `d_model / n_heads`.
    fn head_dim(&self) -> usize {
        self.d_model() / self.n_heads()
    }
}

/// Config for [`RopeSelfAttention`].
///
/// Implements [`RopeSelfAttentionMeta`].
#[derive(Config, Debug)]
pub struct RopeSelfAttentionConfig {
    /// The model width.
    pub d_model: usize,

    /// The head count; must divide `d_model`, into an even head width.
    pub n_heads: usize,

    /// Whether the query, key and value projections carry a bias.
    #[config(default = "false")]
    pub qkv_bias: bool,

    /// Whether the output projection carries a bias.
    #[config(default = "true")]
    pub output_bias: bool,
}

impl RopeSelfAttentionMeta for RopeSelfAttentionConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_heads(&self) -> usize {
        self.n_heads
    }
}

impl ModuleInit<RopeSelfAttention> for RopeSelfAttentionConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<RopeSelfAttention> {
        if self.n_heads == 0 {
            return Err(
                ConstraintError::zero_or_empty("RopeSelfAttentionConfig", "n_heads").into(),
            );
        }
        if !self.d_model.is_multiple_of(2 * self.n_heads) {
            return Err(ConstraintError::new(
                "RopeSelfAttentionConfig",
                "d_model",
                Rule::NotMultiple {
                    value: self.d_model,
                    of: format!("2 * n_heads ({})", 2 * self.n_heads),
                },
            )
            .into());
        }
        let projection = |bias| {
            LinearConfig::new(self.d_model, self.d_model)
                .with_bias(bias)
                .init(device)
        };
        Ok(RopeSelfAttention {
            query: projection(self.qkv_bias),
            key: projection(self.qkv_bias),
            value: projection(self.qkv_bias),
            output: projection(self.output_bias),
            n_heads: self.n_heads,
        })
    }
}

/// Bidirectional multi-head self-attention with rotary position embeddings.
///
/// Queries and keys are projected, split into heads, and rotated by the
/// caller's [`RotaryEmbedding`]; each query then attends to every key that is
/// not padding, `softmax(q k^T / sqrt(head_dim)) v`, and the heads are
/// joined and projected out. No position is hidden from another: this is an
/// encoder's attention, not a decoder's.
///
/// Built by [`RopeSelfAttentionConfig`].
/// Implements [`RopeSelfAttentionMeta`].
#[derive(Module, Debug)]
pub struct RopeSelfAttention {
    /// `[d_model] -> [d_model]`: the queries.
    pub query: Linear,

    /// `[d_model] -> [d_model]`: the keys.
    pub key: Linear,

    /// `[d_model] -> [d_model]`: the values.
    pub value: Linear,

    /// `[d_model] -> [d_model]`: the joined heads, out.
    pub output: Linear,

    /// The head count.
    pub n_heads: usize,
}

impl RopeSelfAttentionMeta for RopeSelfAttention {
    fn d_model(&self) -> usize {
        self.query.weight.dims()[0]
    }

    fn n_heads(&self) -> usize {
        self.n_heads
    }
}

impl RopeSelfAttention {
    /// Attends.
    ///
    /// # Arguments
    /// * `x` - `[batch, time, d_model]`.
    /// * `rope` - the rotary tables over positions `0..time`, `head_dim` wide.
    /// * `valid` - `[batch, time]`, true at the frames that are not padding;
    ///   `None` means every frame is valid. A padding key is never attended to;
    ///   a padding query's output means nothing.
    ///
    /// # Returns
    /// `[batch, time, d_model]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
        rope: &RotaryEmbedding,
        valid: Option<Tensor<2, Bool>>,
    ) -> Tensor<3> {
        let [batch, time] = unpack_shape_contract!(
            ["batch", "time", "d_model"],
            &x.dims(),
            &["batch", "time"],
            &[("d_model", self.d_model())],
        );
        #[cfg(any(debug_assertions, test))]
        {
            crate::contracts::assert_shape_contract_periodically!(
                [1, "time", 1, "half_head"],
                &rope.cos.dims(),
                &[("time", time), ("half_head", self.head_dim() / 2)],
            );
            if let Some(valid) = &valid {
                crate::contracts::assert_shape_contract_periodically!(
                    ["batch", "time"],
                    &valid.dims(),
                    &[("batch", batch), ("time", time)],
                );
            }
        }
        debug_assert_eq!(rope.head_dim(), self.head_dim());

        let (heads, head_dim) = (self.n_heads, self.head_dim());
        let split = |t: Tensor<3>| t.reshape([batch, time, heads, head_dim]);
        // [batch, heads, time, head_dim]
        let q = rope
            .apply(split(self.query.forward(x.clone())))
            .swap_dims(1, 2);
        let k = rope
            .apply(split(self.key.forward(x.clone())))
            .swap_dims(1, 2);
        let v = split(self.value.forward(x)).swap_dims(1, 2);

        let mut scores = q
            .matmul(k.swap_dims(2, 3))
            .div_scalar((head_dim as f64).sqrt());
        if let Some(valid) = valid {
            let padding = valid.bool_not().reshape([batch, 1, 1, time]);
            scores = scores.mask_fill(padding, f32::NEG_INFINITY);
        }
        let y =
            softmax(scores, 3)
                .matmul(v)
                .swap_dims(1, 2)
                .reshape([batch, time, heads * head_dim]);
        let y = self.output.forward(y);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "time", "d_model"],
            &y.dims(),
            &[
                ("batch", batch),
                ("time", time),
                ("d_model", self.d_model())
            ],
        );
        y
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::{
        blocks::transformers::embedding::{
            RotaryDirection,
            RotaryEmbeddingConfig,
        },
        ops::transformers::attention::length_mask,
        support::testing::{
            cpu_device,
            seeded_tensor,
        },
    };

    fn rope(
        time: usize,
        head_dim: usize,
    ) -> RotaryEmbedding {
        RotaryEmbeddingConfig::new(time, head_dim)
            .with_direction(RotaryDirection::Forward)
            .init(&cpu_device())
    }

    #[test]
    fn test_meta() {
        let config = RopeSelfAttentionConfig::new(24, 3)
            .with_qkv_bias(true)
            .with_output_bias(false);
        let module: RopeSelfAttention = config.init(&cpu_device());
        assert_eq!(module.d_model(), config.d_model());
        assert_eq!(module.n_heads(), config.n_heads());
        assert_eq!(module.head_dim(), 8);
        assert!(module.query.bias.is_some());
        assert!(module.output.bias.is_none());
    }

    #[test]
    fn test_refuses_odd_heads() {
        let device = cpu_device();
        for config in [
            RopeSelfAttentionConfig::new(24, 0),
            RopeSelfAttentionConfig::new(24, 5),
            RopeSelfAttentionConfig::new(12, 4),
        ] {
            assert!(ModuleInit::<RopeSelfAttention>::try_init(&config, &device).is_err());
        }
    }

    /// The attention is the textbook computation, head by head.
    #[test]
    fn test_matches_a_head_by_head_reference() {
        let device = cpu_device();
        let module: RopeSelfAttention = RopeSelfAttentionConfig::new(16, 2)
            .with_qkv_bias(true)
            .init(&device);
        let x = seeded_tensor::<3>(3, [1, 5, 16], Distribution::Default, &device);
        let rope = rope(5, 8);
        let got = module.forward(x.clone(), &rope, None);

        let mut heads = Vec::new();
        for h in 0..2 {
            let head = |proj: &Linear| {
                proj.forward(x.clone())
                    .slice_dim(2, h * 8..(h + 1) * 8)
                    .reshape([1, 5, 1, 8])
            };
            let q = rope.apply(head(&module.query)).reshape([5, 8]);
            let k = rope.apply(head(&module.key)).reshape([5, 8]);
            let v = head(&module.value).reshape([5, 8]);
            let w = softmax(q.matmul(k.transpose()).div_scalar(8f64.sqrt()), 1);
            heads.push(w.matmul(v));
        }
        let want = module
            .output
            .forward(Tensor::cat(heads, 1).reshape([1, 5, 16]));
        got.into_data()
            .assert_approx_eq::<f32>(&want.into_data(), Tolerance::absolute(1e-5));
    }

    /// Every position sees every other: changing the last frame changes the
    /// first one's output.
    #[test]
    fn test_is_bidirectional() {
        let device = cpu_device();
        let module: RopeSelfAttention = RopeSelfAttentionConfig::new(8, 2).init(&device);
        let x = seeded_tensor::<3>(5, [1, 4, 8], Distribution::Default, &device);
        let changed = x.clone().slice_assign(
            [0..1, 3..4, 0..8],
            seeded_tensor::<3>(6, [1, 1, 8], Distribution::Default, &device),
        );
        let a = module.forward(x, &rope(4, 4), None).slice_dim(1, 0..1);
        let b = module
            .forward(changed, &rope(4, 4), None)
            .slice_dim(1, 0..1);
        assert!((a - b).abs().max().into_scalar::<f32>() > 1e-4);
    }

    /// Padding keys are never attended to: a padded item's valid frames are
    /// the item's own.
    #[test]
    fn test_padding_is_ignored() {
        let device = cpu_device();
        let module: RopeSelfAttention = RopeSelfAttentionConfig::new(8, 2).init(&device);
        let x = seeded_tensor::<3>(9, [1, 6, 8], Distribution::Default, &device);
        let junk = seeded_tensor::<3>(10, [1, 3, 8], Distribution::Default, &device);
        let padded = Tensor::cat(vec![x.clone(), junk], 1);

        let alone = module.forward(x, &rope(6, 4), None);
        let valid = length_mask(&[6], 9, &device);
        let batched = module
            .forward(padded, &rope(9, 4), Some(valid))
            .slice_dim(1, 0..6);
        batched
            .into_data()
            .assert_approx_eq::<f32>(&alone.into_data(), Tolerance::absolute(1e-5));
    }
}
