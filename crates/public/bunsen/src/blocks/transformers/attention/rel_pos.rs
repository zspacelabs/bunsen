//! # Relative-position self-attention
//!
//! [`RelPosSelfAttention`], Transformer-XL's, built by
//! [`RelPosSelfAttentionConfig`]; the attention of the
//! [`ConformerBlock`](crate::blocks::conformer::ConformerBlock).
use burn::{
    Tensor,
    config::Config,
    module::{
        Module,
        Param,
    },
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
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::transformers::attention::rel_shift,
};

/// Common meta for [`RelPosSelfAttention`] and [`RelPosSelfAttentionConfig`].
pub trait RelPosSelfAttentionMeta {
    /// The model width.
    fn d_model(&self) -> usize;

    /// The head count.
    fn n_heads(&self) -> usize;

    /// The per-head width.
    fn d_head(&self) -> usize {
        self.d_model() / self.n_heads()
    }
}

/// Config for [`RelPosSelfAttention`].
///
/// Implements [`RelPosSelfAttentionMeta`].
#[derive(Config, Debug)]
pub struct RelPosSelfAttentionConfig {
    /// The model width; a multiple of `n_heads`.
    pub d_model: usize,

    /// The head count.
    pub n_heads: usize,

    /// Whether the query, key, value and output projections carry a bias.
    /// The position projection never does.
    #[config(default = "true")]
    pub bias: bool,
}

impl RelPosSelfAttentionMeta for RelPosSelfAttentionConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_heads(&self) -> usize {
        self.n_heads
    }
}

impl ModuleInit<RelPosSelfAttention> for RelPosSelfAttentionConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<RelPosSelfAttention> {
        if self.n_heads == 0 {
            return Err(ConstraintError::new(
                "RelPosSelfAttentionConfig",
                "n_heads",
                Rule::ZeroOrEmpty,
            )
            .into());
        }
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(ConstraintError::new(
                "RelPosSelfAttentionConfig",
                "d_model",
                Rule::NotMultiple {
                    value: self.d_model,
                    of: format!("n_heads ({})", self.n_heads),
                },
            )
            .into());
        }
        let d = self.d_model;
        let linear = |bias| LinearConfig::new(d, d).with_bias(bias).init(device);
        Ok(RelPosSelfAttention {
            query: linear(self.bias),
            key: linear(self.bias),
            value: linear(self.bias),
            output: linear(self.bias),
            pos: linear(false),
            pos_bias_u: Param::from_tensor(Tensor::zeros([self.n_heads, self.d_head()], device)),
            pos_bias_v: Param::from_tensor(Tensor::zeros([self.n_heads, self.d_head()], device)),
            n_heads: self.n_heads,
        })
    }
}

/// Multi-head self-attention with Transformer-XL relative positions
/// (Dai et al. 2019, section 3.3): the Conformer's attention.
///
/// The score of query `i` against key `j` is the sum of a content term and a
/// position term,
///
/// ```text
/// ((q_i + u) . k_j  +  (q_i + v) . P(i - j)) / sqrt(d_head)
/// ```
///
/// where `u` and `v` are learned per-head biases (`pos_bias_u`,
/// `pos_bias_v`) and `P` is the projected relative-position table. The
/// position term is computed against every relative position and aligned
/// with [`rel_shift`].
///
/// Built by [`RelPosSelfAttentionConfig`].
/// Implements [`RelPosSelfAttentionMeta`].
#[derive(Module, Debug)]
pub struct RelPosSelfAttention {
    /// The query projection.
    pub query: Linear,

    /// The key projection.
    pub key: Linear,

    /// The value projection.
    pub value: Linear,

    /// The output projection.
    pub output: Linear,

    /// The relative-position projection.
    pub pos: Linear,

    /// The content bias `u`, `[n_heads, d_head]`.
    pub pos_bias_u: Param<Tensor<2>>,

    /// The position bias `v`, `[n_heads, d_head]`.
    pub pos_bias_v: Param<Tensor<2>>,

    /// The head count.
    pub n_heads: usize,
}

impl RelPosSelfAttentionMeta for RelPosSelfAttention {
    fn d_model(&self) -> usize {
        self.query.weight.dims()[0]
    }

    fn n_heads(&self) -> usize {
        self.n_heads
    }
}

impl RelPosSelfAttention {
    /// Attends over `x`.
    ///
    /// # Arguments
    /// * `x` - `[batch, time, d_model]`.
    /// * `positions` - `[1, 2 * time - 1, d_model]`, the relative-position
    ///   table ([`rel_position_table`](crate::ops::transformers::attention::rel_position_table)).
    /// * `valid` - `[batch, time]`, true at the frames that are not padding;
    ///   padding keys are never attended to. `None` means none is padding.
    ///
    /// # Returns
    /// `[batch, time, d_model]`. Rows at padding frames hold values, but
    /// nothing downstream should read them.
    pub fn forward(
        &self,
        x: Tensor<3>,
        positions: Tensor<3>,
        valid: Option<Tensor<2, Bool>>,
    ) -> Tensor<3> {
        let [batch, time] = crate::contracts::unpack_shape_contract!(
            ["batch", "time", "d_model"],
            &x.dims(),
            &["batch", "time"],
            &[("d_model", self.d_model())],
        );
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            [1, 2 * "time" - 1, "d_model"],
            &positions.dims(),
            &[("time", time), ("d_model", self.d_model())],
        );
        #[cfg(any(debug_assertions, test))]
        if let Some(valid) = &valid {
            crate::contracts::assert_shape_contract_periodically!(
                ["batch", "time"],
                &valid.dims(),
                &[("batch", batch), ("time", time)],
            );
        }
        let d_model = self.d_model();
        let (h, dk) = (self.n_heads, self.d_head());
        let heads = |t: Tensor<3>| {
            t.reshape([batch as i32, -1, h as i32, dk as i32])
                .swap_dims(1, 2)
        };

        let q = heads(self.query.forward(x.clone()));
        let k = heads(self.key.forward(x.clone()));
        let v = heads(self.value.forward(x));
        let p = self
            .pos
            .forward(positions)
            .reshape([1, 2 * time - 1, h, dk])
            .swap_dims(1, 2);

        let bias = |b: &Param<Tensor<2>>| b.val().reshape([1, h, 1, dk]);
        let q_u = q.clone() + bias(&self.pos_bias_u);
        let q_v = q + bias(&self.pos_bias_v);

        let content = q_u.matmul(k.swap_dims(2, 3));
        let position = rel_shift(q_v.matmul(p.swap_dims(2, 3)));
        let mut scores = (content + position).mul_scalar(1.0 / (dk as f64).sqrt());
        if let Some(valid) = valid {
            let masked = valid.bool_not().reshape([batch, 1, 1, time]);
            scores = scores.mask_fill(masked, f32::NEG_INFINITY);
        }

        let context = softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, time, d_model]);
        let y = self.output.forward(context);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "time", "d_model"],
            &y.dims(),
            &[("batch", batch), ("time", time), ("d_model", d_model)],
        );

        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ops::transformers::attention::rel_position_table,
        support::testing::cpu_device,
    };

    #[test]
    fn test_meta() {
        let config = RelPosSelfAttentionConfig::new(12, 3).with_bias(false);
        let module: RelPosSelfAttention = config.init(&cpu_device());
        assert_eq!(module.d_model(), config.d_model());
        assert_eq!(module.n_heads(), config.n_heads());
        assert_eq!(module.d_head(), 4);
        assert!(module.query.bias.is_none());
        assert!(module.pos.bias.is_none());

        let device = cpu_device();
        let y = module.forward(
            Tensor::zeros([2, 5, 12], &device),
            rel_position_table(5, 12, &device),
            None,
        );
        assert_eq!(y.dims(), [2, 5, 12]);
    }

    #[test]
    fn test_heads_must_divide_width() {
        let config = RelPosSelfAttentionConfig::new(10, 3);
        assert!(ModuleInit::<RelPosSelfAttention>::try_init(&config, &cpu_device()).is_err());
    }
}
