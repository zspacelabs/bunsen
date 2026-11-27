//! A learned query's attention over a sequence, pooling it to one vector.

use alloc::{
    format,
    string::ToString,
};

use burn::{
    config::Config,
    module::{
        Module,
        Param,
    },
    nn::attention::{
        MultiHeadAttention,
        MultiHeadAttentionConfig,
    },
    prelude::Tensor,
    tensor::{
        Device,
        Distribution,
    },
};
// `std` has these float methods inherently; `core` has them through `Float`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::{
    blocks::{
        mlp::{
            BunsenMlp,
            BunsenMlpConfig,
        },
        norm::{
            BunsenNormalization,
            BunsenNormalizationConfig,
        },
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::transformers::attention::cross_attn,
};

/// The config of a [`LatentAttentionPool`].
#[derive(Config, Debug)]
pub struct LatentAttentionPoolConfig {
    /// The width of the tokens, the query and the pooled vector.
    pub d_model: usize,

    /// The attention heads.
    pub n_heads: usize,

    /// The MLP after the attention, `d_model` to `d_model`.
    pub mlp: BunsenMlpConfig,

    /// The norm before the MLP, sized to `d_model` at init.
    pub norm: BunsenNormalizationConfig,
}

impl ModuleInit<LatentAttentionPool> for LatentAttentionPoolConfig {
    /// The pool, its latent query drawn from a normal of standard deviation
    /// `d_model^-0.5` (timm's init).
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a width the
    /// heads don't divide, or an MLP that doesn't map `d_model` to
    /// `d_model`; as the norm's and the MLP's `try_init`.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<LatentAttentionPool> {
        let d = self.d_model;
        if self.n_heads == 0 {
            return Err(
                ConstraintError::zero_or_empty("LatentAttentionPoolConfig", "n_heads").into(),
            );
        }
        if !d.is_multiple_of(self.n_heads) {
            return Err(ConstraintError::new(
                "LatentAttentionPoolConfig",
                "d_model",
                Rule::NotMultiple {
                    value: d,
                    of: format!("n_heads ({})", self.n_heads),
                },
            )
            .into());
        }
        for (field, width) in [
            ("mlp.d_input", self.mlp.d_input),
            ("mlp.d_output", self.mlp.d_output),
        ] {
            if width != d {
                return Err(ConstraintError::new(
                    "LatentAttentionPoolConfig",
                    field,
                    Rule::Relation {
                        lhs: (field.into(), width.to_string()),
                        op: "==",
                        rhs: ("d_model".into(), d.to_string()),
                    },
                )
                .into());
            }
        }
        Ok(LatentAttentionPool {
            latent: Param::from_tensor(Tensor::random(
                [1, 1, d],
                Distribution::Normal(0.0, (d as f64).powf(-0.5)),
                device,
            )),
            attn: MultiHeadAttentionConfig::new(d, self.n_heads)
                .with_dropout(0.0)
                .init(device),
            norm: self.norm.clone().with_num_features(d).try_init(device)?,
            mlp: self.mlp.try_init(device)?,
        })
    }
}

/// A learned query's attention over a sequence: the query attends over the
/// tokens, its biased `query`, `key`, `value` and `output` projections
/// those of a [`MultiHeadAttention`], then `x + mlp(norm(x))`, the query's
/// one output token the pooled vector. timm's `AttentionPoolLatent` with one
/// latent (`global_pool='map'`, `SigLIP`'s multi-head attention pooling).
///
/// Built by [`LatentAttentionPoolConfig`].
#[derive(Module, Debug)]
pub struct LatentAttentionPool {
    /// The query, `[1, 1, d_model]` (timm's `latent`).
    pub latent: Param<Tensor<3>>,

    /// The attention of the query over the tokens.
    pub attn: MultiHeadAttention,

    /// The norm before the MLP.
    pub norm: BunsenNormalization,

    /// The MLP, residual over the attention's output.
    pub mlp: BunsenMlp,
}

impl LatentAttentionPool {
    /// The width of the tokens and of the pooled vector.
    pub fn d_model(&self) -> usize {
        self.latent.dims()[2]
    }

    /// The tokens, pooled.
    ///
    /// # Arguments
    ///
    /// - `tokens`: `[batch, seq, d_model]`.
    ///
    /// # Returns
    ///
    /// `[batch, d_model]`.
    pub fn forward(
        &self,
        tokens: Tensor<3>,
    ) -> Tensor<2> {
        let [batch] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "d_model"],
            &tokens.dims(),
            &["batch"],
            &[("d_model", self.d_model())]
        );
        let query = self.latent.val().expand([batch, 1, self.d_model()]);
        let x = cross_attn(&self.attn, query, tokens).context;
        let x = x.clone() + self.mlp.forward(self.norm.forward(x));
        let pooled = x.squeeze_dim::<2>(1);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "d_model"],
            &pooled.dims(),
            &[("batch", batch), ("d_model", self.d_model())]
        );
        pooled
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;
    use crate::{
        blocks::norm::BunsenLayerNormConfig,
        errors::testing::ErrorMatcher,
        support::testing::cpu_device,
    };

    fn config(
        d: usize,
        heads: usize,
    ) -> LatentAttentionPoolConfig {
        LatentAttentionPoolConfig::new(
            d,
            heads,
            BunsenMlpConfig::new(d, 4 * d, d),
            BunsenNormalizationConfig::Layer(BunsenLayerNormConfig::new(d)),
        )
    }

    /// The pool is one vector per sequence, whatever its length; the tokens'
    /// order doesn't change it (the query attends over a set).
    #[test]
    fn test_pools_a_set() {
        let device = cpu_device();
        let pool: LatentAttentionPool = config(8, 2).init(&device);
        let tokens = Tensor::<3>::random([2, 5, 8], Distribution::Default, &device);
        let pooled = pool.forward(tokens.clone());
        assert_eq!(pooled.dims(), [2, 8]);
        let reversed = tokens.flip([1]);
        pooled
            .into_data()
            .assert_approx_eq::<f32>(&pool.forward(reversed).into_data(), Tolerance::default());
        assert_eq!(
            pool.forward(Tensor::zeros([3, 1, 8], &device)).dims(),
            [3, 8]
        );
    }

    #[test]
    fn test_illegal_pools() {
        let device = cpu_device();
        let illegal = |config: LatentAttentionPoolConfig, message: &str| {
            ErrorMatcher::illegal()
                .message_contains(message)
                .assert_err(&ModuleInit::<LatentAttentionPool>::try_init(
                    &config, &device,
                ));
        };
        illegal(config(8, 3), "d_model: 8 is not a multiple of n_heads (3)");
        let mut wide = config(8, 2);
        wide.mlp.d_output = 4;
        illegal(wide, "mlp.d_output (4) must be == d_model (8)");
    }
}
