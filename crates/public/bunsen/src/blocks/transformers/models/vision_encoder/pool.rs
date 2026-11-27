//! The pool over a vision transformer's tokens.

use alloc::string::ToString;

use burn::{
    config::Config,
    module::Module,
    prelude::Tensor,
    tensor::s,
};

use crate::{
    blocks::pooling::LatentAttentionPool,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
};

/// The pool over a vision transformer's tokens, timm's `global_pool` kinds:
/// the config of a [`TokenPool`].
///
/// The reducing kinds and the attention pool take the patch tokens alone,
/// the prefix tokens left out.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum TokenPoolConfig {
    /// The class token, the sequence's first (timm's `token`).
    Token,

    /// The mean of the patch tokens (timm's `avg`).
    Avg,

    /// The max of the patch tokens (timm's `max`).
    Max,

    /// The mean of the two (timm's `avgmax`).
    AvgMax,

    /// The class and distillation tokens, the first two: their mean pooled,
    /// each through its own classifier, the two logits averaged (timm's
    /// `VisionTransformerDistilled`, `DeiT`'s distilled models).
    Distilled,

    /// A learned query's attention over the patch tokens, a
    /// [`LatentAttentionPool`] (timm's `map`, `SigLIP`'s): the one kind with
    /// parameters.
    Map,
}

impl TokenPoolConfig {
    /// The kind timm names `name` (`token`, `avg`, `max`, `avgmax`, `map`);
    /// `None` for a kind the head doesn't build (`prr`, `''`).
    pub fn from_timm(name: &str) -> Option<Self> {
        Some(match name {
            "token" => Self::Token,
            "avg" => Self::Avg,
            "max" => Self::Max,
            "avgmax" => Self::AvgMax,
            "map" => Self::Map,
            _ => return None,
        })
    }

    /// Whether timm normalises after this pool rather than before it: its
    /// `fc_norm` default, on for the reducing pools.
    pub fn timm_fc_norm(self) -> bool {
        matches!(self, Self::Avg | Self::Max | Self::AvgMax)
    }

    /// The prefix tokens the pool reads: the class token for
    /// [`Token`](Self::Token), it and the distillation token for
    /// [`Distilled`](Self::Distilled), none for the reducing pools and the
    /// attention pool.
    pub fn prefix_read(self) -> usize {
        match self {
            Self::Token => 1,
            Self::Distilled => 2,
            Self::Avg | Self::Max | Self::AvgMax | Self::Map => 0,
        }
    }
}

/// A pool over the tokens: [`TokenPool::try_new`], or
/// [`TokenPool::attention`] for the one kind with parameters.
#[derive(Module, Debug)]
pub struct TokenPool {
    /// The kind.
    #[module(skip)]
    pub kind: TokenPoolConfig,

    /// The prefix tokens a reducing pool leaves out.
    pub n_prefix: usize,

    /// The attention pool of a [`Map`](TokenPoolConfig::Map) pool; `None`
    /// for every other kind.
    pub latent: Option<LatentAttentionPool>,
}

impl TokenPool {
    /// The pool of `kind` over a sequence of `n_prefix` prefix tokens.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a pool that
    /// reads more prefix tokens than there are
    /// ([`prefix_read`](TokenPoolConfig::prefix_read)), or a
    /// [`Map`](TokenPoolConfig::Map) pool, which is built with its attention
    /// ([`attention`](Self::attention)).
    pub fn try_new(
        kind: TokenPoolConfig,
        n_prefix: usize,
    ) -> BunsenResult<Self> {
        if kind == TokenPoolConfig::Map {
            return Err(ConstraintError::custom(
                "TokenPool::try_new",
                "kind",
                "a Map pool has parameters: it is built with its latent attention pool",
            )
            .into());
        }
        let read = kind.prefix_read();
        if read > n_prefix {
            return Err(ConstraintError::new(
                "TokenPool::try_new",
                "n_prefix",
                Rule::Relation {
                    lhs: (
                        alloc::format!("the {kind:?} pool's prefix reads"),
                        read.to_string(),
                    ),
                    op: "<=",
                    rhs: ("n_prefix".into(), n_prefix.to_string()),
                },
            )
            .into());
        }
        Ok(Self {
            kind,
            n_prefix,
            latent: None,
        })
    }

    /// The [`Map`](TokenPoolConfig::Map) pool of `latent` over a sequence of
    /// `n_prefix` prefix tokens, which it leaves out.
    pub fn attention(
        latent: LatentAttentionPool,
        n_prefix: usize,
    ) -> Self {
        Self {
            kind: TokenPoolConfig::Map,
            n_prefix,
            latent: Some(latent),
        }
    }

    /// The pooled tokens.
    ///
    /// # Arguments
    ///
    /// - `tokens`: `[batch, n_prefix + patches, d]`.
    ///
    /// # Returns
    ///
    /// `[batch, d]`.
    pub fn forward(
        &self,
        tokens: Tensor<3>,
    ) -> Tensor<2> {
        let [batch, n, d] = crate::contracts::unpack_shape_contract!(
            ["batch", "tokens" = "n_prefix" + "patches", "d"],
            &tokens.dims(),
            &["batch", "tokens", "d"],
            &[("n_prefix", self.n_prefix)]
        );
        let patches = || tokens.clone().slice(s![.., self.n_prefix..n]);
        let pooled: Tensor<3> = match self.kind {
            TokenPoolConfig::Token => tokens.clone().slice(s![.., 0..1]),
            TokenPoolConfig::Distilled => tokens.clone().slice(s![.., 0..2]).mean_dim(1),
            TokenPoolConfig::Avg => patches().mean_dim(1),
            TokenPoolConfig::Max => patches().max_dim(1),
            TokenPoolConfig::AvgMax => (patches().max_dim(1) + patches().mean_dim(1)) * 0.5,
            TokenPoolConfig::Map => self
                .latent
                .as_ref()
                .expect("a Map pool is built with its attention")
                .forward(patches())
                .unsqueeze_dim(1),
        };
        let pooled = pooled.squeeze_dim(1);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "d"],
            &pooled.dims(),
            &[("batch", batch), ("d", d)]
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = (batch, d);
        pooled
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::{
            cpu_device,
            tensor_to_vec_as,
        },
    };

    /// Each pool over `[1, 3, 2]` tokens, one prefix token `(9, 9)` and the
    /// patches `(1, 4)`, `(3, 2)`.
    #[test]
    fn test_pools() {
        let device = cpu_device();
        let tokens = Tensor::<3>::from_data(
            TensorData::new(alloc::vec![9f32, 9., 1., 4., 3., 2.], [1, 3, 2]),
            &device,
        );
        for (kind, want) in [
            (TokenPoolConfig::Token, [9f32, 9.]),
            (TokenPoolConfig::Avg, [2., 3.]),
            (TokenPoolConfig::Max, [3., 4.]),
            (TokenPoolConfig::AvgMax, [2.5, 3.5]),
        ] {
            let pool = TokenPool::try_new(kind, 1).unwrap();
            let got = pool.forward(tokens.clone()).into_data();
            assert_eq!(got.try_to_vec::<f32>().unwrap(), want, "{kind:?}");
        }
        assert_eq!(
            TokenPoolConfig::from_timm("avg"),
            Some(TokenPoolConfig::Avg)
        );
        assert_eq!(
            TokenPoolConfig::from_timm("map"),
            Some(TokenPoolConfig::Map)
        );
        assert_eq!(TokenPoolConfig::from_timm("prr"), None);
        ErrorMatcher::illegal()
            .message_contains("a Map pool has parameters")
            .assert_err(&TokenPool::try_new(TokenPoolConfig::Map, 0));
    }

    /// The distilled pool is the two leading tokens' mean; a pool reading
    /// more prefix tokens than there are is illegal.
    #[test]
    fn test_distilled_pool_and_its_prefix() {
        let device = cpu_device();
        let tokens = Tensor::<3>::from_data(
            TensorData::new(alloc::vec![1f32, 2., 5., 6., 9., 9.], [1, 3, 2]),
            &device,
        );
        let pool = TokenPool::try_new(TokenPoolConfig::Distilled, 2).unwrap();
        assert_eq!(tensor_to_vec_as::<f32, _>(&pool.forward(tokens)), [3., 4.]);
        ErrorMatcher::illegal()
            .message_contains("prefix reads (2) must be <= n_prefix (1)")
            .assert_err(&TokenPool::try_new(TokenPoolConfig::Distilled, 1));
        ErrorMatcher::illegal()
            .message_contains("prefix reads (1) must be <= n_prefix (0)")
            .assert_err(&TokenPool::try_new(TokenPoolConfig::Token, 0));
    }
}
