//! The pool over a vision transformer's tokens.

use alloc::string::ToString;

use burn::{
    config::Config,
    module::Module,
    prelude::Tensor,
    tensor::{
        Device,
        s,
    },
};

use crate::{
    blocks::pooling::{
        LatentAttentionPool,
        LatentAttentionPoolConfig,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        ResultContext,
        Rule,
        WithOkOrPanic,
    },
};

/// The kind of pool over a vision transformer's tokens, timm's
/// `global_pool`: what a policy and a catalog row name. A
/// [`TokenPoolConfig`] builds the pool.
///
/// The reducing kinds and the attention pool take the patch tokens alone,
/// the prefix tokens left out.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum TokenPoolKind {
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

impl TokenPoolKind {
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

/// The pool over a vision transformer's tokens, as a structure holds it:
/// a [`TokenPoolKind`], the [`Map`](Self::Map) kind with its attention.
/// Builds a [`TokenPool`].
#[derive(Config, Debug)]
pub enum TokenPoolConfig {
    /// [`TokenPoolKind::Token`].
    Token,

    /// [`TokenPoolKind::Avg`].
    Avg,

    /// [`TokenPoolKind::Max`].
    Max,

    /// [`TokenPoolKind::AvgMax`].
    AvgMax,

    /// [`TokenPoolKind::Distilled`].
    Distilled,

    /// [`TokenPoolKind::Map`]: a learned query's attention over the patch
    /// tokens.
    Map(LatentAttentionPoolConfig),
}

impl TokenPoolConfig {
    /// The config of a kind with no parameters.
    ///
    /// # Errors
    ///
    /// A [`ConstraintError`] for [`Map`](TokenPoolKind::Map), which is built
    /// with its attention ([`TokenPoolConfig::Map`]).
    pub fn try_from_kind(kind: TokenPoolKind) -> BunsenResult<Self> {
        Ok(match kind {
            TokenPoolKind::Token => Self::Token,
            TokenPoolKind::Avg => Self::Avg,
            TokenPoolKind::Max => Self::Max,
            TokenPoolKind::AvgMax => Self::AvgMax,
            TokenPoolKind::Distilled => Self::Distilled,
            TokenPoolKind::Map => {
                return Err(ConstraintError::custom(
                    "TokenPoolConfig",
                    "kind",
                    "a Map pool has parameters: it is built with its latent attention pool",
                )
                .into());
            }
        })
    }

    /// The pool's kind.
    pub fn kind(&self) -> TokenPoolKind {
        match self {
            Self::Token => TokenPoolKind::Token,
            Self::Avg => TokenPoolKind::Avg,
            Self::Max => TokenPoolKind::Max,
            Self::AvgMax => TokenPoolKind::AvgMax,
            Self::Distilled => TokenPoolKind::Distilled,
            Self::Map(_) => TokenPoolKind::Map,
        }
    }

    /// Checks the pool over a sequence of `n_prefix` prefix tokens.
    ///
    /// # Errors
    ///
    /// A [`ConstraintError`] for a pool that reads more prefix tokens than
    /// there are ([`prefix_read`](TokenPoolKind::prefix_read)).
    pub fn try_check(
        &self,
        n_prefix: usize,
    ) -> BunsenResult<()> {
        let kind = self.kind();
        let read = kind.prefix_read();
        if read > n_prefix {
            return Err(ConstraintError::new(
                "TokenPoolConfig",
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
        Ok(())
    }

    /// The pool over a sequence of `n_prefix` prefix tokens.
    ///
    /// # Errors
    ///
    /// As [`try_check`](Self::try_check), and as the attention pool's build.
    pub fn try_init(
        &self,
        n_prefix: usize,
        device: &Device,
    ) -> BunsenResult<TokenPool> {
        self.try_check(n_prefix)?;
        Ok(TokenPool {
            kind: self.kind(),
            n_prefix,
            latent: match self {
                Self::Map(latent) => Some(latent.try_init(device).context("the attention pool")?),
                _ => None,
            },
        })
    }

    /// [`try_init`](Self::try_init), for a pool known to be sound.
    ///
    /// # Panics
    ///
    /// When it is not.
    pub fn init(
        &self,
        n_prefix: usize,
        device: &Device,
    ) -> TokenPool {
        self.try_init(n_prefix, device).ok_or_panic()
    }
}

/// A pool over the tokens. Built by [`TokenPoolConfig`].
#[derive(Module, Debug)]
pub struct TokenPool {
    /// The kind.
    #[module(skip)]
    pub kind: TokenPoolKind,

    /// The prefix tokens a reducing pool leaves out.
    pub n_prefix: usize,

    /// The attention pool of a [`Map`](TokenPoolKind::Map) pool; `None`
    /// for every other kind.
    pub latent: Option<LatentAttentionPool>,
}

impl TokenPool {
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
            TokenPoolKind::Token => tokens.clone().slice(s![.., 0..1]),
            TokenPoolKind::Distilled => tokens.clone().slice(s![.., 0..2]).mean_dim(1),
            TokenPoolKind::Avg => patches().mean_dim(1),
            TokenPoolKind::Max => patches().max_dim(1),
            TokenPoolKind::AvgMax => (patches().max_dim(1) + patches().mean_dim(1)) * 0.5,
            TokenPoolKind::Map => self
                .latent
                .as_ref()
                .expect("a Map pool is built by TokenPoolConfig::Map, with its attention")
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
            (TokenPoolKind::Token, [9f32, 9.]),
            (TokenPoolKind::Avg, [2., 3.]),
            (TokenPoolKind::Max, [3., 4.]),
            (TokenPoolKind::AvgMax, [2.5, 3.5]),
        ] {
            let pool = TokenPoolConfig::try_from_kind(kind)
                .unwrap()
                .try_init(1, &device)
                .unwrap();
            let got = pool.forward(tokens.clone()).into_data();
            assert_eq!(got.try_to_vec::<f32>().unwrap(), want, "{kind:?}");
        }
        assert_eq!(TokenPoolKind::from_timm("avg"), Some(TokenPoolKind::Avg));
        assert_eq!(TokenPoolKind::from_timm("map"), Some(TokenPoolKind::Map));
        assert_eq!(TokenPoolKind::from_timm("prr"), None);
        ErrorMatcher::illegal()
            .message_contains("a Map pool has parameters")
            .assert_err(&TokenPoolConfig::try_from_kind(TokenPoolKind::Map));
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
        let pool = TokenPoolConfig::try_from_kind(TokenPoolKind::Distilled)
            .unwrap()
            .try_init(2, &device)
            .unwrap();
        assert_eq!(tensor_to_vec_as::<f32, _>(&pool.forward(tokens)), [3., 4.]);
        ErrorMatcher::illegal()
            .message_contains("prefix reads (2) must be <= n_prefix (1)")
            .assert_err(
                &TokenPoolConfig::try_from_kind(TokenPoolKind::Distilled)
                    .unwrap()
                    .try_init(1, &device),
            );
        ErrorMatcher::illegal()
            .message_contains("prefix reads (1) must be <= n_prefix (0)")
            .assert_err(
                &TokenPoolConfig::try_from_kind(TokenPoolKind::Token)
                    .unwrap()
                    .try_init(0, &device),
            );
    }
}
