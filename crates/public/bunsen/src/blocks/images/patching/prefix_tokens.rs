//! # Prefix tokens and the learnt position embedding
//!
//! A vision transformer's learnt tokens before its patch tokens (a class
//! token, a distillation token, register tokens), and the learnt absolute
//! position embedding added to the sequence: timm's
//! `VisionTransformer._pos_embed`.

use alloc::vec::Vec;

use burn::{
    config::Config,
    module::{
        Module,
        Param,
    },
    prelude::Tensor,
    tensor::{
        Device,
        Distribution,
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
    },
};

/// Common meta for [`PrefixTokens`] and [`PrefixTokensConfig`].
pub trait PrefixTokensMeta {
    /// The token width.
    fn d_model(&self) -> usize;

    /// The patch tokens the position embedding covers.
    fn n_patches(&self) -> usize;

    /// Whether a class token leads the sequence.
    fn class_token(&self) -> bool;

    /// Whether a distillation token follows the class token (`DeiT`'s).
    fn dist_token(&self) -> bool;

    /// The register tokens after the class and distillation tokens.
    fn reg_tokens(&self) -> usize;

    /// Whether the position embedding covers the prefix tokens as well as
    /// the patches (timm's default), or only the patches (timm's
    /// `no_embed_class`). `None` when there is no position embedding.
    fn embed_prefix(&self) -> Option<bool>;

    /// The tokens before the patch grid: the class token, the distillation
    /// token, then the register tokens.
    fn n_prefix(&self) -> usize {
        usize::from(self.class_token()) + usize::from(self.dist_token()) + self.reg_tokens()
    }

    /// The position embedding's length: the patches, and the prefix when it
    /// covers them. `None` when there is no position embedding.
    fn pos_embed_len(&self) -> Option<usize> {
        self.embed_prefix().map(|prefix| {
            self.n_patches()
                + match prefix {
                    true => self.n_prefix(),
                    false => 0,
                }
        })
    }
}

/// Config for [`PrefixTokens`].
///
/// Implements [`PrefixTokensMeta`].
#[derive(Config, Debug)]
pub struct PrefixTokensConfig {
    /// The token width.
    pub d_model: usize,

    /// The patch tokens the position embedding covers: the patch grid's
    /// size.
    pub n_patches: usize,

    /// Whether a class token leads the sequence (timm's `class_token`).
    #[config(default = true)]
    pub class_token: bool,

    /// Whether a distillation token follows the class token (timm's
    /// `VisionTransformerDistilled`, `DeiT`'s distilled models).
    #[config(default = false)]
    pub dist_token: bool,

    /// The register tokens after the class token (timm's `reg_tokens`).
    #[config(default = 0)]
    pub reg_tokens: usize,

    /// Whether there is a learnt position embedding (timm's `pos_embed`
    /// `'learn'`, against `'none'`).
    #[config(default = true)]
    pub pos_embed: bool,

    /// Whether the position embedding covers the prefix tokens too (timm's
    /// default), or only the patches, the prefix joining after it (timm's
    /// `no_embed_class`).
    #[config(default = true)]
    pub embed_prefix: bool,
}

impl PrefixTokensMeta for PrefixTokensConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_patches(&self) -> usize {
        self.n_patches
    }

    fn class_token(&self) -> bool {
        self.class_token
    }

    fn dist_token(&self) -> bool {
        self.dist_token
    }

    fn reg_tokens(&self) -> usize {
        self.reg_tokens
    }

    fn embed_prefix(&self) -> Option<bool> {
        self.pos_embed.then_some(self.embed_prefix)
    }
}

/// The tokens, initialised as timm does: the prefix tokens near zero, the
/// position embedding at `std` 0.02.
///
/// # Errors
/// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a zero width or
/// no patches.
impl ModuleInit<PrefixTokens> for PrefixTokensConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<PrefixTokens> {
        for (field, value) in [("d_model", self.d_model), ("n_patches", self.n_patches)] {
            if value == 0 {
                return Err(ConstraintError::zero_or_empty("PrefixTokensConfig", field).into());
            }
        }
        let d = self.d_model;
        let param = |len: usize, std: f64| {
            Param::from_tensor(Tensor::random(
                [1, len, d],
                Distribution::Normal(0.0, std),
                device,
            ))
        };
        Ok(PrefixTokens {
            cls_token: self.class_token.then(|| param(1, 1e-6)),
            dist_token: self.dist_token.then(|| param(1, 0.02)),
            reg_token: (self.reg_tokens > 0).then(|| param(self.reg_tokens, 1e-6)),
            pos_embed: self.pos_embed_len().map(|len| param(len, 0.02)),
            n_patches: self.n_patches,
            embed_prefix: self.embed_prefix,
        })
    }
}

/// A vision transformer's prefix tokens and its learnt absolute position
/// embedding: [`forward`](Self::forward) puts the class token and the
/// register tokens before the patch tokens and adds the position embedding,
/// as timm's `VisionTransformer._pos_embed` does at the model's own size.
///
/// The parameters keep timm's shapes and names: `cls_token` and
/// `dist_token` `[1, 1, d]`, `reg_token` `[1, reg_tokens, d]`, `pos_embed`
/// `[1, len, d]`.
///
/// Built by [`PrefixTokensConfig`]; implements [`PrefixTokensMeta`].
#[derive(Module, Debug)]
pub struct PrefixTokens {
    /// The class token, `[1, 1, d]`.
    pub cls_token: Option<Param<Tensor<3>>>,

    /// The distillation token, `[1, 1, d]`.
    pub dist_token: Option<Param<Tensor<3>>>,

    /// The register tokens, `[1, reg_tokens, d]`.
    pub reg_token: Option<Param<Tensor<3>>>,

    /// The position embedding, `[1, len, d]`: over the prefix and the
    /// patches, or the patches alone ([`embed_prefix`](Self::embed_prefix)).
    pub pos_embed: Option<Param<Tensor<3>>>,

    /// The patch tokens the position embedding covers.
    pub n_patches: usize,

    /// Whether the position embedding covers the prefix tokens.
    pub embed_prefix: bool,
}

impl PrefixTokensMeta for PrefixTokens {
    fn d_model(&self) -> usize {
        [
            &self.cls_token,
            &self.dist_token,
            &self.reg_token,
            &self.pos_embed,
        ]
        .into_iter()
        .flatten()
        .map(|p| p.dims()[2])
        .next()
        .unwrap_or(0)
    }

    fn n_patches(&self) -> usize {
        self.n_patches
    }

    fn class_token(&self) -> bool {
        self.cls_token.is_some()
    }

    fn dist_token(&self) -> bool {
        self.dist_token.is_some()
    }

    fn reg_tokens(&self) -> usize {
        self.reg_token.as_ref().map_or(0, |t| t.dims()[1])
    }

    fn embed_prefix(&self) -> Option<bool> {
        self.pos_embed.as_ref().map(|_| self.embed_prefix)
    }
}

impl PrefixTokens {
    /// The prefix tokens before `patches`, and the position embedding added:
    /// with the prefix when it covers them, else to the patches before the
    /// prefix joins.
    ///
    /// # Arguments
    /// * `patches` - `[batch, n_patches, d]`.
    ///
    /// # Returns
    /// `[batch, n_prefix + n_patches, d]`: the class token, the distillation
    /// token, the register tokens, then the patches.
    pub fn forward(
        &self,
        patches: Tensor<3>,
    ) -> Tensor<3> {
        let [batch] = crate::contracts::unpack_shape_contract!(
            ["batch", "n_patches", "d"],
            &patches,
            &["batch"],
            &[("n_patches", self.n_patches), ("d", self.d_model())],
        );
        let prefix: Vec<Tensor<3>> = [&self.cls_token, &self.dist_token, &self.reg_token]
            .into_iter()
            .flatten()
            .map(|t| t.val().expand([batch as i64, -1, -1]))
            .collect();
        let with_prefix = |x: Tensor<3>| match prefix.is_empty() {
            true => x,
            false => Tensor::cat([prefix.clone(), alloc::vec![x]].concat(), 1),
        };
        match (&self.pos_embed, self.embed_prefix) {
            (None, _) => with_prefix(patches),
            (Some(pos), true) => with_prefix(patches) + pos.val(),
            (Some(pos), false) => with_prefix(patches + pos.val()),
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        TensorData,
        Tolerance,
    };

    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::{
            IDENTICAL,
            ROUNDING,
            assert_tensors_close,
            cpu_device,
        },
    };

    /// A `[1, len, 2]` parameter of `values`.
    fn param(
        values: &[f32],
        device: &Device,
    ) -> Param<Tensor<3>> {
        let len = values.len() / 2;
        Param::from_tensor(Tensor::from_data(
            TensorData::new(values.to_vec(), [1, len, 2]),
            device,
        ))
    }

    /// Two 2-wide patches of a batch of one.
    fn patches(device: &Device) -> Tensor<3> {
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [1, 2, 2]),
            device,
        )
    }

    fn expect(
        values: &[f32],
        device: &Device,
    ) -> Tensor<3> {
        Tensor::from_data(
            TensorData::new(values.to_vec(), [1, values.len() / 2, 2]),
            device,
        )
    }

    /// The meta agrees across the config and the module, and the shapes are
    /// timm's.
    #[test]
    fn test_meta_and_shapes() {
        let device = cpu_device();
        for (config, prefix, len) in [
            (PrefixTokensConfig::new(8, 4), 1, Some(5)),
            (PrefixTokensConfig::new(8, 4).with_reg_tokens(2), 3, Some(7)),
            (
                PrefixTokensConfig::new(8, 4)
                    .with_reg_tokens(2)
                    .with_embed_prefix(false),
                3,
                Some(4),
            ),
            (
                PrefixTokensConfig::new(8, 4).with_class_token(false),
                0,
                Some(4),
            ),
            (PrefixTokensConfig::new(8, 4).with_pos_embed(false), 1, None),
            (
                PrefixTokensConfig::new(8, 4)
                    .with_dist_token(true)
                    .with_reg_tokens(1),
                3,
                Some(7),
            ),
        ] {
            let tokens: PrefixTokens = config.init(&device);
            assert_eq!(config.n_prefix(), prefix, "{config:?}");
            assert_eq!(config.pos_embed_len(), len, "{config:?}");
            assert_eq!(tokens.n_prefix(), prefix, "{config:?}");
            assert_eq!(tokens.pos_embed_len(), len, "{config:?}");
            assert_eq!(tokens.n_patches(), 4);
            if let Some(len) = len {
                assert_eq!(tokens.pos_embed.as_ref().unwrap().dims(), [1, len, 8]);
            }
            let x = tokens.forward(Tensor::zeros([3, 4, 8], &device));
            assert_eq!(x.dims(), [3, prefix + 4, 8]);
        }
    }

    /// The class token, then the registers, then the patches; the position
    /// embedding over all of them, or over the patches before the prefix
    /// joins.
    #[test]
    fn test_forward_orders_and_embeds() {
        let device = cpu_device();
        let tolerance = Tolerance::rel_abs(1e-6, 1e-6);
        let tokens = |pos: Option<&[f32]>, embed_prefix| PrefixTokens {
            cls_token: Some(param(&[10.0, 20.0], &device)),
            dist_token: None,
            reg_token: Some(param(&[30.0, 40.0], &device)),
            pos_embed: pos.map(|p| param(p, &device)),
            n_patches: 2,
            embed_prefix,
        };
        assert_tensors_close(
            &tokens(None, true).forward(patches(&device)),
            &expect(&[10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0], &device),
            tolerance,
        );
        let all = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        assert_tensors_close(
            &tokens(Some(&all), true).forward(patches(&device)),
            &expect(&[10.1, 20.2, 30.3, 40.4, 1.5, 2.6, 3.7, 4.8], &device),
            tolerance,
        );
        assert_tensors_close(
            &tokens(Some(&all[..4]), false).forward(patches(&device)),
            &expect(&[10.0, 20.0, 30.0, 40.0, 1.1, 2.2, 3.3, 4.4], &device),
            tolerance,
        );
    }

    /// The distillation token sits between the class token and the
    /// registers.
    #[test]
    fn test_the_dist_token_follows_the_class_token() {
        let device = cpu_device();
        let tokens = PrefixTokens {
            cls_token: Some(param(&[10.0, 20.0], &device)),
            dist_token: Some(param(&[50.0, 60.0], &device)),
            reg_token: Some(param(&[30.0, 40.0], &device)),
            pos_embed: None,
            n_patches: 2,
            embed_prefix: true,
        };
        assert_tensors_close(
            &tokens.forward(patches(&device)),
            &expect(
                &[10.0, 20.0, 50.0, 60.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0],
                &device,
            ),
            ROUNDING,
        );
    }

    /// Each image of a batch gets the same prefix.
    #[test]
    fn test_the_prefix_broadcasts_over_the_batch() {
        let device = cpu_device();
        let tokens: PrefixTokens = PrefixTokensConfig::new(2, 2).init(&device);
        let x = tokens.forward(Tensor::cat(vec![patches(&device), patches(&device)], 0));
        assert_tensors_close(
            &x.clone().slice(burn::tensor::s![0..1]),
            &x.slice(burn::tensor::s![1..2]),
            IDENTICAL,
        );
    }

    #[test]
    fn test_degenerate_configs_are_illegal() {
        let device = cpu_device();
        for config in [PrefixTokensConfig::new(0, 4), PrefixTokensConfig::new(8, 0)] {
            ErrorMatcher::illegal()
                .assert_err(&ModuleInit::<PrefixTokens>::try_init(&config, &device));
        }
    }
}
