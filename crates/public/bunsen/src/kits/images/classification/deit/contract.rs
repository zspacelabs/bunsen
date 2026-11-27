//! `DeiT`'s policy: a `ViT`'s, and distillation.

use burn::config::Config;

use crate::{
    blocks::transformers::models::vision_encoder::{
        TokenPoolConfig,
        VisionEncoderStructureConfig,
    },
    burner::module::ToStructureConfig,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::images::classification::vit::ViTContractConfig,
};

/// A `DeiT`'s policy: the `ViT` it is, by timm's `VisionTransformer` knobs,
/// and whether it is distilled, lowering to the generic
/// [`VisionEncoderStructureConfig`].
///
/// A `DeiT` is a `ViT` in its geometry: timm builds the plain ones and
/// `DeiT` III as its `VisionTransformer` (`DeiT` III with `no_embed_class`
/// and layer scale). A distilled one is timm's `VisionTransformerDistilled`:
/// a distillation token after the class token, and a second classifier on
/// it, the two classifiers' logits averaged.
#[derive(Config, Debug)]
pub struct DeiTContractConfig {
    /// The `ViT` it is: its sizes, tokens, norms and head.
    pub vit: ViTContractConfig,

    /// Whether it is distilled: a distillation token, and a classifier on it
    /// (timm's `head_dist`).
    #[config(default = "false")]
    pub distilled: bool,
}

impl ToStructureConfig for DeiTContractConfig {
    type Structure = VisionEncoderStructureConfig;

    /// The `ViT`'s structure; distilled, with the distillation token after
    /// the class token and a [`Distilled`](TokenPoolConfig::Distilled) head.
    ///
    /// # Errors
    ///
    /// As the `ViT`'s lowering;
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a distilled
    /// `DeiT` whose `ViT` pools other than by its class token (timm's
    /// distilled head reads the class and distillation tokens) or
    /// normalises after its pool.
    fn try_to_structure(&self) -> BunsenResult<VisionEncoderStructureConfig> {
        let mut structure = self.vit.try_to_structure()?;
        if !self.distilled {
            return Ok(structure);
        }
        if self.vit.global_pool != TokenPoolConfig::Token || self.vit.uses_fc_norm() {
            return Err(BunsenError::illegal(
                "DeiTContractConfig: a distilled head reads the class and distillation tokens, \
                 with no norm after its pool",
            ));
        }
        structure.tokens.dist_token = true;
        structure.pool = TokenPoolConfig::Distilled;
        structure.head.distilled = true;
        Ok(structure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::transformers::models::vision_encoder::{
            VisionEncoderMeta,
            VisionEncoderModel,
        },
        burner::module::ModuleInit,
        errors::testing::ErrorMatcher,
        support::testing::cpu_device,
    };

    fn tiny() -> ViTContractConfig {
        ViTContractConfig::new(16, 4, 8, 2, 2, 5)
    }

    /// A distilled `DeiT` has the distillation token and its classifier; a
    /// plain one is its `ViT`.
    #[test]
    fn test_lowering() {
        let device = cpu_device();
        let plain = DeiTContractConfig::new(tiny()).to_structure();
        assert_eq!(plain.n_prefix(), 1);
        let distilled = DeiTContractConfig::new(tiny()).with_distilled(true);
        let structure = distilled.to_structure();
        assert_eq!(structure.n_prefix(), 2);
        assert_eq!(structure.pool, TokenPoolConfig::Distilled);
        assert!(structure.head.distilled);
        let model: VisionEncoderModel = distilled.init(&device);
        assert!(model.tokens.dist_token.is_some());
        assert!(model.head.fc_dist.is_some());
        assert_eq!(model.num_classes(), 5);
    }

    #[test]
    fn test_a_distilled_avg_pool_is_illegal() {
        ErrorMatcher::illegal()
            .message_contains("reads the class and distillation tokens")
            .assert_err(
                &DeiTContractConfig::new(tiny().with_global_pool(TokenPoolConfig::Avg))
                    .with_distilled(true)
                    .try_to_structure(),
            );
    }
}
