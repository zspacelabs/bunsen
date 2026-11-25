//! `ViT`'s policy, and its lowering onto the generic vision encoder.

use alloc::format;

use burn::config::Config;

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        images::patching::{
            patch_embed::PatchEmbedConfig,
            prefix_tokens::{
                PrefixTokensConfig,
                PrefixTokensMeta,
            },
        },
        mlp::BunsenMlpConfig,
        norm::{
            BunsenLayerNormConfig,
            BunsenNormalizationConfig,
        },
        transformers::{
            attention::csa::{
                AttentionMask,
                CausalSelfAttentionConfig,
            },
            decoders::{
                causal::CausalDecoderBlockConfig,
                stack::UniformStackConfig,
            },
            embedding::RotaryEmbeddingConfig,
            models::vision_encoder::{
                TokenPoolConfig,
                VisionEncoderStructureConfig,
                VisionHeadConfig,
            },
        },
    },
    burner::module::ToStructureConfig,
    errors::{
        BunsenError,
        BunsenResult,
        ResultContext,
    },
};

/// timm's `ViT` layer norm epsilon (its `LayerNorm` default).
pub const VIT_NORM_EPSILON: f64 = 1e-6;

/// CLIP's layer norm epsilon (`nn.LayerNorm`'s default), which timm's CLIP
/// archs keep.
pub const CLIP_NORM_EPSILON: f64 = 1e-5;

/// A `ViT`'s policy: timm's `VisionTransformer` knobs, by their names where
/// bunsen keeps them, lowering to the generic
/// [`VisionEncoderStructureConfig`].
///
/// The model runs at `image_size`, the size its position embedding is learnt
/// for.
#[derive(Config, Debug)]
pub struct ViTContractConfig {
    /// The side of the square images it takes (timm's `img_size`).
    pub image_size: usize,

    /// The side of a square patch.
    pub patch_size: usize,

    /// The token width.
    pub embed_dim: usize,

    /// The transformer layers.
    pub depth: usize,

    /// The attention heads.
    pub num_heads: usize,

    /// The classes; 0 for a headless model.
    pub num_classes: usize,

    /// The image channels (timm's `in_chans`).
    #[config(default = "3")]
    pub in_channels: usize,

    /// The MLP's hidden width over the token width; the hidden width is
    /// rounded down, as timm's `int(dim * mlp_ratio)`.
    #[config(default = "4.0")]
    pub mlp_ratio: f64,

    /// Whether the attention's `qkv` projection has a bias.
    #[config(default = "true")]
    pub qkv_bias: bool,

    /// Whether a class token leads the sequence.
    #[config(default = "true")]
    pub class_token: bool,

    /// The register tokens after the class token.
    #[config(default = "0")]
    pub reg_tokens: usize,

    /// Whether the position embedding covers the patches alone, the prefix
    /// tokens joining after it.
    #[config(default = "false")]
    pub no_embed_class: bool,

    /// CLIP's pre-norm: the tokens normalised before the first layer
    /// (`norm_pre`), and a patch embedding with no bias.
    #[config(default = "false")]
    pub pre_norm: bool,

    /// Whether the norm follows the pool (`fc_norm`) rather than the last
    /// layer; `None`, timm's default: after a reducing pool, before the
    /// class token's.
    #[config(default = "None")]
    pub fc_norm: Option<bool>,

    /// The head's pool (timm's `global_pool`).
    #[config(default = "TokenPoolConfig::Token")]
    pub global_pool: TokenPoolConfig,

    /// Layer scale on each sublayer (timm's `init_values` set).
    #[config(default = "false")]
    pub layer_scale: bool,

    /// The layer norms' epsilon: [`VIT_NORM_EPSILON`], or
    /// [`CLIP_NORM_EPSILON`] for the CLIP archs.
    #[config(default = "VIT_NORM_EPSILON")]
    pub norm_epsilon: f64,

    /// The MLP's activation: `GELU`, or `QuickGelu` for CLIP's
    /// `_quickgelu` archs.
    #[config(default = "BunsenActivationConfig::Gelu")]
    pub activation: BunsenActivationConfig,

    /// The dropout before the classifier (timm's `drop_rate`; training
    /// only).
    #[config(default = "0.0")]
    pub drop_rate: f64,
}

impl ViTContractConfig {
    /// The CLIP image tower's knobs: [`pre_norm`](Self::pre_norm) and
    /// [`CLIP_NORM_EPSILON`].
    pub fn with_clip(self) -> Self {
        self.with_pre_norm(true)
            .with_norm_epsilon(CLIP_NORM_EPSILON)
    }

    /// Whether the norm follows the pool: [`fc_norm`](Self::fc_norm), else
    /// timm's default for the pool.
    pub fn uses_fc_norm(&self) -> bool {
        self.fc_norm
            .unwrap_or_else(|| self.global_pool.timm_fc_norm())
    }

    /// The MLP's hidden width: `embed_dim * mlp_ratio`, rounded down.
    pub fn mlp_hidden(&self) -> usize {
        (self.embed_dim as f64 * self.mlp_ratio) as usize
    }

    /// A layer norm of the model's epsilon.
    fn norm(&self) -> BunsenNormalizationConfig {
        BunsenNormalizationConfig::Layer(
            BunsenLayerNormConfig::new(self.embed_dim).with_epsilon(self.norm_epsilon),
        )
    }

    /// The prefix tokens' config.
    fn tokens(&self) -> PrefixTokensConfig {
        let grid = self.image_size / self.patch_size.max(1);
        PrefixTokensConfig::new(self.embed_dim, grid * grid)
            .with_class_token(self.class_token)
            .with_reg_tokens(self.reg_tokens)
            .with_embed_prefix(!self.no_embed_class)
    }
}

impl ToStructureConfig for ViTContractConfig {
    type Structure = VisionEncoderStructureConfig;

    /// The vision encoder of this `ViT`:
    ///
    /// - a patch embedding of `patch_size` patches, with no norm, biased but
    ///   under [`pre_norm`](Self::pre_norm);
    /// - the class and register tokens, and the position embedding;
    /// - `depth` pre-norm layers: attention over every token, `num_heads` heads
    ///   of `embed_dim / num_heads`, a fused-`qkv` bias as
    ///   [`qkv_bias`](Self::qkv_bias) says and an output bias; an MLP of
    ///   [`mlp_hidden`](Self::mlp_hidden) with biases; layer scale as
    ///   [`layer_scale`](Self::layer_scale) says; an input norm under
    ///   `pre_norm`; the final norm, unless the head's norm follows its pool
    ///   ([`uses_fc_norm`](Self::uses_fc_norm));
    /// - the head: the pool, its norm, the dropout, the classifier.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for images that
    /// don't divide into patches, a width the heads don't divide, a
    /// class-token pool with no class token, or zero layers.
    fn try_to_structure(&self) -> BunsenResult<VisionEncoderStructureConfig> {
        let (d, heads, p) = (self.embed_dim, self.num_heads, self.patch_size);
        let patch_embed = PatchEmbedConfig::new([self.image_size; 2], p, self.in_channels, d)
            .with_bias(!self.pre_norm);
        patch_embed
            .try_validate()
            .context("ViTContractConfig's patches")?;
        if heads == 0 || d % heads != 0 {
            return Err(BunsenError::illegal(format!(
                "ViTContractConfig: {heads} heads don't divide a width of {d}"
            )));
        }
        if self.global_pool == TokenPoolConfig::Token && !self.class_token {
            return Err(BunsenError::illegal(
                "ViTContractConfig: the class-token pool needs a class token",
            ));
        }
        if self.depth == 0 {
            return Err(BunsenError::illegal(
                "ViTContractConfig: a ViT has at least one layer",
            ));
        }
        let fc_norm = self.uses_fc_norm();
        let tokens = self.tokens();
        let mut stack = UniformStackConfig::new(
            self.depth,
            CausalDecoderBlockConfig::new(
                CausalSelfAttentionConfig::new(heads, heads, d)
                    .with_norm(None)
                    .with_qkv_bias(self.qkv_bias)
                    .with_output_bias(true)
                    .with_mask(AttentionMask::Bidirectional)
                    .into(),
                BunsenMlpConfig::new(d, self.mlp_hidden(), d)
                    .with_activation(self.activation.clone())
                    .with_bias(true)
                    .into(),
            )
            .with_norm(self.norm())
            .with_layer_scale(self.layer_scale),
            match fc_norm {
                true => BunsenNormalizationConfig::Identity,
                false => self.norm(),
            },
            RotaryEmbeddingConfig::unrotated(d / heads),
        );
        stack.input_norm = self.pre_norm.then(|| self.norm());
        Ok(VisionEncoderStructureConfig::new(
            patch_embed,
            tokens.clone(),
            stack.try_to_structure()?,
            VisionHeadConfig::new(d, self.num_classes, tokens.n_prefix())
                .with_pool(self.global_pool)
                .with_fc_norm(fc_norm.then(|| self.norm()))
                .with_drop_rate(self.drop_rate),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::transformers::{
            decoders::stack::DecoderStackMeta,
            models::vision_encoder::{
                VisionEncoderMeta,
                VisionEncoderModel,
            },
        },
        burner::module::ModuleInit,
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        support::testing::cpu_device,
    };

    fn tiny() -> ViTContractConfig {
        ViTContractConfig::new(16, 4, 8, 2, 2, 5)
    }

    /// The lowering's knobs land where timm puts them.
    #[test]
    fn test_lowering() {
        let s = tiny().to_structure();
        assert_eq!(s.grid(), [4, 4]);
        assert_eq!(s.n_prefix(), 1);
        assert_eq!(s.stack.n_layers(), 2);
        assert!(s.stack.input_norm.is_none());
        assert!(matches!(s.stack.norm, BunsenNormalizationConfig::Layer(_)));
        assert!(s.head.fc_norm.is_none());

        let avg = tiny().with_global_pool(TokenPoolConfig::Avg).to_structure();
        assert!(matches!(
            avg.stack.norm,
            BunsenNormalizationConfig::Identity
        ));
        assert!(avg.head.fc_norm.is_some());

        let clip = tiny().with_clip().to_structure();
        assert!(clip.stack.input_norm.is_some());
        match &clip.stack.norm {
            BunsenNormalizationConfig::Layer(norm) => assert_eq!(norm.epsilon, 1e-5),
            other => panic!("{other:?}"),
        }

        let reg = tiny()
            .with_reg_tokens(4)
            .with_no_embed_class(true)
            .to_structure();
        assert_eq!(reg.n_prefix(), 5);
        assert_eq!(reg.tokens.pos_embed_len(), Some(16));

        assert_eq!(tiny().with_mlp_ratio(2.5).mlp_hidden(), 20);
    }

    /// A pre-norm model's patch embedding has no bias; the policy builds.
    #[test]
    fn test_init() {
        let device = cpu_device();
        let model: VisionEncoderModel = tiny().with_clip().init(&device);
        assert!(model.patch_embed.projection.bias.is_none());
        assert_eq!(model.num_classes(), 5);
        let model: VisionEncoderModel = tiny().init(&device);
        assert!(model.patch_embed.projection.bias.is_some());
    }

    #[test]
    fn test_illegal_policies() {
        let illegal = |config: ViTContractConfig, message: &str| {
            ErrorMatcher::kind(BunsenErrorKind::Illegal)
                .message_contains(message)
                .assert_err(&config.try_to_structure());
        };
        illegal(
            ViTContractConfig::new(18, 4, 8, 2, 2, 5),
            "18x18 images don't divide into 4x4 patches",
        );
        illegal(
            ViTContractConfig::new(16, 4, 8, 2, 3, 5),
            "3 heads don't divide a width of 8",
        );
        illegal(tiny().with_class_token(false), "needs a class token");
        illegal(
            ViTContractConfig::new(16, 4, 8, 0, 2, 5),
            "at least one layer",
        );
    }
}
