//! `ViT`'s policy, and its lowering onto the generic vision encoder.

use alloc::format;

use burn::{
    config::Config,
    nn::RmsNormConfig,
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        images::{
            models::{
                classifier_head::ClassifierHeadConfig,
                conv_net::ConvNetMeta,
            },
            patching::{
                patch_embed::{
                    PatchEmbedConfig,
                    PatchEmbedMeta,
                },
                prefix_tokens::PrefixTokensConfig,
            },
        },
        mlp::{
            BunsenGatedMlpConfig,
            BunsenMlpConfig,
        },
        norm::{
            BunsenLayerNormConfig,
            BunsenNormalizationConfig,
        },
        pooling::LatentAttentionPoolConfig,
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
            mlp::TransformerFeedForwardConfig,
            models::vision_encoder::{
                TokenPoolConfig,
                VisionBackboneConfig,
                VisionEncoderStructureConfig,
            },
            sublayer::NormPlacement,
        },
    },
    burner::module::ToStructureConfig,
    errors::{
        BunsenResult,
        ConstraintError,
        ResultContext,
        Rule,
    },
    kits::images::classification::resnetv2::{
        BIT_STD_CONV_EPSILON,
        ResNetV2ContractConfig,
        ResNetV2Stem,
    },
};

/// timm's `ViT` layer norm epsilon (its `LayerNorm` default).
pub const VIT_NORM_EPSILON: f64 = 1e-6;

/// CLIP's layer norm epsilon (`nn.LayerNorm`'s default), which timm's CLIP
/// archs keep.
pub const CLIP_NORM_EPSILON: f64 = 1e-5;

/// A `ViT` block's MLP, timm's `mlp_layer`.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ViTMlpLayer {
    /// timm's `Mlp`: `fc2(act(fc1(x)))`.
    Mlp,

    /// timm's `SwiGLU`: `fc2(act(fc1_g(x)) * fc1_x(x))`, the gate and the
    /// input in two projections of the hidden width (`AIMv2`'s).
    SwiGlu,

    /// timm's `SwiGLUPacked` (`GluMlp`, its gate first): the gate and the
    /// input in one `fc1`, each half its rows, so the hidden width is half
    /// [`mlp_hidden`](ViTContractConfig::mlp_hidden) (the giant `DINOv2`'s).
    SwiGluPacked,
}

/// A `ViT`'s norms, timm's `norm_layer`.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ViTNormLayer {
    /// timm's `LayerNorm`.
    LayerNorm,

    /// timm's `RmsNorm` (`AIMv2`'s).
    RmsNorm,
}

/// A `ViT`'s policy: timm's `VisionTransformer` knobs, by their names where
/// bunsen keeps them, lowering to the generic
/// [`VisionEncoderStructureConfig`].
///
/// The model runs at `image_size`, the size its position embedding is learnt
/// for. As timm's, an `image_size` that isn't a whole number of patches is
/// taken: the patch embedding drops the pixels past the last whole patch.
#[derive(Config, Debug)]
pub struct ViTContractConfig {
    /// The side of the square images it takes (timm's `img_size`).
    pub image_size: usize,

    /// The side of a square patch: of the images, or of the
    /// [`backbone`](Self::backbone)'s map.
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

    /// Whether the attention's output projection and the MLP's projections
    /// have biases (timm's `proj_bias`).
    #[config(default = "true")]
    pub proj_bias: bool,

    /// The blocks' MLP (timm's `mlp_layer`).
    #[config(default = "ViTMlpLayer::Mlp")]
    pub mlp_layer: ViTMlpLayer,

    /// The norms' kind (timm's `norm_layer`), every norm's: the blocks', the
    /// final one, the pool's and the patch embedding's.
    #[config(default = "ViTNormLayer::LayerNorm")]
    pub norm_layer: ViTNormLayer,

    /// Whether each patch token is normalized (timm's `embed_norm_layer`, of
    /// the model's norm kind).
    #[config(default = "false")]
    pub embed_norm: bool,

    /// Whether the attention normalizes its output before the output
    /// projection (timm's `scale_attn_norm`, `BEiT3`'s sub-LN).
    #[config(default = "false")]
    pub scale_attn_norm: bool,

    /// Whether the MLP normalizes its hidden width before `fc2` (timm's
    /// `scale_mlp_norm`, `BEiT3`'s sub-LN); a plain MLP's only.
    #[config(default = "false")]
    pub scale_mlp_norm: bool,

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

    /// The head's pool (timm's `global_pool`); a
    /// [`Map`](TokenPoolConfig::Map) pool is an attention pool of the
    /// model's heads, MLP, norm and activation (timm's `AttentionPoolLatent`).
    #[config(default = "TokenPoolConfig::Token")]
    pub global_pool: TokenPoolConfig,

    /// Layer scale on each sublayer (timm's `init_values` set, on its
    /// `Block`).
    #[config(default = "false")]
    pub layer_scale: bool,

    /// Where each block normalizes: before each sublayer (timm's `Block`),
    /// or [`ResPost`](NormPlacement::ResPost), on each branch's output
    /// (its `ResPostBlock`, the `rpn` archs').
    #[config(default = "NormPlacement::Pre")]
    pub norm_placement: NormPlacement,

    /// Whether the tokens are normalized after the last layer, before the
    /// head (timm's `final_norm`); `false`, no norm there or after the pool
    /// (C-RADIO's).
    #[config(default = "true")]
    pub final_norm: bool,

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

    /// The conv backbone the images go through before the patch embedding
    /// (timm's `HybridEmbed`: the hybrid `ViT`'s headless, non-preact
    /// `ResNetV2`, [`resnetv2_backbone`](Self::resnetv2_backbone)); none when
    /// `None`. The patch embedding then cuts the backbone's map.
    #[config(default = "None")]
    pub backbone: Option<ResNetV2ContractConfig>,
}

impl ViTContractConfig {
    /// The hybrid `ViT`'s backbone of `layers` (timm's
    /// `vision_transformer_hybrid._resnetv2`): a `ResNetV2` that is not
    /// pre-activation, its convs weight-standardized (eps
    /// [`BIT_STD_CONV_EPSILON`]) and SAME-padded, the `same` stem, a stage of
    /// `ResNetV2`'s widths per depth, and no head; no `layers`, its stem
    /// alone.
    pub fn resnetv2_backbone(layers: &[usize]) -> ResNetV2ContractConfig {
        let channels = ResNetV2ContractConfig::new(vec![], 0).channels;
        ResNetV2ContractConfig::new(layers.to_vec(), 0)
            .with_channels(channels[..layers.len()].to_vec())
            .with_stem_type(ResNetV2Stem::Same)
            .with_std_conv(Some(BIT_STD_CONV_EPSILON))
            .with_same_pad(true)
            .with_preact(false)
    }

    /// This `ViT` as a hybrid on [`resnetv2_backbone`](Self::resnetv2_backbone)
    /// of `layers`.
    pub fn with_resnetv2_backbone(
        self,
        layers: &[usize],
    ) -> Self {
        self.with_backbone(Some(Self::resnetv2_backbone(layers)))
    }

    /// The CLIP image tower's knobs: [`pre_norm`](Self::pre_norm) and
    /// [`CLIP_NORM_EPSILON`].
    pub fn with_clip(self) -> Self {
        self.with_pre_norm(true)
            .with_norm_epsilon(CLIP_NORM_EPSILON)
    }

    /// Whether the norm follows the pool: [`fc_norm`](Self::fc_norm), else
    /// timm's default for the pool; never without a
    /// [`final_norm`](Self::final_norm).
    pub fn uses_fc_norm(&self) -> bool {
        self.final_norm
            && self
                .fc_norm
                .unwrap_or_else(|| self.global_pool.timm_fc_norm())
    }

    /// Whether the norm follows the last layer: a
    /// [`final_norm`](Self::final_norm) the pool's norm doesn't take
    /// ([`uses_fc_norm`](Self::uses_fc_norm)).
    pub fn uses_last_norm(&self) -> bool {
        self.final_norm && !self.uses_fc_norm()
    }

    /// The MLP's hidden width: `embed_dim * mlp_ratio`, rounded down.
    pub fn mlp_hidden(&self) -> usize {
        (self.embed_dim as f64 * self.mlp_ratio) as usize
    }

    /// The attention pool's MLP: timm's `Mlp`, `embed_dim` to
    /// [`mlp_hidden`](Self::mlp_hidden) and back, biased, of the model's
    /// activation, whatever the blocks' MLP.
    fn pool_mlp(&self) -> BunsenMlpConfig {
        BunsenMlpConfig::new(self.embed_dim, self.mlp_hidden(), self.embed_dim)
            .with_activation(self.activation.clone())
            .with_bias(true)
    }

    /// A block's MLP: [`mlp_layer`](Self::mlp_layer)'s, biased under
    /// [`proj_bias`](Self::proj_bias), of the model's activation.
    fn block_mlp(&self) -> TransformerFeedForwardConfig {
        let (d, hidden, act) = (self.embed_dim, self.mlp_hidden(), self.activation.clone());
        let gated = |d_ff| {
            BunsenGatedMlpConfig::new(d, d_ff)
                .with_bias(self.proj_bias)
                .with_activation(act.clone())
                .into()
        };
        match self.mlp_layer {
            ViTMlpLayer::Mlp => BunsenMlpConfig::new(d, hidden, d)
                .with_activation(act.clone())
                .with_bias(self.proj_bias)
                .with_hidden_norm(self.scale_mlp_norm.then(|| self.norm()))
                .into(),
            ViTMlpLayer::SwiGlu => gated(hidden),
            ViTMlpLayer::SwiGluPacked => gated(hidden / 2),
        }
    }

    /// A norm of the model's kind and epsilon.
    fn norm(&self) -> BunsenNormalizationConfig {
        match self.norm_layer {
            ViTNormLayer::LayerNorm => BunsenNormalizationConfig::Layer(
                BunsenLayerNormConfig::new(self.embed_dim).with_epsilon(self.norm_epsilon),
            ),
            ViTNormLayer::RmsNorm => BunsenNormalizationConfig::Rms(
                RmsNormConfig::new(self.embed_dim).with_epsilon(self.norm_epsilon),
            ),
        }
    }

    /// The prefix tokens' config, over `n_patches` patches.
    fn tokens(
        &self,
        n_patches: usize,
    ) -> PrefixTokensConfig {
        PrefixTokensConfig::new(self.embed_dim, n_patches)
            .with_class_token(self.class_token)
            .with_reg_tokens(self.reg_tokens)
            .with_embed_prefix(!self.no_embed_class)
    }
}

impl ToStructureConfig for ViTContractConfig {
    type Structure = VisionEncoderStructureConfig;

    /// The vision encoder of this `ViT`:
    ///
    /// - the [`backbone`](Self::backbone)'s conv net, at `image_size`;
    /// - a patch embedding of `patch_size` patches (of the images, or the
    ///   backbone's map), the pixels past the last whole patch dropped, normed
    ///   under [`embed_norm`](Self::embed_norm), biased but under
    ///   [`pre_norm`](Self::pre_norm);
    /// - the class and register tokens, and the position embedding;
    /// - `depth` layers: attention over every token, `num_heads` heads of
    ///   `embed_dim / num_heads`, a fused-`qkv` bias as
    ///   [`qkv_bias`](Self::qkv_bias) says and an output bias as
    ///   [`proj_bias`](Self::proj_bias) says; the
    ///   [`mlp_layer`](Self::mlp_layer)'s MLP of
    ///   [`mlp_hidden`](Self::mlp_hidden), biased as `proj_bias` says; the
    ///   inner norms [`scale_attn_norm`](Self::scale_attn_norm) and
    ///   [`scale_mlp_norm`](Self::scale_mlp_norm) say; every norm of
    ///   [`norm_layer`](Self::norm_layer)'s kind; layer scale as
    ///   [`layer_scale`](Self::layer_scale) says; the norms where
    ///   [`norm_placement`](Self::norm_placement) puts them; an input norm
    ///   under `pre_norm`; the final norm, as
    ///   [`uses_last_norm`](Self::uses_last_norm) says;
    /// - the head: the pool (under [`Map`](TokenPoolConfig::Map), an attention
    ///   pool of `num_heads` heads, an MLP of [`mlp_hidden`](Self::mlp_hidden)
    ///   and the model's norm), its norm, the dropout, the classifier.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for images
    /// smaller than a patch, a width the heads don't divide, a
    /// class-token pool with no class token, a packed `SwiGLU` of an odd
    /// hidden width, a hidden norm on a gated MLP, or zero layers.
    fn try_to_structure(&self) -> BunsenResult<VisionEncoderStructureConfig> {
        let (d, heads, p) = (self.embed_dim, self.num_heads, self.patch_size);
        let backbone = self
            .backbone
            .as_ref()
            .map(|backbone| -> BunsenResult<VisionBackboneConfig> {
                if backbone.in_channels != self.in_channels {
                    return Err(ConstraintError::new(
                        "ViTContractConfig",
                        "backbone.in_channels",
                        Rule::Relation {
                            lhs: (
                                "backbone.in_channels".into(),
                                backbone.in_channels.to_string(),
                            ),
                            op: "==",
                            rhs: ("in_channels".into(), self.in_channels.to_string()),
                        },
                    )
                    .into());
                }
                Ok(VisionBackboneConfig::new(
                    backbone.try_to_structure()?,
                    [self.image_size; 2],
                ))
            })
            .transpose()
            .context("ViTContractConfig's backbone")?;
        let (input, d_input) = match &backbone {
            Some(backbone) => (backbone.try_feature_size()?, backbone.net.d_features()),
            None => ([self.image_size; 2], self.in_channels),
        };
        let patch_embed = PatchEmbedConfig::new(input, p, d_input, d)
            .with_bias(!self.pre_norm)
            .with_norm(self.embed_norm.then(|| self.norm()))
            .with_allow_remainder(true);
        patch_embed
            .try_validate()
            .context("ViTContractConfig's patches")?;
        if heads == 0 {
            return Err(ConstraintError::zero_or_empty("ViTContractConfig", "num_heads").into());
        }
        if !d.is_multiple_of(heads) {
            return Err(ConstraintError::new(
                "ViTContractConfig",
                "embed_dim",
                Rule::NotMultiple {
                    value: d,
                    of: format!("num_heads ({heads})"),
                },
            )
            .into());
        }
        if self.global_pool == TokenPoolConfig::Token && !self.class_token {
            return Err(ConstraintError::custom(
                "ViTContractConfig",
                "class_token",
                "the class-token pool needs a class token",
            )
            .into());
        }
        if self.scale_mlp_norm && self.mlp_layer != ViTMlpLayer::Mlp {
            return Err(ConstraintError::custom(
                "ViTContractConfig",
                "scale_mlp_norm",
                format!(
                    "the MLP's hidden norm is a plain MLP's, not {:?}'s",
                    self.mlp_layer
                ),
            )
            .into());
        }
        if self.mlp_layer == ViTMlpLayer::SwiGluPacked && !self.mlp_hidden().is_multiple_of(2) {
            return Err(ConstraintError::new(
                "ViTContractConfig",
                "mlp_ratio",
                Rule::NotMultiple {
                    value: self.mlp_hidden(),
                    of: "2, as a packed SwiGLU halves its hidden width".into(),
                },
            )
            .into());
        }
        if self.depth == 0 {
            return Err(ConstraintError::zero_or_empty("ViTContractConfig", "depth").into());
        }
        let fc_norm = self.uses_fc_norm();
        let tokens = self.tokens(patch_embed.num_patches());
        let mut stack = UniformStackConfig::new(
            self.depth,
            CausalDecoderBlockConfig::new(
                CausalSelfAttentionConfig::new(heads, heads, d)
                    .with_norm(None)
                    .with_qkv_bias(self.qkv_bias)
                    .with_output_bias(self.proj_bias)
                    .with_output_norm(self.scale_attn_norm.then(|| self.norm()))
                    .with_mask(AttentionMask::Bidirectional)
                    .into(),
                self.block_mlp(),
            )
            .with_norm(self.norm())
            .with_norm_placement(self.norm_placement)
            .with_layer_scale(self.layer_scale),
            match self.uses_last_norm() {
                true => self.norm(),
                false => BunsenNormalizationConfig::Identity,
            },
            RotaryEmbeddingConfig::unrotated(d / heads),
        );
        stack.input_norm = self.pre_norm.then(|| self.norm());
        let latent_pool = (self.global_pool == TokenPoolConfig::Map)
            .then(|| LatentAttentionPoolConfig::new(d, heads, self.pool_mlp(), self.norm()));
        Ok(VisionEncoderStructureConfig::new(
            patch_embed,
            tokens,
            stack.try_to_structure()?,
            ClassifierHeadConfig::new(d, self.num_classes)
                .with_fc_norm(fc_norm.then(|| self.norm()))
                .with_drop_rate(self.drop_rate),
        )
        .with_pool(self.global_pool)
        .with_latent_pool(latent_pool)
        .with_backbone(backbone))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::{
            images::patching::prefix_tokens::PrefixTokensMeta,
            transformers::{
                decoders::stack::DecoderStackMeta,
                models::vision_encoder::{
                    VisionEncoderMeta,
                    VisionEncoderModel,
                },
            },
        },
        burner::module::ModuleInit,
        data::pretrained::Prefabs,
        errors::testing::ErrorMatcher,
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

        let bare = tiny()
            .with_global_pool(TokenPoolConfig::Avg)
            .with_final_norm(false)
            .to_structure();
        assert!(matches!(
            bare.stack.norm,
            BunsenNormalizationConfig::Identity
        ));
        assert!(bare.head.fc_norm.is_none());

        // 18 pixels in 4-pixel patches: 4 patches a side, 2 pixels dropped.
        let remainder = ViTContractConfig::new(18, 4, 8, 2, 2, 5).to_structure();
        assert_eq!(remainder.grid(), [4, 4]);
        assert_eq!(remainder.tokens.pos_embed_len(), Some(17));
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
            ErrorMatcher::illegal()
                .message_contains(message)
                .assert_err(&config.try_to_structure());
        };
        illegal(
            ViTContractConfig::new(3, 4, 8, 2, 2, 5),
            "3x3 images don't hold a 4x4 patch",
        );
        illegal(
            ViTContractConfig::new(16, 4, 8, 2, 3, 5),
            "embed_dim: 8 is not a multiple of num_heads (3)",
        );
        illegal(tiny().with_class_token(false), "needs a class token");
        illegal(
            ViTContractConfig::new(16, 4, 8, 0, 2, 5),
            "ViTContractConfig.depth: must not be zero",
        );
    }

    /// Each hybrid prefab's backbone map and patch grid are timm's (its
    /// `patch_embed.feature_size`, `feature_dim` and `grid_size`, read from
    /// timm's built models), and its position embedding covers the class
    /// token and the grid.
    #[test]
    fn test_hybrid_prefabs_are_timms() {
        for (name, map, d_features, grid) in [
            ("vit_tiny_r_s16_p8_224", 56, 64, 7),
            ("vit_tiny_r_s16_p8_384", 96, 64, 12),
            ("vit_small_r26_s32_224", 7, 2048, 7),
            ("vit_small_r26_s32_384", 12, 2048, 12),
            ("vit_base_r50_s16_224", 14, 1024, 14),
            ("vit_base_r50_s16_384", 24, 1024, 24),
            ("vit_large_r50_s32_224", 7, 2048, 7),
            ("vit_large_r50_s32_384", 12, 2048, 12),
        ] {
            let s = ViTContractConfig::expect_prefab(name).to_structure();
            let backbone = s.backbone.as_ref().expect("a hybrid");
            assert_eq!(backbone.try_feature_size().unwrap(), [map; 2], "{name}");
            assert_eq!(s.patch_embed.d_input(), d_features, "{name}");
            assert_eq!(s.grid(), [grid; 2], "{name}");
            assert_eq!(s.tokens.n_patches, grid * grid, "{name}");
            assert_eq!(s.in_channels(), 3, "{name}");
        }
    }

    /// A small hybrid builds and runs at its image size; a backbone of other
    /// image channels than the `ViT`'s is refused.
    #[test]
    fn test_hybrid() {
        let config = ViTContractConfig::new(64, 1, 32, 1, 2, 5).with_backbone(Some(
            ViTContractConfig::resnetv2_backbone(&[1])
                .with_channels(vec![128])
                .with_stem_chs(32),
        ));
        let device = cpu_device();
        let model: VisionEncoderModel = config.init(&device);
        assert_eq!(model.grid(), [16, 16]);
        let images = burn::Tensor::<4>::zeros([1, 3, 64, 64], &device);
        assert_eq!(model.forward(images).dims(), [1, 5]);

        let mut gray = config;
        gray.in_channels = 1;
        ErrorMatcher::illegal()
            .message_contains("backbone.in_channels (3) must be == in_channels (1)")
            .assert_err(&gray.try_to_structure());
    }
}
