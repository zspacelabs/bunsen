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
            models::classifier_head::ClassifierHeadConfig,
            patching::{
                patch_embed::PatchEmbedConfig,
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
                VisionEncoderStructureConfig,
            },
            sublayer::NormPlacement,
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
}

impl ViTContractConfig {
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
    /// - a patch embedding of `patch_size` patches, the pixels past the last
    ///   whole patch dropped, normed under [`embed_norm`](Self::embed_norm),
    ///   biased but under [`pre_norm`](Self::pre_norm);
    /// - the class and register tokens, and the position embedding;
    /// - `depth` layers: attention over every token, `num_heads` heads of
    ///   `embed_dim / num_heads`, a fused-`qkv` bias as
    ///   [`qkv_bias`](Self::qkv_bias) says and an output bias as
    ///   [`proj_bias`](Self::proj_bias) says; the
    ///   [`mlp_layer`](Self::mlp_layer)'s MLP of
    ///   [`mlp_hidden`](Self::mlp_hidden), biased as `proj_bias` says; every
    ///   norm of [`norm_layer`](Self::norm_layer)'s kind; layer scale as
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
    /// hidden width, or zero layers.
    fn try_to_structure(&self) -> BunsenResult<VisionEncoderStructureConfig> {
        let (d, heads, p) = (self.embed_dim, self.num_heads, self.patch_size);
        let patch_embed = PatchEmbedConfig::new([self.image_size; 2], p, self.in_channels, d)
            .with_bias(!self.pre_norm)
            .with_norm(self.embed_norm.then(|| self.norm()))
            .with_allow_remainder(true);
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
        if self.mlp_layer == ViTMlpLayer::SwiGluPacked && !self.mlp_hidden().is_multiple_of(2) {
            return Err(BunsenError::illegal(format!(
                "ViTContractConfig: a packed SwiGLU halves its hidden width, {}",
                self.mlp_hidden()
            )));
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
                    .with_output_bias(self.proj_bias)
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
        .with_latent_pool(latent_pool))
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
            ErrorMatcher::kind(BunsenErrorKind::Illegal)
                .message_contains(message)
                .assert_err(&config.try_to_structure());
        };
        illegal(
            ViTContractConfig::new(3, 4, 8, 2, 2, 5),
            "3x3 images don't hold a 4x4 patch",
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
