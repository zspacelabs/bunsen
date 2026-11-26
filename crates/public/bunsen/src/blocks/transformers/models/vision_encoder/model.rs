//! The vision encoder: its structure and the model.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    config::Config,
    module::Module,
    prelude::Tensor,
    tensor::{
        Device,
        s,
    },
};

use super::{
    TokenPool,
    TokenPoolConfig,
};
use crate::{
    blocks::{
        images::{
            models::classifier_head::{
                ClassifierHead,
                ClassifierHeadConfig,
            },
            patching::{
                patch_embed::{
                    PatchEmbed,
                    PatchEmbedConfig,
                    PatchEmbedMeta,
                },
                prefix_tokens::{
                    PrefixTokens,
                    PrefixTokensConfig,
                    PrefixTokensMeta,
                },
            },
        },
        transformers::decoders::stack::{
            DecoderStack,
            DecoderStackMeta,
            DecoderStackStructureConfig,
        },
    },
    burner::module::{
        CanonicalKeys,
        ModuleInit,
        Rekey,
    },
    errors::{
        BunsenError,
        BunsenResult,
        ResultContext,
    },
};

/// The view shared by [`VisionEncoderStructureConfig`] and
/// [`VisionEncoderModel`].
pub trait VisionEncoderMeta {
    /// The image channels the patch embedding takes.
    fn in_channels(&self) -> usize;

    /// The `[height, width]` of the images it takes.
    fn image_size(&self) -> [usize; 2];

    /// The patch size.
    fn patch_size(&self) -> usize;

    /// The `[height, width]` of the patch grid.
    fn grid(&self) -> [usize; 2];

    /// The token width.
    fn d_model(&self) -> usize;

    /// The transformer layers.
    fn n_layers(&self) -> usize;

    /// The tokens before the patch grid.
    fn n_prefix(&self) -> usize;

    /// The classes of the head; 0 for a headless model.
    fn num_classes(&self) -> usize;
}

/// [`VisionEncoderModel`]'s structure: the patch embedding, the prefix
/// tokens and position embedding, the transformer, the token pool and the
/// head.
///
/// A family's policy (`ViTContractConfig`) lowers to it. `try_init` checks
/// that the parts agree: one width, the tokens over the patch grid, a class
/// token for a class-token pool.
///
/// Implements [`VisionEncoderMeta`].
#[derive(Config, Debug)]
pub struct VisionEncoderStructureConfig {
    /// The patch embedding.
    pub patch_embed: PatchEmbedConfig,

    /// The prefix tokens and the position embedding.
    pub tokens: PrefixTokensConfig,

    /// The transformer: bidirectional, unrotated layers (the position
    /// embedding is absolute), an input norm when the model normalises the
    /// tokens before the first layer (timm's `norm_pre`), and the final norm
    /// (`Identity` when the head normalises after its pool).
    pub stack: DecoderStackStructureConfig,

    /// The pool over the output tokens.
    #[config(default = "TokenPoolConfig::Token")]
    pub pool: TokenPoolConfig,

    /// The head, over the pooled features: distilled for a
    /// [`Distilled`](TokenPoolConfig::Distilled) pool.
    pub head: ClassifierHeadConfig,
}

impl VisionEncoderMeta for VisionEncoderStructureConfig {
    fn in_channels(&self) -> usize {
        self.patch_embed.d_input()
    }

    fn image_size(&self) -> [usize; 2] {
        self.patch_embed.input_resolution()
    }

    fn patch_size(&self) -> usize {
        self.patch_embed.patch_size()
    }

    fn grid(&self) -> [usize; 2] {
        self.patch_embed.patches_resolution()
    }

    fn d_model(&self) -> usize {
        self.stack.d_model()
    }

    fn n_layers(&self) -> usize {
        self.stack.n_layers()
    }

    fn n_prefix(&self) -> usize {
        self.tokens.n_prefix()
    }

    fn num_classes(&self) -> usize {
        self.head.num_classes
    }
}

impl VisionEncoderStructureConfig {
    /// Checks that the parts agree.
    ///
    /// See the [errors convention](crate::errors#convention-try_x-and-x).
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when the patch
    /// embedding, the tokens, the transformer and the head are not one
    /// width; when the tokens' patches are not the patch grid's; when the
    /// pool reads more prefix tokens than there are, or the head is distilled
    /// and the pool not (or the other way); when the stack has
    /// attention-residual routes; or when the images don't divide into
    /// patches.
    pub fn try_validate(&self) -> BunsenResult<()> {
        let d = self.stack.d_model();
        let widths = [
            ("the patch embedding", self.patch_embed.d_output()),
            ("the prefix tokens", self.tokens.d_model),
            ("the head", self.head.d_features),
        ];
        for (part, width) in widths {
            if width != d {
                return Err(BunsenError::illegal(format!(
                    "{part} is {width} wide; the transformer is {d}"
                )));
            }
        }
        self.patch_embed.try_validate()?;
        if self.tokens.n_patches != self.patch_embed.num_patches() {
            return Err(BunsenError::illegal(format!(
                "the prefix tokens cover {} patches; the grid has {}",
                self.tokens.n_patches,
                self.patch_embed.num_patches()
            )));
        }
        TokenPool::try_new(self.pool, self.tokens.n_prefix())?;
        if self.head.distilled != (self.pool == TokenPoolConfig::Distilled) {
            return Err(BunsenError::illegal(format!(
                "a {:?} pool with a head that is {}distilled: a distilled head reads the \
                 distilled pool's two tokens",
                self.pool,
                if self.head.distilled { "" } else { "not " }
            )));
        }
        if self.stack.routes.is_some() {
            return Err(BunsenError::illegal(
                "a vision encoder's transformer has no attention-residual routes",
            ));
        }
        Ok(())
    }
}

impl ModuleInit<VisionEncoderModel> for VisionEncoderStructureConfig {
    /// Builds the model.
    ///
    /// # Errors
    ///
    /// As [`try_validate`](VisionEncoderStructureConfig::try_validate), and
    /// as each part's `try_init`.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<VisionEncoderModel> {
        self.try_validate()?;
        Ok(VisionEncoderModel {
            patch_embed: self.patch_embed.try_init(device)?,
            tokens: self.tokens.try_init(device).context("the prefix tokens")?,
            stack: self.stack.try_init(device).context("the transformer")?,
            pool: TokenPool::try_new(self.pool, self.tokens.n_prefix())?,
            head: self.head.try_init(device).context("the head")?,
        })
    }
}

/// The generic vision encoder: a [`PatchEmbed`], [`PrefixTokens`] with the
/// learnt position embedding, a [`DecoderStack`] of bidirectional layers,
/// a [`TokenPool`] and a [`ClassifierHead`]: timm's `VisionTransformer`.
///
/// Every `ViT` family is this type, built from its policy.
/// [`forward`](Self::forward) maps `[batch, in_channels, height, width]`
/// images to `[batch, num_classes]` logits;
/// [`feature_tokens`](Self::feature_tokens),
/// [`pooled_features`](Self::pooled_features) and
/// [`layer_maps`](Self::layer_maps) are the steps on the way.
///
/// The model runs at its own image size: the position embedding is learnt
/// for its patch grid.
///
/// Implements [`VisionEncoderMeta`].
#[derive(Module, Debug)]
pub struct VisionEncoderModel {
    /// The patch embedding.
    pub patch_embed: PatchEmbed,

    /// The prefix tokens and the position embedding.
    pub tokens: PrefixTokens,

    /// The transformer and its norms.
    pub stack: DecoderStack,

    /// The pool over the output tokens.
    pub pool: TokenPool,

    /// The head, over the pooled features.
    pub head: ClassifierHead,
}

impl VisionEncoderMeta for VisionEncoderModel {
    fn in_channels(&self) -> usize {
        self.patch_embed.d_input()
    }

    fn image_size(&self) -> [usize; 2] {
        self.patch_embed.input_resolution()
    }

    fn patch_size(&self) -> usize {
        self.patch_embed.patch_size()
    }

    fn grid(&self) -> [usize; 2] {
        self.patch_embed.patches_resolution()
    }

    fn d_model(&self) -> usize {
        self.stack.d_model()
    }

    fn n_layers(&self) -> usize {
        self.stack.n_layers()
    }

    fn n_prefix(&self) -> usize {
        self.tokens.n_prefix()
    }

    fn num_classes(&self) -> usize {
        self.head.num_classes()
    }
}

/// The transformer's canonical names under `stack.` (its layers' `attn_norm`,
/// `attn.c_q`, `mlp.linear1`, …); every other path is its own.
impl CanonicalKeys for VisionEncoderModel {
    fn canonical_rekey(&self) -> Rekey {
        Rekey::keep_rest().mount_canonical("stack.", &self.stack)
    }
}

impl VisionEncoderModel {
    /// The logits: the head over the feature tokens; for a headless model,
    /// its pooled features.
    ///
    /// # Arguments
    ///
    /// - `images`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, num_classes]`, or `[batch, d_model]` for a headless model.
    pub fn forward(
        &self,
        images: Tensor<4>,
    ) -> Tensor<2> {
        #[cfg(any(debug_assertions, test))]
        let [batch] = self.unpack_images(&images);
        let logits = self.logits(self.feature_tokens(images));
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "d_output"],
            &logits.dims(),
            &[("batch", batch), ("d_output", self.head.d_output())]
        );
        logits
    }

    /// The batch of `images`, checked against the model's input:
    /// `[batch, in_channels, height, width]`.
    #[cfg(any(debug_assertions, test))]
    fn unpack_images(
        &self,
        images: &Tensor<4>,
    ) -> [usize; 1] {
        let [height, width] = self.image_size();
        crate::contracts::unpack_shape_contract!(
            ["batch", "in_channels", "height", "width"],
            &images.dims(),
            &["batch"],
            &[
                ("in_channels", self.in_channels()),
                ("height", height),
                ("width", width)
            ]
        )
    }

    /// The head's logits of the output tokens: the pool, then the head; for a
    /// distilled head, its two classifiers over the class and distillation
    /// tokens.
    ///
    /// # Arguments
    ///
    /// - `tokens`: `[batch, n_prefix + patches, d_model]`.
    pub fn logits(
        &self,
        tokens: Tensor<3>,
    ) -> Tensor<2> {
        #[cfg(any(debug_assertions, test))]
        {
            let [gh, gw] = self.grid();
            crate::contracts::assert_shape_contract_periodically!(
                ["batch", "n_prefix" + "grid_h" * "grid_w", "d_model"],
                &tokens.dims(),
                &[
                    ("n_prefix", self.n_prefix()),
                    ("grid_h", gh),
                    ("grid_w", gw),
                    ("d_model", self.d_model())
                ]
            );
        }
        if self.head.is_distilled() {
            let token = |i: usize| tokens.clone().slice(s![.., i..i + 1]).squeeze_dim::<2>(1);
            return self.head.forward_distilled(token(0), token(1));
        }
        self.head.forward(self.pool.forward(tokens))
    }

    /// The tokens into the transformer: the patches embedded, the prefix
    /// tokens before them, the position embedding added.
    ///
    /// # Arguments
    ///
    /// - `images`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, n_prefix + patches, d_model]`.
    pub fn embed(
        &self,
        images: Tensor<4>,
    ) -> Tensor<3> {
        self.tokens.forward(self.patch_embed.forward(images))
    }

    /// The transformer's output tokens, normalised (timm's
    /// `forward_features`).
    ///
    /// # Arguments
    ///
    /// - `images`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, n_prefix + patches, d_model]`: the prefix tokens, then the
    /// patch grid, row by row.
    pub fn feature_tokens(
        &self,
        images: Tensor<4>,
    ) -> Tensor<3> {
        self.stack.encoder_forward(self.embed(images), None)
    }

    /// The pooled features, the classifier's input (timm's
    /// `forward_head(..., pre_logits=True)`).
    ///
    /// # Arguments
    ///
    /// - `images`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, d_model]`.
    pub fn pooled_features(
        &self,
        images: Tensor<4>,
    ) -> Tensor<2> {
        #[cfg(any(debug_assertions, test))]
        let [batch] = self.unpack_images(&images);
        let pooled = self
            .head
            .pre_logits(self.pool.forward(self.feature_tokens(images)));
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "d_model"],
            &pooled.dims(),
            &[("batch", batch), ("d_model", self.d_model())]
        );
        pooled
    }

    /// Each layer's output as a map over the patch grid, its prefix tokens
    /// left out: timm's `forward_intermediates` (no final norm).
    ///
    /// # Arguments
    ///
    /// - `images`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `n_layers` maps, `[batch, d_model, grid_h, grid_w]` each.
    pub fn layer_maps(
        &self,
        images: Tensor<4>,
    ) -> Vec<Tensor<4>> {
        let (_, states) = self.stack.encoder_forward_layers(self.embed(images), None);
        states
            .into_iter()
            .map(|tokens| self.grid_map(tokens))
            .collect()
    }

    /// The patch tokens of `tokens` as a `[batch, d_model, grid_h, grid_w]`
    /// map, the prefix left out.
    ///
    /// # Arguments
    ///
    /// - `tokens`: `[batch, n_prefix + patches, d_model]`.
    pub fn grid_map(
        &self,
        tokens: Tensor<3>,
    ) -> Tensor<4> {
        let [gh, gw] = self.grid();
        let n_prefix = self.n_prefix();
        let [batch, d] = crate::contracts::unpack_shape_contract!(
            ["batch", "n_prefix" + "grid_h" * "grid_w", "d_model"],
            &tokens.dims(),
            &["batch", "d_model"],
            &[("n_prefix", n_prefix), ("grid_h", gh), ("grid_w", gw)]
        );
        tokens
            .slice(s![.., n_prefix..n_prefix + gh * gw])
            .reshape([batch, gh, gw, d])
            .permute([0, 3, 1, 2])
    }

    /// The model with its head re-initialized for `num_classes` classes (0:
    /// headless).
    pub fn with_classes(
        self,
        num_classes: usize,
    ) -> Self {
        let device = self.patch_embed.projection.weight.device();
        Self {
            head: self.head.with_classes(num_classes, &device),
            ..self
        }
    }
}
