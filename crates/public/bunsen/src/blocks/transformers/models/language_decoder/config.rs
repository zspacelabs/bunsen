//! The configs of an [`LmDecoderModel`].
use alloc::format;
use core::marker::PhantomData;

use burn::{
    config::Config,
    tensor::Device,
};

use super::{
    LmDecodeKind,
    LmDecoderMeta,
    LmDecoderModel,
};
use crate::{
    blocks::transformers::{
        decoders::stack::{
            DecoderStackMeta,
            DecoderStackStructureConfig,
            DecoderStackVariantConfig,
        },
        embedding::TokenEmbeddingConfig,
        head::LmHeadConfig,
    },
    burner::module::{
        ModuleInit,
        ToStructureConfig,
    },
    errors::{
        BunsenResult,
        ConstraintError,
        ResultContext,
    },
};

/// The positions an [`LmDecoderModel`]'s KV cache allocates up front by
/// default: one growth chunk. A caller that knows a decode's length sizes
/// the cache to it.
pub const LM_DECODER_KV_CACHE_PRE_SIZE: usize = 1024;

/// Config for [`LmDecoderModel`], its stack in any of its configs' shapes:
/// what a named model's config lowers to
/// ([`ToStructureConfig`], `Structure = LmDecoderVariantConfig`), so it
/// builds the kit's [`LmDecoderHolder`](super::LmDecoderHolder) through the
/// blanket `ModuleInit`s.
///
/// Lowers to [`LmDecoderStructureConfig`], and builds the model through it.
#[derive(Config, Debug)]
pub struct LmDecoderVariantConfig {
    /// The token embedding.
    pub embedding: TokenEmbeddingConfig,

    /// The output head: tied to the embedding, or a projection of its own.
    pub head: LmHeadConfig,

    /// The decoder stack.
    pub stack: DecoderStackVariantConfig,

    /// The positions a KV cache allocates up front by default.
    #[config(default = "LM_DECODER_KV_CACHE_PRE_SIZE")]
    pub kv_cache_pre_size: usize,
}

impl ToStructureConfig for LmDecoderVariantConfig {
    type Structure = LmDecoderStructureConfig;

    /// # Errors
    /// As the stack's lowering, with a frame naming it.
    fn try_to_structure(&self) -> BunsenResult<LmDecoderStructureConfig> {
        Ok(LmDecoderStructureConfig::new(
            self.embedding.clone(),
            self.head.clone(),
            self.stack
                .try_to_structure()
                .context("lowering an LmDecoderVariantConfig's stack")?,
        )
        .with_kv_cache_pre_size(self.kv_cache_pre_size))
    }
}

/// Structure config for [`LmDecoderModel`]: the embedding, the head, and
/// the stack unrolled.
///
/// Builds a model of any [`LmDecodeKind`] its stack is of.
/// Implements [`LmDecoderMeta`] (once its stack has a layer that attends).
#[derive(Config, Debug)]
pub struct LmDecoderStructureConfig {
    /// The token embedding.
    pub embedding: TokenEmbeddingConfig,

    /// The output head: tied to the embedding, or a projection of its own.
    pub head: LmHeadConfig,

    /// The decoder stack.
    pub stack: DecoderStackStructureConfig,

    /// The positions a KV cache allocates up front by default.
    #[config(default = "LM_DECODER_KV_CACHE_PRE_SIZE")]
    pub kv_cache_pre_size: usize,
}

impl LmDecoderMeta for LmDecoderStructureConfig {
    fn n_vocab(&self) -> usize {
        self.embedding.n_vocab
    }

    fn n_logits(&self) -> usize {
        use crate::blocks::transformers::head::LmHeadMeta;
        self.head.n_vocab()
    }

    fn d_model(&self) -> usize {
        self.embedding.d_model
    }

    fn n_layers(&self) -> usize {
        self.stack.n_layers()
    }

    fn n_kv_heads(&self) -> usize {
        self.stack.n_kv_heads()
    }

    fn head_dim(&self) -> usize {
        self.stack.head_dim()
    }
}

impl<K: LmDecodeKind> ModuleInit<LmDecoderModel<K>> for LmDecoderStructureConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, for a stack that is not of kind `K`; and
    /// as the embedding's, head's and stack's builds.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<LmDecoderModel<K>> {
        if let Some(why) = K::refusal(&self.stack.layer_attends()) {
            return Err(ConstraintError::custom(
                "LmDecoderStructureConfig",
                "stack",
                format!("not of kind {}: {why}", K::NAME),
            )
            .into());
        }
        Ok(LmDecoderModel {
            embedding: self.embedding.try_init(device)?,
            head: self.head.try_init(device)?,
            stack: self.stack.try_init(device)?,
            kv_cache_pre_size: self.kv_cache_pre_size,
            kind: PhantomData,
        })
    }
}

/// A holder is built around the model the config builds
/// ([`LmDecoderHolder::from_lm`](super::LmDecoderHolder::from_lm)).
impl<W: super::LmDecoderHolder + burn::module::Module> ModuleInit<W> for LmDecoderStructureConfig {
    /// # Errors
    /// As the model's build.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<W> {
        Ok(W::from_lm(ModuleInit::<LmDecoderModel<W::Kind>>::try_init(
            self, device,
        )?))
    }
}
