//! [`LmDecoderHolder`] and [`LmDecoderWrapper`]: a kit's model around an [`LmDecoderModel`].
use burn::Tensor;

use super::{
    LmAttentionOnlyKind,
    LmDecodeKind,
    LmDecoderMeta,
    LmDecoderModel,
    LmHybridKind,
    LmKvForwardHidden,
    LmUncachedForwardHidden,
};
use crate::{
    blocks::transformers::embedding::RotarySet,
    ops::transformers::attention::{
        KVCache,
        KVCacheConfig,
        KVCacheModel,
    },
};

/// A named model that holds exactly an [`LmDecoderModel`]: a kit's model,
/// its own names, config and quirks around one, and no parameters of its
/// own.
///
/// The model's forward-free API comes through blanket impls:
/// [`LmDecoderOps`](super::LmDecoderOps), [`LmDecoderMeta`], an
/// [`LmAttentionOnlyKind`] one's [`KVCacheModel`] (the cache's geometry is
/// its stack's), and its build from an
/// [`LmDecoderStructureConfig`](super::LmDecoderStructureConfig)
/// ([`from_lm`](Self::from_lm)). Its forward is its own unless it is an
/// [`LmDecoderWrapper`] (a model that conditions its layers on inputs
/// beside the stack's has its own).
///
/// A holder loads its weights into [`lm_mut`](Self::lm_mut), under a rekey
/// onto the [`LmDecoderModel`]'s own paths.
pub trait LmDecoderHolder {
    /// The held model's decode kind.
    type Kind: LmDecodeKind;

    /// The held model.
    fn lm(&self) -> &LmDecoderModel<Self::Kind>;

    /// The held model, to load into.
    fn lm_mut(&mut self) -> &mut LmDecoderModel<Self::Kind>;

    /// The holder around `lm`: what an
    /// [`LmDecoderStructureConfig`](super::LmDecoderStructureConfig) builds the
    /// holder as (its blanket `ModuleInit`), so a kit's config that lowers to
    /// one builds the kit's model.
    fn from_lm(lm: LmDecoderModel<Self::Kind>) -> Self;
}

/// An [`LmDecoderHolder`] whose forward is its [`LmDecoderModel`]'s: the
/// bare stack, from the embedding.
///
/// The forward comes through blanket impls, by the kind: every wrapper is
/// [`LmPlainStackOps`](super::LmPlainStackOps) and a
/// [`StackedLm`](crate::kits::support::decoding::StackedLm); an
/// [`LmAttentionOnlyKind`] one has [`LmKvForwardHidden`], an
/// [`LmHybridKind`] one [`LmUncachedForwardHidden`].
pub trait LmDecoderWrapper: LmDecoderHolder {}

impl<W: LmDecoderWrapper<Kind = LmAttentionOnlyKind>> LmKvForwardHidden for W {
    fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
        kv_cache: Option<&mut KVCache>,
    ) -> Tensor<3> {
        self.lm().forward_hidden(embeds, r_emb, kv_cache)
    }
}

impl<W: LmDecoderWrapper<Kind = LmHybridKind>> LmUncachedForwardHidden for W {
    fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
    ) -> Tensor<3> {
        self.lm().forward_hidden(embeds, r_emb)
    }
}

/// The held model's meta.
impl<W: LmDecoderHolder> LmDecoderMeta for W {
    fn n_vocab(&self) -> usize {
        self.lm().n_vocab()
    }

    fn n_logits(&self) -> usize {
        self.lm().n_logits()
    }

    fn d_model(&self) -> usize {
        self.lm().d_model()
    }

    fn n_layers(&self) -> usize {
        self.lm().n_layers()
    }

    fn n_kv_heads(&self) -> usize {
        self.lm().n_kv_heads()
    }

    fn head_dim(&self) -> usize {
        self.lm().head_dim()
    }
}

/// The held model's cache.
impl<W: LmDecoderHolder<Kind = LmAttentionOnlyKind>> KVCacheModel for W {
    fn kv_cache_config(&self) -> KVCacheConfig {
        self.lm().kv_cache_config()
    }
}
