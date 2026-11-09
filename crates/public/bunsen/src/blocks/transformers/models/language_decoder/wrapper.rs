//! [`LmDecoderWrapper`]: a kit's model around an [`LmDecoderModel`].
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

/// A named model that wraps an [`LmDecoderModel`]: a kit's model, which
/// adds its own names, config and quirks around one.
///
/// The model's API comes through blanket impls, by its kind: every wrapper
/// is [`LmDecoderOps`](super::LmDecoderOps),
/// [`LmPlainStackOps`](super::LmPlainStackOps), [`LmDecoderMeta`] and a
/// [`StackedLm`](crate::kits::support::decoding::StackedLm);
/// an [`LmAttentionOnlyKind`] one has [`LmKvForwardHidden`] and
/// [`KVCacheModel`], an [`LmHybridKind`] one [`LmUncachedForwardHidden`].
///
/// A wrapper loads its weights into [`lm_mut`](Self::lm_mut), under a rekey
/// onto the [`LmDecoderModel`]'s own paths.
pub trait LmDecoderWrapper {
    /// The wrapped model's decode kind.
    type Kind: LmDecodeKind;

    /// The wrapped model.
    fn lm(&self) -> &LmDecoderModel<Self::Kind>;

    /// The wrapped model, to load into.
    fn lm_mut(&mut self) -> &mut LmDecoderModel<Self::Kind>;
}

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

/// The wrapped model's meta.
impl<W: LmDecoderWrapper> LmDecoderMeta for W {
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

/// The wrapped model's cache.
impl<W: LmDecoderWrapper<Kind = LmAttentionOnlyKind>> KVCacheModel for W {
    fn kv_cache_config(&self) -> KVCacheConfig {
        self.lm().kv_cache_config()
    }
}
