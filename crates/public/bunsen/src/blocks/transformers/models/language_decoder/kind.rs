//! The decode kinds.
use alloc::{
    format,
    string::String,
};
use core::fmt::Debug;

/// How an [`LmDecoderModel`](super::LmDecoderModel)'s stack decodes, as a type:
/// which caches can hold its state, and so which forward API it has.
///
/// A zero-size marker; building a model of a kind refuses a stack that is
/// not of it.
pub trait LmDecodeKind:
    Clone + Copy + Debug + Default + PartialEq + Eq + Send + Sync + 'static
{
    /// The kind's name, in errors.
    const NAME: &'static str;

    /// Why a stack whose layers attend as `layer_attends` says (`true` for
    /// attention, `false` for linear attention) is not of this kind; `None`
    /// when it is.
    fn refusal(layer_attends: &[bool]) -> Option<String>;
}

/// Every layer attends: a flat
/// [`KVCache`](crate::ops::transformers::attention::KVCache) holds the decode's
/// state.
///
/// Its models implement [`LmKvForwardHidden`](super::LmKvForwardHidden) and
/// [`KVCacheModel`](crate::ops::transformers::attention::KVCacheModel), and
/// decode through a
/// [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache)
/// as every kind does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LmAttentionOnlyKind;

impl LmDecodeKind for LmAttentionOnlyKind {
    const NAME: &'static str = "LmAttentionOnlyKind";

    fn refusal(layer_attends: &[bool]) -> Option<String> {
        layer_attends
            .iter()
            .position(|attends| !attends)
            .map(|i| format!("layer {i} is linear attention; every layer must attend"))
    }
}

/// Any stack, linear-attention layers among its layers: only a
/// [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache),
/// which carries a state for each linear layer, holds the decode's state.
///
/// Its models implement
/// [`LmUncachedForwardHidden`](super::LmUncachedForwardHidden).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LmHybridKind;

impl LmDecodeKind for LmHybridKind {
    const NAME: &'static str = "LmHybridKind";

    fn refusal(_layer_attends: &[bool]) -> Option<String> {
        None
    }
}
