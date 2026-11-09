//! # The language decoder
//!
//! [`LmDecoderModel`], the generic decoder-only language model: a token
//! embedding, a
//! [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//! and a head, which a named model's kit wraps
//! ([`LmDecoderWrapper`]).
//!
//! - Configs: [`LmDecoderVariantConfig`], the stack in any of its configs'
//!   shapes, lowers to [`LmDecoderStructureConfig`], which builds the model.
//! - Kinds: [`LmDecodeKind`], a type parameter of the model, says how it
//!   decodes: [`LmAttentionOnlyKind`] through a flat
//!   [`KVCache`](crate::ops::transformers::attention::KVCache) too
//!   ([`LmKvForwardHidden`]), [`LmHybridKind`] (linear-attention layers
//!   allowed) only through a
//!   [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache)
//!   ([`LmUncachedForwardHidden`] besides).
//! - API: [`LmDecoderOps`], every kind's (the stack, rotary tables, embedding,
//!   head, attention weights), and [`LmDecoderMeta`].
//! - Wrappers: a kit's model implements [`LmDecoderWrapper`], and gets the
//!   model's API, the forward by its kind, through blanket impls; it writes no
//!   pass-throughs.

mod config;
mod kind;
mod model;
mod ops;
mod wrapper;

pub use config::*;
pub use kind::*;
pub use model::*;
pub use ops::*;
pub use wrapper::*;

#[cfg(test)]
mod tests;
