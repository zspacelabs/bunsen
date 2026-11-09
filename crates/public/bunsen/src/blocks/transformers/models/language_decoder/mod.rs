//! # The language decoder
//!
//! [`LmDecoderModel`], the generic decoder-only language model: a token
//! embedding, a
//! [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//! and a head, which a named model's kit wraps
//! ([`LmDecoderHolder`], [`LmDecoderWrapper`]).
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
//! - API, by layer: [`LmDecoderOps`] and [`LmDecoderMeta`], which say nothing
//!   of the forward (the embedding, head, rotary tables, dtype); and
//!   [`LmPlainStackOps`], for a model whose forward is its stack's alone (the
//!   stack, the attention weights), which the kinds' forwards and
//!   [`StackedLm`](crate::kits::support::decoding::StackedLm) build on.
//! - Holders and wrappers: a kit's model that holds exactly an
//!   [`LmDecoderModel`] implements [`LmDecoderHolder`], and gets the
//!   forward-free API and its build through blanket impls; one whose forward is
//!   the model's also implements [`LmDecoderWrapper`], and gets the forward, by
//!   its kind. It writes no pass-throughs.

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
