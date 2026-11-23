//! # The language decoder
//!
//! [`LmDecoderModel`], the generic decoder-only language model: a token
//! embedding and an [`LmStackDecoder`], a
//! [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//! and a head, which a named model's kit wraps
//! ([`LmDecoderHolder`], [`LmDecoderWrapper`]). A model whose rows are not one
//! table's lookups holds an [`LmStackDecoder`] alone, with its own embedder
//! ([`LmStackDecoderConfig`]).
//!
//! - Configs: [`LmDecoderVariantConfig`], the stack in any of its configs'
//!   shapes, lowers to [`LmDecoderStructureConfig`], which builds the model.
//! - Decoding: through a
//!   [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache)
//!   ([`StackedLm`](crate::kits::support::decoding::StackedLm)), any stack; or
//!   through a flat [`KVCache`](crate::ops::transformers::attention::KVCache)
//!   ([`LmKvForwardHidden`]) when every layer attends (a stack with
//!   linear-attention layers refuses one).
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
mod model;
mod ops;
mod stack_decoder;
mod wrapper;

pub use config::*;
pub use model::*;
pub use ops::*;
pub use stack_decoder::*;
pub use wrapper::*;

#[cfg(test)]
mod tests;
