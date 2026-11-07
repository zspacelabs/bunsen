//! # `ModernBERT`
//!
//! Answer.AI's `ModernBERT` encoder, as `transformers`' `ModernBertModel`
//! defines it: a [`ModernBertModel`], built from a [`ModernBertConfig`], is
//! a token embedding with a `LayerNorm`, then a
//! [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//! of bidirectional pre-norm layers with a `GeGLU` (exact `GELU`), every
//! third layer global and the others local (a two-sided window of
//! `local_attention / 2` positions), each kind with its own rotary base; the
//! first layer has no attention norm, the embedding's norm standing in for
//! it. Its norms are `LayerNorm`s without a bias, and no projection has one.
//!
//! - [`blocks`]: the model, built from a [`ModernBertConfig`].
//! - [`weights`] (feature `store`): loading tensors named as a `transformers`
//!   `ModernBertModel` state dict (a bare model's, or a `ModernBertFor*`'s
//!   under `model.`, its head dropped), and reading its `config.json`.
//!
//! The model gives the final hidden states; a head of its user's reads them.
//! A tokenizer is not part of the kit.

pub mod blocks;

#[cfg(feature = "store")]
pub mod weights;

pub use blocks::model::*;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
