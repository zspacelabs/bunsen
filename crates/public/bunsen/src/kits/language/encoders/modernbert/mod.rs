//! # `ModernBERT`
//!
//! Answer.AI's `ModernBERT` encoder, as `transformers`' `ModernBertModel`
//! defines it: a [`ModernBertModel`], built from a
//! [`ModernBertContractConfig`], is a token embedding with a `LayerNorm`, then
//! a
//! [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//! of bidirectional pre-norm layers with a `GeGLU` (exact `GELU`), every
//! third layer global and the others local (a two-sided window of
//! `local_attention / 2` positions), each kind with its own rotary base; the
//! first layer has no attention norm, the embedding's norm standing in for
//! it. Its norms are `LayerNorm`s without a bias, and no projection has one.
//!
//! - [`blocks`]: the model, built from a [`ModernBertContractConfig`].
//! - [`interop`]: its checkpoints' ecosystem, [`interop::transformers`]: a
//!   `transformers` `ModernBertModel`'s `config.json` and state-dict names (a
//!   bare model's, or a `ModernBertFor*`'s under `model.`, its head dropped).
//!
//! The model gives the final hidden states; a head of its user's reads them.
//! A tokenizer is not part of the kit.

pub mod blocks;

pub mod interop;

pub use blocks::model::*;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
