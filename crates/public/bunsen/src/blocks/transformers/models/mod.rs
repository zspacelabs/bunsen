//! # Transformer models
//!
//! Whole models assembled from the blocks, generic over what tells their
//! users apart: [`LmDecoderModel`], a decoder-only language model (a token
//! embedding, a [`DecoderStack`](super::decoders::stack::DecoderStack) and a
//! head). A named model's kit wraps one, with its config, weights and quirks.

mod lm_decoder_model;

pub use lm_decoder_model::*;
