//! # Transformer models
//!
//! Whole models assembled from the blocks, generic over what tells their
//! users apart:
//! [`language_decoder`], the decoder-only language model
//! ([`LmDecoderModel`](language_decoder::LmDecoderModel): a token embedding,
//! a [`DecoderStack`](super::decoders::stack::DecoderStack) and a head). A
//! named model's kit wraps one, with its config, weights and quirks.

pub mod language_decoder;
