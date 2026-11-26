//! # Transformer models
//!
//! Whole models assembled from the blocks, generic over what tells their
//! users apart:
//!
//! - [`language_decoder`], the decoder-only language model
//!   ([`LmDecoderModel`](language_decoder::LmDecoderModel): a token embedding,
//!   a [`DecoderStack`](super::decoders::stack::DecoderStack) and a head). A
//!   named model's kit wraps one, with its config, weights and quirks.
//! - [`vision_encoder`], the vision transformer
//!   ([`VisionEncoderModel`](vision_encoder::VisionEncoderModel): a patch
//!   embedding, prefix tokens, a bidirectional `DecoderStack` and a token
//!   head). A `ViT` family's kit holds no model type of its own: its model is
//!   this one, built from its config.
//! - [`staged_vision_encoder`], the hierarchical vision transformer
//!   ([`StagedVisionEncoderModel`](staged_vision_encoder::StagedVisionEncoderModel):
//!   a patch embedding, stages of a patch merge and a `DecoderStack` of
//!   window-attention blocks, a token pool and the head): the Swin
//!   Transformers'.

pub mod language_decoder;
pub mod staged_vision_encoder;
pub mod vision_encoder;
