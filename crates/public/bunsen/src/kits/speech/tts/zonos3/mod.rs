//! # ZONOS3-Mini
//!
//! Zyphra's ZONOS3-Mini text-to-speech model: a language model over rows
//! of ten audio codebooks and a text token, and a codec that decodes the
//! codes to 44.1 kHz audio.
//!
//! - [`lm`]: the language model, [`Zonos3Lm`](lm::Zonos3Lm). Each row's ids are
//!   summed through their own tables (a speaker row and the condition rows
//!   replaced), run through a stack of gated `DeltaNet` and global attention
//!   layers without a residual stream
//!   ([`AttnResStack`](crate::blocks::transformers::decoders::attn_res::AttnResStack)),
//!   their feed-forwards a dense `SwiGLU` and then latent mixtures of experts,
//!   and read out as every codebook's logits.
//!
//! The weights are `Zyphra/ZONOS3-Mini-hf`'s `model.safetensors`, whose
//! `lm.*` tensors are upstream's release language model as it is
//! ([`lm::weights`]).

pub mod lm;
