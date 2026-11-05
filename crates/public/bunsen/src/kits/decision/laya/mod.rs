//! # Laya
//!
//! Laya (`convaiinnovations/laya`, Apache-2.0), an open decision model: a
//! state and typed questions in (a choice among named options, a level on a
//! scale, a yes/no probability), calibrated probabilities over each
//! question's options out, in one forward pass. A [`LayaModel`], built from a
//! [`LayaConfig`], is upstream's `DecisionModel`: a
//! [`ModernBertModel`](crate::kits::text::encoders::modernbert::ModernBertModel)
//! encoder, the question kind's type embedding, two `PyTorch`-layout encoder
//! layers, and a scorer over each option's `[MASK]` marker; and an act head.
//!
//! - [`blocks`]: the model, built from a [`LayaConfig`].
//! - [`LayaAgentConfig`]: `rl_agent_config.json`, the head's geometry, the
//!   prompt budgets, and the temperatures that calibrate each question's logits
//!   by its kind and option count.
//! - [`weights`] (feature `store`): loading a checkpoint's `model.safetensors`
//!   (the encoder under `encoder.`, the fused projections split), and the
//!   config from its two files.
//!
//! A tokenizer and the prompt's layout are not part of this module yet.

mod agent_config;
pub mod blocks;

#[cfg(feature = "store")]
pub mod weights;

pub use agent_config::*;
pub use blocks::model::*;

#[cfg(all(test, feature = "store_safetensors"))]
mod toy_tests;
