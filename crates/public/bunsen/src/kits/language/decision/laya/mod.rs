//! # Laya
//!
//! Laya (`convaiinnovations/laya`, Apache-2.0), an open decision model: a
//! state and typed questions in (a choice among named options, a level on a
//! scale, a yes/no probability), calibrated probabilities over each
//! question's options out, in one forward pass. A [`LayaModel`], built from a
//! [`LayaConfig`], is upstream's `DecisionModel`: a
//! [`ModernBertModel`](crate::kits::language::encoders::modernbert::ModernBertModel)
//! encoder, the question kind's type embedding, two `PyTorch`-layout encoder
//! layers, and a scorer over each option's `[MASK]` marker; and an act head.
//!
//! - [`blocks`]: the model, built from a [`LayaConfig`].
//! - [`LayaAgentConfig`]: the head's geometry, the prompt budgets, and the
//!   temperatures that calibrate each question's logits by its kind and option
//!   count; a release's is its `rl_agent_config.json`'s.
//! - [`LayaQuestion`] and [`LayaAnswer`]: a typed question (its kind, what is
//!   asked, its options), and its answer (each option's calibrated probability,
//!   the most probable, its confidence).
//! - [`build_prompt`]: a question's prompt over a state, as upstream's
//!   `build_sequence` lays it out and cuts it to its budgets, the specials
//!   ([`LayaSpecials`]) from the checkpoint's `tokenizer_config.json`.
//! - [`LayaModel::decide`]: questions about a state answered, in one padded
//!   batch.
//! - [`LayaConstruct`]: a model with its head's config and its checkpoint's
//!   tokenizer files, unread.
//! - [`pretrained`] (features `store` and `cache`): the named checkpoints,
//!   `laya/laya` the default, and the factory that loads one.
//! - [`interop`]: its checkpoints' ecosystem, [`interop::laya`]: the head's
//!   `rl_agent_config.json`, the model's config from its two files, and the
//!   `model.safetensors`' names (the encoder under `encoder.`, the fused
//!   projections split).
//!
//! The tokenizer is a [`Tokenizer`](crate::kits::support::tokens::Tokenizer)
//! the caller brings: `BunsenTokenizer` (feature `tokenizer`) reads a
//! checkpoint's `tokenizer.json`.

mod agent_config;
pub mod blocks;
mod construct;
mod prompt;

pub mod interop;

pub use agent_config::*;
pub use blocks::model::*;
pub use construct::*;
pub use prompt::*;

#[cfg(feature = "store")]
pub mod pretrained;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
