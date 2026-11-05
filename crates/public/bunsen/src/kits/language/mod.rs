//! # Language models
//!
//! Whole decoder-only language models. Their parts are elsewhere:
//! attention, the MLP and the rotary embedding in
//! [`crate::blocks::transformers`], the KV cache in
//! [`crate::ops::transformers::attention`]. A kit here assembles them into
//! a model, with its configs and, where it trains on a public corpus, a
//! table of that corpus as a [`data::shards`](crate::data::shards) shard
//! set.
//!
//! - [`gemma`]: Google's Gemma, generations 1 to 3 (their text models).
//! - [`nanochat`]: the GPT in karpathy's nanochat, upstream's 2025 and 2026
//!   models: the community checkpoints of both, loaded from upstream's own
//!   files, and the corpus it trains on.
//! - [`qwen3`]: the Qwen3 decoder: the dense Qwen3 checkpoints, and the text
//!   model of [`speech::asr::qwen3_asr`](crate::kits::speech::asr::qwen3_asr).
//! - [`driver`]: the model-generic parts of a driver that streams a model's
//!   output: text deltas, stop strings, and the cached prefix a multi-turn
//!   session keeps.
//!
//! Tokenizers and training loops are not part of a kit:
//! `examples/train-chat` brings its own (a `wordchipper` tokenizer, burn's
//! `Learner`, and bunsen's optimizer groups) around
//! [`NanoChatGpt`](nanochat::NanoChatGpt).

pub mod driver;
pub mod gemma;
pub mod nanochat;
pub mod qwen3;
