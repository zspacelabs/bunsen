//! # GPT language models
//!
//! Whole decoder-only language models. Their parts are elsewhere:
//! attention, the MLP and the rotary embedding in
//! [`crate::blocks::transformers`], the KV cache in
//! [`crate::ops::transformers::attention`]. A kit here assembles them into
//! a model, with its configs and, where it trains on a public corpus, a
//! table of that corpus as a [`data::shards`](crate::data::shards) shard
//! set.
//!
//! - [`nanochat`]: the GPT in karpathy's nanochat, upstream's 2025 and 2026
//!   models: the community checkpoints of both, loaded from upstream's own
//!   files, and the corpus it trains on.
//! - [`qwen3`]: the Qwen3 decoder: the dense Qwen3 checkpoints, and the text
//!   model of [`speech::qwen3_asr`](crate::kits::speech::qwen3_asr).
//! - [`driver_support`]: the model-generic parts of a driver that streams a
//!   model's output: text deltas, stop strings, and the cached prefix a
//!   multi-turn session keeps.
//!
//! Tokenizers and training loops are not part of a kit:
//! `examples/train-chat` brings its own (a `wordchipper` tokenizer, burn's
//! `Learner`, and bunsen's optimizer groups) around
//! [`NanoChatGpt`](nanochat::NanoChatGpt).

pub mod driver_support;
pub mod nanochat;
pub mod qwen3;
