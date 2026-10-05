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
//! - [`nanochat`]: a port of the GPT in karpathy's nanochat, and the corpus it
//!   trains on. Work in progress.
//!
//! Tokenizers and training loops are not part of a kit:
//! `examples/train-chat` brings its own (a `wordchipper` tokenizer, burn's
//! `Learner`, and bunsen's optimizer groups) around
//! [`NanoChatGpt`](nanochat::NanoChatGpt).

pub mod nanochat;
