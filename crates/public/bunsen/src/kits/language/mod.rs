//! # Language models
//!
//! Models of text: the decoder-only language models; the text [`encoders`],
//! which read a whole input at once and give a hidden state for each token,
//! for a head of their user's to read; and the [`decision`] models, an encoder
//! and a head answering typed questions. Their parts are elsewhere:
//! attention, the MLP and the rotary embedding in
//! [`crate::blocks::transformers`], the KV cache in
//! [`crate::ops::transformers::attention`]. A kit here assembles them into
//! a model, with its configs and, where it trains on a public corpus, a
//! table of that corpus as a [`data::datasets`](crate::data::datasets) shard
//! set.
//!
//! - [`gemma`]: Google's Gemma, generations 1 to 3 (their text models).
//! - [`nanochat`]: the GPT in karpathy's nanochat, upstream's 2025 and 2026
//!   models: the community checkpoints of both, loaded from upstream's own
//!   files, and the corpus it trains on.
//! - [`qwen3`]: the Qwen3 decoder: the dense Qwen3 checkpoints, and the text
//!   model of [`speech::asr::qwen3_asr`](crate::kits::speech::asr::qwen3_asr).
//! - [`qwen3_5`]: the dense Qwen3.5 decoder, a hybrid of gated `DeltaNet`
//!   (linear-attention) and gated full-attention layers.
//! - [`driver`]: the model-generic parts of a driver that streams a model's
//!   output: text deltas, stop strings, and the cached prefix a multi-turn
//!   session keeps.
//! - [`decision`]: models that answer typed questions about a state, in
//!   probabilities over each question's options.
//! - [`encoders`]: the text encoders; an encoder's stack is a
//!   [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//!   whose attention is bidirectional.
//!
//! Tokenizers and training loops are not part of a kit:
//! `examples/train-chat` brings its own (a `wordchipper` tokenizer, burn's
//! `Learner`, and bunsen's optimizer groups) around
//! [`NanoChatGpt`](nanochat::NanoChatGpt).

pub mod decision;
pub mod driver;
pub mod encoders;
pub mod gemma;
pub mod nanochat;
pub mod qwen3;
pub mod qwen3_5;
