//! # Language models
//!
//! Models of text, grouped by the task each does. Each task module holds its
//! kits and the parts of the task that do not depend on the model:
//!
//! - [`generation`]: whole decoder-only language models, and the model-generic
//!   parts of a driver that streams their output ([`generation::driver`]).
//! - [`encoders`]: text encoders, which read a whole input at once and give a
//!   hidden state for each token, for a head of their user's to read; an
//!   encoder's stack is a
//!   [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//!   whose attention is bidirectional.
//! - [`decision`]: models that answer typed questions about a state, in
//!   probabilities over each question's options: an encoder and a head.
//!
//! The speech kits build on these: the Qwen3 decoder is
//! [`speech::asr::qwen3_asr`](crate::kits::speech::asr::qwen3_asr)'s text
//! model and the Qwen3-TTS talker's stack, and the ASR kits decode through
//! [`generation::driver`]. A language kit never depends on a speech one.

pub mod decision;
pub mod encoders;
pub mod generation;
