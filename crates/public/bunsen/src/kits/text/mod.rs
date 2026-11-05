//! # Text models
//!
//! Models of text that are not decoder-only language models: the encoders,
//! which read a whole input at once and give a hidden state for each token,
//! for a head of their user's to read. Their parts are the language models':
//! attention, the MLP and the rotary embedding in
//! [`crate::blocks::transformers`], the stack in
//! [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack),
//! whose attention is bidirectional here.
//!
//! - [`encoders`]: the text encoders.

pub mod encoders;
