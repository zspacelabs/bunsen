//! # bunsen-tool
//!
//! `bunsen-tool` is bunsen's models on the command line: `bunsen-tool asr`
//! (speech to text, and who spoke when), `bunsen-tool chat` (chat with a
//! Qwen3 or nanochat checkpoint) and `bunsen-tool cache` (the weights
//! cache). Its README is the binary's guide.
//!
//! The library is what the binary's `chat` is built on, for a program of its
//! own: [`chat`], the chat driver.

pub mod chat;
