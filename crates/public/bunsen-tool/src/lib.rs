//! # bunsen-tool
//!
//! `bunsen-tool` is bunsen's models on the command line: `bunsen-tool asr`
//! (speech to text, and who spoke when), `bunsen-tool chat` (chat with a
//! Qwen3, Qwen3.5, Gemma or nanochat checkpoint), `bunsen-tool decide`
//! (typed questions about a context, answered by a Laya decision model),
//! `bunsen-tool cache` (the weights cache), `bunsen-tool card` and
//! `bunsen-tool devices`. Its README is the binary's guide.
//!
//! The library is all of it, one module per command, each with what the
//! command runs and, in its `cli`, the command tree the binary parses:
//!
//! - `asr` (feature `speech`): presets and model families for speech to text,
//!   the stream drivers, the publishers; `asr::cli`.
//! - [`chat`]: the chat driver; [`chat::cli`].
//! - [`decide`]: Laya decisions on typed questions; [`decide::cli`].
//! - [`card`] and [`devices`]: the build, its cache and the models it knows;
//!   the backends and their devices.
//! - [`support`]: what the command trees share.

#[cfg(feature = "speech")]
pub mod asr;
pub mod card;
pub mod chat;
pub mod decide;
pub mod devices;
pub mod support;
