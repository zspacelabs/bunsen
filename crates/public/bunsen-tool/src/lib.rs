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
//! command runs and its command tree, nested as the commands are: a
//! `{group}_command_group` per group, a `{group}_{command}` per command, every
//! type named by its command path (`DecideRunCommand`, `ChatKitCommandGroup`)
//! and re-exported at the command's module:
//!
//! - `asr` (feature `speech`): presets and model families for speech to text,
//!   the stream drivers, the publishers; `asr::AsrCommandGroup`.
//! - [`chat`]: the chat driver; [`chat::ChatCommandGroup`].
//! - [`decide`]: Laya decisions on typed questions;
//!   [`decide::DecideCommandGroup`].
//! - [`card`] and [`devices`]: the build, its cache and the models it knows;
//!   the backends and their devices.
//! - [`select`]: choosing a model by its tags: `--select`, selection
//!   expressions, and each command's defaults and named selections.
//! - [`support`]: what the command trees share.

#[cfg(feature = "speech")]
pub mod asr;
pub mod card;
pub mod chat;
pub mod decide;
pub mod devices;
pub mod select;
pub mod support;
