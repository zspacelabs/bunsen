//! Silero VAD blocks: the model, the two-rate collection a checkpoint
//! holds, and the per-stream context.
//!
//! [`SileroVadSignalConfig`] (or [`SileroVadStftConfig`]) lowers to a
//! [`SileroVadStructureConfig`], which builds a [`SileroVad`] for one
//! sample rate; [`SileroVad`] documents the pipeline. A
//! [`SileroVadCollection`] holds one model per rate, as the checkpoint
//! does. A [`SileroVadContext`], opened by [`SileroVadContextConfig`], is
//! one stream's state, passed into the model and returned updated: the
//! model holds none.

mod collection;
mod context;
mod module;

pub use collection::*;
pub use context::*;
pub use module::*;
