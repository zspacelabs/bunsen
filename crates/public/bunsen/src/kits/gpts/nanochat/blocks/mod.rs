//! # `NanoChat` blocks.
//!
//! The model, [`NanoChatGpt`] with its configs and [`NanoChatGptMeta`], and
//! its repeated layer, [`NanoChatGptBlock`]. The attention, MLP and rotary
//! embedding inside them come from [`crate::blocks::transformers`]. The
//! [`nanochat`](super) module docs cover the lifecycle.

mod block;
pub(crate) mod model;

pub use block::*;
pub use model::*;
