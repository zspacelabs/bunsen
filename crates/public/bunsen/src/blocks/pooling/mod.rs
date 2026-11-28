//! # Pooling
//!
//! Blocks that pool a sequence into one vector:
//!
//! - [`AttentiveStatsPooling`]: each channel's attention-weighted mean and
//!   standard deviation over time, the pooling of speaker encoders (ASTP's form
//!   and a form with a global context and a centered deviation, by config).
//! - [`LatentAttentionPool`]: a learned query's attention over the tokens, then
//!   an MLP: a vision transformer's attention pooling (timm's
//!   `AttentionPoolLatent`).
//!
//! ## Discussion
//!
//! The global-context form of [`AttentiveStatsPooling`] is ECAPA-TDNN's, and
//! [`LatentAttentionPool`] is `SigLIP`'s multi-head attention pooling.

mod attentive_stats;
mod latent_attention;

pub use attentive_stats::*;
pub use latent_attention::*;
