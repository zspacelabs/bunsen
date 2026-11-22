//! # Pooling
//!
//! Blocks that pool a sequence over time into one vector:
//!
//! - [`AttentiveStatsPooling`]: each channel's attention-weighted mean and
//!   standard deviation, the pooling of speaker encoders (ASTP's form and
//!   ECAPA-TDNN's, by config).

mod attentive_stats;

pub use attentive_stats::*;
