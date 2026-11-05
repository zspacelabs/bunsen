//! # Decision models
//!
//! Models that answer typed questions about a state with probabilities over
//! each question's options, in one forward pass, and generate no text. Their
//! encoders are [`crate::kits::text::encoders`]'.
//!
//! - [`laya`]: Laya (`convaiinnovations/laya`), a `ModernBERT` encoder and a
//!   decision head.

pub mod laya;
