//! # Vision blocks
//!
//! Blocks for image models: structured drop layers ([`mod@drop`]), patch
//! embedding ([`patching`]), and pooling that burn's defaults do not cover
//! ([`pool`]). The convolution blocks are domain neutral, so they live in
//! [`crate::blocks::conv`].

pub mod drop;
pub mod patching;
pub mod pool;
