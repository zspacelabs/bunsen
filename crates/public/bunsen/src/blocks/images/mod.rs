//! # Vision blocks
//!
//! Blocks for image models: structured drop layers ([`mod@drop`]) and patch
//! embedding ([`patching`]). The convolution blocks are domain neutral, so they
//! live in [`crate::blocks::conv`].

pub mod drop;
pub mod patching;
