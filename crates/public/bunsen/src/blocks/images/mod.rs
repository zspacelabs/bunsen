//! # Vision blocks
//!
//! Blocks for image models: structured drop layers ([`mod@drop`]), patch
//! embedding ([`patching`]), and whole models ([`models`]: the generic conv
//! net). The convolution blocks are domain neutral, so they live in
//! [`crate::blocks::conv`].

pub mod drop;
pub mod models;
pub mod patching;
