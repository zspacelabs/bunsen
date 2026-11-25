//! # Vision blocks
//!
//! Blocks for image models: channel attention ([`attention`]), structured
//! drop layers ([`mod@drop`]), patch embedding ([`patching`]), and whole
//! models ([`models`]: the generic conv net). The convolution blocks are domain
//! neutral, so they live in [`crate::blocks::conv`].

pub mod attention;
pub mod drop;
pub mod models;
pub mod patching;
