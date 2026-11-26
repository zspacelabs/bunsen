//! # Vision blocks
//!
//! Blocks for image models: anti-aliased downsampling ([`anti_alias`]),
//! channel attention ([`attention`]), structured
//! drop layers ([`mod@drop`]), the residual units' spatial mixers
//! ([`mixers`]), patch embedding ([`patching`]), and whole
//! models ([`models`]: the generic conv net). The convolution blocks are domain
//! neutral, so they live in [`crate::blocks::conv`].

pub mod anti_alias;
pub mod attention;
pub mod drop;
pub mod mixers;
pub mod models;
pub mod patching;
