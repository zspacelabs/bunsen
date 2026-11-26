//! # The Swin Transformer family
//!
//! Swin Transformers are vision transformers that attend within local
//! windows of a patch grid. Successive blocks shift the windows, so
//! information crosses window borders, and each stage merges `2x2`
//! neighbourhoods of patches, halving the grid and doubling the width. The
//! cost of attention then grows with the image's area, not its square.
//!
//! Each version is the generic
//! [`StagedVisionEncoderModel`](crate::blocks::transformers::models::staged_vision_encoder::StagedVisionEncoderModel),
//! its blocks
//! [`WindowAttention`](crate::blocks::transformers::attention::window::WindowAttention)
//! mixers; a kit is its policy, its prefabs and timm's checkpoints.
//!
//! - [`v2`]: Swin Transformer V2.

pub mod interop;
pub mod v2;

mod common;

pub use common::SWIN_NORM_EPSILON;
