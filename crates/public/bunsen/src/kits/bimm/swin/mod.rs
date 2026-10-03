//! # The Swin Transformer family
//!
//! Swin Transformers are vision transformers that attend within local
//! windows of a patch grid. Successive blocks shift the windows, so
//! information crosses window borders, and each stage merges `2x2`
//! neighbourhoods of patches, halving the grid and doubling the width. The
//! cost of attention then grows with the image's area, not its square.
//!
//! - [`v2`]: Swin Transformer V2, the only version here. There is no V1.

pub mod v2;
