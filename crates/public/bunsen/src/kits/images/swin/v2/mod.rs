//! # Swin Transformer V2
//!
//! Swin Transformer V2 ([arXiv:2111.09883](https://arxiv.org/abs/2111.09883)),
//! after Microsoft's [reference
//! implementation](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py).
//! V2 changes V1's block to train stably at larger sizes: scaled cosine
//! attention with a learned logit scale, a continuous relative position
//! bias computed by a small MLP over log-spaced offsets, and a norm after
//! each residual branch rather than before it. [`blocks`] has the parts.
//!
//! # Lifecycle
//!
//! Swin is a [Stacked
//! config](crate::burner::module::ModuleInit#stacked-config) whose
//! lowering can fail:
//!
//! 1. [`SwinTransformerV2ContractConfig`] is the policy: the image size, patch
//!    size, channels, classes and embedding width, one [`LayerConfig`] (block
//!    depth and head count) per stage, the window size, and the dropout rates.
//! 2. [`try_to_structure`](crate::burner::module::ToStructureConfig::try_to_structure)
//!    checks that the stages fit the image and the window, and resolves each
//!    stage's grid, width and per-block drop-path rates into a
//!    [`SwinTransformerV2StructureConfig`]. A policy whose stages do not fit is a
//!    [`Illegal`](crate::errors::BunsenErrorKind::Illegal) error that says
//!    why, not a panic.
//! 3. `init` builds the [`SwinTransformerV2`]: from the structure, or from the
//!    policy in one step through the blanket
//!    [`ModuleInit`](crate::burner::module::ModuleInit), whose `try_init`
//!    returns the same error.
//! 4. [`forward`](SwinTransformerV2::forward) maps an image batch to
//!    classification logits.
//!
//! [`SwinTransformerV2Meta`] answers the same questions of all three.
//!
//! What fits: stage `i` of `n` runs on a grid of `input/(patch_size*2^i)`
//! patches per side, `d_embed*2^i` wide. The last stage's grid must be a
//! non-zero multiple of the window size, and must scale back up to the
//! input exactly, so each side of the image is
//! `k*window_size*2^(n-1)*patch_size` for a whole `k`.
//!
//! Swin has no prefabs and no pretrained weights, and nothing reads
//! upstream's checkpoints: the kit builds models to train.
//! `examples/swin_tiny` trains one on CINIC-10.
//!
//! # Example
//!
//! ```rust
//! use bunsen::{
//!     burner::module::{
//!         ModuleInit,
//!         ToStructureConfig,
//!     },
//!     kits::images::swin::v2::{
//!         LayerConfig,
//!         SwinTransformerV2,
//!         SwinTransformerV2ContractConfig,
//!         SwinTransformerV2Meta,
//!     },
//!     support::testing::cpu_device,
//! };
//! use burn::backend::Flex;
//!
//! let device = cpu_device();
//!
//! // A 256x256 RGB image in 4x4 patches: a 64x64 grid, then 32x32 after
//! // one merge, which the 8x8 window divides.
//! let policy = SwinTransformerV2ContractConfig::new(
//!     [256, 256],
//!     4,
//!     3,
//!     10,
//!     96,
//!     vec![LayerConfig::new(2, 3), LayerConfig::new(2, 6)],
//! )
//! .with_window_size(8)
//! .with_attn_drop_rate(0.2)
//! .with_drop_rate(0.2);
//!
//! let swin_model: SwinTransformerV2<Flex> = policy.init(&device);
//! assert_eq!(swin_model.num_classes(), 10);
//!
//! // A 7x7 window does not divide the last grid: an error, not a panic.
//! assert!(policy.with_window_size(7).try_to_structure().is_err());
//! ```

pub mod blocks;

pub use blocks::swin_model::*;
