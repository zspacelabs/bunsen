//! # Swin Transformer V2
//!
//! Swin Transformer V2 ([arXiv:2111.09883](https://arxiv.org/abs/2111.09883)),
//! with the checkpoints `timm` publishes. V2 changes V1's block to train
//! stably at larger sizes: scaled cosine attention with a learned logit
//! scale, a continuous relative position bias computed by a small MLP over
//! log-spaced offsets (scaled by the window the weights were pretrained at),
//! and a norm after each residual branch rather than before it. The width,
//! the depths and heads, the window and the image size name the variant:
//! `swinv2_tiny_window8_256` to `swinv2_large_window12to24_192to384`.
//!
//! [`pretrained`] turns names into loaded models.
//! [`SwinV2ContractConfig`]'s prefabs name the 12 geometries, by timm's
//! architecture names, and
//! [`default_swin_v2_factory`](pretrained::default_swin_v2_factory) is the
//! index of the timm hub repos (`timm/swinv2_tiny_window8_256.ms_in1k`), and
//! loads any other of the prefabs' architectures as `hf:timm/<repo>`.
//!
//! # The model
//!
//! A Swin V2 is the generic staged vision encoder,
//! [`StagedVisionEncoderModel`](crate::blocks::transformers::models::staged_vision_encoder::StagedVisionEncoderModel):
//! the kit has no model type of its own. [`SwinV2ContractConfig`] is the
//! policy in timm's terms (the image and patch sizes, the width, the depths
//! and heads, the window and the pretraining windows, the MLP ratio, the
//! `qkv` bias, the pool, the training regularizers) and lowers to the
//! generic structure, its blocks
//! [`WindowAttention`](crate::blocks::transformers::attention::window::WindowAttention)
//! mixers in res-post-norm decoder blocks; `init` builds the model through
//! it. A policy whose stages don't fit is an
//! [`Illegal`](crate::errors::BunsenErrorKind::Illegal) error, not a panic.
//!
//! The model's parameter paths are the staged encoder's own; the family's
//! [`interop`](super::interop) maps timm's names onto them
//! ([`swin_rekey`](super::interop::timm::swin_rekey)), and [`interop`]
//! reads a timm `config.json`.
//!
//! # Compatibility
//!
//! The kit aims for parity with `timm`'s `swin_transformer_v2.py`. Missing
//! today:
//!
//! * running at another size than the model's (`set_input_size`,
//!   `dynamic_img_size`), and a rectangular image or window
//! * `always_partition`, and the per-stage window sizes
//! * the original Microsoft checkpoints' layout (merges at the end of a stage,
//!   `head.weight`), which timm's `checkpoint_filter_fn` converts
//!
//! # Example
//!
//! ```rust
//! use bunsen::{
//!     blocks::transformers::models::staged_vision_encoder::{
//!         StagedVisionEncoderMeta,
//!         StagedVisionEncoderModel,
//!     },
//!     burner::module::{
//!         ModuleInit,
//!         ToStructureConfig,
//!     },
//!     kits::images::classification::swin::v2::SwinV2ContractConfig,
//!     support::testing::cpu_device,
//! };
//!
//! let device = cpu_device();
//!
//! // 32x32 RGB images in 4x4 patches: an 8x8 grid, then 4x4 after the
//! // merge, in 4x4 windows; 10 classes.
//! let policy =
//!     SwinV2ContractConfig::new(32, 16, vec![2, 2], vec![2, 4], 4, 10)
//!         .with_attn_drop_rate(0.1)
//!         .with_drop_path_rate(0.1);
//!
//! let model: StagedVisionEncoderModel = policy.init(&device);
//! assert_eq!(model.num_classes(), 10);
//! assert_eq!(model.stage_grids(), [[8, 8], [4, 4]]);
//!
//! // Three heads don't divide a stage 16 wide: an error, not a panic.
//! assert!(
//!     SwinV2ContractConfig {
//!         num_heads: vec![3, 4],
//!         ..policy
//!     }
//!     .try_to_structure()
//!     .is_err()
//! );
//! ```

pub mod pretrained;

pub mod interop;

mod contract;

#[cfg(all(test, feature = "store"))]
mod toy_tests;

pub use contract::*;
