//! # Swin Transformer (V1)
//!
//! The Swin Transformer ([arXiv:2103.14030](https://arxiv.org/abs/2103.14030)),
//! with the checkpoints `timm` publishes: shifted-window self-attention with
//! a learnt relative position bias table, in pre-norm blocks, over a grid
//! that each stage after the first halves (a patch merge, normed before its
//! reduction) and widens. The width, the depths and heads, the window and the
//! image size name the variant: `swin_tiny_patch4_window7_224` to
//! `swin_large_patch4_window12_384`, and the Swin-S3 geometries
//! ([arXiv:2111.14725](https://arxiv.org/abs/2111.14725)), a window per
//! stage (`swin_s3_tiny_224`).
//!
//! [`pretrained`] turns names into loaded models.
//! [`SwinV1ContractConfig`]'s prefabs name the 9 geometries, by timm's
//! architecture names, and
//! [`default_swin_v1_factory`](pretrained::default_swin_v1_factory) is the
//! index of the timm hub repos (`timm/swin_tiny_patch4_window7_224.ms_in1k`),
//! and loads any other of the prefabs' architectures as `hf:timm/<repo>`.
//!
//! # The model
//!
//! A Swin is the generic staged vision encoder,
//! [`StagedVisionEncoderModel`](crate::blocks::transformers::models::staged_vision_encoder::StagedVisionEncoderModel):
//! the kit has no model type of its own. [`SwinV1ContractConfig`] is the
//! policy in timm's terms (the image and patch sizes, the width, the depths
//! and heads, the window or a window per stage, the MLP ratio, the `qkv`
//! bias, the pool, the training regularizers) and lowers to the generic
//! structure, its blocks
//! [`WindowAttention`](crate::blocks::transformers::attention::window::WindowAttention)
//! mixers in pre-norm decoder blocks; `init` builds the model through it. A
//! policy whose stages don't fit is an
//! [`Illegal`](crate::errors::BunsenErrorKind::Illegal) error, not a panic.
//!
//! The model's parameter paths are the staged encoder's own; the family's
//! [`interop`](super::interop) maps timm's names onto them
//! ([`swin_rekey`](super::interop::timm::swin_rekey)), and [`interop`]
//! reads a timm `config.json`.
//!
//! # Compatibility
//!
//! The kit aims for parity with `timm`'s `swin_transformer.py`. Missing
//! today:
//!
//! * running at another size than the model's (`set_input_size`,
//!   `dynamic_img_size`, a resized position bias table), and a rectangular
//!   image or window
//! * `always_partition`, and `head_dim` other than the width over the heads
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
//!     burner::module::ModuleInit,
//!     kits::images::classification::swin::v1::SwinV1ContractConfig,
//!     support::testing::cpu_device,
//! };
//!
//! let device = cpu_device();
//!
//! // 32x32 RGB images in 4x4 patches: an 8x8 grid in 4x4 windows, then 4x4
//! // after the merge in 2x2 windows; 10 classes.
//! let policy =
//!     SwinV1ContractConfig::new(32, 16, vec![2, 2], vec![2, 4], 4, 10)
//!         .with_stage_window_sizes(vec![4, 2])
//!         .with_drop_path_rate(0.1);
//!
//! let model: StagedVisionEncoderModel = policy.init(&device);
//! assert_eq!(model.num_classes(), 10);
//! assert_eq!(model.stage_grids(), [[8, 8], [4, 4]]);
//! ```

pub mod pretrained;

pub mod interop;

mod contract;

#[cfg(all(test, feature = "store"))]
mod toy_tests;

pub use contract::*;
