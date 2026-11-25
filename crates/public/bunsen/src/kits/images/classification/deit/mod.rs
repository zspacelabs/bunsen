//! # `DeiT`
//!
//! The Data-efficient Image Transformers
//! ([arXiv:2012.12877](https://arxiv.org/abs/2012.12877)) and `DeiT` III
//! ([arXiv:2204.07118](https://arxiv.org/abs/2204.07118)), with the
//! checkpoints `timm` publishes. A `DeiT` is a `ViT` trained on `ImageNet`
//! alone: the plain ones are timm's `ViT`s of their sizes; the distilled
//! ones put a distillation token after the class token and a second
//! classifier on it, the two classifiers' logits averaged; `DeiT` III adds
//! the position embedding over the patches alone and layer scale.
//!
//! [`pretrained`] turns names into loaded models. [`DeiTContractConfig`]'s
//! prefabs name the 16 geometries, by timm's architecture names
//! (`deit_tiny_patch16_224` to `deit_base_distilled_patch16_384`,
//! `deit3_small_patch16_224` to `deit3_huge_patch14_224`), and
//! [`default_deit_factory`](pretrained::default_deit_factory) loads timm's
//! hub repos of them (`timm/deit_base_distilled_patch16_224.fb_in1k`).
//!
//! # The model
//!
//! A `DeiT` is the generic vision encoder,
//! [`VisionEncoderModel`](crate::blocks::transformers::models::vision_encoder::VisionEncoderModel):
//! the kit has no model type of its own. [`DeiTContractConfig`] is the
//! policy: the `ViT` it is
//! ([`ViTContractConfig`](crate::kits::images::classification::vit::ViTContractConfig)),
//! and whether it is distilled. The checkpoints are timm's `ViT` state
//! dicts, which the vit kit's interop reads ([`interop`]).

pub mod pretrained;

pub mod interop;

mod contract;

#[cfg(all(test, feature = "store"))]
mod toy_tests;

pub use contract::*;
