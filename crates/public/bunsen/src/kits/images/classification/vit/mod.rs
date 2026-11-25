//! # `ViT`
//!
//! The Vision Transformer family of image models
//! ([arXiv:2010.11929](https://arxiv.org/abs/2010.11929)), with the
//! checkpoints `timm` publishes. A `ViT` cuts an image into square patches,
//! embeds each as a token, puts a class token (and register tokens) before
//! them, adds a learnt position embedding, runs a transformer encoder over
//! the sequence, and pools it for a classifier head. The patch size, the
//! width, the depth, the heads and the image size name the variant:
//! `vit_tiny_patch16_224` to `vit_huge_patch14_224`, the MIIL
//! `vit_base_patch16_224_miil`, and the CLIP image towers
//! (`vit_base_patch32_clip_224`, …), which normalise the tokens before the
//! first layer.
//!
//! [`pretrained`] turns names into loaded models.
//! [`ViTContractConfig`]'s prefabs name the 25 geometries, by timm's
//! architecture names, and
//! [`default_vit_factory`](pretrained::default_vit_factory) is the index of
//! 96 timm hub repos (`timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`), and
//! loads any other of the prefabs' architectures as `hf:timm/<repo>`.
//!
//! # The model
//!
//! A `ViT` is the generic vision encoder,
//! [`VisionEncoderModel`](crate::blocks::transformers::models::vision_encoder::VisionEncoderModel):
//! the kit has no model type of its own. [`ViTContractConfig`] is the policy
//! in timm's terms (the image and patch sizes, the width, depth and heads,
//! the MLP ratio, the `qkv` bias, the class and register tokens, the position
//! embedding's reach, CLIP's pre-norm, the head's pool and norm, layer scale,
//! the norms' epsilon, the activation) and lowers to the generic structure;
//! `init` builds the model through it.
//!
//! The model's parameter paths are the vision encoder's own; [`interop`]
//! maps timm's names onto them
//! ([`vit_rekey`](interop::timm::vit_rekey)).
//!
//! # Compatibility
//!
//! The kit aims for parity with `timm`'s `vision_transformer.py`. Missing
//! today:
//!
//! * the attention pool (`global_pool='map'`, `SigLIP`'s) and `prr`
//! * running at another size than the position embedding's
//!   (`resample_abs_pos_embed`, `dynamic_img_size`)
//! * the hybrid stems (`vit_base_r50_s16_384`) and the other block kinds
//!   (`ParallelScalingBlock`, `ResPostBlock`)
//! * QK-norm, `SwiGLU` MLPs and RMS norms (the post-2021 configs)

pub mod pretrained;

pub mod interop;

mod contract;

#[cfg(all(test, feature = "store"))]
mod toy_tests;

pub use contract::*;
