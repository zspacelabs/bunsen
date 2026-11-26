//! # `ViT`
//!
//! The Vision Transformer family of image models
//! ([arXiv:2010.11929](https://arxiv.org/abs/2010.11929)), with the
//! checkpoints `timm` publishes. A `ViT` cuts an image into square patches,
//! embeds each as a token, puts a class token (and register tokens) before
//! them, adds a learnt position embedding, runs a transformer encoder over
//! the sequence, and pools it for a classifier head. The patch size, the
//! width, the depth, the heads and the image size name the variant
//! (`vit_base_patch16_224`); later weight sets keep the tree and turn its
//! knobs: the CLIP image towers normalise the tokens before the first layer,
//! `DINOv2` adds register tokens and layer scale, `SigLIP`'s `gap` towers drop
//! the class token for the mean, and its `map` towers pool by a learned query's
//! attention; C-RADIO drops the final norm; `AIMv2` takes a `SwiGLU` MLP, RMS
//! norms and no biases; `BEiT3` normalizes inside its attention and its MLP.
//!
//! [`pretrained`] turns names into loaded models.
//! [`ViTContractConfig`]'s prefabs name the 134 geometries, by timm's
//! architecture names, generated from timm's own entrypoints
//! (`tools/gen_vit_prefabs.py`), and
//! [`default_vit_factory`](pretrained::default_vit_factory) is the index of
//! 294 timm hub repos (`timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`), and
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
//! the blocks' norm placement, the final norm, the MLP's and the norms'
//! kinds, the projections' biases, the norms' epsilon, the activation) and
//! lowers to the generic structure;
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
//! * the `prr` pool
//! * running at another size than the position embedding's
//!   (`resample_abs_pos_embed`, `dynamic_img_size`)
//! * the hybrid stems (`vit_base_r50_s16_384`), the parallel blocks
//!   (`ParallelScalingBlock`, `DiffParallelScalingBlock`) and differential
//!   attention
//! * QK-norm

pub mod pretrained;

pub mod interop;

mod contract;

#[cfg(all(test, feature = "store"))]
mod toy_tests;

pub use contract::*;
