//! # Image models
//!
//! The image-model kits: image-recognition model families for `burn`, after
//! [`timm`](https://github.com/huggingface/pytorch-image-models) (`PyTorch`
//! Image Models) and torchvision. The goal is parity with them: the same
//! architectures, configured with the same knobs, reading the checkpoints
//! they publish, so that a model trained in `PyTorch` loads, runs and
//! fine-tunes in `burn`. The port is incremental, one family at a time,
//! and each family's docs list what of upstream it lacks.
//!
//! The kits are grouped by the task each does:
//!
//! - [`classification`]: image classifiers, the `ResNet` family and Swin
//!   Transformer V2.
//!
//! What the kits share sits in [`support`]: the eval transform that turns a
//! resized image into a model's input.
//!
//! Both families are Stacked configs: a policy config, what the model
//! means, lowers to a structure config, the unrolled tree, which builds
//! the module
//! ([`ModuleInit`](crate::burner::module::ModuleInit#stacked-config)). A
//! conv-net family has no model type of its own: its policy lowers to the
//! generic conv net's, and the model it builds is a
//! [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel).
//! `ResNet` adds the pretrained pathway on top: a name resolves to a row,
//! the row's prefab is the policy, and the kit's hook builds the model and
//! reads the checkpoint into it.
//!
//! `ResNet`'s parity is checked: the `timm-model-validation` crate compares
//! its rows' logits and features with timm's on `ImageNet`'s validation
//! images, and its toy tests compare every stage with timm's own model.
//! Swin's is not checked yet. `examples/resnet_finetune`,
//! `examples/resnet_tiny` and `examples/swin_tiny` train them.

pub mod classification;
pub mod support;
