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
//! - [`classification`]: image classifiers, ten kits: the conv nets (`ResNet`,
//!   `ResNetV2`, `RegNet`, `Res2Net`, `ResNeSt`, `SKNet`), the vision
//!   transformers (`ViT`, `DeiT`) and the staged ones (Swin V1 and V2), each
//!   with `timm`'s checkpoints by name.
//!
//! What the kits share sits in [`support`]: the eval transform that turns a
//! resized image into a model's input.
//!
//! Every family is a Stacked config: a policy config, what the model
//! means, lowers to a structure config, the unrolled tree, which builds
//! the module
//! ([`ModuleInit`](crate::burner::module::ModuleInit#stacked-config)). A
//! family has no model type of its own: its policy lowers to a generic
//! model's, the
//! [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel),
//! the
//! [`VisionEncoderModel`](crate::blocks::transformers::models::vision_encoder::VisionEncoderModel)
//! or the
//! [`StagedVisionEncoderModel`](crate::blocks::transformers::models::staged_vision_encoder::StagedVisionEncoderModel).
//! Each kit adds the pretrained pathway on top: a name resolves to a row,
//! the row's prefab is the policy, and the kit's hook builds the model and
//! reads the checkpoint into it.
//!
//! Parity is checked: the `timm-model-validation` crate compares every kit's
//! rows' logits and features with timm's on `ImageNet`'s validation images,
//! and the kits' toy tests compare their models with timm's own on toy
//! fixtures.
//! `examples/resnet_finetune`, `examples/resnet_tiny` and `examples/swin_tiny`
//! train them.

pub mod classification;
pub mod support;
