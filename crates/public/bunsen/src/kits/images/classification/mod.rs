//! # Image classification
//!
//! Image classifiers: an image in, a score for each class out, after
//! [`timm`](https://github.com/huggingface/pytorch-image-models)'s and
//! torchvision's.
//!
//! The task's model-generic part is [`driver`]: the
//! [`ImageClassifier`](driver::ImageClassifier) seam every classifier
//! implements, and the top classes of its logits. The kits:
//!
//! - [`resnet`]: the `ResNet` family, built as the generic conv net
//!   ([`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel)).
//!   76 prefabs (`resnet18` to `resnet152`, `resnet50_gn`, the C, D, S and T
//!   variants, the `ResNeXt`s, the wide `ResNet`s, the squeeze-excite and ECA
//!   nets, the anti-aliased nets, and the `ResNet-RS`s), and 145 pretrained
//!   rows, torchvision's and `timm`'s hub repos, loaded by name through
//!   [`default_resnet_factory`](resnet::pretrained::default_resnet_factory).
//!   Its [compatibility](resnet#compatibility) section lists what of `timm`'s
//!   `ResNet` is missing.
//! - [`vit`]: the Vision Transformer, built as the generic vision encoder
//!   ([`VisionEncoderModel`](crate::blocks::transformers::models::vision_encoder::VisionEncoderModel)).
//!   117 prefabs, generated from timm's `vision_transformer.py` (from
//!   `vit_tiny_patch16_224` and the CLIP image towers to the `SigLIP`,
//!   `DINOv2`, sbb and C-RADIO archs), and 269 pretrained rows, timm's hub
//!   repos, loaded by name through
//!   [`default_vit_factory`](vit::pretrained::default_vit_factory).
//! - [`deit`]: `DeiT` and `DeiT` III, `ViT`s built the same way, the distilled
//!   ones with a distillation token and a second classifier. 16 prefabs, loaded
//!   from timm's hub repos through
//!   [`default_deit_factory`](deit::pretrained::default_deit_factory).
//! - [`swin`]: the Swin Transformer, V1 (with the Swin-S3 geometries) and V2,
//!   built as the generic staged vision encoder, with timm's hub repos.

pub mod deit;
pub mod driver;
pub mod resnet;
pub mod swin;
pub mod vit;
