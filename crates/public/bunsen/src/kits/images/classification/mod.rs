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
//!   25 prefabs (`vit_tiny_patch16_224` to `vit_huge_patch14_224`, the MIIL
//!   `ViT` and the CLIP image towers), and 99 pretrained rows, timm's hub
//!   repos, loaded by name through
//!   [`default_vit_factory`](vit::pretrained::default_vit_factory).
//! - [`swin::v2`]: Swin Transformer V2. The model and its configs, with no
//!   prefabs and no pretrained weights yet.

pub mod driver;
pub mod resnet;
pub mod swin;
pub mod vit;
