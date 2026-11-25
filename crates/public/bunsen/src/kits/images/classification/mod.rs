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
//!   33 prefabs (`resnet18` to `resnet152`, `resnet50_gn`, the C, D, S and T
//!   variants, the `ResNeXt`s and the wide `ResNet`s), and 103 pretrained rows,
//!   torchvision's and `timm`'s hub repos, loaded by name through
//!   [`default_resnet_factory`](resnet::pretrained::default_resnet_factory).
//!   Its [compatibility](resnet#compatibility) section lists what of `timm`'s
//!   `ResNet` is missing.
//! - [`swin::v2`]: Swin Transformer V2. The model and its configs, with no
//!   prefabs and no pretrained weights yet.

pub mod driver;
pub mod resnet;
pub mod swin;
