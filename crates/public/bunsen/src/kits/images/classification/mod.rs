//! # Image classification
//!
//! Image classifiers: an image in, a score for each class out, after
//! [`timm`](https://github.com/huggingface/pytorch-image-models)'s and
//! torchvision's.
//!
//! - [`resnet`]: the `ResNet` family. Six prefabs, `resnet18` to `resnet152`,
//!   and 14 pretrained rows, torchvision's and `timm`'s, loaded by name through
//!   [`default_resnet_factory`](resnet::pretrained::default_resnet_factory).
//!   Its [compatibility](resnet#compatibility) section lists what of `timm`'s
//!   `ResNet` is missing.
//! - [`swin::v2`]: Swin Transformer V2. The model and its configs, with no
//!   prefabs and no pretrained weights yet.

pub mod resnet;
pub mod swin;
