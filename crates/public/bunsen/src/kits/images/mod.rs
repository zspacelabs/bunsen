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
//! - [`resnet`]: the `ResNet` family. Six prefabs, `resnet18` to `resnet152`,
//!   and 14 pretrained rows, torchvision's and `timm`'s, loaded by name through
//!   [`default_resnet_factory`](resnet::default_resnet_factory). Its
//!   [compatibility](resnet#compatibility) section lists what of `timm`'s
//!   `ResNet` is missing.
//! - [`swin::v2`]: Swin Transformer V2. The model and its configs, with no
//!   prefabs and no pretrained weights yet.
//!
//! Both families are Stacked configs: a policy config, what the model
//! means, lowers to a structure config, the unrolled tree, which builds
//! the module
//! ([`ModuleInit`](crate::burner::module::ModuleInit#stacked-config)). `ResNet`
//! adds the pretrained pathway on top: a name resolves to a row,
//! the row's prefab is the policy, and the kit's hook builds the model and
//! reads the checkpoint into it.
//!
//! Parity is a goal, not yet a checked property: no validation crate
//! compares these models' outputs with `PyTorch`'s.
//! `examples/resnet_finetune`, `examples/resnet_tiny` and
//! `examples/swin_tiny` train them.

pub mod resnet;
pub mod swin;
