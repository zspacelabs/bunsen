//! # `RegNet`
//!
//! The `RegNet` family of image models (`RegNetX` and `RegNetY`,
//! [arXiv:2003.13678](https://arxiv.org/abs/2003.13678); `RegNetZ`,
//! [arXiv:2103.06877](https://arxiv.org/abs/2103.06877); and timm's
//! pre-activation `RegNetV`), with the checkpoints `timm` publishes: pycls's,
//! torchvision's v2, timm's own, and the large `SEER` and `SWAG` towers. A
//! `RegNet` is a `3x3` stem and four stages of bottleneck units whose `3x3`
//! mixer is grouped; `RegNetY` adds a squeeze-excite after the mixer,
//! `RegNetV` pre-activates its units, and `RegNetZ` inverts its bottleneck,
//! ends its units linear and has no shortcut convs. Its widths are not listed
//! but generated from four numbers (`w0`, `wa`, `wm`, `depth`), a linear rule
//! quantized to a geometric one.
//!
//! [`pretrained`] turns names into loaded models.
//! [`RegNetContractConfig`]'s prefabs name the geometries, by timm's
//! architecture names, generated from timm's own configs
//! (`tools/gen_regnet_prefabs.py`), and
//! [`default_regnet_factory`](pretrained::default_regnet_factory) is the
//! index of timm's hub repos (`timm/regnety_032.ra_in1k`), and loads any
//! other of the prefabs' architectures as `hf:timm/<repo>`.
//!
//! # The model
//!
//! A `RegNet` is the generic conv net,
//! [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel):
//! the kit has no model type of its own. [`RegNetContractConfig`] is the
//! policy in timm's terms (`RegNetCfg`: the width rule, the group size, the
//! bottleneck ratio, the squeeze-excite, the shortcut, pre-activation, the
//! linear output, the final conv, the activation) and lowers to the
//! conv net's; its stages are timm's to the channel, computed in float32 as
//! torch computes them ([`RegNetContractConfig::try_stages`]).
//!
//! The model's parameter paths are the conv net's own; [`interop`] maps
//! timm's names onto them
//! ([`regnet_rekey`](interop::timm::regnet_rekey)).
//!
//! # Compatibility
//!
//! The kit aims for parity with `timm`'s `regnet.py`, and builds every one
//! of its architectures with hub weights. Missing today:
//!
//! * group norm (`regnety_040_sgn`, which has no hub weights)

pub mod pretrained;

pub mod interop;

mod contract;

#[cfg(all(test, feature = "store"))]
mod toy_tests;

pub use contract::*;
