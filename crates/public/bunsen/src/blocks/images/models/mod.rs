//! # Image models
//!
//! Whole image models assembled from the blocks, each the one type a family
//! of checkpoints builds: [`conv_net`], the generic residual conv net
//! ([`ConvNetModel`](conv_net::ConvNetModel)), which `ResNet` and its kin
//! are. A family's kit holds no model of its own: its policy lowers to the
//! generic one, and its interop maps its checkpoints' names onto the generic
//! model's paths.

pub mod conv_net;
