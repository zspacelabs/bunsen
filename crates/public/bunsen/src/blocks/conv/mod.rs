//! # Convolution blocks
//!
//! Conv / norm / activation composites, in 1D and 2D. They are domain
//! neutral: the 2D blocks build the `ResNet` kit, and the 1D blocks build the
//! Whisper audio encoder and Silero VAD.
//!
//! - [`ConvBlock1d`] / [`ConvBlock2d`]: a burn convolution ([`Conv1d`] /
//!   [`Conv2d`]), an optional [`Normalization`], and an optional
//!   [`Activation`], applied in that order.
//!   [`map_forward`](ConvBlock2d::map_forward) runs a caller's function between
//!   the norm and the activation; `ResNet` uses it to place a `DropBlock2d`
//!   there.
//! - [`ConvSeq1d`] / [`ConvSeq2d`]: blocks run in sequence. Building one
//!   [validates](ConvSeq2dMeta::validate) that the sequence is non-empty and
//!   that each block's output channels match the next block's input channels.
//!
//! ## Lifecycle
//!
//! - [`ConvBlock2dConfig`] holds a [`Conv2dConfig`] plus the norm and
//!   activation configs; [`ModuleInit`] builds a [`ConvBlock2d`] from it,
//!   sizing the norm to the conv's output channels.
//! - [`AbstractConvBlock2dConfig`] is the norm-and-activation policy on its
//!   own, before a conv is chosen. A model states it once (for example "batch
//!   norm, then `ReLU`") and calls
//!   [`build_config`](AbstractConvBlock2dConfig::build_config) with each
//!   `Conv2dConfig` to get a matched `ConvBlock2dConfig`.
//! - [`ConvBlock2dMeta`] is implemented by both the config and the module, so
//!   channels, kernel, stride and the predicted output size
//!   ([`try_output_resolution`](ConvBlock2dMeta::try_output_resolution),
//!   computed with [`crate::ops::conv`]) read the same before and after init.
//!   [`ConvSeq2dMeta`] folds them through a sequence.
//!
//! The 1D types follow the same pattern, with lengths in place of
//! resolutions.
//!
//! [`Conv1d`]: burn::nn::conv::Conv1d
//! [`Conv2d`]: burn::nn::conv::Conv2d
//! [`Conv2dConfig`]: burn::nn::conv::Conv2dConfig
//! [`Normalization`]: burn::nn::norm::Normalization
//! [`Activation`]: burn::nn::activation::Activation
//! [`ModuleInit`]: crate::burner::module::ModuleInit

mod conv_block_1d;
mod conv_block_2d;
mod conv_seq_1d;
mod conv_seq_2d;

pub use conv_block_1d::*;
pub use conv_block_2d::*;
pub use conv_seq_1d::*;
pub use conv_seq_2d::*;
