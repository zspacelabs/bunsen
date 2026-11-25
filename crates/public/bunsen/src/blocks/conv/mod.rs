//! # Convolution blocks
//!
//! Conv / norm / activation composites, in 1D and 2D. They are domain
//! neutral: the 2D blocks build the generic conv net, and the 1D blocks build
//! the Whisper audio encoder and Silero VAD.
//!
//! - [`ConvBlock1d`] / [`ConvBlock2d`]: a burn convolution ([`Conv1d`] /
//!   [`Conv2d`]), an optional [`BunsenNormalization`], and an optional
//!   [`BunsenActivation`], applied in that order.
//!   [`map_forward`](ConvBlock2d::map_forward) runs a caller's function between
//!   the norm and the activation; the conv net's residual unit uses it to place
//!   a `DropBlock2d` there.
//! - [`CausalConv1d`] / [`CausalConvTranspose1d`]: 1-d convolutions over each
//!   frame's past, channels first: padded before (zeros or the edge
//!   [replicated](CausalPadding)) and convolved unpadded; and an upsampler
//!   whose kernel's overlap past the last frame is trimmed. Encodec's, DAC's,
//!   Mimi's and Qwen3-TTS's codec's. Both also run as streams, in chunks of any
//!   size through a
//!   [`CausalConvStreamContext`](crate::ops::conv::CausalConvStreamContext) or
//!   a [`CausalConvTransposeStreamContext`](crate::ops::conv::CausalConvTransposeStreamContext),
//!   their outputs the whole forward's.
//! - [`CausalConvNext1d`]: a causal 1-d `ConvNeXt` block, channels last: a
//!   depthwise convolution over each channel's past, a norm, a pointwise
//!   expansion and projection, an optional per-channel layer scale, residual;
//!   it carries its convolution's state across chunks. ZONOS3's codec's, and
//!   (with the layer scale) Qwen3-TTS's.
//! - [`ConvSeq1d`] / [`ConvSeq2d`]: blocks run in sequence. Building one
//!   [validates](ConvSeq2dMeta::validate) that the sequence is non-empty and
//!   that each block's output channels match the next block's input channels.
//!
//! ## Lifecycle
//!
//! - [`ConvBlock2dConfig`] holds a [`Conv2dConfig`] plus the norm and
//!   activation configs; [`ModuleInit`] builds a [`ConvBlock2d`] from it,
//!   sizing the norm to the conv's output channels.
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
//! [`BunsenNormalization`]: crate::blocks::norm::BunsenNormalization
//! [`BunsenActivation`]: crate::blocks::activation::BunsenActivation
//! [`ModuleInit`]: crate::burner::module::ModuleInit

mod causal_conv_1d;
mod causal_convnext;
mod conv_block_1d;
mod conv_block_2d;
mod conv_seq_1d;
mod conv_seq_2d;

pub use causal_conv_1d::*;
pub use causal_convnext::*;
pub use conv_block_1d::*;
pub use conv_block_2d::*;
pub use conv_seq_1d::*;
pub use conv_seq_2d::*;
