//! # Vector quantization
//!
//! The decode side of a neural audio codec's residual vector quantizer (RVQ):
//! codes back to the latent they stand for.
//!
//! - [`EuclideanCodebook`]: a codebook kept as its running sums,
//!   `embedding_sum` and `cluster_usage`; a code's vector is their quotient.
//! - [`ResidualVectorQuantizer`]: codebooks in sequence, each coding what the
//!   ones before left; the decode sums their vectors, then projects.
//! - [`SplitResidualVectorQuantizer`]: two of them, the first codebooks (the
//!   "semantic" ones) apart from the rest, each with its own projection, their
//!   latents added.
//!
//! Codes are `[B, K, T]` (`K` codebooks over `T` frames) and latents
//! `[B, D, T]`, channels first, as the codecs lay them out.
//!
//! ## Discussion
//!
//! Encodec's, DAC's, Mimi's and Qwen3-TTS's codecs quantize so; the split
//! quantizer's "semantic" codebooks are Mimi's.

mod codebook;
mod residual;

pub use codebook::*;
pub use residual::*;
