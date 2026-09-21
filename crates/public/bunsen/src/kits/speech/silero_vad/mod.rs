//! # Silero VAD
//!
//! The Silero voice-activity detector as a burn module: a small recurrent
//! model that gives, for each chunk of audio (512 samples at 16 kHz, 256
//! at 8 kHz), the probability that it holds speech, carrying its state
//! from one chunk to the next.
//!
//! - [`blocks`]: the model, [`SileroVad`], for one sample rate; the two-rate
//!   [`SileroVadCollection`] a checkpoint holds; and the [`SileroVadContext`]
//!   that carries what a stream needs between chunks (the tail of the last
//!   chunk and the recurrent state).
//! - [`pretrained`] (feature `store`): how the weights arrive.
//!   `default_silero_factory()` is the index: with the `silero-weights`
//!   feature, one row, `bundled:silero/vad`, the burnpack linked into the
//!   binary and written into the cache under its digest on first use. The
//!   loaders in `pretrained::load` read the same bytes with no cache at all,
//!   for a binary that wants nothing else.
//!
//! The ONNX reference this was transliterated from, and the cross-checks
//! against it, live in the `silero-model-validation` crate. There is
//! *some* bug in the burn CUDA backend, which we see by model divergence
//! from the golden tests on that backend alone.
//!
//! # Example
//!
//! The default pathway, over any backend: a cache, the factory, the
//! collection by name, the branch for the stream's rate, and a context fed
//! chunk by chunk.
//!
//! ```rust,no_run
//! # #[cfg(feature = "store")] {
//! use std::sync::Arc;
//!
//! use bunsen::{
//!     data::pretrained::{
//!         PretrainedCache,
//!         PretrainedCacheOptions,
//!     },
//!     errors::BunsenResult,
//!     kits::speech::silero_vad::{
//!         SileroVad,
//!         SileroVadCollection,
//!         SileroVadContextConfig,
//!         SileroVadMeta,
//!         pretrained::default_silero_factory,
//!     },
//! };
//! use burn::{
//!     prelude::Backend,
//!     tensor::{
//!         ElementConversion,
//!         Tensor,
//!     },
//! };
//!
//! /// The bundled model, both rates, on `device`: from the cache, or
//! /// written into it from the binary on first use.
//! fn load_vad<B: Backend>(
//!     device: &B::Device
//! ) -> BunsenResult<Arc<SileroVadCollection<B>>> {
//!     let cache = PretrainedCache::new(PretrainedCacheOptions::default())?;
//!     Ok(default_silero_factory()?
//!         .load::<B>("bundled:silero/vad", &cache, device)?
//!         .handle)
//! }
//!
//! /// One stream through one branch: mono samples at the branch's rate,
//! /// a chunk at a time, each answered with the probability it holds
//! /// speech.
//! fn speech_probabilities<B: Backend>(
//!     vad: &SileroVad<B>,
//!     samples: &[f32],
//!     device: &B::Device,
//! ) -> Vec<f32> {
//!     // One stream at the branch's rate, and the 64-sample tail the model
//!     // looks back over.
//!     let mut ctx =
//!         SileroVadContextConfig::new(vad.sample_rate()).init(vad, device);
//!     let mut probabilities = Vec::new();
//!     for chunk in samples.chunks_exact(vad.chunk_size()) {
//!         let chunk: Tensor<B, 2> =
//!             Tensor::<B, 1>::from_floats(chunk, device).unsqueeze();
//!         let (probability, next) = vad.context_forward(chunk, ctx);
//!         ctx = next;
//!         probabilities.push(probability.into_scalar().elem::<f32>());
//!     }
//!     probabilities
//! }
//!
//! fn run<B: Backend>(
//!     device: &B::Device,
//!     samples: &[f32],
//! ) -> BunsenResult<()> {
//!     let vad = load_vad::<B>(device)?;
//!     let branch = vad.expect_branch(16000);
//!     for (i, p) in speech_probabilities(branch, samples, device)
//!         .iter()
//!         .enumerate()
//!     {
//!         let at = i * branch.chunk_size();
//!         println!("{at:>8}: {p:.3}");
//!     }
//!     Ok(())
//! }
//! # }
//! ```
//!
//! The Whisper stream driver's real-time emission presets take the model
//! the same way, and turn its probabilities into speech regions through a
//! [`VoiceActivityFilterConfig`](crate::kits::speech::whisper::driver::VoiceActivityFilterConfig):
//!
//! ```rust,ignore
//! let vad = load_vad::<B>(device)?;
//! let driver = driver.with_vad(vad.expect_branch(16000).clone(), Default::default())?;
//! ```

#[cfg(feature = "store")]
pub mod pretrained;

pub mod blocks;

#[doc(inline)]
pub use blocks::*;
