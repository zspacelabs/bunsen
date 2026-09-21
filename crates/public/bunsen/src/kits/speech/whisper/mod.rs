//! # Whisper
//!
//! `OpenAI`'s Whisper speech-recognition model as a burn module, with what
//! it takes to use one: an index of pretrained checkpoints, a decoder, and
//! a stream driver that turns audio pushed in chunks into timed transcript
//! segments.
//!
//! - [`blocks`]: the model, [`Whisper`], and its configs ([`WhisperApiConfig`]
//!   describes it as upstream's `ModelDimensions` does; [`WhisperGeometry`] is
//!   the part a checkpoint fixes).
//! - [`pretrained`]: names to loaded models. `default_whisper_factory()` is the
//!   index: `openai/base`, `large`, `well-known:openai/tiny.en`,
//!   `hf:openai/whisper-large-v3` for a Hugging Face repo, and
//!   `bundled:openai/base` when the weights are built in. It resolves a name to
//!   a [`WhisperBundle`](driver::WhisperBundle): the model at the precision it
//!   ships in, its token layout, and its vocabulary.
//! - [`decode`]: one window of audio to tokens, with upstream's temperature
//!   fallback.
//! - [`logit_filters`]: what the decode may and may not emit.
//! - [`driver`]: [`WhisperStreamDriver`](driver::WhisperStreamDriver) over a
//!   bundle, and the [context](driver::WhisperStreamContext) it opens on a
//!   stream: samples in, [`TranscriptEvent`](driver::TranscriptEvent)s out.
//!
//! # Example
//!
//! The default pathway, as `examples/whisper-cli` walks it, over any
//! backend: a cache, the factory, a bundle by name, a driver over it, and a
//! context fed a stream. The first run fetches `openai/base` (145 MB,
//! digest-checked) and its vocabulary into the cache; later runs find them
//! there.
//!
//! ```rust,no_run
//! # #[cfg(all(feature = "store_pytorch", feature = "cache"))] {
//! use bunsen::{
//!     data::pretrained::{
//!         PretrainedCache,
//!         PretrainedCacheOptions,
//!     },
//!     errors::BunsenResult,
//!     kits::speech::whisper::{
//!         driver::{
//!             RunningMaxClamp,
//!             StreamClock,
//!             TranscriptSegment,
//!             WhisperStreamDriver,
//!             WhisperStreamDriverConfig,
//!         },
//!         pretrained::default_whisper_factory,
//!     },
//! };
//! use burn::prelude::Backend;
//!
//! /// The named model as a driver on `device`, with the defaults: the
//! /// language detected from the first window, a transcription, greedy,
//! /// and the offline emission policy, which decodes whole windows as
//! /// they fill and commits each once.
//! fn load_driver<B: Backend>(
//!     name: &str,
//!     device: &B::Device,
//! ) -> BunsenResult<WhisperStreamDriver<B>> {
//!     let cache = PretrainedCache::new(PretrainedCacheOptions::default())?;
//!
//!     // A name to a bundle: the checkpoint read through the reader its
//!     // row names, checked against the geometry the name promises, and
//!     // the vocabulary its token layout selects.
//!     let bundle = default_whisper_factory()?
//!         .load_bundle::<B>(name, &cache, device)?;
//!
//!     WhisperStreamDriverConfig::new().init_from_bundle(bundle, device)
//! }
//!
//! /// One stream through the driver: mono samples at the model's rate
//! /// (16 kHz), pushed as a live loop would feed them; the segments as
//! /// they became final.
//! fn transcribe<B: Backend>(
//!     driver: &WhisperStreamDriver<B>,
//!     samples: &[f32],
//! ) -> BunsenResult<Vec<TranscriptSegment>> {
//!     // A bare stream: a clock from zero at the model's rate, and the
//!     // running maximum as each window's dynamic-range reference.
//!     let mut ctx = driver.new_context(
//!         StreamClock::uniform(driver.sample_rate()),
//!         RunningMaxClamp::new(),
//!     )?;
//!
//!     let mut segments = Vec::new();
//!     for block in samples.chunks(driver.sample_rate() / 10) {
//!         for event in ctx.write_read(block)? {
//!             segments.push(event.segment().clone());
//!         }
//!     }
//!     // The end of the stream: whatever is pending is decoded and
//!     // committed.
//!     for event in ctx.end_read()? {
//!         segments.push(event.segment().clone());
//!     }
//!     Ok(segments)
//! }
//!
//! fn run<B: Backend>(
//!     device: &B::Device,
//!     samples: &[f32],
//! ) -> BunsenResult<()> {
//!     let driver = load_driver::<B>("openai/base", device)?;
//!     for segment in transcribe(&driver, samples)? {
//!         println!(
//!             "[{:7.2} --> {:7.2}] {}",
//!             segment.start,
//!             segment.end,
//!             segment.text.as_deref().unwrap_or("")
//!         );
//!     }
//!     Ok(())
//! }
//! # }
//! ```
//!
//! The same driver serves the other ways in. A checkpoint on disk is a
//! given map, with the same hook the factory would attach:
//!
//! ```rust,ignore
//! use bunsen::{data::pretrained::{Deferred, ResourceMap}, kits::speech::whisper::pretrained::{CHECKPOINT, WhisperConstruct}};
//!
//! fn load_from_path<B: Backend>(path: &Path, cache: &PretrainedCache, device: &B::Device) -> BunsenResult<Arc<WhisperBundle<B>>> {
//!     Deferred::<WhisperConstruct>::from_map(ResourceMap::given("mine", CHECKPOINT, path))?
//!         .load_bundle::<B>(cache, device)
//! }
//! ```
//!
//! A Hugging Face repo is a ref of the `hf` provider, resolved through the
//! cache (the hub's file listing is fetched once and kept), and a
//! vocabulary of one's own is an overlay on the resolved model:
//!
//! ```rust,ignore
//! let factory = default_whisper_factory()?;
//! let bundle = factory.load_bundle::<B>("hf:openai/whisper-large-v3", &cache, device)?;
//!
//! let model = factory
//!     .resolve("openai/tiny.en", &cache)?
//!     .with_overlay(ResourceMap::given("--vocab", VOCABULARY, "/vocab/gpt2.tiktoken"))?;
//! let bundle = model.load_bundle::<B>(&cache, device)?;
//! ```
//!
//! The real-time emission presets decode on speech endpoints and want a
//! voice-activity model, which the Silero kit's factory provides the same
//! way; `examples/whisper-cli` attaches it with
//! [`with_vad`](driver::WhisperStreamDriver::with_vad).

pub mod blocks;
pub mod decode;
pub mod driver;
pub mod logit_filters;
pub mod pretrained;

#[doc(inline)]
pub use blocks::{
    Whisper,
    WhisperApiConfig,
    WhisperGeometry,
    WhisperMeta,
    WhisperStructuralConfig,
};
#[doc(inline)]
pub use decode::*;
