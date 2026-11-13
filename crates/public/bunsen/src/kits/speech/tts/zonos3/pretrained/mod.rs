//! # Pretrained ZONOS3 models
//!
//! [`default_zonos3_factory`] is the index: one row,
//! `catalog:zyphra/zonos3/zonos3-mini`, Zyphra's `transformers` packaging of
//! ZONOS3-Mini (`Zyphra/ZONOS3-Mini-hf`, Apache-2.0 weights), every file
//! pinned to its digest at a fixed revision: `model.safetensors` (the
//! language model in bf16 under `lm.`, the codec decoder in f32 under
//! `codec.`; 2.4 GB), `config.json`, `tokenizer.json` and
//! `voices.safetensors`.
//!
//! [`Zonos3ConstructConfig`] reads them into a
//! [`Zonos3Construct`](super::Zonos3Construct). The kit has no prefab map:
//! the geometry is `config.json`'s.
//!
//! The whole module needs the `store` and `cache` features, both default.

mod construct;
mod providers;

pub use construct::*;
pub use providers::*;
