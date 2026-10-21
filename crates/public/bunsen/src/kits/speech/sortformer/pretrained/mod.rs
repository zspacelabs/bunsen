//! # Pretrained Sortformer models
//!
//! From a model's name to a loaded
//! [`SortformerBundle`](super::SortformerBundle), through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)). The
//! whole module needs the `store_safetensors` and `cache` features.
//!
//! # The index
//!
//! [`default_sortformer_factory`] is the index a caller holds, over
//! [`WELL_KNOWN_TABLE`] and Hugging Face repos:
//!
//! - `nvidia/nemotron-3-diarization` (`nemotron-3-diarization`): NVIDIA's
//!   Nemotron-3-Diarization, 8 speakers, 10 ms frames, `transformers`' 397 MB
//!   `model.safetensors`. The default.
//! - `nvidia/diar_streaming_sortformer_4spk-v2` (`sortformer-v2`) and
//!   `nvidia/diar_streaming_sortformer_4spk-v2.1` (`sortformer-v2.1`): the
//!   streaming Sortformer v2 and v2.1, 4 speakers, 80 ms frames, each a 471 MB
//!   `.nemo` whose config gives the model and its front end. Needs the `nemo`
//!   feature.
//! - `hf:org/repo`: a repo in `transformers`' `Nemotron3Diarization` layout,
//!   whose `config.json` gives the geometry; it is read before the checkpoint
//!   is fetched, so a repo of another model is refused first.
//!
//! A bundle's [`family`](super::SortformerBundle::family) names the
//! streaming presets it runs under.
//!
//! Every well-known file is pinned to its digest and fetched from a fixed
//! revision of its Hugging Face repo.

mod construct;
mod providers;

pub use construct::*;
pub use providers::*;
