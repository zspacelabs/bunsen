//! # Pretrained Sortformer models
//!
//! From a model's name to a loaded
//! [`SortformerConstruct`](super::SortformerConstruct), through the data
//! layer's pretrained machinery
//! ([`data::pretrained`](crate::data::pretrained)). The whole module needs the
//! `store` and `cache` features.
//!
//! # The index
//!
//! [`default_sortformer_factory`] is the index a caller holds, over
//! the built-in catalog's builds of [`SORTFORMER_CATALOG_CONTRACT`] and
//! Hugging Face repos:
//!
//! - `nvidia/nemotron-3-diarization` (`nemotron-3-diarization`): NVIDIA's
//!   Nemotron-3-Diarization, 8 speakers, 10 ms frames, `transformers`' 397 MB
//!   `model.safetensors`. The default.
//! - `nvidia/diar_streaming_sortformer_4spk-v2` (`sortformer-v2`) and
//!   `nvidia/diar_streaming_sortformer_4spk-v2.1` (`sortformer-v2.1`): the
//!   streaming Sortformer v2 and v2.1, 4 speakers, 80 ms frames, each a 471 MB
//!   `.nemo` whose config gives the model and its front end. Needs the `store`
//!   feature.
//! - `hf:org/repo`: a repo in `transformers`' `Nemotron3Diarization` layout,
//!   whose `config.json` gives the geometry, or one holding a Sortformer v2
//!   `.nemo` alone (`hf:nvidia/diar_streaming_sortformer_4spk-v2.1`), whose own
//!   config does. Either config is read before the checkpoint is fetched (a
//!   `.nemo`'s from the archive's head, by a range request), so a repo of
//!   another model is refused first.
//!
//! Every model takes its family's presets (the catalog's `families`, which
//! a row inherits and an `hf:` repo is given by its
//! [`family`](super::SortformerConstruct::family)): a `latency` axis, the model
//! card's streaming latencies (`ultra-low`, `very-low` for Nemotron, `low`,
//! the default, `high` for v2, `offline`), and a `postprocess` axis, `NeMo`'s
//! turn post-processing (`none`, the default, `dihard3`, `callhome`). So
//! `'nvidia/nemotron-3-diarization[very-low]'` runs 0.64 s behind the audio,
//! and the construct's [`streaming`](super::SortformerConstruct::streaming)
//! and [`turns`](super::SortformerConstruct::turns) are what the presets
//! resolved to.
//!
//! Every catalog file is pinned to its digest and fetched from a fixed
//! revision of its Hugging Face repo.

mod construct;
mod providers;

pub use construct::*;
pub use providers::*;
