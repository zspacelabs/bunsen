//! # Qwen3-ASR's interop: its checkpoints' ecosystems
//!
//! - [`transformers`]: a `Qwen3ASRForConditionalGeneration` checkpoint, its
//!   `config.json` and state-dict names.
//! - [`phonon`]: `FermionResearch`'s Phonon-1 archive of one.

pub mod phonon;
pub mod transformers;
