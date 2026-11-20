//! # Qwen3-ASR as `FermionResearch`'s Phonon-1 archive packs it
//!
//! A Phonon-1 archive holds a Qwen3-ASR's weights, packed (MLX affine and
//! quint5 codes inside byte-plane-split safetensors members, under a
//! digest-checked manifest), and its vocabulary. The container is
//! [`data::formats::fermion`](crate::data::formats::fermion); this module
//! reads the manifest's mirrors and expands the packing to plain f32 tensors
//! under `transformers`' names, which
//! [`interop::transformers`](super::transformers)' loader maps onto the
//! model.
//!
//! - **Loading** (feature `store`): `read_phonon1_archive`; with `cache`,
//!   `load_qwen3_asr`, a resource set's archive.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;
