//! # `FermionResearch`'s model formats
//!
//! The Phonon speech models ship in formats of their own. Every one is read
//! here to plain f32 values, exactly as the format's reference reader
//! expands it, and kits build on the results:
//!
//! - `PhononArchive` (feature `fermion`): the release archive of every Phonon
//!   model, a zstd tar with a digest-checked manifest whose safetensors members
//!   may be stored byte-plane split.
//! - [`read_five_value_container`]: the `fermion-five-value-parakeet-v1`
//!   container inside Phonon-2's archive, the weights of
//!   [`kits::speech::parakeet`](crate::kits::speech::parakeet).
//! - [`decode_quint5`]: the `ten-base5-per-24bit-v1` codes Phonon-1 stores its
//!   text-decoder linears in, inside safetensors members whose other records
//!   are bf16 or MLX affine-quantized
//!   ([`dequantize_mlx_affine`](super::mlx::dequantize_mlx_affine)).
//!
//! The fixtures in `testdata/fermion/` were written by
//! `tools/gen_fermion_fixtures.py` with the formats' own readers and writer.

#[cfg(feature = "fermion")]
mod archive;
mod five_value;
mod quint5;

#[cfg(feature = "fermion")]
pub use archive::*;
pub use five_value::*;
pub use quint5::*;
