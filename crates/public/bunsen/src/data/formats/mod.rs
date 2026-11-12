//! # Formats: reading the files a build is made of
//!
//! A kit's hook reads its checkpoint's own format with the readers here: the
//! safetensors checkpoint and header ([`SafetensorsCheckpoint`]), the
//! `PyTorch` reader ([`read_pytorch_tensors`]), MLX's affine-quantized
//! tensors ([`mlx`]), `FermionResearch`'s Phonon formats ([`fermion`]), and
//! [`apply_named_tensors`], which loads any of them into a module under a
//! kit's names, or [`apply_rekeyed_tensors`], under a
//! [`Rekey`](crate::burner::module::Rekey) whose targets may also change a
//! tensor's data on the way. A [`LoadPolicy`] is the load call's choice of the
//! float dtype the parameters are placed in.
//!
//! The readers need the `store` feature, and [`LoadPolicy`] the `cache`
//! feature; [`mlx`] is plain tensor code, always here.

#[cfg(feature = "store")]
mod apply_named;
#[cfg(feature = "store")]
pub mod fermion;
#[cfg(feature = "cache")]
mod load_policy;
pub mod mlx;
#[cfg(feature = "store")]
mod pytorch;
#[cfg(feature = "store")]
mod safetensors;

#[cfg(feature = "store")]
pub use apply_named::*;
#[cfg(feature = "cache")]
pub use load_policy::*;
#[cfg(feature = "store")]
pub use pytorch::*;
#[cfg(feature = "store")]
pub use safetensors::*;
