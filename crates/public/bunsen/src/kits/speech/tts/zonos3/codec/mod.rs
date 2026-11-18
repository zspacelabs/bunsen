//! ZONOS3-Mini's codec decoder: [`Zonos3Codec`], its config
//! ([`Zonos3CodecConfig`], from `config.json`'s `codec` section,
//! [`Zonos3CodecJson`]), and the loader of the `-hf` file's `codec.*`
//! tensors ([`weights`]).

mod config;
pub use config::*;

mod model;
pub use model::*;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
