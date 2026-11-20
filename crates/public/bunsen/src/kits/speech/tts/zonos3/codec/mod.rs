//! ZONOS3-Mini's codec decoder: [`Zonos3Codec`] and its config
//! ([`Zonos3CodecConfig`]; a release's, from `config.json`'s `codec`
//! section,
//! [`Zonos3CodecMirror::codec_config`](super::interop::zonos::Zonos3CodecMirror::codec_config)).

mod config;
pub use config::*;

mod model;
pub use model::*;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
