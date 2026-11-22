//! ZONOS3-Mini's speaker encoder: [`Zonos3SpeakerEncoder`], a clip to the
//! raw speaker embedding the prompt's speaker row projects, and its config
//! ([`Zonos3SpeakerEncoderConfig`]; a release's, from `config.json`'s
//! `speaker_encoder` section,
//! [`Zonos3SpeakerEncoderMirror::speaker_config`](super::interop::zonos::Zonos3SpeakerEncoderMirror::speaker_config)).

mod config;
pub use config::*;

mod model;
pub use model::*;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
