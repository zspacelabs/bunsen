//! Voxtral Realtime component blocks: the audio encoder, its projector, the
//! text model, and the model.
//!
//! [`VoxtralRealtimeConfig`] builds a [`VoxtralRealtime`]: a
//! [`VoxtralAudioEncoder`] (a causal convolution stem, then the shared
//! [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock)),
//! a [`VoxtralProjector`] (the shared
//! [`FeatureStacking`](crate::blocks::signals::FeatureStacking)), and a
//! [`VoxtralTextModel`] (delay-conditioned `CausalDecoderBlock`s).

mod audio_encoder;
mod model;
mod projector;
mod text_model;

pub use audio_encoder::*;
pub use model::*;
pub use projector::*;
pub use text_model::*;
