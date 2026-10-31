//! # Voxtral Realtime
//!
//! Mistral's Voxtral Realtime speech recognizer
//! (`mistralai/Voxtral-Mini-4B-Realtime-2602`): a causal audio encoder whose
//! frames are added, one group per 80 ms, to the token embeddings of a
//! Mistral text model that writes the transcript a fixed delay behind the
//! audio. Every part is causal, so the model streams natively: each 80 ms of
//! audio is one decoder step.
//!
//! - [`blocks`]: the model, [`VoxtralRealtime`], built from a
//!   [`VoxtralRealtimeConfig`]: a
//!   [`VoxtralAudioEncoder`](blocks::VoxtralAudioEncoder) and a
//!   [`VoxtralTextModel`](blocks::VoxtralTextModel) on the shared
//!   [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock),
//!   and a [`VoxtralProjector`](blocks::VoxtralProjector) between them.
//! - [`VoxtralFrontEnd`]: the feature extractor, Whisper's log-mels under a
//!   fixed level.
//! - [`VoxtralTokenLayout`]: the stream's special tokens, its delay (480 ms by
//!   default; any multiple of 80 ms up to 1200, or 2400), the padding around an
//!   utterance, and the prompt.
//! - [`VoxtralGreedyDecoder`]: greedy decoding through a KV cache, one id per
//!   position, each position's audio added to its token
//!   ([`VoxtralDecodeStep`]).
//! - [`VoxtralTranscript`]: a decode's ids as timed [`VoxtralWord`]s and text.
//! - [`VoxtralBundle`]: a model with its front end, vocabulary
//!   ([`TekkenVocabulary`](crate::kits::tokens::TekkenVocabulary)), layout and
//!   decoder; [`transcribe`](VoxtralBundle::transcribe) one utterance.
//! - [`weights`] (feature `store`): loading tensors named as the published
//!   `model.safetensors` names them, and reading its `config.json`.

pub mod blocks;

mod bundle;
mod decode;
mod front_end;
mod layout;
mod transcript;
pub use bundle::*;
pub use decode::*;
pub use front_end::*;
pub use layout::*;
pub use transcript::*;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(test, feature = "store_safetensors"))]
mod toy_tests;

#[doc(inline)]
pub use blocks::{
    VoxtralRealtime,
    VoxtralRealtimeConfig,
};
