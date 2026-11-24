//! # ZONOS3-Mini's blocks
//!
//! The Modules ZONOS3-Mini is built from, each with its config (a release's,
//! from its `config.json`, through [`interop::zonos`](super::interop::zonos)).
//!
//! - **Language model.** [`Zonos3Lm`] and [`Zonos3LmConfig`] (`config.json`'s
//!   `lm` section,
//!   [`Zonos3LmMirror::lm_config`](super::interop::zonos::Zonos3LmMirror::lm_config)).
//! - **Codec decoder.** [`Zonos3Codec`] and [`Zonos3CodecConfig`] (the `codec`
//!   section,
//!   [`Zonos3CodecMirror::codec_config`](super::interop::zonos::Zonos3CodecMirror::codec_config)):
//!   its codebooks' embeddings ([`Zonos3CodebookDecoder`]), its trunk stages
//!   ([`Zonos3CodecStageConfig`], each a
//!   [`ProjectedDecoderStack`](crate::blocks::transformers::decoders::projected::ProjectedDecoderStack)), and its head of inverse STFTs.
//!   It also runs as a stream ([`Zonos3CodecStreamContext`]): its causal
//!   trunk and heads carry their state, and its tail (the iSTFTs, the
//!   centred mixer, the resampler) returns each sample once it is final, so
//!   the stream lags by a fixed number of samples and, joined with its
//!   finish, is the whole decode.
//! - **Speaker encoder.** [`Zonos3SpeakerEncoder`] and
//!   [`Zonos3SpeakerEncoderConfig`] (the `speaker_encoder` section,
//!   [`Zonos3SpeakerEncoderMirror::speaker_config`](super::interop::zonos::Zonos3SpeakerEncoderMirror::speaker_config)):
//!   a clip to the raw speaker embedding the prompt's speaker row projects,
//!   through its MECT-B2 stages ([`Zonos3MectBlock`]: residual convolutions
//!   ([`Zonos3MectResBlock`]), then a transformer layer
//!   ([`Zonos3MectTransformerLayer`]) with a token soft mixture of experts
//!   ([`Zonos3MectTokenSoftMoe`])), a mix of the stages and
//!   [`AttentiveStatsPooling`](crate::blocks::pooling::AttentiveStatsPooling).

mod codec;
mod codec_stage;
mod codec_stream;
mod lm;
mod mect;
mod speaker_encoder;

pub use codec::*;
pub use codec_stage::*;
pub use codec_stream::*;
pub use lm::*;
pub use mect::*;
pub use speaker_encoder::*;
