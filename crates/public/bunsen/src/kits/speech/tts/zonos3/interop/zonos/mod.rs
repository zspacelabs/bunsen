//! # `zonos`: ZONOS3's own files, as its release reads them
//!
//! ZONOS3-Mini ships as `Zyphra/ZONOS3-Mini-hf`: a `config.json`, one
//! `model.safetensors` (the language model under `lm.`, the codec decoder
//! under `codec.`), and the stock voices, `voices.safetensors` and
//! `voices.json`.
//!
//! - **Mirrors:** [`Zonos3Mirror`] (`config.json`), its sections
//!   [`Zonos3LmMirror`], [`Zonos3CodecMirror`] and [`Zonos3GenerationMirror`],
//!   and [`Zonos3VoiceInfoMirror`] (`voices.json`). A mirror's conversions are
//!   its methods ([`lm_config`](Zonos3LmMirror::lm_config),
//!   [`codec_config`](Zonos3CodecMirror::codec_config),
//!   [`engine_config`](Zonos3GenerationMirror::engine_config),
//!   [`prompt_config`](Zonos3Mirror::prompt_config)). A `config.json` is read
//!   whole, by [`json_mirror_from_path`](crate::kits::support::interop::json_mirror_from_path).
//! - **Names:** the `-hf` file's `lm.*` tensors are upstream's reference
//!   language model's state dict (`zonos3.reference.lm`); its projections are
//!   packed where bunsen's blocks hold them apart, so three tensors are split
//!   by rows first ([`zonos3_lm_row_splits`]), and the names then map onto the
//!   blocks' canonical names layer by layer, as each layer's kind spells them
//!   ([`zonos3_lm_rekey`]). Its `codec.*` tensors are the `-hf` port's
//!   `Zonos3MiniCodec`, weight norm already fused: each trunk layer's packed
//!   `qkv` is split by rows ([`zonos3_codec_row_splits`]), and the stage list's
//!   odd entries are the parameterless patch-ups, so stage `2 i` is
//!   [`Zonos3Codec::stages`](crate::kits::speech::tts::zonos3::codec::Zonos3Codec::stages)`[i]`
//!   ([`zonos3_codec_rekey`]).
//! - **Voices:** each stock voice is two tensors of `voices.safetensors`,
//!   `<Name>.codes`, `[F, n]` integers, its reference codes, and
//!   `<Name>.embedding`, `[S]` floats, its speaker embedding; `voices.json`
//!   maps each name to its language, style and gender, and its files' stem.
//! - **Loading** (feature `store`): `load_zonos3_lm_tensors`,
//!   `load_zonos3_codec_tensors`, `zonos3_voices`; with `cache`, the parts of a
//!   resource set, `load_zonos3_lm_and_codec` and `load_zonos3_voices`.

mod codec;
mod lm;
mod model;
mod voices;
#[cfg(feature = "store")]
mod weights;

pub use codec::*;
pub use lm::*;
pub use model::*;
pub use voices::*;
#[cfg(feature = "store")]
pub use weights::*;
