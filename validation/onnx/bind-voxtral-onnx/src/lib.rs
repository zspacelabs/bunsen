//! Voxtral Realtime validation assets.
//!
//! **This crate hosts nothing.** `mistralai/Voxtral-Mini-4B-Realtime-2602` is
//! 8.9 GB and its fp32 ONNX export 19 GB, far too large to commit, so this
//! crate *fetches* them, pins each file to a SHA-256, and keeps them in
//! `OUT_DIR`. Only `voxtral-model-validation` depends on it.
//!
//! The checkpoint is laid out in `OUT_DIR` as a **pretrained cache**,
//! `pretrained/voxtral/mistralai/<sha256>/<file>`, which is how bunsen's
//! `data::pretrained::PretrainedCache` roots a resource. A cache pointed at
//! `cache_dir()`, offline, hits the row without knowing it was bundled.
//!
//! ## What it exposes
//!
//! Each item exists only under the feature that fetches its asset:
//!
//! | Item | Feature | What it is |
//! |---|---|---|
//! | `cache_dir()` | `checkpoint` | `OUT_DIR`, laid out as a pretrained cache |
//! | `onnx_gen::VoxtralAudioEncoderOnnx` | `onnx_gen` | the audio encoder (stem, layers, norm, projector), streamed through its caches |
//! | `onnx_gen::VoxtralDecoderOnnx` | `onnx_gen` | the text decoder, from input embeddings to logits, through its KV cache |
//! | `onnx_gen::VoxtralEmbedTokensOnnx` | `onnx_gen` | the token embedding |
//!
//! All three are generated from
//! `onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX`
//! after `rewrite-genai-onnx` rewrites its onnxruntime ops. The decoder's
//! adaptive scales are baked into the export at a delay of 6 tokens (480 ms).
//!
//! ## Using local copies
//!
//! Each file has a variable naming a local file to use instead, read only
//! when the feature that fetches it is on: `VOXTRAL_MODEL_SAFETENSORS` and
//! `VOXTRAL_TEKKEN_JSON` for the checkpoint, and for the export
//! `VOXTRAL_ONNX_` and the file's name, in capitals with `_` for `.`
//! (`VOXTRAL_ONNX_AUDIO_ENCODER_ONNX`,
//! `VOXTRAL_ONNX_DECODER_MODEL_MERGED_ONNX_DATA_3`). A file that is the
//! pinned asset is put where the download would have gone; any other file is
//! used where it is, with a warning.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

/// `OUT_DIR` laid out as a pretrained cache: the root a
/// `bunsen::data::pretrained::PretrainedCache` is pointed at to read the
/// bundled row. A cache that fetches other models into it will change it, so
/// a test wants it offline.
#[cfg(feature = "checkpoint")]
pub fn cache_dir() -> &'static std::path::Path {
    std::path::Path::new(env!("VOXTRAL_CACHE_DIR"))
}

/// The reference models, generated from the rewritten export.
#[cfg(feature = "onnx_gen")]
pub mod onnx_gen {
    use burn::prelude::*;

    fn out_dir() -> &'static std::path::Path {
        std::path::Path::new(env!("VOXTRAL_ONNX_OUT_DIR"))
    }

    /// The audio encoder: `input_features` (`[batch, 128, frames]` log-mels,
    /// `frames` a multiple of 8), `attention_mask` and `position_ids` over
    /// the encoder frames, the packed conv carry `past_padding_cache`
    /// (`[batch, 1408, 2]`), then each layer's past key and value; to the
    /// `[batch, frames/8, 3072]` audio embeddings, the next carry, and each
    /// layer's present key and value.
    // Machine-generated: not held to this crate's lint bar.
    #[allow(warnings, clippy::all)]
    mod voxtral_audio_encoder {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/voxtral_audio_encoder.rs"));

        impl Model {
            /// Loads the reference model from the generated weights.
            pub fn load_pretrained(device: &Device) -> Self {
                Self::from_file(out_dir().join("voxtral_audio_encoder.bpk"), device)
            }
        }
    }
    pub use voxtral_audio_encoder::Model as VoxtralAudioEncoderOnnx;

    /// The text decoder: `inputs_embeds` (`[batch, seq, 3072]`) and
    /// `attention_mask`, then each layer's past key and value; to the
    /// `[batch, seq, 131072]` logits and each layer's present key and value.
    // Machine-generated: not held to this crate's lint bar.
    #[allow(warnings, clippy::all)]
    mod voxtral_decoder {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/voxtral_decoder.rs"));

        impl Model {
            /// Loads the reference model from the generated weights.
            pub fn load_pretrained(device: &Device) -> Self {
                Self::from_file(out_dir().join("voxtral_decoder.bpk"), device)
            }
        }
    }
    pub use voxtral_decoder::Model as VoxtralDecoderOnnx;

    /// The token embedding: `input_ids` (`[batch, seq]`) to
    /// `[batch, seq, 3072]`.
    // Machine-generated: not held to this crate's lint bar.
    #[allow(warnings, clippy::all)]
    mod voxtral_embed_tokens {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/voxtral_embed_tokens.rs"));

        impl Model {
            /// Loads the reference model from the generated weights.
            pub fn load_pretrained(device: &Device) -> Self {
                Self::from_file(out_dir().join("voxtral_embed_tokens.bpk"), device)
            }
        }
    }
    pub use voxtral_embed_tokens::Model as VoxtralEmbedTokensOnnx;
}
