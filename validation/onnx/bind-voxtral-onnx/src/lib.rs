//! Voxtral Realtime validation assets.
//!
//! **This crate hosts nothing.** The fp32 ONNX export of
//! `mistralai/Voxtral-Mini-4B-Realtime-2602` is 19 GB, far too large to commit,
//! so this crate *fetches* it under `onnx_gen`, pins each file to a SHA-256,
//! keeps them in `OUT_DIR`, and generates the reference models from it. Only
//! `voxtral-model-validation` depends on it, and its tests load the
//! checkpoint themselves, through bunsen's own pinned row, at run time, into
//! the weights cache.
//!
//! ## What it exposes
//!
//! Each item exists only under `onnx_gen`, which fetches its asset:
//!
//! | Item | Feature | What it is |
//! |---|---|---|
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
//! under `onnx_gen`: `VOXTRAL_ONNX_` and the file's name, in capitals with
//! `_` for `.`
//! (`VOXTRAL_ONNX_AUDIO_ENCODER_ONNX`,
//! `VOXTRAL_ONNX_DECODER_MODEL_MERGED_ONNX_DATA_3`). A file that is the
//! pinned asset is put where the download would have gone; any other file is
//! used where it is, with a warning.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

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
