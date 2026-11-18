//! A loaded ZONOS3 model.

use alloc::{
    collections::BTreeMap,
    string::String,
};
use std::path::PathBuf;

use super::{
    Zonos3GenerationJson,
    codec::{
        Zonos3Codec,
        Zonos3CodecConfig,
    },
    lm::{
        Zonos3Lm,
        Zonos3LmConfig,
    },
    prompt::Zonos3PromptConfig,
    voices::Zonos3Voice,
};

/// What a ZONOS3 pretrained loads to: the language model and the codec
/// decoder with their configs, the prompt layout, the generation settings,
/// the stock voices, and the tokenizer's file, unread.
#[derive(Debug)]
pub struct Zonos3Construct {
    /// The language model.
    pub lm: Zonos3Lm,

    /// Its config.
    pub lm_config: Zonos3LmConfig,

    /// The codec decoder.
    pub codec: Zonos3Codec,

    /// Its config.
    pub codec_config: Zonos3CodecConfig,

    /// The prompt layout.
    pub prompt: Zonos3PromptConfig,

    /// The release's generation settings.
    pub generation: Zonos3GenerationJson,

    /// The output's sample rate.
    pub sample_rate: u32,

    /// The stock voices, by name.
    pub voices: BTreeMap<String, Zonos3Voice>,

    /// The `tokenizer.json`: ids already offset past the four special ids
    /// ([`zonos3_text_ids`](super::prompt::zonos3_text_ids)).
    pub tokenizer: PathBuf,
}
