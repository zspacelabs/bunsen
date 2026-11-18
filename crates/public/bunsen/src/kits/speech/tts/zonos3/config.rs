//! `Zyphra/ZONOS3-Mini-hf`'s `config.json`.

use alloc::{
    collections::BTreeMap,
    string::String,
    vec::Vec,
};

use serde::Deserialize;

use super::{
    codec::Zonos3CodecJson,
    lm::Zonos3LmJson,
};

/// The `generation` section of `config.json`, under upstream's names: how
/// the release generates and outputs a take.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Zonos3GenerationJson {
    /// Frames of context a streamed decode carries.
    pub decode_context_frames: usize,
    /// The output gain's headroom, in dB.
    pub headroom_db: f64,
    /// Request language codes to text-normalization grammars.
    pub languages: BTreeMap<String, String>,
    /// The markup tags the normalizer keeps.
    pub markup_tags: Vec<String>,
    /// The sampler's min-p.
    pub min_p: f64,
    /// The output limiter's knee, in dBFS.
    pub output_limit_dbfs: f64,
    /// Frames a streamed decode overlaps.
    pub overlap_frames: usize,
    /// The codebooks the repetition penalty covers.
    pub repetition_codebooks: usize,
    /// The repetition penalty.
    pub repetition_penalty: f64,
    /// The rows the repetition penalty looks back over.
    pub repetition_window: usize,
    /// The codec's silence frame.
    pub silence_frame: Vec<u32>,
    /// The default silence prefill.
    pub silence_prefill_frames: usize,
    /// The longest silence prefill.
    pub silence_prefill_max_frames: usize,
    /// The sampler's temperature.
    pub temperature: f64,
    /// The sampler's top-p.
    pub top_p: f64,
    /// The sampler's top-k.
    pub topk: usize,
}

/// `Zyphra/ZONOS3-Mini-hf`'s `config.json`: its language model's, codec
/// decoder's and generation's sections (the speaker encoder's is not read).
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Zonos3Json {
    /// The language model.
    pub lm: Zonos3LmJson,
    /// The codec decoder.
    pub codec: Zonos3CodecJson,
    /// How a take is generated and output.
    pub generation: Zonos3GenerationJson,
    /// The output's sample rate.
    pub sampling_rate: u32,
}
