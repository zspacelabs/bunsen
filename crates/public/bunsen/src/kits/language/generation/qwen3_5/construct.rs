//! A loaded Qwen3.5 model, with the files a decode reads beside it.
pub use crate::kits::support::construct::{
    CHAT_TEMPLATE,
    GENERATION_CONFIG,
    TOKENIZER,
    TOKENIZER_CONFIG,
};
use crate::{
    blocks::transformers::models::language_decoder::LmDecoderModel,
    kits::support::construct::{
        LmConstruct,
        LmPolicy,
    },
};

/// How a Qwen3.5 pretrained is run: the [`LmPolicy`] its presets and probes
/// patch.
pub type Qwen3_5Policy = LmPolicy;

/// A Qwen3.5 model, the positions it was trained on, and its checkpoint's
/// other files, unread: what a pretrained loads to ([`LmConstruct`]).
pub type Qwen3_5Construct = LmConstruct<LmDecoderModel>;
