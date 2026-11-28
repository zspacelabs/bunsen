//! A loaded Qwen3 model, with the files a decode reads beside it.
pub use crate::kits::support::construct::{
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

/// How a Qwen3 pretrained is run: the [`LmPolicy`] its presets and probes
/// patch, the template's kwargs its `thinking-mode` presets set among them.
pub type Qwen3Policy = LmPolicy;

/// A Qwen3 model, the positions it was trained on, and its checkpoint's
/// other files, unread: what a pretrained loads to
/// ([`LmConstruct`]).
///
/// [`default_qwen3_factory`](super::pretrained::default_qwen3_factory)
/// builds one from a name.
pub type Qwen3Construct = LmConstruct<LmDecoderModel>;
