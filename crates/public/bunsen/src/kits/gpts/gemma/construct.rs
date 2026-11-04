//! A loaded Gemma model, with the files a decode reads beside it.
use super::GemmaModel;
pub use crate::kits::support::construct::{
    CHAT_TEMPLATE,
    GENERATION_CONFIG,
    TOKENIZER,
    TOKENIZER_CONFIG,
};
use crate::kits::support::construct::{
    LmConstruct,
    LmPolicy,
};

/// How a Gemma pretrained is run: the [`LmPolicy`] its presets and probes
/// patch (the stop tokens and sampling its `generation_config.json` names).
pub type GemmaPolicy = LmPolicy;

/// A Gemma model, the positions it was trained on, and its checkpoint's
/// other files, unread: what a pretrained loads to ([`LmConstruct`]).
///
/// [`default_gemma_factory`](super::pretrained::default_gemma_factory)
/// builds one from a name.
pub type GemmaConstruct = LmConstruct<GemmaModel>;
