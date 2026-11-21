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

/// What ends a turn in a Gemma's chat template, `<end_of_turn>`: a reply
/// stops there as well as at the end of the text. Its id is the
/// tokenizer's (106 in Gemma 3's, 107 in Gemma 1's and 2's), which the
/// `tokenizer_config` probe reads by this name into a row's stop tokens: the
/// `generation_config.json` of some instruction-tuned checkpoints (Gemma 1's,
/// Gemma 2 9B's) names only the end of the text.
pub const END_OF_TURN: &str = "<end_of_turn>";

/// How a Gemma pretrained is run: the [`LmPolicy`] its presets and probes
/// patch (the stop tokens and sampling its `generation_config.json` names).
pub type GemmaPolicy = LmPolicy;

/// A Gemma model, the positions it was trained on, and its checkpoint's
/// other files, unread: what a pretrained loads to ([`LmConstruct`]).
///
/// [`default_gemma_factory`](super::pretrained::default_gemma_factory)
/// builds one from a name.
pub type GemmaConstruct = LmConstruct<GemmaModel>;
