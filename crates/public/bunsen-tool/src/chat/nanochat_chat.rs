//! A nanochat checkpoint as a chat's model.
use std::sync::Arc;

use bunsen::{
    errors::BunsenResult,
    kits::{
        language::nanochat::NanoChatGpt,
        support::{
            construct::LmConstruct,
            model_config::GenerationPolicy,
            tokens::{
                BunsenTokenizer,
                Tokenizer,
            },
        },
    },
};

use crate::chat::{
    ChatLm,
    ChatMessage,
    ChatParts,
    ChatRenderer,
    LmChat,
    NanoChatLayout,
    backend::missing,
};

/// A chat over a loaded nanochat: its tokenizer (its `tokenizer.pkl`, as a
/// [`BunsenTokenizer`]), its conversation layout ([`NanoChatLayout`]), and
/// upstream's chat defaults. A reply never chooses an id the tokenizer cannot
/// spell.
///
/// nanochat's checkpoints carry no generation config: its sampling is the
/// kit's policy
/// ([`nanochat_default_policy`](bunsen::kits::language::nanochat::nanochat_default_policy),
/// upstream's chat scripts'), and a reply ends at its tokenizer's
/// `<|assistant_end|>` or `<|bos|>`. It has no thinking switch.
pub type NanoChatChat = LmChat<NanoChatGpt>;

/// nanochat's parts: its `tokenizer.pkl`, its layout, and the kit's policy
/// with the layout's stop tokens.
///
/// # Errors
/// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
/// when the checkpoint lacks `tokenizer.pkl`, or its tokenizer the layout's
/// special tokens; as [`BunsenTokenizer::from_tiktoken_pickle_file`].
impl ChatLm for NanoChatGpt {
    fn chat_parts(bundle: &LmConstruct<Self>) -> BunsenResult<ChatParts> {
        let path = bundle.tokenizer().ok_or_else(|| missing("tokenizer.pkl"))?;
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(BunsenTokenizer::from_tiktoken_pickle_file(path)?);
        let layout = NanoChatLayout::from_tokenizer(tokenizer.as_ref())?;
        // The kit's policy, the sampling among it; the stops are the
        // tokenizer's, which the layout reads.
        let generation = GenerationPolicy {
            stop_ids: layout.stop_ids(),
            ..bundle.policy.generation.clone()
        };
        Ok(ChatParts {
            tokenizer,
            renderer: Arc::new(layout),
            generation,
        })
    }
}

/// Upstream's layout renders the conversation, with no thinking switch. A
/// nanochat's state rewinds to any position, so its prompt needs no split
/// point ([`closed_len`](ChatRenderer::closed_len)'s default).
impl ChatRenderer for NanoChatLayout {
    fn render_prompt(
        &self,
        tokenizer: &dyn Tokenizer,
        messages: &[ChatMessage],
        _enable_thinking: Option<bool>,
    ) -> BunsenResult<Vec<i64>> {
        self.render(tokenizer, messages)
    }
}
