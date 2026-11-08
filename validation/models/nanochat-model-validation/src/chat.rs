//! `bunsen_tool::chat`'s nanochat tokenizer (a `tokenizer.pkl` read as an
//! `HfTokenizer`) against upstream's `RustBPETokenizer`.

use bunsen::kits::{
    language::nanochat::TOKENIZER,
    support::tokens::{
        Detokenizer,
        Tokenizer,
    },
};
use bunsen_tool::chat::{
    ChatMessage,
    ChatTokenizer,
    HfTokenizer,
    NanoChatLayout,
};

use super::*;

/// Each tokenizer encodes every fixture text to upstream's ids, decodes
/// them back to upstream's text, and has upstream's special tokens.
fn check_tokenizer(generation: &str) {
    let fixture = chat_reference();
    let record = &fixture[generation];
    let path = row_file(tokenizer_row(generation), TOKENIZER);
    let tok = HfTokenizer::from_tiktoken_pickle_file(&path).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(tok.len(), record.vocab_size, "{generation}");
    for (name, &id) in &record.specials {
        assert_eq!(tok.token_id(name), Some(id), "{generation}: {name}");
    }
    for case in &record.encode {
        // Upstream's `encode`, a special token's spelling text.
        let ids = tok.encode_plain(&case.text).unwrap();
        assert_eq!(ids, case.ids, "{generation}: {:?}", case.text);
        assert_eq!(
            tok.detokenize(&ids).unwrap(),
            case.decoded,
            "{generation}: {:?}",
            case.text
        );
    }
    eprintln!(
        "{generation}: {} texts encode as upstream's",
        record.encode.len()
    );
}

/// Each tokenizer's layout renders every fixture conversation to upstream's
/// ids for the assistant's next turn: the system message folded, the
/// markers, and a special token's spelling as text.
fn check_conversations(generation: &str) {
    let fixture = chat_reference();
    let record = &fixture[generation];
    let path = row_file(tokenizer_row(generation), TOKENIZER);
    let tok = HfTokenizer::from_tiktoken_pickle_file(&path).unwrap_or_else(|e| panic!("{e}"));
    let layout = NanoChatLayout::from_tokenizer(&tok).unwrap();
    for case in &record.conversations {
        let messages: Vec<ChatMessage> = case
            .messages
            .iter()
            .map(|m| ChatMessage::new(m.role.clone(), m.content.clone()))
            .collect();
        assert_eq!(
            layout.render(&tok, &messages).unwrap(),
            case.ids,
            "{generation}: {:?}",
            case.messages
        );
    }
    eprintln!(
        "{generation}: {} conversations render as upstream's",
        record.conversations.len()
    );
}

/// A tokenizer's tests, under the id of the row it is read from.
macro_rules! tokenizer_tests {
    ($id:ident, $generation:literal) => {
        mod $id {
            use super::*;

            #[test]
            fn tokenizer() {
                check_tokenizer($generation);
            }

            #[test]
            fn conversations() {
                check_conversations($generation);
            }
        }
    };
}

tokenizer_tests!(nanochat_d20_rl, "2025");
tokenizer_tests!(nanochat_d18_sft, "2026");
