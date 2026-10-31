//! `bunsen-chat`'s nanochat tokenizer against upstream's `RustBPETokenizer`.

use bunsen::kits::{
    gpts::nanochat::TOKENIZER,
    tokens::Detokenizer,
};
use bunsen_chat::{
    ChatMessage,
    NanoChatLayout,
    NanoChatTokenizer,
};

use super::*;

/// Each tokenizer encodes every fixture text to upstream's ids, decodes
/// them back to upstream's text, and has upstream's special tokens.
#[test]
fn test_tokenizers_encode_as_upstream() {
    for (generation, record) in chat_reference() {
        let path = row_file(tokenizer_row(&generation), TOKENIZER);
        let tok = NanoChatTokenizer::from_file(&path).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(tok.len(), record.vocab_size, "{generation}");
        for (name, &id) in &record.specials {
            assert_eq!(tok.special(name), Some(id), "{generation}: {name}");
        }
        for case in &record.encode {
            let ids = tok.encode(&case.text).unwrap();
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
}

/// Each tokenizer's layout renders every fixture conversation to upstream's
/// ids for the assistant's next turn: the system message folded, the
/// markers, and a special token's spelling as text.
#[test]
fn test_conversations_render_as_upstream() {
    for (generation, record) in chat_reference() {
        let path = row_file(tokenizer_row(&generation), TOKENIZER);
        let tok = NanoChatTokenizer::from_file(&path).unwrap_or_else(|e| panic!("{e}"));
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
}
