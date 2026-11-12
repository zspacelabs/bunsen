//! bunsen's tokenizer (`BunsenTokenizer`) and `bunsen_tool::chat`'s chat
//! template against `transformers`'.

use std::path::PathBuf;

use bunsen::{
    data::pretrained::ResourceBundle,
    kits::{
        language::qwen3_5::{
            CHAT_TEMPLATE,
            TOKENIZER,
            TOKENIZER_CONFIG,
            pretrained::default_qwen3_5_factory,
        },
        support::tokens::{
            BunsenTokenizer,
            Detokenizer,
            Tokenizer,
        },
    },
};
use bunsen_tool::chat::{
    ChatMessage,
    ChatTemplate,
};

use super::*;

/// A row's tokenizer files, and its `chat_template.jinja` where it has one,
/// from the weights cache, without its weights.
fn tokenizer_files(row: &str) -> (PathBuf, PathBuf, Option<PathBuf>) {
    let (_, pretrained) = default_qwen3_5_factory()
        .expect("the qwen3.5 factory")
        .lookup(row)
        .unwrap_or_else(|e| panic!("{row}: {e}"));
    let mut map = ResourceBundle::new(pretrained.resources.name.clone());
    for key in [TOKENIZER, TOKENIZER_CONFIG, CHAT_TEMPLATE] {
        if let Some(r) = pretrained.resources.get(key) {
            map.insert(r.clone());
        }
    }
    let loaded = weights_cache()
        .load(&map)
        .unwrap_or_else(|e| panic!("{row}: {e}"));
    (
        loaded.expect(TOKENIZER).unwrap().to_path_buf(),
        loaded.expect(TOKENIZER_CONFIG).unwrap().to_path_buf(),
        loaded.get(CHAT_TEMPLATE).map(|p| p.path.clone()),
    )
}

/// Every text encodes to `transformers`' ids (the 248k-token vocabulary, its
/// NFC normalizer and `\p{M}` split, and its added tokens, through
/// `BunsenTokenizer`), and decodes to a text that encodes to them again; every
/// conversation, under every setting of the thinking switch, renders to
/// `transformers`' text, which encodes to its ids.
fn check_row(row: &str) {
    let fixture = &chat_reference()[row];
    let (tokenizer, config, jinja) = tokenizer_files(row);
    let tok = BunsenTokenizer::from_files(&tokenizer, Some(&config)).unwrap();
    for case in &fixture.encode {
        let ids = tok.encode(&case.text).unwrap();
        assert_eq!(ids, case.ids, "{row}: {:?}", case.text);
        // The normalizer composes what it can (NFC), so a text decodes to its
        // normal form, which encodes to the same ids.
        let text = tok.detokenize(&ids).unwrap();
        assert_eq!(
            tok.encode(&text).unwrap(),
            ids,
            "{row}: {:?} decodes to {text:?}",
            case.text
        );
    }
    if !fixture.templates.is_empty() {
        let template = match &jinja {
            Some(jinja) => ChatTemplate::from_jinja_file(jinja, &config),
            None => ChatTemplate::from_file(&config),
        }
        .unwrap();
        for case in &fixture.templates {
            let messages: Vec<ChatMessage> = case
                .messages
                .iter()
                .map(|m| ChatMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                    reasoning_content: m.reasoning_content.clone(),
                })
                .collect();
            let what = format!(
                "{row}: {} (generation prompt {}, thinking {:?})",
                case.name, case.add_generation_prompt, case.enable_thinking
            );
            let text = template
                .render(&messages, case.add_generation_prompt, case.enable_thinking)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(text, case.text, "{what}");
            assert_eq!(tok.encode(&text).unwrap(), case.ids, "{what}");
        }
    }
    eprintln!(
        "{row}: {} texts encode alike, {} renders alike",
        fixture.encode.len(),
        fixture.templates.len()
    );
}

#[test]
fn qwen3_5_0_8b() {
    check_row("qwen3.5-0.8b");
}

#[test]
fn qwen3_5_4b() {
    check_row("qwen3.5-4b");
}

#[test]
fn qwen3_5_0_8b_base() {
    check_row("qwen3.5-0.8b-base");
}
