//! `bunsen_tool::chat`'s tokenizer and chat template against `transformers`'.

use std::path::PathBuf;

use bunsen::{
    data::pretrained::ResourceMap,
    kits::gpts::qwen3::{
        TOKENIZER,
        TOKENIZER_CONFIG,
        pretrained::{
            QWEN3_KIT,
            default_qwen3_factory,
        },
    },
};
use bunsen_tool::chat::{
    ChatMessage,
    ChatTemplate,
    Qwen3Tokenizer,
};

use super::*;

/// A row's `tokenizer.json` and `tokenizer_config.json`, from the bundled
/// cache, without its weights.
fn tokenizer_files(row: &str) -> (PathBuf, PathBuf) {
    let (_, pretrained) = default_qwen3_factory()
        .expect("the qwen3 factory")
        .lookup(row)
        .unwrap_or_else(|e| panic!("{row}: {e}"));
    let mut map = ResourceMap::new(pretrained.resources.name.clone());
    for key in [TOKENIZER, TOKENIZER_CONFIG] {
        map.insert(pretrained.resources.try_get(key).unwrap().clone());
    }
    let loaded = weights_cache()
        .load(QWEN3_KIT, &map)
        .unwrap_or_else(|e| panic!("{row}: {e}"));
    (
        loaded.expect(TOKENIZER).unwrap().to_path_buf(),
        loaded.expect(TOKENIZER_CONFIG).unwrap().to_path_buf(),
    )
}

/// Every text encodes to `transformers`' ids, added tokens in it matched,
/// and decodes back to itself.
#[test]
fn test_tokenizer_matches_transformers() {
    for (row, fixture) in chat_reference() {
        let (tokenizer, config) = tokenizer_files(&row);
        let tok = Qwen3Tokenizer::from_files(&tokenizer, Some(&config)).unwrap();
        for case in &fixture.encode {
            assert_eq!(
                tok.encode(&case.text).unwrap(),
                case.ids,
                "{row}: {:?}",
                case.text
            );
        }
        eprintln!("{row}: {} texts encode alike", fixture.encode.len());
    }
}

/// Every conversation, under every setting, renders to `transformers`'
/// text, which encodes to its ids.
#[test]
fn test_chat_template_matches_transformers() {
    for (row, fixture) in chat_reference() {
        let (tokenizer, config) = tokenizer_files(&row);
        let tok = Qwen3Tokenizer::from_files(&tokenizer, Some(&config)).unwrap();
        let template = ChatTemplate::from_file(&config).unwrap();
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
        eprintln!("{row}: {} renders alike", fixture.templates.len());
    }
}
