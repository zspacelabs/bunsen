//! bunsen's tokenizer (`BunsenTokenizer`) and `bunsen_tool::chat`'s chat
//! template against `transformers`'.

use std::path::PathBuf;

use bunsen::{
    data::pretrained::ResourceBundle,
    kits::{
        language::gemma::{
            CHAT_TEMPLATE,
            END_OF_TURN,
            GemmaPolicy,
            TOKENIZER,
            TOKENIZER_CONFIG,
            pretrained::{
                GEMMA_CATALOG,
                GEMMA_FAMILY,
                GEMMA_KIT,
                default_gemma_factory,
            },
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
    let (_, pretrained) = default_gemma_factory()
        .expect("the gemma factory")
        .lookup(row)
        .unwrap_or_else(|e| panic!("{row}: {e}"));
    let mut map = ResourceBundle::new(pretrained.resources.name.clone());
    for key in [TOKENIZER, TOKENIZER_CONFIG, CHAT_TEMPLATE] {
        if let Some(r) = pretrained.resources.get(key) {
            map.insert(r.clone());
        }
    }
    let loaded = weights_cache()
        .load(GEMMA_KIT, &map)
        .unwrap_or_else(|e| panic!("{row}: {e}"));
    (
        loaded.expect(TOKENIZER).unwrap().to_path_buf(),
        loaded.expect(TOKENIZER_CONFIG).unwrap().to_path_buf(),
        loaded.get(CHAT_TEMPLATE).map(|p| p.path.clone()),
    )
}

/// Every text encodes to `transformers`' ids and decodes back to itself;
/// every conversation renders to `transformers`' text, which encodes to its
/// ids; the [`GEMMA_FAMILY`]'s own template, which a pretrained row renders
/// with, renders every conversation as the row's own does; and the row's
/// stop tokens, as its presets resolve them, include the end of a turn its
/// tokenizer spells ([`END_OF_TURN`]).
fn check_row(row: &str) {
    let fixture = &chat_reference()[row];
    let (tokenizer, config, jinja) = tokenizer_files(row);
    let tok = BunsenTokenizer::from_files(&tokenizer, Some(&config)).unwrap();
    let stops = default_gemma_factory()
        .unwrap()
        .resolve_presets(row, &GemmaPolicy::default())
        .unwrap()
        .config
        .generation
        .stop_ids;
    let end = tok
        .token_id(END_OF_TURN)
        .expect("the tokenizer spells a turn's end");
    assert!(
        stops.contains(&end),
        "{row}: stops {stops:?} lack {END_OF_TURN} ({end})"
    );
    for case in &fixture.encode {
        let ids = tok.encode(&case.text).unwrap();
        assert_eq!(ids, case.ids, "{row}: {:?}", case.text);
        assert_eq!(
            tok.detokenize(&ids).unwrap(),
            case.text,
            "{row}: {:?}",
            case.text
        );
    }
    let template = match &jinja {
        Some(jinja) => ChatTemplate::from_jinja_file(jinja, &config),
        None => ChatTemplate::from_file(&config),
    }
    .unwrap();
    let family = GEMMA_CATALOG.family_presets(GEMMA_FAMILY).base["generation"]["chat_template"]
        .as_str()
        .expect("the family's template")
        .to_string();
    let family = ChatTemplate::new(&family)
        .unwrap()
        .with_special_tokens([("bos_token".to_string(), "<bos>".to_string())].into());
    for case in &fixture.templates {
        let messages: Vec<ChatMessage> = case
            .messages
            .iter()
            .map(|m| ChatMessage::new(m.role.clone(), m.content.clone()))
            .collect();
        let what = format!(
            "{row}: {} (generation prompt {})",
            case.name, case.add_generation_prompt
        );
        let text = template
            .render(&messages, case.add_generation_prompt, None)
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!(text, case.text, "{what}");
        assert_eq!(tok.encode(&text).unwrap(), case.ids, "{what}");
        let text = family
            .render(&messages, case.add_generation_prompt, None)
            .unwrap_or_else(|e| panic!("{what}, the family's template: {e}"));
        assert_eq!(text, case.text, "{what}, the family's template");
    }
    eprintln!(
        "{row}: {} texts encode alike, {} renders alike",
        fixture.encode.len(),
        fixture.templates.len()
    );
}

#[test]
fn gemma_3_1b_it() {
    check_row("gemma-3-1b-it");
}

#[test]
fn gemma_2_2b_it() {
    check_row("gemma-2-2b-it");
}
