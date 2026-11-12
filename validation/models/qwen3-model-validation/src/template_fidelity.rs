//! The `thinking-mode` presets' default against each chat template's own.

use bunsen::kits::language::qwen3::{
    Qwen3Policy,
    TOKENIZER_CONFIG,
    pretrained::default_qwen3_factory,
};
use bunsen_tool::chat::{
    ChatMessage,
    ChatTemplate,
};

use super::*;

/// The rows that think or answer at once: the catalog's `hybrid` family.
const HYBRID: &[&str] = &[
    "qwen/qwen3-0.6b",
    "qwen/qwen3-1.7b",
    "qwen/qwen3-4b",
    "qwen/qwen3-8b",
    "qwen/qwen3-4b-saferl",
];

/// Each hybrid row's default `enable_thinking` is what its chat template does
/// when the switch is absent, so upstream's default is tested here rather
/// than delegated to the template; and the switch is real, the two values
/// rendering differently. Fetches each row's `tokenizer_config.json` alone.
#[test]
fn test_the_default_thinking_mode_is_the_templates() {
    let factory = default_qwen3_factory().unwrap();
    let cache = weights_cache();
    let messages = [ChatMessage::user("Why is the sky blue?")];
    for row in HYBRID {
        let bundle = factory.resolve(row, &cache).unwrap().to_resource_set();
        let config = bundle
            .get(TOKENIZER_CONFIG)
            .expect("a tokenizer_config.json");
        let path = cache.resolve(config).unwrap().path;
        let template = ChatTemplate::from_file(&path).unwrap();
        let render = |switch| template.render(&messages, true, switch).unwrap();

        let (absent, on, off) = (render(None), render(Some(true)), render(Some(false)));
        assert_ne!(on, off, "{row}: the template reads `enable_thinking`");
        let upstream = if absent == on {
            true
        } else {
            assert_eq!(
                absent, off,
                "{row}: no switch renders as one value or the other"
            );
            false
        };

        let policy = factory
            .resolve_presets(row, &Qwen3Policy::default())
            .unwrap();
        assert_eq!(policy.presets, ["thinking"], "{row}");
        assert_eq!(
            policy.config.generation.template_kwargs["enable_thinking"].as_bool(),
            Some(upstream),
            "{row}: the default preset's switch is the template's own"
        );
    }
}
