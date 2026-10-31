//! bunsen's tekken vocabulary and transcript parser on `transformers`'
//! decodes: the published `tekken.json`, no model.

use bunsen::kits::{
    speech::voxtral::{
        VoxtralTokenLayoutConfig,
        VoxtralTranscript,
    },
    tokens::TekkenVocabulary,
};

use super::*;

/// The bundled `tekken.json`.
fn tekken_json() -> String {
    fn find(dir: &std::path::Path) -> Option<std::path::PathBuf> {
        for entry in std::fs::read_dir(dir).ok()? {
            let path = entry.ok()?.path();
            if path.is_dir() {
                if let Some(found) = find(&path) {
                    return Some(found);
                }
            } else if path.file_name().is_some_and(|n| n == "tekken.json") {
                return Some(path);
            }
        }
        None
    }
    let path = find(bind_voxtral_onnx::cache_dir()).expect("the bundled tekken.json");
    std::fs::read_to_string(&path).expect("read tekken.json")
}

/// The published `tekken.json` lays the stream out as the processor did, and
/// the ids `transformers` decoded read back as its text.
#[test]
fn test_transcripts_read_as_transformers_text() {
    let json = tekken_json();
    let vocab = TekkenVocabulary::from_tekken_json(&json).unwrap();
    assert_eq!(vocab.vocab_size(), 131_072);
    assert_eq!(vocab.special_id("[STREAMING_PAD]"), Some(32));
    let layout = VoxtralTokenLayoutConfig::from_tekken_json(&json)
        .unwrap()
        .init();
    assert_eq!(layout.config(), &VoxtralTokenLayoutConfig::new());
    for window in speech_reference().windows {
        assert_eq!(layout.prompt(), window.prompt_ids);
        let transcript = VoxtralTranscript::parse(window.ids, &layout, &vocab).unwrap();
        assert_eq!(transcript.text, window.text.trim());
        assert!(!transcript.words.is_empty());
    }
}
