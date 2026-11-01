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

/// The published row's `tekken.json`, fetched into the weights cache alone.
fn tekken_json() -> String {
    use bunsen::{
        data::pretrained::ResourceMap,
        kits::speech::voxtral::pretrained::{
            DEFAULT_VOXTRAL_MODEL,
            VOCABULARY,
            VOXTRAL_KIT,
            default_voxtral_factory,
        },
    };
    let (_, row) = default_voxtral_factory()
        .expect("the voxtral factory")
        .lookup(DEFAULT_VOXTRAL_MODEL)
        .expect("the published row");
    let mut map = ResourceMap::new(row.resources.name.clone());
    map.insert(
        row.resources
            .try_get(VOCABULARY)
            .expect("its vocabulary")
            .clone(),
    );
    let loaded = weights_cache()
        .load(VOXTRAL_KIT, &map)
        .expect("fetch tekken.json");
    let path = loaded.expect(VOCABULARY).expect("the fetched file");
    std::fs::read_to_string(path).expect("read tekken.json")
}

mod voxtral_mini_4b_realtime_2602 {
    use super::*;

    /// The published `tekken.json` lays the stream out as the processor did,
    /// and the ids `transformers` decoded read back as its text.
    #[test]
    fn transcripts_read_as_transformers_text() {
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
}
