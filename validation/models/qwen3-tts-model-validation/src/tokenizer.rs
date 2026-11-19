//! bunsen's tokenizer, built from the repository's slow files, against the
//! processor's, without the weights.

use bunsen::{
    data::resources::ResourceSet,
    kits::{
        speech::tts::qwen3_tts::pretrained::{
            MERGES,
            TOKENIZER_CONFIG,
            VOCAB,
            default_qwen3_tts_factory,
        },
        support::tokens::{
            BunsenTokenizer,
            Tokenizer,
        },
    },
};
use bunsen_test_audio::testdata_dir;

use super::*;
use crate::fixtures::{
    REFERENCE,
    REQUESTS,
    TOKENIZER,
};

/// The release's tokenizer, from its files in the weights cache, without
/// the weights.
fn release_tokenizer() -> BunsenTokenizer {
    let (_, pretrained) = default_qwen3_tts_factory()
        .expect("the qwen3_tts factory")
        .lookup(MODEL)
        .unwrap_or_else(|e| panic!("{MODEL}: {e}"));
    let mut map = ResourceSet::new(pretrained.resource_set().name.clone());
    for key in [VOCAB, MERGES, TOKENIZER_CONFIG] {
        map.insert(pretrained.resource_set().get(key).unwrap().clone());
    }
    let loaded = weights_cache()
        .load(&map)
        .unwrap_or_else(|e| panic!("{MODEL}: {e}"));
    BunsenTokenizer::from_qwen2_files(
        loaded.expect(VOCAB).unwrap(),
        loaded.expect(MERGES).unwrap(),
        loaded.expect(TOKENIZER_CONFIG).unwrap(),
    )
    .expect("the release's tokenizer")
}

/// The texts the fixture holds ids for, by name.
fn texts() -> Vec<(String, String)> {
    let dir = testdata_dir();
    let mut texts: Vec<(String, String)> = ["de", "es", "fr"]
        .iter()
        .map(|lang| {
            let name = format!("udhr_{lang}");
            let text = std::fs::read_to_string(dir.join(format!("{name}.txt"))).unwrap();
            (name, text)
        })
        .collect();
    texts.push((
        "jfk_moon".into(),
        std::fs::read_to_string(dir.join("jfk_moon.txt")).unwrap(),
    ));
    texts.push((
        "chinese".into(),
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。".into(),
    ));
    texts
}

mod qwen3_tts_12hz_0_6b_customvoice {
    use super::*;

    /// The test texts' ids are the processor's, and each request's
    /// chat-wrapped text's, its specials among them, too.
    #[test]
    fn tokenizer() {
        let tokenizer = release_tokenizer();
        let fixture = FixtureFile::load(TOKENIZER);
        for (name, text) in texts() {
            let ids = tokenizer.encode(&text).unwrap();
            assert_eq!(ids, fixture.ints(&format!("{name}.ids")), "{name}");
        }
        let reference = FixtureFile::load(REFERENCE);
        for name in REQUESTS {
            let text = reference.request(name).text;
            let wrapped =
                format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
            assert_eq!(
                tokenizer.encode(&wrapped).unwrap(),
                reference.ints(&format!("{name}.input_ids")),
                "{name}"
            );
        }
    }
}
