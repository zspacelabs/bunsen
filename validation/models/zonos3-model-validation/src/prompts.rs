//! bunsen's tokenizer and prompt layout against the reference's, without the
//! weights.

use std::path::PathBuf;

use bunsen::{
    data::resources::ResourceSet,
    kits::{
        speech::tts::zonos3::{
            Zonos3Json,
            lm::Zonos3LmConfig,
            pretrained::{
                CONFIG,
                TOKENIZER,
                default_zonos3_factory,
            },
            prompt::{
                Zonos3PromptConfig,
                Zonos3PromptRequest,
                Zonos3Speaker,
                condition_bucket,
                zonos3_text_ids,
            },
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
    TOKENIZER as TOKENIZER_FIXTURE,
};

/// The release's config and tokenizer, from the weights cache, without its
/// weights.
fn release_files() -> (Zonos3Json, PathBuf) {
    let (_, pretrained) = default_zonos3_factory()
        .expect("the zonos3 factory")
        .lookup(MODEL)
        .unwrap_or_else(|e| panic!("{MODEL}: {e}"));
    let mut map = ResourceSet::new(pretrained.resource_set().name.clone());
    for key in [CONFIG, TOKENIZER] {
        map.insert(pretrained.resource_set().get(key).unwrap().clone());
    }
    let loaded = weights_cache()
        .load(&map)
        .unwrap_or_else(|e| panic!("{MODEL}: {e}"));
    let config = loaded.expect(CONFIG).unwrap();
    let json = serde_json::from_str(&std::fs::read_to_string(config).unwrap())
        .unwrap_or_else(|e| panic!("{}: {e}", config.display()));
    (json, loaded.expect(TOKENIZER).unwrap().to_path_buf())
}

mod zonos3_mini {
    use super::*;

    /// The `-hf` tokenizer's ids for each UDHR text, as `tokenizers` gives
    /// them.
    #[test]
    fn tokenizer() {
        let (_, path) = release_files();
        let tokenizer = BunsenTokenizer::from_file(&path).unwrap();
        let fixture = FixtureFile::load(TOKENIZER_FIXTURE);
        let keys: Vec<&str> = fixture.keys().collect();
        assert_eq!(keys.len(), 6);
        for key in keys {
            let stem = key.strip_suffix(".ids").unwrap();
            let text = std::fs::read_to_string(testdata_dir().join(format!("udhr_{stem}.txt")))
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert_eq!(tokenizer.encode(&text).unwrap(), fixture.ints(key), "{key}");
        }
    }

    /// Each request's prompt, laid out from its normalized text, is the
    /// reference's: its rows, its speaker row, its condition rows and their
    /// buckets (from the values, through the release's edges), its silence
    /// prefill.
    #[test]
    fn prompts() {
        let (json, path) = release_files();
        let tokenizer = BunsenTokenizer::from_file(&path).unwrap();
        let lm = Zonos3LmConfig::from_json(&json.lm).unwrap();
        let config = Zonos3PromptConfig::new(
            &lm,
            json.lm.text_vocab as u32,
            json.generation.silence_frame.clone(),
            json.lm.reference_max_frames,
        );
        let reference = FixtureFile::load(REFERENCE);
        for name in REQUESTS {
            let facts = reference.request(name);
            let request = &facts.request;
            let text_ids = zonos3_text_ids(&tokenizer, &facts.normalized_text).unwrap();
            assert_eq!(
                text_ids.iter().map(|&id| i64::from(id)).collect::<Vec<_>>(),
                reference.ints(&format!("{name}.text_ids")),
                "{name}: the text ids"
            );
            let speaker = match request["speaker_mode"].as_str() {
                Some("embedding") => {
                    Zonos3Speaker::Embedding(reference.floats(&format!("{name}.speaker_embedding")))
                }
                Some("reference") => Zonos3Speaker::Reference(
                    serde_json::from_value(request["speaker_reference_codes"].clone()).unwrap(),
                ),
                None => Zonos3Speaker::None,
                Some(other) => panic!("{name}: speaker mode {other}"),
            };
            let condition_buckets = match request.get("condition_values") {
                Some(values) => json
                    .lm
                    .condition_names
                    .iter()
                    .map(|c| {
                        values[c]
                            .as_f64()
                            .map(|value| condition_bucket(&json.lm.condition_edges[c], value))
                    })
                    .collect(),
                None => Vec::new(),
            };
            let silence_prefill_frames = request["silence_prefill_frames"]
                .as_u64()
                .map_or(json.generation.silence_prefill_frames, |f| f as usize)
                .min(json.generation.silence_prefill_max_frames);
            let prompt = config
                .build(&Zonos3PromptRequest {
                    text_ids,
                    speaker,
                    condition_buckets,
                    silence_prefill_frames,
                })
                .unwrap();
            assert_eq!(prompt, reference_prompt(&reference, name), "{name}");
        }
    }
}
