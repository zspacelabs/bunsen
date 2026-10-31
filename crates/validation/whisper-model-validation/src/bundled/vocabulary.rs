//! The rank files, against upstream's `whisper.tokenizer`.

use bunsen::{
    data::pretrained::{
        CacheStatus,
        Provenance,
    },
    kits::{
        speech::whisper::{
            blocks::WhisperTokenLayoutConfig,
            driver::{
                WhisperSpecialIds,
                WhisperTokenLayout,
            },
            logit_filters::{
                default_filters,
                default_suppress_tokens,
            },
            pretrained::{
                VOCABULARY,
                WHISPER_KIT,
                WHISPER_PREFABS,
                WhisperVocabulary,
            },
        },
        tokens::{
            Detokenizer,
            blank_token,
            non_speech_tokens,
        },
    },
};

use super::{
    bundled_vocabulary,
    bundled_vocabulary_path,
    offline_cache,
};

fn ids_of(prefab: &str) -> WhisperSpecialIds {
    let cfg = WHISPER_PREFABS.expect_lookup_prefab(prefab).to_config();
    *cfg.token_layout
        .policy_for_vocab(cfg.vocab_size)
        .unwrap()
        .ids()
}

/// With the bundle, both rank files resolve offline from its
/// directory and parse to the layout's base rank count.
#[test]
fn test_bundled_vocabularies_resolve_offline() {
    let cache = offline_cache();
    for prefab in ["tiny", "tiny.en"] {
        let ids = ids_of(prefab);
        let vocabulary = WhisperVocabulary::for_layout(&ids);
        let map = vocabulary.map().to_map();
        let r = map.try_get(VOCABULARY).unwrap();
        assert_eq!(
            cache.status(WHISPER_KIT, r),
            CacheStatus::Cached,
            "{prefab}"
        );
        let resolved = cache.resolve(WHISPER_KIT, r).unwrap();
        assert_eq!(resolved.provenance, Provenance::Cached);
        assert_eq!(resolved.path, bundled_vocabulary_path(&ids));
        assert!(resolved.path.starts_with(bind_whisper_onnx::cache_dir()));
        let ranks = vocabulary.load(&cache).unwrap();
        // The layout's first special id is the rank count: 50257 for
        // `multilingual.tiktoken`, 50256 for `gpt2.tiktoken`.
        assert_eq!(ranks.len(), ids.n_base, "{prefab}");
    }
}

/// Against upstream's own `non_speech_tokens` and `encode(" ")`.
#[test]
fn test_multilingual_matches_upstream() {
    let ids = *WhisperTokenLayout::from_vocab_size(51865).unwrap().ids();
    let ranks = bundled_vocabulary(&ids);
    assert_eq!(blank_token(&ranks), Some(220));
    assert_eq!(
        non_speech_tokens(&ranks),
        vec![
            1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93,
            359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246,
            3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929,
            11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362,
            18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470,
            36865, 42863, 47425, 49870, 50254,
        ]
    );

    let all = default_suppress_tokens(&ranks, &ids);
    assert!(all.windows(2).all(|w| w[0] < w[1]), "sorted and unique");
    for id in [
        ids.transcribe,
        ids.translate,
        ids.sot,
        ids.sot_prev,
        ids.sot_lm,
        ids.no_speech,
    ] {
        assert!(all.contains(&id));
    }
    assert_eq!(all.len(), 82 + 6);
    assert_eq!(default_filters(&ranks, &ids).len(), 2);
}

#[test]
fn test_english_matches_upstream() {
    let ids = *WhisperTokenLayout::from_vocab_size(51864).unwrap().ids();
    let ranks = bundled_vocabulary(&ids);
    assert_eq!(blank_token(&ranks), Some(220));
    assert_eq!(
        non_speech_tokens(&ranks),
        vec![
            1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93,
            357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391,
            1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329,
            7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306,
            16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409,
            34949, 40283, 40493, 40549, 47282, 49146,
        ]
    );
}

/// Detokenized, with ids produced by `whisper.tokenizer`. The files come
/// through the pretrained resolver, as a caller's would.
#[test]
fn test_multilingual_vocabulary() {
    let ranks = bundled_vocabulary(&WhisperSpecialIds::from_vocab_size(51865).unwrap());
    assert_eq!(
        ranks.len(),
        WhisperTokenLayoutConfig::new().multilingual_base_ranks
    );
    assert_eq!(ranks.get(50256), Some(&b""[..]), "the empty token");
    assert_eq!(
        ranks.get(50255),
        Some("场".as_bytes()),
        "rank 50255 is `5Zy6`"
    );

    // The layout the checkpoint implies is the layout the file has.
    let token_layout = WhisperTokenLayout::from_vocab_size(51865).unwrap();
    let detok = token_layout.detokenizer(&ranks).unwrap();
    assert_eq!(detok.vocab_size(), 51865);

    // Plain text, including non-ASCII.
    let sentence = [
        15947, 1002, 11, 341, 307, 257, 1500, 295, 264, 1667, 15487, 303, 979, 19866, 3466, 220,
        27311, 31348, 886, 13,
    ];
    assert_eq!(
        detok.detokenize(&sentence).unwrap(),
        "Hello world, this is a test of the naïve decoder — 日本語 too.",
    );

    // A codepoint split across two ids: 19034 is `b" \xf0\x9f\x8e"`
    // and 231 is `b"\x89"`.
    assert_eq!(
        detok.detokenize(&[453, 19034, 231, 1096]).unwrap(),
        "ok 🎉 done"
    );

    // A window as the model emits it, with and without its
    // specials.
    let window = [50258, 50259, 50359, 50364, 15947, 1002, 13, 50464, 50257];
    assert_eq!(
        detok.detokenize(&window).unwrap(),
        "<|startoftranscript|><|en|><|transcribe|><|0.00|>Hello world.<|2.00|><|endoftext|>",
    );
    assert_eq!(
        detok.detokenize(&token_layout.text_ids(&window)).unwrap(),
        "Hello world."
    );

    // Every id renders; none is a hole.
    let all: Vec<i64> = (0..51865).collect();
    assert!(detok.detokenize(&all).is_ok());
}

#[test]
fn test_english_vocabulary() {
    let ids = WhisperSpecialIds::from_vocab_size(51864).unwrap();
    let ranks = bundled_vocabulary(&ids);
    assert_eq!(
        ranks.len(),
        WhisperTokenLayoutConfig::new().english_base_ranks
    );
    assert_eq!(ranks.get(50255), Some(&b" gazed"[..]));

    let policy = WhisperTokenLayout::from_vocab_size(51864).unwrap();
    let detok = policy
        .load_detokenizer(bundled_vocabulary_path(&ids))
        .unwrap();
    assert_eq!(detok.vocab_size(), 51864);

    // The same text, tokenized differently: this vocabulary splits
    // the party popper across three ids.
    assert_eq!(detok.detokenize(&[15496, 995, 13]).unwrap(), "Hello world.");
    assert_eq!(
        detok.detokenize(&[482, 12520, 236, 231, 1760]).unwrap(),
        "ok 🎉 done"
    );
    assert_eq!(
        detok.detokenize(&[50257, 50362, 50363, 50256]).unwrap(),
        "<|startoftranscript|><|notimestamps|><|0.00|><|endoftext|>",
    );

    // The multilingual layout does not fit this file.
    let wrong = WhisperTokenLayout::from_vocab_size(51865).unwrap();
    assert!(wrong.detokenizer(&ranks).is_err());
}
