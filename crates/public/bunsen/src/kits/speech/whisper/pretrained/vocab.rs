//! # Whisper vocabularies
//!
//! The two `.tiktoken` rank files behind Whisper's tokenizer, as pretrained
//! descriptors in an `openai` vocabulary provider: named, pinned to the
//! commit that last touched them, fetched into the cache like weights, and
//! read from the bundle first when the `whisper-weights` feature is on.
//!
//! The two files number their tokens differently, and a checkpoint decoded
//! through the wrong one produces text that is wrong without being
//! obviously so. [`vocabulary_for`] takes the token layout, which comes from
//! the checkpoint's vocabulary size, so the pairing stays out of the
//! caller's hands.

use crate::{
    data::pretrained::{
        StaticPretrainedProvider,
        StaticPretrainedWeightsDescriptor,
        StaticWeightsSource,
        WeightsCache,
        WeightsFormat,
    },
    errors::BunsenResult,
    kits::{
        speech::whisper::{
            driver::WhisperSpecialIds,
            pretrained::WHISPER_KIT,
        },
        tokens::TiktokenRanks,
    },
};

/// The commit of `openai/whisper` the rank files are pinned to: the one that
/// last touched them ("Use tiktoken", openai/whisper #1044), so the URL names
/// one file forever.
pub const OPENAI_VOCAB_REVISION: &str = "839639a223b92ad61851baae9ad8a695ccb41ce5";

/// The token layout a vocabulary instantiates, named where a checkpoint's
/// descriptor names its prefab.
pub const MULTILINGUAL_LAYOUT: &str = "multilingual";

/// The English-only token layout, one rank shorter than the multilingual one.
pub const ENGLISH_ONLY_LAYOUT: &str = "english-only";

/// `multilingual.tiktoken`: the vocabulary of every multilingual checkpoint.
/// Its last line is `= 50256`, base64 of nothing, and that empty token is
/// real; bunsen's parser reads it as such.
pub static MULTILINGUAL_TIKTOKEN: StaticPretrainedWeightsDescriptor<'static> =
    StaticPretrainedWeightsDescriptor {
        name: "multilingual",
        description: "the multilingual vocabulary: GPT-2's ranks plus one, as every multilingual checkpoint numbers them",
        license: Some("MIT"),
        origin: Some(
            "https://github.com/openai/whisper/tree/839639a223b92ad61851baae9ad8a695ccb41ce5/whisper/assets",
        ),
        prefab: MULTILINGUAL_LAYOUT,
        aliases: &[],
        file: "multilingual.tiktoken",
        sha256: Some("b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126"),
        format: WeightsFormat::Tiktoken,
        sources: &[
            #[cfg(feature = "whisper-weights")]
            StaticWeightsSource::File(bunsen_bundled_whisper::multilingual_tiktoken),
            StaticWeightsSource::Url(
                "https://raw.githubusercontent.com/openai/whisper/839639a223b92ad61851baae9ad8a695ccb41ce5/whisper/assets/multilingual.tiktoken",
            ),
        ],
    };

/// `gpt2.tiktoken`: the vocabulary of the English-only (`*.en`) checkpoints,
/// GPT-2's ranks as they are.
pub static GPT2_TIKTOKEN: StaticPretrainedWeightsDescriptor<'static> =
    StaticPretrainedWeightsDescriptor {
        name: "gpt2",
        description: "the English-only vocabulary: GPT-2's ranks, as the `*.en` checkpoints number them",
        license: Some("MIT"),
        origin: Some(
            "https://github.com/openai/whisper/tree/839639a223b92ad61851baae9ad8a695ccb41ce5/whisper/assets",
        ),
        prefab: ENGLISH_ONLY_LAYOUT,
        aliases: &[],
        file: "gpt2.tiktoken",
        sha256: Some("306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930"),
        format: WeightsFormat::Tiktoken,
        sources: &[
            #[cfg(feature = "whisper-weights")]
            StaticWeightsSource::File(bunsen_bundled_whisper::gpt2_tiktoken),
            StaticWeightsSource::Url(
                "https://raw.githubusercontent.com/openai/whisper/839639a223b92ad61851baae9ad8a695ccb41ce5/whisper/assets/gpt2.tiktoken",
            ),
        ],
    };

/// `OpenAI`'s Whisper vocabularies, by the token layout each instantiates.
pub static OPENAI_VOCABULARIES: StaticPretrainedProvider<'static> = StaticPretrainedProvider {
    name: "openai",
    description: "OpenAI's Whisper vocabularies: the tiktoken rank files behind the tokenizer",
    license: Some("MIT"),
    origin: Some(
        "https://github.com/openai/whisper/tree/839639a223b92ad61851baae9ad8a695ccb41ce5/whisper/assets",
    ),
    items: &[&MULTILINGUAL_TIKTOKEN, &GPT2_TIKTOKEN],
};

/// The vocabulary that matches a token layout: `multilingual.tiktoken` for a
/// multilingual layout, `gpt2.tiktoken` for an English-only one.
pub fn vocabulary_descriptor(
    ids: &WhisperSpecialIds
) -> &'static StaticPretrainedWeightsDescriptor<'static> {
    if ids.is_multilingual() {
        &MULTILINGUAL_TIKTOKEN
    } else {
        &GPT2_TIKTOKEN
    }
}

/// The rank file that matches a token layout, brought local through the
/// cache: the bundled copy in place when the `whisper-weights` feature is on,
/// the cache, or one 800 KB fetch from the pinned commit.
///
/// # Errors
/// As [`WeightsCache::resolve`], and [`TiktokenRanks::load`] if the file
/// cannot be parsed.
pub fn vocabulary_for(
    ids: &WhisperSpecialIds,
    cache: &WeightsCache,
) -> BunsenResult<TiktokenRanks> {
    let desc = vocabulary_descriptor(ids).to_descriptor();
    let resolved = cache.resolve(WHISPER_KIT, OPENAI_VOCABULARIES.name, &desc)?;
    TiktokenRanks::load(&resolved.path)
}

/// Test support: the bundled rank files, through the cache, offline.
///
/// With `whisper-weights` on, both files are `File` sources, so an offline
/// cache in a scratch directory resolves them in place: nothing is written,
/// nothing is fetched, and the path handed back is the bundle's. Reachable
/// as `crate::kits::speech::whisper::pretrained::testing::*`.
#[cfg(all(test, feature = "whisper-weights"))]
pub(crate) mod testing {
    use std::path::PathBuf;

    use super::*;
    use crate::data::{
        cache::BunsenDiskCacheOptions,
        pretrained::WeightsCacheOptions,
    };

    /// An offline cache in a scratch directory, which lives as long as the
    /// returned guard.
    pub fn offline_cache() -> (tempfile::TempDir, WeightsCache) {
        let dir = tempfile::tempdir().unwrap();
        let cache = WeightsCache::new(
            WeightsCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap();
        (dir, cache)
    }

    /// The bundled rank file a layout selects.
    pub fn bundled_vocabulary_path(ids: &WhisperSpecialIds) -> PathBuf {
        let (_dir, cache) = offline_cache();
        let desc = vocabulary_descriptor(ids).to_descriptor();
        cache
            .resolve(WHISPER_KIT, OPENAI_VOCABULARIES.name, &desc)
            .unwrap()
            .path
    }

    /// The bundled rank file a layout selects, parsed.
    pub fn bundled_vocabulary(ids: &WhisperSpecialIds) -> TiktokenRanks {
        let (_dir, cache) = offline_cache();
        vocabulary_for(ids, &cache).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::pretrained::WeightsSource,
        kits::speech::whisper::pretrained::WHISPER_PREFABS,
    };

    fn ids_of(prefab: &str) -> WhisperSpecialIds {
        let cfg = WHISPER_PREFABS.expect_lookup_prefab(prefab).to_config();
        *cfg.token_layout
            .policy_for_vocab(cfg.vocab_size)
            .unwrap()
            .ids()
    }

    #[test]
    fn test_the_table_validates_and_pins_the_commit() {
        for d in OPENAI_VOCABULARIES.items {
            let owned = d.to_descriptor();
            owned.validate().unwrap();
            assert_eq!(owned.format, WeightsFormat::Tiktoken);
            assert!(owned.is_pinned());
            for url in owned.urls() {
                assert!(url.contains(OPENAI_VOCAB_REVISION), "{url}");
                assert!(url.ends_with(&owned.file), "{url}");
            }
            assert!(
                owned
                    .sources
                    .iter()
                    .any(|s| matches!(s, WeightsSource::Url(_)))
            );
        }
        assert_eq!(
            OPENAI_VOCABULARIES.ids(),
            vec!["openai/multilingual", "openai/gpt2"]
        );
        assert_eq!(OPENAI_VOCABULARIES.for_prefab(MULTILINGUAL_LAYOUT).len(), 1);
        assert_eq!(OPENAI_VOCABULARIES.for_prefab(ENGLISH_ONLY_LAYOUT).len(), 1);
    }

    /// The layout picks the file: multilingual checkpoints get
    /// `multilingual.tiktoken`, English-only ones `gpt2.tiktoken`.
    #[test]
    fn test_layout_picks_the_vocabulary() {
        assert_eq!(vocabulary_descriptor(&ids_of("tiny")).name, "multilingual");
        assert_eq!(vocabulary_descriptor(&ids_of("tiny.en")).name, "gpt2");
        assert_eq!(
            vocabulary_descriptor(&ids_of("large-v3")).name,
            "multilingual"
        );
    }

    /// With the bundle on, both rank files resolve offline in place and parse.
    #[cfg(feature = "whisper-weights")]
    #[test]
    fn test_bundled_vocabularies_resolve_offline() {
        use crate::{
            data::pretrained::{
                CacheStatus,
                Provenance,
            },
            kits::speech::whisper::pretrained::testing::{
                bundled_vocabulary_path,
                offline_cache,
            },
        };
        let (dir, cache) = offline_cache();
        for prefab in ["tiny", "tiny.en"] {
            let ids = ids_of(prefab);
            let desc = vocabulary_descriptor(&ids).to_descriptor();
            assert_eq!(
                cache.status(WHISPER_KIT, "openai", &desc),
                CacheStatus::File
            );
            let resolved = cache.resolve(WHISPER_KIT, "openai", &desc).unwrap();
            assert_eq!(resolved.provenance, Provenance::File);
            assert_eq!(resolved.path, bundled_vocabulary_path(&ids));
            let ranks = vocabulary_for(&ids, &cache).unwrap();
            // The layout's first special id is the rank count: 50257 for
            // `multilingual.tiktoken`, 50256 for `gpt2.tiktoken`.
            assert_eq!(ranks.len(), ids.n_base, "{prefab}");
        }
        assert!(!dir.path().join("cache").join("weights").exists());
    }
}
