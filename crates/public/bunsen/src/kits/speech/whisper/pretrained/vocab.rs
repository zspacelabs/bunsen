//! # Whisper vocabularies
//!
//! The rank file a token layout selects, brought local through the cache
//! and parsed: [`WhisperVocabulary::load`]. The files themselves are the
//! two maps [`MULTILINGUAL_VOCABULARY`](super::MULTILINGUAL_VOCABULARY) and
//! [`GPT2_VOCABULARY`](super::GPT2_VOCABULARY), pinned to the commit that
//! last touched them.

use crate::{
    data::pretrained::PretrainedCache,
    errors::BunsenResult,
    kits::{
        speech::whisper::pretrained::{
            VOCABULARY,
            WHISPER_KIT,
            WhisperVocabulary,
        },
        tokens::TiktokenRanks,
    },
};

/// The commit of `openai/whisper` the rank files are pinned to: the one that
/// last touched them ("Use tiktoken", openai/whisper #1044), so the URL names
/// one file forever.
pub const OPENAI_VOCAB_REVISION: &str = "839639a223b92ad61851baae9ad8a695ccb41ce5";

impl WhisperVocabulary {
    /// The rank file, brought local through the cache and parsed: the
    /// cache, or one 800 KB fetch from the pinned commit.
    ///
    /// # Errors
    /// As [`PretrainedCache::resolve`], and [`TiktokenRanks::load`] if the
    /// file cannot be parsed.
    pub fn load(
        self,
        cache: &PretrainedCache,
    ) -> BunsenResult<TiktokenRanks> {
        let map = self.map().to_map();
        let resolved = cache.resolve(WHISPER_KIT, map.try_get(VOCABULARY)?)?;
        TiktokenRanks::load(&resolved.path)
    }
}

/// Test support: the bundled files, through a cache rooted at the bundle's
/// directory, offline.
///
/// `bunsen-bundled-whisper` lays its build output out as a pretrained
/// cache, so a cache pointed at it finds every bundled asset `Cached`:
/// nothing is fetched, nothing is written. Reachable as
/// `crate::kits::speech::whisper::pretrained::testing::*`.
#[cfg(all(test, feature = "whisper-weights"))]
pub(crate) mod testing {
    use std::path::PathBuf;

    use super::*;
    use crate::{
        data::{
            cache::BunsenDiskCacheOptions,
            pretrained::PretrainedCacheOptions,
        },
        kits::speech::whisper::driver::WhisperSpecialIds,
    };

    /// A cache rooted at the bundle's directory, offline.
    pub fn offline_cache() -> PretrainedCache {
        PretrainedCache::new(
            PretrainedCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(bunsen_bundled_whisper::cache_dir().to_path_buf()))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap()
    }

    /// The bundled rank file a layout selects.
    pub fn bundled_vocabulary_path(ids: &WhisperSpecialIds) -> PathBuf {
        let map = WhisperVocabulary::for_layout(ids).map().to_map();
        offline_cache()
            .resolve(WHISPER_KIT, map.try_get(VOCABULARY).unwrap())
            .unwrap()
            .path
    }

    /// The bundled rank file a layout selects, parsed.
    pub fn bundled_vocabulary(ids: &WhisperSpecialIds) -> TiktokenRanks {
        WhisperVocabulary::for_layout(ids)
            .load(&offline_cache())
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::whisper::pretrained::OPENAI_VOCABULARIES_MAPS;

    #[cfg(feature = "whisper-weights")]
    fn ids_of(prefab: &str) -> crate::kits::speech::whisper::driver::WhisperSpecialIds {
        use crate::kits::speech::whisper::pretrained::WHISPER_PREFABS;
        let cfg = WHISPER_PREFABS.expect_lookup_prefab(prefab).to_config();
        *cfg.token_layout
            .policy_for_vocab(cfg.vocab_size)
            .unwrap()
            .ids()
    }

    /// Both maps fetch from the pinned commit, by the file's own name.
    #[test]
    fn test_the_maps_pin_the_commit() {
        for map in OPENAI_VOCABULARIES_MAPS {
            let owned = map.to_map();
            let r = owned.try_get(VOCABULARY).unwrap();
            assert!(r.is_pinned(), "{}", map.name);
            for url in r.urls() {
                assert!(url.contains(OPENAI_VOCAB_REVISION), "{url}");
                assert!(url.ends_with(&format!("/{}", r.file)), "{url}");
            }
        }
    }

    /// With the bundle, both rank files resolve offline from its
    /// directory and parse to the layout's base rank count.
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
            assert!(
                resolved
                    .path
                    .starts_with(bunsen_bundled_whisper::cache_dir())
            );
            let ranks = vocabulary.load(&cache).unwrap();
            // The layout's first special id is the rank count: 50257 for
            // `multilingual.tiktoken`, 50256 for `gpt2.tiktoken`.
            assert_eq!(ranks.len(), ids.n_base, "{prefab}");
        }
    }
}
