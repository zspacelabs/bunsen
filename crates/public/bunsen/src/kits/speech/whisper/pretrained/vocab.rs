//! # Whisper vocabularies

use crate::{
    data::pretrained::PretrainedCache,
    errors::BunsenResult,
    kits::{
        speech::whisper::pretrained::{
            VOCABULARY,
            WHISPER_KIT,
            WhisperVocabulary,
        },
        support::tokens::TiktokenRanks,
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
    /// The files are the two vocabulary bundles of
    /// [`WHISPER_CATALOG`](super::WHISPER_CATALOG), pinned to
    /// [`OPENAI_VOCAB_REVISION`]. A loaded construct already carries its
    /// vocabulary; this is for a caller that wants the ranks alone.
    ///
    /// # Errors
    /// As [`PretrainedCache::resolve`], and [`TiktokenRanks::load`] if the
    /// file cannot be parsed.
    pub fn load(
        self,
        cache: &PretrainedCache,
    ) -> BunsenResult<TiktokenRanks> {
        let map = self.resource_bundle();
        let resolved = cache.resolve(WHISPER_KIT, map.try_get(VOCABULARY)?)?;
        TiktokenRanks::load(&resolved.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both bundles fetch from the pinned commit, by the file's own name.
    #[test]
    fn test_the_bundles_pin_the_commit() {
        for vocabulary in WhisperVocabulary::ALL {
            let owned = vocabulary.resource_bundle();
            let r = owned.try_get(VOCABULARY).unwrap();
            assert!(r.is_pinned(), "{}", owned.name);
            for url in r.urls() {
                assert!(url.contains(OPENAI_VOCAB_REVISION), "{url}");
                assert!(url.ends_with(&format!("/{}", r.file)), "{url}");
            }
        }
    }
}
