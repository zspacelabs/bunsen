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
    /// The files are the two maps
    /// [`MULTILINGUAL_VOCABULARY`](super::MULTILINGUAL_VOCABULARY) and
    /// [`GPT2_VOCABULARY`](super::GPT2_VOCABULARY), pinned to
    /// [`OPENAI_VOCAB_REVISION`]. A loaded bundle already carries its
    /// vocabulary; this is for a caller that wants the ranks alone.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::whisper::pretrained::OPENAI_VOCABULARIES_MAPS;

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
}
