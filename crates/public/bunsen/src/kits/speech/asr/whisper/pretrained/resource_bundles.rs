//! # Whisper resource bundles

use crate::{
    data::pretrained::ResourceBundle,
    kits::speech::asr::whisper::driver::WhisperSpecialIds,
};

/// The key of the checkpoint in a Whisper resource bundle.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of the `.tiktoken` vocabulary in a Whisper resource bundle.
pub const VOCABULARY: &str = "vocabulary";

/// The namespace every `openai` resource is cached under:
/// `pretrained/whisper/openai/<sha256>/<file>`.
pub const OPENAI_NAMESPACE: &str = "openai";

/// The label on every `openai` checkpoint.
pub const PYTORCH_FP16: &str = "pytorch fp16";

/// The label on the vocabularies.
pub const TIKTOKEN: &str = "tiktoken";

/// Which of the two rank files a checkpoint decodes through:
/// `multilingual.tiktoken` for a multilingual layout, `gpt2.tiktoken` for
/// an English-only one.
///
/// [`for_layout`](Self::for_layout) is the rule a path model's vocabulary
/// is derived by, and a named row's is checked against;
/// [`resource_bundle`](Self::resource_bundle) is the file, as a resource
/// bundle. The two files number their tokens differently, and a checkpoint
/// decoded through the wrong one produces text that is wrong without being
/// obviously so, so the rule takes the token layout, which comes from the
/// checkpoint's vocabulary size, and the pairing stays out of the caller's
/// hands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhisperVocabulary {
    /// `multilingual.tiktoken`: every multilingual checkpoint's.
    Multilingual,

    /// `gpt2.tiktoken`: the English-only (`*.en`) checkpoints'.
    Gpt2,
}

impl WhisperVocabulary {
    /// Both, multilingual first.
    pub const ALL: [Self; 2] = [Self::Multilingual, Self::Gpt2];

    /// The vocabulary a token layout selects.
    pub fn for_layout(ids: &WhisperSpecialIds) -> Self {
        if ids.is_multilingual() {
            Self::Multilingual
        } else {
            Self::Gpt2
        }
    }

    /// The rank file's full name in the built-in catalog:
    /// `openai/whisper/multilingual.tiktoken` or
    /// `openai/whisper/gpt2.tiktoken`.
    pub fn resource_name(self) -> &'static str {
        match self {
            Self::Multilingual => "openai/whisper/multilingual.tiktoken",
            Self::Gpt2 => "openai/whisper/gpt2.tiktoken",
        }
    }

    /// The rank file, as a resource bundle, pinned to
    /// [`OPENAI_VOCAB_REVISION`](super::OPENAI_VOCAB_REVISION).
    /// `multilingual.tiktoken`'s last line is `= 50256`, base64 of nothing,
    /// and that empty token is real; bunsen's parser reads it as such.
    ///
    /// # Panics
    /// When the built-in catalog lacks it, a bug in the catalog.
    pub fn resource_bundle(self) -> ResourceBundle {
        crate::data::pretrained::builtin_resource_bundle(self.resource_name(), VOCABULARY)
            .unwrap_or_else(|e| panic!("the Whisper vocabulary {}: {e}", self.resource_name()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::pretrained::{
            LocalDirBase,
            Source,
            resolve_dir_template,
        },
        kits::speech::asr::whisper::pretrained::{
            OPENAI_LOCAL_DIR,
            WHISPER_CATALOG,
            WHISPER_PREFABS,
        },
    };

    fn ids_of(prefab: &str) -> WhisperSpecialIds {
        let cfg = WHISPER_PREFABS.expect_lookup_prefab(prefab).to_config();
        *cfg.token_layout
            .policy_for_vocab(cfg.vocab_size)
            .unwrap()
            .ids()
    }

    /// Every checkpoint resource bundle is one pinned, labeled `checkpoint`
    /// under the `openai` namespace, found under upstream's root before
    /// upstream's URL, and that URL is addressed by the digest that pins it;
    /// upstream's root is `$XDG_CACHE_HOME/whisper`, else `~/.cache/whisper`.
    #[test]
    fn test_checkpoint_bundles_validate_and_pin_their_urls() {
        let checkpoints: Vec<_> = WHISPER_CATALOG
            .catalog()
            .bundles
            .iter()
            .filter(|b| b.name.ends_with(".pt"))
            .collect();
        assert_eq!(checkpoints.len(), 12);

        for bundle in checkpoints {
            let owned = bundle.to_resource_bundle();
            owned.validate().unwrap();
            assert_eq!(owned.keys(), [CHECKPOINT], "{}", bundle.name);
            let r = owned.get(CHECKPOINT).unwrap();
            assert_eq!(r.namespace, OPENAI_NAMESPACE, "{}", bundle.name);
            assert_eq!(r.kind.as_deref(), Some(PYTORCH_FP16), "{}", bundle.name);
            let sha256 = r.sha256.as_deref().expect("every checkpoint is pinned");
            assert_eq!(bundle.name, format!("openai/{}", r.file));
            assert_eq!(r.sources.len(), 2, "{}", bundle.name);
            assert!(
                matches!(&r.sources[0], Source::LocalDir { name, .. } if name == OPENAI_LOCAL_DIR),
                "{}: upstream's root comes first",
                bundle.name
            );
            assert_eq!(
                r.urls(),
                vec![format!(
                    "https://openaipublic.azureedge.net/main/whisper/models/{sha256}/{}",
                    r.file
                )],
                "{}",
                bundle.name
            );
            let crate::data::pretrained::CatalogBase::LocalDir(LocalDirBase { default, .. }) =
                &bundle.bases[0]
            else {
                panic!("{}: upstream's root is the first base", bundle.name);
            };
            assert_eq!(default, &["$XDG_CACHE_HOME/whisper", "~/.cache/whisper"]);
        }
        assert!(
            resolve_dir_template("~/.cache/whisper").is_some_and(|d| d.ends_with(".cache/whisper"))
        );
    }

    /// The vocabulary resource bundles validate, are pinned to the commit,
    /// and the layout picks between them.
    #[test]
    fn test_vocabulary_bundles_and_the_rule() {
        for vocabulary in WhisperVocabulary::ALL {
            let owned = vocabulary.resource_bundle();
            owned.validate().unwrap();
            assert_eq!(owned.keys(), [VOCABULARY], "{}", owned.name);
            let r = owned.get(VOCABULARY).unwrap();
            assert_eq!(r.namespace, OPENAI_NAMESPACE);
            assert_eq!(r.kind.as_deref(), Some(TIKTOKEN));
            assert!(r.is_pinned());
            assert_eq!(owned.name, vocabulary.resource_name());
            assert_eq!(owned.name, format!("openai/whisper/{}", r.file));
            let urls = r.urls();
            assert_eq!(urls.len(), 1);
            assert!(
                urls[0].contains("839639a223b92ad61851baae9ad8a695ccb41ce5"),
                "{}",
                urls[0]
            );
            assert!(urls[0].ends_with(&format!("/{}", r.file)), "{}", urls[0]);
        }

        let rule = |prefab: &str| WhisperVocabulary::for_layout(&ids_of(prefab));
        assert_eq!(rule("tiny"), WhisperVocabulary::Multilingual);
        assert_eq!(rule("tiny.en"), WhisperVocabulary::Gpt2);
        assert_eq!(rule("large-v3"), WhisperVocabulary::Multilingual);
        assert_eq!(
            rule("tiny").resource_name(),
            "openai/whisper/multilingual.tiktoken"
        );
        assert_eq!(
            rule("tiny.en").resource_name(),
            "openai/whisper/gpt2.tiktoken"
        );
    }
}
