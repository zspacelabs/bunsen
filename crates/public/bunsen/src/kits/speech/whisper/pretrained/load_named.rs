//! # From a model's name to a loaded model
//!
//! [`resolve_model`] takes what a `--model` flag was given &mdash;
//! `well-known:openai/base`, `openai/base`, `large`, or a path &mdash; and
//! [`load_named`] takes it the rest of the way through
//! [`WhisperConstruct`]: every resource of its map into the cache, the
//! checkpoint through the scanner and past a check that it has the
//! geometry its prefab promised, the vocabulary through the rank parser,
//! and out as a
//! [`WhisperBundle`](crate::kits::speech::whisper::driver::WhisperBundle)
//! behind an `Arc`.

use std::sync::Arc;

use burn::prelude::Backend;

use crate::{
    data::pretrained::{
        Loaded,
        PretrainedCache,
        PretrainedFactory,
        PretrainedRef,
    },
    errors::BunsenResult,
    kits::speech::whisper::{
        driver::WhisperBundle,
        pretrained::{
            CHECKPOINT,
            WELL_KNOWN_TABLE,
            WHISPER_KIT,
            WhisperConstruct,
        },
    },
};

/// The factory over the well-known table.
fn factory() -> PretrainedFactory {
    PretrainedFactory::new(WHISPER_KIT)
        .with_provider(Arc::new(WELL_KNOWN_TABLE.to_table()))
        .unwrap_or_else(|e| panic!("{e}"))
}

/// Resolves a model spec against the well-known table:
/// `well-known:openai/base`, `openai/base`, a bare name or alias, or a
/// path to a checkpoint, which becomes a one-resource map under
/// [`CHECKPOINT`].
///
/// # Errors
/// As [`PretrainedFactory::resolve`].
pub fn resolve_model(spec: &str) -> BunsenResult<PretrainedRef> {
    factory().resolve(spec, Some(CHECKPOINT))
}

/// [`resolve_model`] then [`PretrainedRef::load`]: a name or a path to a
/// loaded bundle, with every resource that went into it.
///
/// # Errors
/// As [`resolve_model`] and [`PretrainedRef::load`].
pub fn load_named<B: Backend>(
    spec: &str,
    cache: &PretrainedCache,
    hook: &WhisperConstruct,
    device: &B::Device,
) -> BunsenResult<Loaded<WhisperBundle<B>>> {
    resolve_model(spec)?.load::<B, _>(cache, hook, device)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::errors::BunsenError;

    #[test]
    fn test_resolve_names_aliases_and_paths() {
        match resolve_model("openai/tiny.en").unwrap() {
            PretrainedRef::Named {
                provider,
                pretrained,
            } => {
                assert_eq!(provider, "well-known");
                assert_eq!(pretrained.name, "openai/tiny.en");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            resolve_model("well-known:openai/tiny.en").unwrap().id(),
            "well-known:openai/tiny.en"
        );
        match resolve_model("turbo").unwrap() {
            PretrainedRef::Named { pretrained, .. } => {
                assert_eq!(pretrained.name, "openai/large-v3-turbo");
            }
            other => panic!("{other:?}"),
        }

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("ckpt.pt");
        fs::write(&file, b"x").unwrap();
        match resolve_model(file.to_str().unwrap()).unwrap() {
            PretrainedRef::Given(map) => {
                assert_eq!(map.keys(), [CHECKPOINT]);
                assert_eq!(map.get(CHECKPOINT).unwrap().file, "ckpt.pt");
            }
            other => panic!("{other:?}"),
        }

        assert!(matches!(
            resolve_model("openai/gigantic"),
            Err(BunsenError::ResourceNotFound(_))
        ));
        assert!(matches!(
            resolve_model("/no/such/file.pt"),
            Err(BunsenError::ResourceNotFound(_))
        ));
    }
}
