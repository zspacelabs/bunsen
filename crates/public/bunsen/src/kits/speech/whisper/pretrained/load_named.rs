//! # From a model's name to a loaded model
//!
//! [`resolve_model`] takes what a `--model` flag was given &mdash;
//! `openai/base`, `large`, or a path &mdash; and [`load_named`] takes it the
//! rest of the way through [`WhisperConstruct`](super::WhisperConstruct):
//! every resource of its map into the cache, the checkpoint through the
//! scanner and past a check that it has the geometry its prefab promised,
//! the vocabulary through the rank parser, and out as a
//! [`WhisperBundle`](crate::kits::speech::whisper::driver::WhisperBundle)
//! behind an `Arc`.
//!
//! [`scan_model`] is the read-only half, for a listing that wants the
//! geometry without the weights.

use std::path::Path;

use burn::prelude::Backend;

use crate::{
    data::pretrained::{
        Loaded,
        PreFabConfig,
        PretrainedCache,
        PretrainedRef,
        ResourceMap,
        load_map,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::whisper::{
        WhisperApiConfig,
        driver::WhisperBundle,
        pretrained::{
            CHECKPOINT,
            PytorchWhisperScanner,
            WHISPER_PREFABS,
            WHISPER_PROVIDERS,
            WhisperConstruct,
        },
    },
};

/// Resolves a model spec against [`WHISPER_PROVIDERS`]: `provider/name`, a
/// bare name or alias, or a path to a checkpoint, which becomes a
/// one-resource map under [`CHECKPOINT`].
///
/// # Errors
/// As [`PretrainedRef::resolve`].
pub fn resolve_model(spec: &str) -> BunsenResult<PretrainedRef> {
    PretrainedRef::resolve(WHISPER_PROVIDERS, spec, CHECKPOINT)
}

/// Checks a scanned config against the prefab a name promised.
///
/// # Errors
/// [`BunsenError::Invalid`] naming both geometries: the file at that name is
/// not the model it claims to be.
pub fn check_geometry(
    id: &str,
    prefab: &PreFabConfig<WhisperApiConfig>,
    scanned: &WhisperApiConfig,
) -> BunsenResult<()> {
    let expected = prefab.to_config().geometry();
    let found = scanned.geometry();
    if expected == found {
        Ok(())
    } else {
        Err(BunsenError::Invalid(format!(
            "{id}: the checkpoint's geometry is {found:?}, not prefab {:?}'s {expected:?}",
            prefab.name
        )))
    }
}

/// Scans a checkpoint for its config without loading its weights, and
/// checks it against the prefab if the model was named.
///
/// # Errors
/// As [`PytorchWhisperScanner::scan_cfg`] and [`check_geometry`].
pub fn scan_model(
    model: &PretrainedRef,
    path: &Path,
) -> BunsenResult<WhisperApiConfig> {
    scan_model_with(model, path, &PytorchWhisperScanner::new())
}

/// [`scan_model`] through a configured scanner: for a checkpoint whose
/// tensors are not under `model_state_dict`, or a front end or token layout
/// that is not upstream's.
///
/// # Errors
/// As [`scan_model`].
pub fn scan_model_with(
    model: &PretrainedRef,
    path: &Path,
    scanner: &PytorchWhisperScanner,
) -> BunsenResult<WhisperApiConfig> {
    let (_, cfg) = scanner.scan_cfg(path)?;
    if let Some(prefab) = model.prefab(&WHISPER_PREFABS) {
        check_geometry(&model.id(), &prefab, &cfg)?;
    }
    Ok(cfg)
}

/// Loads a resolved model through `hook` from `map`.
///
/// The map is the model's, possibly overlaid by the caller (a `--vocab`);
/// the hook is the caller's, and is told the geometry the model's prefab
/// promises, so a checkpoint under the wrong name is refused before its
/// tensors are read.
///
/// # Errors
/// As [`load_map`].
pub fn load_model<B: Backend>(
    model: &PretrainedRef,
    map: ResourceMap,
    cache: &PretrainedCache,
    hook: &WhisperConstruct,
    device: &B::Device,
) -> BunsenResult<Loaded<WhisperBundle<B>>> {
    let hook = hook.clone().expecting(model);
    load_map::<B, _>(map, cache, &hook, device)
}

/// [`resolve_model`] then [`load_model`]: a name or a path to a loaded
/// bundle, with every resource that went into it.
///
/// # Errors
/// As [`resolve_model`] and [`load_model`].
pub fn load_named<B: Backend>(
    spec: &str,
    cache: &PretrainedCache,
    hook: &WhisperConstruct,
    device: &B::Device,
) -> BunsenResult<Loaded<WhisperBundle<B>>> {
    let model = resolve_model(spec)?;
    load_model::<B>(&model, model.to_map(), cache, hook, device)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn test_resolve_names_aliases_and_paths() {
        match resolve_model("openai/tiny.en").unwrap() {
            PretrainedRef::Named {
                provider,
                pretrained,
            } => {
                assert_eq!(provider.name, "openai");
                assert_eq!(pretrained.name, "tiny.en");
            }
            other => panic!("{other:?}"),
        }
        match resolve_model("turbo").unwrap() {
            PretrainedRef::Named { pretrained, .. } => {
                assert_eq!(pretrained.name, "large-v3-turbo");
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

    #[test]
    fn test_check_geometry_accepts_its_own_prefab() {
        for prefab in WHISPER_PREFABS.iter() {
            let prefab = prefab.to_prefab();
            check_geometry("x", &prefab, &prefab.to_config()).unwrap();
        }
    }

    /// The bundled checkpoint is `openai/base`; scanning it must agree with
    /// the `base` prefab, which is what pins the prefab table to a real
    /// file.
    #[cfg(feature = "whisper-weights")]
    #[test]
    fn test_the_bundled_base_scans_as_the_base_prefab() {
        use crate::kits::speech::whisper::pretrained::prefab_for_geometry;
        let model = resolve_model("openai/base").unwrap();
        let cfg = scan_model(&model, bunsen_bundled_whisper::base_pt()).unwrap();

        let geometry = cfg.geometry();
        assert_eq!(prefab_for_geometry(&geometry).map(|p| p.name), Some("base"));
        assert_eq!(geometry.n_heads(), 8);
    }

    /// The scanner is honored: one that declares another head size scans
    /// the bundled file to another geometry, which a named model rejects
    /// and a given one reports.
    #[cfg(feature = "whisper-weights")]
    #[test]
    fn test_scan_model_with_honors_the_scanner() {
        let base = bunsen_bundled_whisper::base_pt();
        let scanner = PytorchWhisperScanner::new().with_d_head(32);

        let named = resolve_model("openai/base").unwrap();
        let err = scan_model_with(&named, base, &scanner).unwrap_err();
        assert!(matches!(err, BunsenError::Invalid(_)), "{err}");

        let given = PretrainedRef::Given(ResourceMap::given("base", CHECKPOINT, base));
        let cfg = scan_model_with(&given, base, &scanner).unwrap();
        assert_eq!(cfg.geometry().d_head, 32);
        assert_eq!(cfg.geometry().n_heads(), 16);
    }

    /// `base.pt` scanned as if it were `openai/tiny`: same file, wrong
    /// promise.
    #[cfg(feature = "whisper-weights")]
    #[test]
    fn test_a_checkpoint_under_the_wrong_name_is_rejected() {
        let model = resolve_model("openai/tiny").unwrap();
        let err = scan_model(&model, bunsen_bundled_whisper::base_pt()).unwrap_err();
        assert!(matches!(err, BunsenError::Invalid(_)), "{err}");
        assert!(err.to_string().contains("openai/tiny"), "{err}");
    }
}
