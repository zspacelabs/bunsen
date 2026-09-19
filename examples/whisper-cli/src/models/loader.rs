//! From a model's name to a loaded model.
//!
//! [`ModelRef::resolve`] takes what `--model` was given &mdash; `openai/base`,
//! `large`, or a path &mdash; and [`load_model`] takes it the rest of the
//! way: into the cache, through bunsen's scanner, and past a check that the
//! checkpoint has the geometry its prefab promised.
//!
//! The index it resolves against is bunsen's:
//! [`WHISPER_PROVIDERS`] over [`WHISPER_PREFABS`], through a
//! [`WeightsCache`].

use std::path::{
    Path,
    PathBuf,
};

use bunsen::{
    data::pretrained::{
        PreFabConfig,
        Provenance,
        ResolvedWeights,
        StaticPretrainedProvider,
        StaticPretrainedWeightsDescriptor,
        WeightsCache,
        available_ids,
        lookup_pretrained,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::whisper::{
        Whisper,
        WhisperApiConfig,
        pretrained::{
            PytorchWhisperScanner,
            WHISPER_KIT,
            WHISPER_PREFABS,
            WHISPER_PROVIDERS,
        },
    },
};
use burn::prelude::Backend;

/// What a model name resolved to.
#[derive(Debug, Clone)]
pub enum ModelRef {
    /// An entry in the pretrained index.
    Pretrained {
        /// Its provider.
        provider: &'static StaticPretrainedProvider<'static>,
        /// The entry.
        pretrained: &'static StaticPretrainedWeightsDescriptor<'static>,
    },

    /// A checkpoint on disk, as upstream's `load_model` also accepts. No
    /// prefab is promised; the scan says what it is.
    Path(PathBuf),
}

impl ModelRef {
    /// Resolves a `--model` argument.
    ///
    /// `provider/name` looks up that provider; a bare name looks across all
    /// of them; an alias (`large`, `turbo`) is honoured. Failing those, a
    /// path to an existing file is taken as a checkpoint. The index wins
    /// over the file system, as upstream's does.
    ///
    /// # Errors
    /// [`BunsenError::ResourceNotFound`], naming what is available.
    pub fn resolve(spec: &str) -> BunsenResult<Self> {
        let (provider, name) = match spec.rsplit_once('/') {
            Some((provider, name)) => (Some(provider), name),
            None => (None, spec),
        };
        if let Some((provider, pretrained)) = lookup_pretrained(WHISPER_PROVIDERS, provider, name) {
            return Ok(Self::Pretrained {
                provider,
                pretrained,
            });
        }

        let path = Path::new(spec);
        if path.is_file() {
            return Ok(Self::Path(path.to_path_buf()));
        }

        Err(BunsenError::ResourceNotFound(format!(
            "no model {spec:?}: not a pretrained name and not a file; the names are {}",
            available_ids(WHISPER_PROVIDERS).join(", ")
        )))
    }

    /// The qualified id, or the path.
    pub fn id(&self) -> String {
        match self {
            Self::Pretrained {
                provider,
                pretrained,
            } => provider.id(pretrained),
            Self::Path(path) => path.display().to_string(),
        }
    }

    /// The prefab the name promised, if it was a name.
    ///
    /// # Panics
    /// If the table names a prefab that does not exist; bunsen's tests pin
    /// that every entry's does.
    pub fn prefab(&self) -> Option<PreFabConfig<WhisperApiConfig>> {
        match self {
            Self::Pretrained { pretrained, .. } => {
                Some(WHISPER_PREFABS.expect_lookup_prefab(pretrained.prefab))
            }
            Self::Path(_) => None,
        }
    }

    /// Brings the weights local.
    ///
    /// # Errors
    /// As [`WeightsCache::resolve`]. A path is local already.
    pub fn locate(
        &self,
        cache: &WeightsCache,
    ) -> BunsenResult<ResolvedWeights> {
        match self {
            Self::Pretrained {
                provider,
                pretrained,
            } => cache.resolve(WHISPER_KIT, provider.name, &pretrained.to_descriptor()),
            Self::Path(path) => Ok(ResolvedWeights {
                path: path.clone(),
                provenance: Provenance::Given,
            }),
        }
    }
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
    model: &ModelRef,
    path: &Path,
) -> BunsenResult<WhisperApiConfig> {
    let (_, cfg) = PytorchWhisperScanner::new().scan_cfg(path)?;
    if let Some(prefab) = model.prefab() {
        check_geometry(&model.id(), &prefab, &cfg)?;
    }
    Ok(cfg)
}

/// Loads a model: locates the weights, scans them, checks the prefab, and
/// materializes the module at the precision the checkpoint ships in.
///
/// The scan runs before the load so a mismatch is caught before 3 GB of
/// tensors are read for nothing.
///
/// # Errors
/// As [`ModelRef::locate`], [`scan_model`] and
/// [`PytorchWhisperScanner::load`].
pub fn load_model<B: Backend>(
    model: &ModelRef,
    cache: &WeightsCache,
    device: &B::Device,
) -> BunsenResult<(Whisper<B>, WhisperApiConfig)> {
    let located = model.locate(cache)?;
    log::info!(
        "{}: {} ({})",
        model.id(),
        located.path.display(),
        located.provenance
    );
    scan_model(model, &located.path)?;
    PytorchWhisperScanner::new().load::<B, _>(&located.path, device)
}

#[cfg(test)]
mod tests {
    use bunsen::kits::speech::whisper::pretrained::{
        bundled,
        prefab_for_geometry,
    };

    use super::*;

    #[test]
    fn test_resolve_names_aliases_and_paths() {
        match ModelRef::resolve("openai/tiny.en").unwrap() {
            ModelRef::Pretrained {
                provider,
                pretrained,
            } => {
                assert_eq!(provider.name, "openai");
                assert_eq!(pretrained.name, "tiny.en");
            }
            other => panic!("{other:?}"),
        }
        match ModelRef::resolve("turbo").unwrap() {
            ModelRef::Pretrained { pretrained, .. } => {
                assert_eq!(pretrained.name, "large-v3-turbo");
            }
            other => panic!("{other:?}"),
        }

        let bundled = bundled::base_pt();
        match ModelRef::resolve(bundled.to_str().unwrap()).unwrap() {
            ModelRef::Path(path) => assert_eq!(path, bundled),
            other => panic!("{other:?}"),
        }

        assert!(matches!(
            ModelRef::resolve("openai/gigantic"),
            Err(BunsenError::ResourceNotFound(_))
        ));
        assert!(matches!(
            ModelRef::resolve("/no/such/file.pt"),
            Err(BunsenError::ResourceNotFound(_))
        ));
    }

    /// The bundled checkpoint is `openai/base`; scanning it must agree with
    /// the `base` prefab, which is what pins the prefab table to a real
    /// file.
    #[test]
    fn test_the_bundled_base_scans_as_the_base_prefab() {
        let model = ModelRef::resolve("openai/base").unwrap();
        let cfg = scan_model(&model, bundled::base_pt()).unwrap();

        let geometry = cfg.geometry();
        assert_eq!(prefab_for_geometry(&geometry).map(|p| p.name), Some("base"));
        assert_eq!(geometry.n_heads(), 8);
    }

    #[test]
    fn test_a_checkpoint_under_the_wrong_name_is_rejected() {
        // `base.pt` scanned as if it were `openai/tiny`: same file, wrong
        // promise.
        let model = ModelRef::resolve("openai/tiny").unwrap();
        let err = scan_model(&model, bundled::base_pt()).unwrap_err();
        assert!(matches!(err, BunsenError::Invalid(_)), "{err}");
        assert!(err.to_string().contains("openai/tiny"), "{err}");
    }

    #[test]
    fn test_check_geometry_accepts_its_own_prefab() {
        for prefab in WHISPER_PREFABS.items {
            let prefab = prefab.to_prefab();
            check_geometry("x", &prefab, &prefab.to_config()).unwrap();
        }
    }
}
