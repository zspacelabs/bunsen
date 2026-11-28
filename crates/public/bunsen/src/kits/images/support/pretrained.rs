//! The pretrained pathway the timm image kits share.

use alloc::{
    format,
    sync::Arc,
};
use core::fmt::Debug;

use burn::tensor::Device;

use crate::{
    data::{
        builds::{
            BuildRef,
            CONFIG,
            ConstructInit,
            GEOMETRY,
        },
        formats::LoadPolicy,
        resources::{
            LoadedResourceSet,
            ResourceSet,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
        ResultContext,
        Rule,
    },
    kits::images::support::{
        card::{
            ImageModelCard,
            ImageModelConstruct,
        },
        timm::interop::{
            TimmModelConfigMirror,
            timm_model_config_from_path,
        },
    },
};

/// An image family's policy config that a timm hub repo can state and fill,
/// and the family's checkpoint reader: what [`TimmConstructConfig`] needs of
/// a kit.
pub trait TimmImageContract:
    Clone + Debug + serde::de::DeserializeOwned + Send + Sync + 'static
{
    /// The model the config builds.
    type Model;

    /// The kit's name: its rows' `kit` tag.
    const KIT: &'static str;

    /// The kit's construct config, by name, for its errors.
    const CONSTRUCT: &'static str;

    /// The classes of the model's head.
    fn num_classes(&self) -> usize;

    /// The config of a timm `config.json`: its architecture's prefab, with
    /// its classes, pool and input onto it.
    ///
    /// # Errors
    /// The kit's: an architecture it doesn't build, a setting it can't take.
    fn from_timm_config(config: &TimmModelConfigMirror) -> BunsenResult<Self>;

    /// The model of `loaded`'s checkpoint, built from `config`.
    ///
    /// # Errors
    /// The kit's loader's.
    fn load(
        loaded: &LoadedResourceSet,
        config: &Self,
        device: &Device,
        policy: LoadPolicy,
    ) -> BunsenResult<Self::Model>;
}

/// How a timm image family's pretrained is built: the config the model is
/// initialised from before the checkpoint is read into it.
///
/// The [`ConstructInit`] hook of every timm image kit
/// (`SKNetConstructConfig` is `TimmConstructConfig<SKNetContractConfig>`). It
/// builds the family's model from the config, reads the checkpoint into it
/// ([`TimmImageContract::load`]), and reads the build's [`ImageModelCard`]:
/// its eval transform and labels, checked against the head.
///
/// The config comes from [`with_config`](Self::with_config), which wins and
/// is what a given path needs; else from the geometry the build states;
/// else, for a timm hub repo, from its `config.json`
/// ([`TimmImageContract::from_timm_config`]).
#[derive(Clone, Debug)]
pub struct TimmConstructConfig<C> {
    /// The config to build from, overriding the build's.
    pub config: Option<C>,
}

impl<C> Default for TimmConstructConfig<C> {
    fn default() -> Self {
        Self { config: None }
    }
}

impl<C: TimmImageContract> TimmConstructConfig<C> {
    /// Builds from the build's config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the config to build from, which wins over the build's.
    pub fn with_config(
        mut self,
        config: C,
    ) -> Self {
        self.config = Some(config);
        self
    }

    /// The config `model` is built from: the explicit one, else the
    /// geometry the build states, else its loaded timm `config.json`'s.
    ///
    /// # Errors
    /// As [`TimmImageContract::from_timm_config`];
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause ([`Rule::Missing`] on `config`), when there
    /// is none: a given checkpoint needs [`with_config`](Self::with_config).
    pub fn config_for(
        &self,
        model: &BuildRef,
        loaded: Option<&LoadedResourceSet>,
    ) -> BunsenResult<C> {
        if let Some(config) = &self.config {
            return Ok(config.clone());
        }
        if let Some(config) = model.stated(GEOMETRY)? {
            return Ok(config);
        }
        if let Some(timm) = loaded.and_then(|l| l.get(CONFIG)) {
            return C::from_timm_config(&timm_model_config_from_path(&timm.path)?);
        }
        Err(
            BunsenError::from(ConstraintError::new(C::CONSTRUCT, "config", Rule::Missing)).context(
                format!(
                    "{}: states no geometry and has no timm `config.json`; a given checkpoint \
                     needs `with_config`",
                    model.id()
                ),
            ),
        )
    }
}

impl<C: TimmImageContract> ConstructInit for TimmConstructConfig<C> {
    type Construct = ImageModelConstruct<C::Model>;

    const KIT: &'static str = C::KIT;

    /// Every row is a timm checkpoint; the config comes at construction.
    fn for_resource_set(_map: &ResourceSet) -> BunsenResult<Self> {
        Ok(Self::new())
    }

    /// Initialises the config's model, reads the checkpoint into it, and
    /// reads the build's card, checked against the head.
    fn try_init(
        &self,
        model: &BuildRef,
        loaded: &LoadedResourceSet,
        device: &Device,
        load: LoadPolicy,
    ) -> BunsenResult<Arc<Self::Construct>> {
        let config = self.config_for(model, Some(loaded))?;
        let card = ImageModelCard::try_from_build(model, loaded)?
            .try_for_classes(config.num_classes())
            .with_context(|| format!("{}'s labels", model.id()))?;
        let model = C::load(loaded, &config, device, load)?;
        Ok(Arc::new(ImageModelConstruct { model, card }))
    }
}

/// A timm image kit's compiled-in providers, in search order: the built-in
/// catalog's provider of `catalog_contract` (the kit's `catalog:` rows), then
/// `hf:org/repo`, a timm hub repo: its checkpoint (one file or shards) and its
/// `config.json`, whose architecture names the geometry.
pub fn timm_providers(
    catalog_contract: &str
) -> alloc::vec::Vec<Arc<dyn crate::data::builds::BuildProvider>> {
    alloc::vec![
        Arc::new(crate::data::catalog::BuiltinCatalog::expect_provider(
            catalog_contract
        )),
        Arc::new(crate::data::builds::HfProvider::new()),
    ]
}

/// Test helpers for the timm image kits' pretrained pathways: the audits
/// every kit's provider tests run.
///
/// This module exists with the `testing` feature, and in bunsen's own unit
/// tests.
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    use alloc::{
        collections::BTreeMap,
        string::{
            String,
            ToString,
        },
        vec::Vec,
    };

    use super::{
        TimmConstructConfig,
        TimmImageContract,
    };
    use crate::{
        data::{
            builds::{
                BuildFactory,
                CHECKPOINT,
                CONFIG,
            },
            catalog::resolve::ResolvedBuild,
            pretrained::Prefabs,
        },
        kits::images::support::card::{
            LABEL_NAMES,
            LABELS,
        },
    };

    /// What [`assert_timm_rows`] holds the rows' labels to.
    #[derive(Clone, Copy, Debug)]
    pub enum LabelRule<'a> {
        /// The rows by label set, `""` for the unlabelled ones (a headless
        /// trunk, a pretraining-only repo).
        Counts(&'a [(&'a str, usize)]),
        /// Only that a row with labels names them too; the kit checks the
        /// rest on the rows returned.
        Any,
    }

    /// Asserts that `factory` lists `n_rows` rows, each a timm hub repo of an
    /// architecture `C` has a prefab of (`timm/{arch}.{tag}`), its
    /// checkpoint pinned, with its `config.json`, and its labels (where it
    /// has them) with their names, the label sets as `labels` says.
    ///
    /// # Returns
    /// The rows, by name, for the kit's own further checks.
    ///
    /// # Panics
    /// When a row breaks any of that; the message names the row.
    #[track_caller]
    pub fn assert_timm_rows<C: TimmImageContract + Prefabs>(
        factory: &BuildFactory<TimmConstructConfig<C>>,
        n_rows: usize,
        labels: LabelRule<'_>,
    ) -> Vec<(String, ResolvedBuild)> {
        let rows = factory.list();
        assert_eq!(rows.len(), n_rows, "{}'s rows", C::KIT);
        let architectures = C::prefab_names();
        let mut by_labels: BTreeMap<String, usize> = BTreeMap::new();
        let mut named = Vec::new();
        for (_, row) in rows {
            let name = row.name.to_string();
            let map = row.resource_set();
            map.validate().unwrap_or_else(|e| panic!("{name}: {e:#}"));
            let arch = name
                .strip_prefix("timm/")
                .and_then(|n| n.split_once('.'))
                .map(|(arch, _)| arch)
                .unwrap_or_else(|| panic!("{name}: a timm repo's name"));
            assert!(
                architectures.contains(&arch),
                "{name}: an architecture {} builds",
                C::KIT
            );
            assert!(
                map.get(CHECKPOINT)
                    .and_then(|c| c.sha256.as_ref())
                    .is_some(),
                "{name}: a pinned checkpoint"
            );
            assert!(map.get(CONFIG).is_some(), "{name}: a config.json");
            let label_set = map.get(LABELS).map(|l| l.name.to_string());
            assert_eq!(
                label_set.is_some(),
                map.get(LABEL_NAMES).is_some(),
                "{name}: labels and their names together"
            );
            *by_labels.entry(label_set.unwrap_or_default()).or_default() += 1;
            named.push((name, row));
        }
        if let LabelRule::Counts(counts) = labels {
            let want: BTreeMap<String, usize> =
                counts.iter().map(|(k, n)| (k.to_string(), *n)).collect();
            assert_eq!(by_labels, want, "{}'s rows by label set", C::KIT);
        }
        named
    }

    /// Asserts that `factory` serves `C`'s kit through the built-in catalog,
    /// then `hf:`.
    ///
    /// # Panics
    /// When it doesn't.
    #[track_caller]
    pub fn assert_timm_defaults_register<C: TimmImageContract>(
        factory: &BuildFactory<TimmConstructConfig<C>>
    ) {
        assert_eq!(factory.kit(), C::KIT);
        let names: Vec<&str> = factory.providers().iter().map(|p| p.name()).collect();
        assert_eq!(names, ["catalog", "hf"], "{}'s providers", C::KIT);
    }
}
