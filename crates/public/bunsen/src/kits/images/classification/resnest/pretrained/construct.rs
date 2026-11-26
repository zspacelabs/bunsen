//! # Constructing a `ResNeSt` pretrained

use alloc::sync::Arc;

use burn::tensor::Device;

use crate::{
    blocks::images::models::conv_net::ConvNetModel,
    data::{
        builds::{
            BuildRef,
            CONFIG,
            ConstructInit,
            GEOMETRY,
        },
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
    kits::images::{
        classification::resnest::{
            ResNeStContractConfig,
            interop::timm::{
                load_resnest,
                resnest_contract_config,
            },
            pretrained::RESNEST_KIT,
        },
        support::{
            card::{
                ImageModelCard,
                ImageModelConstruct,
            },
            interop::timm::timm_model_config_from_path,
        },
    },
};

/// How a `ResNeSt` pretrained is built: the config the model is initialised
/// from before the checkpoint is read into it.
///
/// `ResNeSt`'s [`ConstructInit`] hook, behind
/// [`default_resnest_factory`](super::default_resnest_factory). It builds a
/// [`ConvNetModel`] from the config, reads the checkpoint into it with
/// [`load_resnest`], and reads the build's [`ImageModelCard`]: its eval
/// transform and labels.
///
/// The config comes from [`with_config`](Self::with_config), which wins and
/// is what a given path needs; else from the geometry the build states;
/// else, for a timm hub repo, from its `config.json`
/// ([`resnest_contract_config`]): its architecture, classes and pool.
#[derive(Clone, Debug, Default)]
pub struct ResNeStConstructConfig {
    /// The config to build from, overriding the build's.
    pub config: Option<ResNeStContractConfig>,
}

impl ResNeStConstructConfig {
    /// Builds from the build's config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the config to build from, which wins over the build's.
    pub fn with_config(
        mut self,
        config: ResNeStContractConfig,
    ) -> Self {
        self.config = Some(config);
        self
    }

    /// The config `model` is built from: the explicit one, else the
    /// geometry the build states, else its loaded timm `config.json`'s.
    ///
    /// # Errors
    /// As [`resnest_contract_config`];
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause ([`Rule::Missing`] on `config`), when there
    /// is none: a given checkpoint needs [`with_config`](Self::with_config).
    pub fn config_for(
        &self,
        model: &BuildRef,
        loaded: Option<&LoadedResourceSet>,
    ) -> BunsenResult<ResNeStContractConfig> {
        if let Some(config) = &self.config {
            return Ok(config.clone());
        }
        if let Some(config) = model.stated(GEOMETRY)? {
            return Ok(config);
        }
        if let Some(timm) = loaded.and_then(|l| l.get(CONFIG)) {
            return resnest_contract_config(&timm_model_config_from_path(&timm.path)?);
        }
        Err(BunsenError::from(ConstraintError::new(
            "ResNeStConstructConfig",
            "config",
            Rule::Missing,
        ))
        .context(format!(
            "{}: states no geometry and has no timm `config.json`; a given checkpoint needs \
             `with_config`",
            model.id()
        )))
    }
}

impl ConstructInit for ResNeStConstructConfig {
    type Construct = ImageModelConstruct<ConvNetModel>;

    const KIT: &'static str = RESNEST_KIT;

    /// Every row is a timm safetensors checkpoint; the config comes at
    /// construction.
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
        load: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<Arc<ImageModelConstruct<ConvNetModel>>> {
        let config = self.config_for(model, Some(loaded))?;
        let card = ImageModelCard::try_from_build(model, loaded)?
            .try_for_classes(config.resnet.num_classes)
            .with_context(|| format!("{}'s labels", model.id()))?;
        let model = load_resnest(loaded, &config, device, load)?;
        Ok(Arc::new(ImageModelConstruct { model, card }))
    }
}
