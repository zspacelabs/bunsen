//! # Constructing a Swin V2 pretrained

use alloc::sync::Arc;

use burn::tensor::Device;

use crate::{
    blocks::transformers::models::staged_vision_encoder::StagedVisionEncoderModel,
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
        classification::swin::{
            interop::timm::load_swin,
            v2::{
                SwinV2ContractConfig,
                interop::timm::swin_v2_contract_config,
                pretrained::{
                    CHECKPOINT,
                    SWIN_V2_KIT,
                },
            },
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

/// How a Swin V2 pretrained is built: the config the model is initialised from
/// before the checkpoint is read into it.
///
/// Swin V2's [`ConstructInit`] hook, behind
/// [`default_swin_v2_factory`](super::default_swin_v2_factory). It builds a
/// [`StagedVisionEncoderModel`] from the config, reads the checkpoint into it
/// with [`load_swin`], and reads the build's [`ImageModelCard`]: its eval
/// transform and labels.
///
/// The config comes from [`with_config`](Self::with_config), which wins and
/// is what a given path needs; else from the geometry the build states;
/// else, for a timm hub repo, from its `config.json`
/// ([`swin_v2_contract_config`]): its architecture, classes, pool and size.
#[derive(Clone, Debug, Default)]
pub struct SwinV2ConstructConfig {
    /// The config to build from, overriding the build's.
    pub config: Option<SwinV2ContractConfig>,
}

impl SwinV2ConstructConfig {
    /// Builds from the build's config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the config to build from, which wins over the build's.
    pub fn with_config(
        mut self,
        config: SwinV2ContractConfig,
    ) -> Self {
        self.config = Some(config);
        self
    }

    /// The config `model` is built from: the explicit one, else the
    /// geometry the build states, else its loaded timm `config.json`'s.
    ///
    /// # Errors
    /// As [`swin_v2_contract_config`];
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause ([`Rule::Missing`] on `config`), when there
    /// is none: a given checkpoint needs [`with_config`](Self::with_config).
    pub fn config_for(
        &self,
        model: &BuildRef,
        loaded: Option<&LoadedResourceSet>,
    ) -> BunsenResult<SwinV2ContractConfig> {
        if let Some(config) = &self.config {
            return Ok(config.clone());
        }
        if let Some(config) = model.stated(GEOMETRY)? {
            return Ok(config);
        }
        if let Some(timm) = loaded.and_then(|l| l.get(CONFIG)) {
            return swin_v2_contract_config(&timm_model_config_from_path(&timm.path)?);
        }
        Err(BunsenError::from(ConstraintError::new(
            "SwinV2ConstructConfig",
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

impl ConstructInit for SwinV2ConstructConfig {
    type Construct = ImageModelConstruct<StagedVisionEncoderModel>;

    const KIT: &'static str = SWIN_V2_KIT;

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
    ) -> BunsenResult<Arc<ImageModelConstruct<StagedVisionEncoderModel>>> {
        let config = self.config_for(model, Some(loaded))?;
        let card = ImageModelCard::try_from_build(model, loaded)?
            .try_for_classes(config.num_classes)
            .with_context(|| format!("{}'s labels", model.id()))?;
        let model = load_swin(loaded, CHECKPOINT, &config, device, load)?;
        Ok(Arc::new(ImageModelConstruct { model, card }))
    }
}
