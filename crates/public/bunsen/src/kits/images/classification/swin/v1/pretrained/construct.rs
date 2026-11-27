//! # Constructing a Swin V1 pretrained

use burn::tensor::Device;

use crate::{
    blocks::transformers::models::staged_vision_encoder::StagedVisionEncoderModel,
    data::resources::LoadedResourceSet,
    errors::BunsenResult,
    kits::images::{
        classification::swin::{
            interop::timm::load_swin,
            v1::{
                SwinV1ContractConfig,
                interop::timm::swin_v1_contract_config,
                pretrained::{
                    CHECKPOINT,
                    SWIN_V1_KIT,
                },
            },
        },
        support::{
            interop::timm::TimmModelConfigMirror,
            pretrained::{
                TimmConstructConfig,
                TimmImageContract,
            },
        },
    },
};

/// How a Swin V1 pretrained is built: the config the model is initialised from
/// before the checkpoint is read into it.
///
/// Swin V1's [`ConstructInit`](crate::data::builds::ConstructInit) hook, behind
/// [`default_swin_v1_factory`](super::default_swin_v1_factory). It builds a
/// [`StagedVisionEncoderModel`] from the config, reads the checkpoint into it
/// with [`load_swin`], and reads the build's [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard): its eval
/// transform and labels.
///
/// The config comes from [`with_config`](Self::with_config), which wins and
/// is what a given path needs; else from the geometry the build states;
/// else, for a timm hub repo, from its `config.json`
/// ([`swin_v1_contract_config`]): its architecture, classes, pool and size.
pub type SwinV1ConstructConfig = TimmConstructConfig<SwinV1ContractConfig>;

impl TimmImageContract for SwinV1ContractConfig {
    type Model = StagedVisionEncoderModel;

    const CONSTRUCT: &'static str = "SwinV1ConstructConfig";
    const KIT: &'static str = SWIN_V1_KIT;

    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn from_timm_config(config: &TimmModelConfigMirror) -> BunsenResult<Self> {
        swin_v1_contract_config(config)
    }

    fn load(
        loaded: &LoadedResourceSet,
        config: &Self,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<StagedVisionEncoderModel> {
        load_swin(loaded, CHECKPOINT, config, device, policy)
    }
}
