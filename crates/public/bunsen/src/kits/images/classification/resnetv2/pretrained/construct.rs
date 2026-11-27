//! # Constructing a `ResNetV2` pretrained

use burn::tensor::Device;

use crate::{
    blocks::images::models::conv_net::ConvNetModel,
    data::resources::LoadedResourceSet,
    errors::BunsenResult,
    kits::images::{
        classification::resnetv2::{
            ResNetV2ContractConfig,
            interop::timm::{
                load_resnetv2,
                resnetv2_contract_config,
            },
            pretrained::RESNETV2_KIT,
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

/// How a `ResNetV2` pretrained is built: the config the model is initialised
/// from before the checkpoint is read into it.
///
/// `ResNetV2`'s [`ConstructInit`](crate::data::builds::ConstructInit) hook, behind
/// [`default_resnetv2_factory`](super::default_resnetv2_factory). It builds a
/// [`ConvNetModel`] from the config, reads the checkpoint into it with
/// [`load_resnetv2`], and reads the build's [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard): its eval
/// transform and labels.
///
/// The config comes from [`with_config`](Self::with_config), which wins and
/// is what a given path needs; else from the geometry the build states;
/// else, for a timm hub repo, from its `config.json`
/// ([`resnetv2_contract_config`]): its architecture, classes and pool.
pub type ResNetV2ConstructConfig = TimmConstructConfig<ResNetV2ContractConfig>;

impl TimmImageContract for ResNetV2ContractConfig {
    type Model = ConvNetModel;

    const CONSTRUCT: &'static str = "ResNetV2ConstructConfig";
    const KIT: &'static str = RESNETV2_KIT;

    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn from_timm_config(config: &TimmModelConfigMirror) -> BunsenResult<Self> {
        resnetv2_contract_config(config)
    }

    fn load(
        loaded: &LoadedResourceSet,
        config: &Self,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<ConvNetModel> {
        load_resnetv2(loaded, config, device, policy)
    }
}
