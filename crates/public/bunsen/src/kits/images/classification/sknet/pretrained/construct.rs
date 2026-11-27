//! # Constructing a `SKNet` pretrained

use burn::tensor::Device;

use crate::{
    blocks::images::models::conv_net::ConvNetModel,
    data::resources::LoadedResourceSet,
    errors::BunsenResult,
    kits::images::{
        classification::sknet::{
            SKNetContractConfig,
            interop::timm::{
                load_sknet,
                sknet_contract_config,
            },
            pretrained::SKNET_KIT,
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

/// How a `SKNet` pretrained is built: the config the model is initialised
/// from before the checkpoint is read into it.
///
/// `SKNet`'s [`ConstructInit`](crate::data::builds::ConstructInit) hook, behind
/// [`default_sknet_factory`](super::default_sknet_factory). It builds a
/// [`ConvNetModel`] from the config, reads the checkpoint into it with
/// [`load_sknet`], and reads the build's [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard): its eval
/// transform and labels.
///
/// The config comes from [`with_config`](Self::with_config), which wins and
/// is what a given path needs; else from the geometry the build states;
/// else, for a timm hub repo, from its `config.json`
/// ([`sknet_contract_config`]): its architecture, classes and pool.
pub type SKNetConstructConfig = TimmConstructConfig<SKNetContractConfig>;

impl TimmImageContract for SKNetContractConfig {
    type Model = ConvNetModel;

    const CONSTRUCT: &'static str = "SKNetConstructConfig";
    const KIT: &'static str = SKNET_KIT;

    fn num_classes(&self) -> usize {
        self.resnet.num_classes
    }

    fn from_timm_config(config: &TimmModelConfigMirror) -> BunsenResult<Self> {
        sknet_contract_config(config)
    }

    fn load(
        loaded: &LoadedResourceSet,
        config: &Self,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<ConvNetModel> {
        load_sknet(loaded, config, device, policy)
    }
}
