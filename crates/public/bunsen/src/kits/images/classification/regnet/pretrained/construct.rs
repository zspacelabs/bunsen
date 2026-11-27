//! # Constructing a `RegNet` pretrained

use burn::tensor::Device;

use crate::{
    blocks::images::models::conv_net::ConvNetModel,
    data::resources::LoadedResourceSet,
    errors::BunsenResult,
    kits::images::{
        classification::regnet::{
            RegNetContractConfig,
            interop::timm::{
                load_regnet,
                regnet_contract_config,
            },
            pretrained::REGNET_KIT,
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

/// How a `RegNet` pretrained is built: the config the model is initialised from
/// before the checkpoint is read into it.
///
/// `RegNet`'s [`ConstructInit`](crate::data::builds::ConstructInit) hook, behind
/// [`default_regnet_factory`](super::default_regnet_factory). It builds a
/// [`ConvNetModel`] from the config, reads the checkpoint into it with
/// [`load_regnet`], and reads the build's [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard): its eval
/// transform and labels.
///
/// The config comes from [`with_config`](Self::with_config), which wins and
/// is what a given path needs; else from the geometry the build states;
/// else, for a timm hub repo, from its `config.json`
/// ([`regnet_contract_config`]): its architecture, classes and pool.
pub type RegNetConstructConfig = TimmConstructConfig<RegNetContractConfig>;

impl TimmImageContract for RegNetContractConfig {
    type Model = ConvNetModel;

    const CONSTRUCT: &'static str = "RegNetConstructConfig";
    const KIT: &'static str = REGNET_KIT;

    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn from_timm_config(config: &TimmModelConfigMirror) -> BunsenResult<Self> {
        regnet_contract_config(config)
    }

    fn load(
        loaded: &LoadedResourceSet,
        config: &Self,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<ConvNetModel> {
        load_regnet(loaded, config, device, policy)
    }
}
