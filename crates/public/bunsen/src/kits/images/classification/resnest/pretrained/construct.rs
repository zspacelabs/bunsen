//! # Constructing a `ResNeSt` pretrained

use burn::tensor::Device;

use crate::{
    blocks::images::models::conv_net::ConvNetModel,
    data::resources::LoadedResourceSet,
    errors::BunsenResult,
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
            interop::timm::TimmModelConfigMirror,
            pretrained::{
                TimmConstructConfig,
                TimmImageContract,
            },
        },
    },
};

/// How a `ResNeSt` pretrained is built: the config the model is initialised
/// from before the checkpoint is read into it.
///
/// `ResNeSt`'s [`ConstructInit`](crate::data::builds::ConstructInit) hook, behind
/// [`default_resnest_factory`](super::default_resnest_factory). It builds a
/// [`ConvNetModel`] from the config, reads the checkpoint into it with
/// [`load_resnest`], and reads the build's [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard): its eval
/// transform and labels.
///
/// The config comes from [`with_config`](Self::with_config), which wins and
/// is what a given path needs; else from the geometry the build states;
/// else, for a timm hub repo, from its `config.json`
/// ([`resnest_contract_config`]): its architecture, classes and pool.
pub type ResNeStConstructConfig = TimmConstructConfig<ResNeStContractConfig>;

impl TimmImageContract for ResNeStContractConfig {
    type Model = ConvNetModel;

    const CONSTRUCT: &'static str = "ResNeStConstructConfig";
    const KIT: &'static str = RESNEST_KIT;

    fn num_classes(&self) -> usize {
        self.resnet.num_classes
    }

    fn from_timm_config(config: &TimmModelConfigMirror) -> BunsenResult<Self> {
        resnest_contract_config(config)
    }

    fn load(
        loaded: &LoadedResourceSet,
        config: &Self,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<ConvNetModel> {
        load_resnest(loaded, config, device, policy)
    }
}
