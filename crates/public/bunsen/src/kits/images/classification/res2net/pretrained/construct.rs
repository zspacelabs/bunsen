//! # Constructing a `Res2Net` pretrained

use burn::tensor::Device;

use crate::{
    blocks::images::models::conv_net::ConvNetModel,
    data::resources::LoadedResourceSet,
    errors::BunsenResult,
    kits::images::{
        classification::res2net::{
            Res2NetContractConfig,
            interop::timm::{
                load_res2net,
                res2net_contract_config,
            },
            pretrained::RES2NET_KIT,
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

/// How a `Res2Net` pretrained is built: the config the model is initialised
/// from before the checkpoint is read into it.
///
/// `Res2Net`'s [`ConstructInit`](crate::data::builds::ConstructInit) hook, behind
/// [`default_res2net_factory`](super::default_res2net_factory). It builds a
/// [`ConvNetModel`] from the config, reads the checkpoint into it with
/// [`load_res2net`], and reads the build's [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard): its eval
/// transform and labels.
///
/// The config comes from [`with_config`](Self::with_config), which wins and
/// is what a given path needs; else from the geometry the build states;
/// else, for a timm hub repo, from its `config.json`
/// ([`res2net_contract_config`]): its architecture, classes and pool.
pub type Res2NetConstructConfig = TimmConstructConfig<Res2NetContractConfig>;

impl TimmImageContract for Res2NetContractConfig {
    type Model = ConvNetModel;

    const CONSTRUCT: &'static str = "Res2NetConstructConfig";
    const KIT: &'static str = RES2NET_KIT;

    fn num_classes(&self) -> usize {
        self.resnet.num_classes
    }

    fn from_timm_config(config: &TimmModelConfigMirror) -> BunsenResult<Self> {
        res2net_contract_config(config)
    }

    fn load(
        loaded: &LoadedResourceSet,
        config: &Self,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<ConvNetModel> {
        load_res2net(loaded, config, device, policy)
    }
}
