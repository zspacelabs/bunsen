//! # Constructing a `DeiT` pretrained

use burn::tensor::Device;

use crate::{
    blocks::transformers::models::vision_encoder::VisionEncoderModel,
    data::resources::LoadedResourceSet,
    errors::BunsenResult,
    kits::images::{
        classification::deit::{
            DeiTContractConfig,
            interop::timm::{
                deit_contract_config,
                load_deit,
            },
            pretrained::DEIT_KIT,
        },
        support::{
            pretrained::{
                TimmConstructConfig,
                TimmImageContract,
            },
            timm::interop::TimmModelConfigMirror,
        },
    },
};

/// How a `DeiT` pretrained is built: the config the model is initialised from
/// before the checkpoint is read into it.
///
/// `DeiT`'s [`ConstructInit`](crate::data::builds::ConstructInit) hook, behind
/// [`default_deit_factory`](super::default_deit_factory). It builds a
/// [`VisionEncoderModel`] from the config, reads the checkpoint into it with
/// [`load_deit`], and reads the build's [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard): its eval transform
/// and labels.
///
/// The config comes from [`with_config`](Self::with_config), which wins and
/// is what a given path needs; else from the geometry the build states;
/// else, for a timm hub repo, from its `config.json`
/// ([`deit_contract_config`]): its architecture, classes, pool and size.
pub type DeiTConstructConfig = TimmConstructConfig<DeiTContractConfig>;

impl TimmImageContract for DeiTContractConfig {
    type Model = VisionEncoderModel;

    const CONSTRUCT: &'static str = "DeiTConstructConfig";
    const KIT: &'static str = DEIT_KIT;

    fn num_classes(&self) -> usize {
        self.vit.num_classes
    }

    fn from_timm_config(config: &TimmModelConfigMirror) -> BunsenResult<Self> {
        deit_contract_config(config)
    }

    fn load(
        loaded: &LoadedResourceSet,
        config: &Self,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<VisionEncoderModel> {
        load_deit(loaded, config, device, policy)
    }
}
