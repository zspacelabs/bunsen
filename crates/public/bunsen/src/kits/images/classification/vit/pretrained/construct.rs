//! # Constructing a `ViT` pretrained

use burn::tensor::Device;

use crate::{
    blocks::transformers::models::vision_encoder::VisionEncoderModel,
    data::resources::LoadedResourceSet,
    errors::BunsenResult,
    kits::images::{
        classification::vit::{
            ViTContractConfig,
            interop::timm::{
                load_vit,
                vit_contract_config,
            },
            pretrained::VIT_KIT,
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

/// How a `ViT` pretrained is built: the config the model is initialised from
/// before the checkpoint is read into it.
///
/// `ViT`'s [`ConstructInit`](crate::data::builds::ConstructInit) hook, behind
/// [`default_vit_factory`](super::default_vit_factory). It builds a
/// [`VisionEncoderModel`] from the config, reads the checkpoint into it with
/// [`load_vit`], and reads the build's [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard): its eval transform
/// and labels.
///
/// The config comes from [`with_config`](Self::with_config), which wins and
/// is what a given path needs; else from the geometry the build states;
/// else, for a timm hub repo, from its `config.json`
/// ([`vit_contract_config`]): its architecture, classes, pool and size.
pub type ViTConstructConfig = TimmConstructConfig<ViTContractConfig>;

impl TimmImageContract for ViTContractConfig {
    type Model = VisionEncoderModel;

    const CONSTRUCT: &'static str = "ViTConstructConfig";
    const KIT: &'static str = VIT_KIT;

    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn from_timm_config(config: &TimmModelConfigMirror) -> BunsenResult<Self> {
        vit_contract_config(config)
    }

    fn load(
        loaded: &LoadedResourceSet,
        config: &Self,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<VisionEncoderModel> {
        load_vit(loaded, config, device, policy)
    }
}
