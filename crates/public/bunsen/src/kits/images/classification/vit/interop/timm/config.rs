//! The config conversion.

use alloc::format;

use crate::{
    blocks::transformers::models::vision_encoder::TokenPoolKind,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::images::{
        classification::vit::ViTContractConfig,
        support::timm::interop::TimmModelConfigMirror,
    },
};

/// The [`ViTContractConfig`] a timm `config.json` describes: the prefab of
/// its architecture's name (`vit_base_patch16_224`, …), with the config's
/// classes, pool and size ([`with_timm_config`]).
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of; as [`with_timm_config`].
pub fn vit_contract_config(config: &TimmModelConfigMirror) -> BunsenResult<ViTContractConfig> {
    let contract: ViTContractConfig =
        crate::kits::images::support::timm::interop::timm_prefab(config, "vit")?;
    with_timm_config(contract, config)
}

/// `contract`, the geometry of a timm `config.json`'s architecture, with
/// the config's class count (0 for a headless tower), its global pool, and
/// its input size, the size the position embedding is learnt for.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a pool
/// the head doesn't build (`prr`, `''`), or an input that isn't square or
/// has other channels than the architecture's.
pub fn with_timm_config(
    mut contract: ViTContractConfig,
    config: &TimmModelConfigMirror,
) -> BunsenResult<ViTContractConfig> {
    let architecture = config.architecture.as_str();
    if let Some(pool) = config.global_pool.as_deref() {
        contract.global_pool = TokenPoolKind::from_timm(pool).ok_or_else(|| {
            BunsenError::unsupported(format!(
                "timm's {architecture} with a {pool:?} pool: the vit kit pools by token, avg, \
                 max, avgmax or map"
            ))
        })?;
    }
    let [channels, height, width] = config.pretrained_cfg.input_size;
    if height != width || channels != contract.in_channels {
        return Err(BunsenError::unsupported(format!(
            "timm's {architecture} at {channels}x{height}x{width}: the vit kit takes square \
             images of {} channels",
            contract.in_channels
        )));
    }
    contract.image_size = height;
    contract.num_classes = config.num_classes;
    Ok(contract)
}
