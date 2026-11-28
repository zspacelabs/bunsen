//! The config conversion.

use crate::{
    errors::BunsenResult,
    kits::images::{
        classification::{
            deit::DeiTContractConfig,
            vit::interop::timm::with_timm_config,
        },
        support::interop::timm::TimmModelConfigMirror,
    },
};

/// The [`DeiTContractConfig`] a timm `config.json` describes: the prefab of
/// its architecture's name (`deit_base_patch16_224`, …), its `ViT` with the
/// config's classes, pool and size
/// ([`with_timm_config`]).
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of; as [`with_timm_config`].
pub fn deit_contract_config(config: &TimmModelConfigMirror) -> BunsenResult<DeiTContractConfig> {
    let contract: DeiTContractConfig =
        crate::kits::images::support::interop::timm::timm_prefab(config, "deit")?;
    Ok(DeiTContractConfig {
        vit: with_timm_config(contract.vit, config)?,
        ..contract
    })
}
