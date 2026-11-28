//! The config conversion.

use crate::{
    errors::BunsenResult,
    kits::images::{
        classification::regnet::RegNetContractConfig,
        support::timm::interop::TimmModelConfigMirror,
    },
};

/// The [`RegNetContractConfig`] a timm `config.json` describes: the prefab
/// of its architecture's name (`regnety_032`, …), with its class count (0
/// for a headless trunk) and its global pool.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of, or a pool the head doesn't build
/// (`''`).
pub fn regnet_contract_config(
    config: &TimmModelConfigMirror
) -> BunsenResult<RegNetContractConfig> {
    use crate::kits::images::support::timm::interop::{
        timm_conv_global_pool,
        timm_prefab,
    };
    let mut contract: RegNetContractConfig = timm_prefab(config, "regnet")?;
    if let Some(pool) = timm_conv_global_pool(config, "regnet")? {
        contract.global_pool = pool;
    }
    contract.num_classes = config.num_classes;
    Ok(contract)
}
