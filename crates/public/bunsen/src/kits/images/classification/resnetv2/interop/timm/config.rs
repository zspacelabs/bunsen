//! The config conversion.

use crate::{
    errors::BunsenResult,
    kits::images::{
        classification::resnetv2::ResNetV2ContractConfig,
        support::interop::timm::TimmModelConfigMirror,
    },
};

/// The [`ResNetV2ContractConfig`] a timm `config.json` describes: the
/// prefab of its architecture's name (`resnetv2_50x1_bit`, …), with its class
/// count and its global pool.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of, or a pool the head doesn't build
/// (`''`).
pub fn resnetv2_contract_config(
    config: &TimmModelConfigMirror
) -> BunsenResult<ResNetV2ContractConfig> {
    use crate::kits::images::support::interop::timm::{
        timm_conv_global_pool,
        timm_prefab,
    };
    let mut contract: ResNetV2ContractConfig = timm_prefab(config, "resnetv2")?;
    if let Some(pool) = timm_conv_global_pool(config, "resnetv2")? {
        contract.global_pool = pool;
    }
    contract.num_classes = config.num_classes;
    Ok(contract)
}
