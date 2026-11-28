//! The config conversion.

use crate::{
    errors::BunsenResult,
    kits::images::{
        classification::resnet::ResNetContractConfig,
        support::interop::timm::TimmModelConfigMirror,
    },
};

/// The [`ResNetContractConfig`] a timm `config.json` describes: the prefab of
/// its architecture's name (`resnet50`, `resnet50_gn`, …), with its class
/// count and its global pool.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of (`resnest50d`, …), a headless model
/// (0 classes), or a pool timm's `SelectAdaptivePool2d` names but the head
/// doesn't build (`''`, no pool).
pub fn resnet_contract_config(
    config: &TimmModelConfigMirror
) -> BunsenResult<ResNetContractConfig> {
    use crate::kits::images::support::interop::timm::{
        timm_conv_global_pool,
        timm_prefab,
    };
    let mut contract: ResNetContractConfig = timm_prefab(config, "resnet")?;
    if let Some(pool) = timm_conv_global_pool(config, "resnet")? {
        contract.global_pool = pool;
    }
    contract.num_classes = config.num_classes;
    Ok(contract)
}
