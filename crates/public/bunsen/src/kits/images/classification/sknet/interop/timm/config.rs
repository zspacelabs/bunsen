//! The config conversion.

use crate::{
    errors::BunsenResult,
    kits::images::{
        classification::sknet::SKNetContractConfig,
        support::timm::interop::TimmModelConfigMirror,
    },
};

/// The [`SKNetContractConfig`] a timm `config.json` describes: the prefab
/// of its architecture's name (`skresnet18`, …), with its class count
/// and its global pool.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of, a headless model (0 classes), or a
/// pool the head doesn't build (`''`).
pub fn sknet_contract_config(config: &TimmModelConfigMirror) -> BunsenResult<SKNetContractConfig> {
    use crate::kits::images::support::timm::interop::{
        timm_conv_global_pool,
        timm_prefab,
    };
    let mut contract: SKNetContractConfig = timm_prefab(config, "sknet")?;
    if let Some(pool) = timm_conv_global_pool(config, "sknet")? {
        contract.resnet.global_pool = pool;
    }
    contract.resnet.num_classes = config.num_classes;
    Ok(contract)
}
