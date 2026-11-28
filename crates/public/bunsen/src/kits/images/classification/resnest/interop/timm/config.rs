//! The config conversion.

use crate::{
    errors::BunsenResult,
    kits::images::{
        classification::resnest::ResNeStContractConfig,
        support::timm::interop::TimmModelConfigMirror,
    },
};

/// The [`ResNeStContractConfig`] a timm `config.json` describes: the prefab
/// of its architecture's name (`resnest50d`, …), with its class count
/// and its global pool.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of, a headless model (0 classes), or a
/// pool the head doesn't build (`''`).
pub fn resnest_contract_config(
    config: &TimmModelConfigMirror
) -> BunsenResult<ResNeStContractConfig> {
    use crate::kits::images::support::timm::interop::{
        timm_conv_global_pool,
        timm_prefab,
    };
    let mut contract: ResNeStContractConfig = timm_prefab(config, "resnest")?;
    if let Some(pool) = timm_conv_global_pool(config, "resnest")? {
        contract.resnet.global_pool = pool;
    }
    contract.resnet.num_classes = config.num_classes;
    Ok(contract)
}
