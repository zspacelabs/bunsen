//! The config conversion.

use crate::{
    errors::BunsenResult,
    kits::images::{
        classification::res2net::Res2NetContractConfig,
        support::interop::timm::TimmModelConfigMirror,
    },
};

/// The [`Res2NetContractConfig`] a timm `config.json` describes: the prefab
/// of its architecture's name (`res2net50_26w_4s`, …), with its class count
/// and its global pool.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of, a headless model (0 classes), or a
/// pool the head doesn't build (`''`).
pub fn res2net_contract_config(
    config: &TimmModelConfigMirror
) -> BunsenResult<Res2NetContractConfig> {
    use crate::kits::images::support::interop::timm::{
        timm_conv_global_pool,
        timm_prefab,
    };
    let mut contract: Res2NetContractConfig = timm_prefab(config, "res2net")?;
    if let Some(pool) = timm_conv_global_pool(config, "res2net")? {
        contract.resnet.global_pool = pool;
    }
    contract.resnet.num_classes = config.num_classes;
    Ok(contract)
}
