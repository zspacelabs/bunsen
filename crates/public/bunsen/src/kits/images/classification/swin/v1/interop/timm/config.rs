//! The config conversion.

use crate::{
    errors::BunsenResult,
    kits::images::{
        classification::swin::{
            interop::timm::swin_timm_input,
            v1::SwinV1ContractConfig,
        },
        support::timm::interop::TimmModelConfigMirror,
    },
};

/// The [`SwinV1ContractConfig`] a timm `config.json` describes: the prefab of
/// its architecture's name (`swin_tiny_patch4_window7_224`, …), with its class
/// count, its pool and its input size.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of, a pool the head doesn't build, or
/// an input that isn't square or has other channels than the
/// architecture's.
pub fn swin_v1_contract_config(
    config: &TimmModelConfigMirror
) -> BunsenResult<SwinV1ContractConfig> {
    let mut contract: SwinV1ContractConfig =
        crate::kits::images::support::timm::interop::timm_prefab(config, "swin v1")?;
    let (image_size, pool) = swin_timm_input(config, "swin v1", contract.in_channels)?;
    contract.image_size = image_size;
    if let Some(pool) = pool {
        contract.global_pool = pool;
    }
    contract.num_classes = config.num_classes;
    Ok(contract)
}
