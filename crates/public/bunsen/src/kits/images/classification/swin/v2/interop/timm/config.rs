//! The config conversion.

use crate::{
    errors::BunsenResult,
    kits::images::{
        classification::swin::{
            interop::timm::swin_timm_input,
            v2::SwinV2ContractConfig,
        },
        support::timm::interop::TimmModelConfigMirror,
    },
};

/// The [`SwinV2ContractConfig`] a timm `config.json` describes: the prefab of
/// its architecture's name (`swinv2_tiny_window8_256`, …), with its class
/// count, its pool and its input size.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of, a pool the head doesn't build, or
/// an input that isn't square or has other channels than the
/// architecture's.
pub fn swin_v2_contract_config(
    config: &TimmModelConfigMirror
) -> BunsenResult<SwinV2ContractConfig> {
    let mut contract: SwinV2ContractConfig =
        crate::kits::images::support::timm::interop::timm_prefab(config, "swin v2")?;
    let (image_size, pool) = swin_timm_input(config, "swin v2", contract.in_channels)?;
    contract.image_size = image_size;
    if let Some(pool) = pool {
        contract.global_pool = pool;
    }
    contract.num_classes = config.num_classes;
    Ok(contract)
}
