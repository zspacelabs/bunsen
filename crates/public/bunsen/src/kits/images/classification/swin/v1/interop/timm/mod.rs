//! # Swin V1 as timm publishes it
//!
//! A timm hub repo's Swin V1 (`timm/swin_tiny_patch4_window7_224.ms_in1k`): its
//! `config.json` names the architecture, one of the kit's prefabs, and its
//! `model.safetensors` holds timm's `SwinTransformer` state dict.
//!
//! - **Conversion:** [`swin_v1_contract_config`], a timm `config.json` as the
//!   [`SwinV1ContractConfig`] it builds.
//! - **Names and loading:** the family's
//!   [`swin::interop::timm`](crate::kits::images::classification::swin::interop::timm),
//!   V1's and V2's alike.

use alloc::format;

use crate::{
    data::pretrained::Prefabs,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::images::{
        classification::swin::{
            interop::timm::swin_timm_input,
            v1::SwinV1ContractConfig,
        },
        support::interop::timm::TimmModelConfigMirror,
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
    let architecture = config.architecture.as_str();
    let mut contract = SwinV1ContractConfig::prefab(architecture).ok_or_else(|| {
        BunsenError::unsupported(format!(
            "timm's {architecture}: the swin v1 kit builds {}",
            SwinV1ContractConfig::prefab_names().join(", ")
        ))
    })?;
    let (image_size, pool) = swin_timm_input(config, "swin v1", contract.in_channels)?;
    contract.image_size = image_size;
    if let Some(pool) = pool {
        contract.global_pool = pool;
    }
    contract.num_classes = config.num_classes;
    Ok(contract)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        kits::images::support::interop::timm::timm_model_config_from_str,
    };

    fn config(
        architecture: &str,
        num_classes: usize,
        size: usize,
        pool: &str,
    ) -> TimmModelConfigMirror {
        timm_model_config_from_str(&format!(
            r#"{{"architecture": "{architecture}", "num_classes": {num_classes},
                "num_features": 768, "global_pool": "{pool}", "pretrained_cfg": {{
                "input_size": [3, {size}, {size}], "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5]}}}}"#
        ))
        .unwrap()
    }

    /// Each prefab is a timm architecture of its name; the classes are the
    /// config's.
    #[test]
    fn test_prefab_architectures_map() {
        for name in SwinV1ContractConfig::prefab_names() {
            let mut want = SwinV1ContractConfig::expect_prefab(name);
            want.num_classes = 21841;
            assert_eq!(
                format!(
                    "{:?}",
                    swin_v1_contract_config(&config(name, 21841, want.image_size, "avg")).unwrap()
                ),
                format!("{want:?}"),
                "{name}"
            );
        }
        let unsupported = || ErrorMatcher::kind(BunsenErrorKind::Unsupported);
        unsupported()
            .message_contains("timm's swinv2_tiny_window8_256")
            .assert_err(&swin_v1_contract_config(&config(
                "swinv2_tiny_window8_256",
                1000,
                256,
                "avg",
            )));
        unsupported()
            .message_contains("\"token\" pool")
            .assert_err(&swin_v1_contract_config(&config(
                "swin_tiny_patch4_window7_224",
                1000,
                224,
                "token",
            )));
    }
}
