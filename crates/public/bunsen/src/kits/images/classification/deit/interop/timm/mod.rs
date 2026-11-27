//! # `DeiT` as timm publishes it
//!
//! A timm hub repo's `DeiT` (`timm/deit_base_distilled_patch16_224.fb_in1k`):
//! its `config.json` names the architecture, which is one of the kit's
//! prefabs, and its `model.safetensors` holds timm's `VisionTransformer`
//! state dict, or its `VisionTransformerDistilled`'s.
//!
//! - **Conversion:** [`deit_contract_config`], a timm `config.json` as the
//!   [`DeiTContractConfig`] it builds.
//! - **Loading** (feature `store`): `load_deit`, a resource set's checkpoint,
//!   through the vit kit's
//!   [`load_vit_tensors`](crate::kits::images::classification::vit::interop::timm::load_vit_tensors).

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

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::pretrained::Prefabs,
        errors::testing::ErrorMatcher,
        kits::images::support::interop::timm::timm_model_config_from_str,
        support::testing::assert_config_eq,
    };

    fn config(
        architecture: &str,
        size: usize,
    ) -> TimmModelConfigMirror {
        timm_model_config_from_str(&format!(
            r#"{{"architecture": "{architecture}", "num_classes": 1000,
                "num_features": 768, "global_pool": "token", "pretrained_cfg": {{
                "input_size": [3, {size}, {size}], "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5]}}}}"#
        ))
        .unwrap()
    }

    /// Each prefab is a timm architecture of its name.
    #[test]
    fn test_prefab_architectures_map() {
        for name in DeiTContractConfig::prefab_names() {
            let want = DeiTContractConfig::expect_prefab(name);
            assert_config_eq(
                name,
                &deit_contract_config(&config(name, want.vit.image_size)).unwrap(),
                &want,
            );
        }
        ErrorMatcher::unsupported()
            .message_contains("timm's vit_base_patch16_224: the deit kit builds")
            .assert_err(&deit_contract_config(&config("vit_base_patch16_224", 224)));
    }
}
