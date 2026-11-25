//! # `ResNet` as timm publishes it
//!
//! A timm hub repo's `ResNet` (`timm/resnet50.a1_in1k`): its `config.json`
//! names the architecture, which is one of the kit's prefabs, and its
//! `model.safetensors` uses torchvision's names
//! ([`resnet_rekey`](super::torchvision::resnet_rekey)).
//!
//! - **Conversion:** [`resnet_contract_config`], a timm `config.json` as the
//!   [`ResNetContractConfig`] it builds.
//! - **Loading** (feature `store`): `load_resnet`, a resource set's safetensors
//!   checkpoint.

use crate::{
    data::pretrained::Prefabs,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::images::{
        classification::resnet::ResNetContractConfig,
        support::interop::timm::TimmModelConfigMirror,
    },
};

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

/// The [`ResNetContractConfig`] a timm `config.json` describes: the prefab of
/// its architecture's name (`resnet50`, `resnet50_gn`, …), with its class
/// count.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of (`resnest50d`, …), a headless model
/// (0 classes), or a pool other than `avg`.
pub fn resnet_contract_config(
    config: &TimmModelConfigMirror
) -> BunsenResult<ResNetContractConfig> {
    let architecture = config.architecture.as_str();
    let mut contract = ResNetContractConfig::prefab(architecture).ok_or_else(|| {
        BunsenError::unsupported(format!(
            "timm's {architecture}: the resnet kit builds {}",
            ResNetContractConfig::prefab_names().join(", ")
        ))
    })?;
    if config.num_classes == 0 {
        return Err(BunsenError::unsupported(format!(
            "timm's headless {architecture}: the resnet kit builds a classifier"
        )));
    }
    if let Some(pool) = config.global_pool.as_deref()
        && pool != "avg"
    {
        return Err(BunsenError::unsupported(format!(
            "timm's {architecture} with a {pool:?} pool: the resnet kit pools by average"
        )));
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

    /// A timm config of `architecture` with `num_classes` classes.
    fn config(
        architecture: &str,
        num_classes: usize,
    ) -> TimmModelConfigMirror {
        timm_model_config_from_str(&format!(
            r#"{{"architecture": "{architecture}", "num_classes": {num_classes},
                "num_features": 2048, "pretrained_cfg": {{"input_size": [3, 224, 224],
                "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}}}}"#
        ))
        .unwrap()
    }

    /// Each prefab is a timm architecture of its name; the class count is
    /// the config's.
    #[test]
    fn test_prefab_architectures_map() {
        for name in ResNetContractConfig::prefab_names() {
            let mut want = ResNetContractConfig::expect_prefab(name);
            want.num_classes = 11821;
            assert_eq!(
                format!(
                    "{:?}",
                    resnet_contract_config(&config(name, 11821)).unwrap()
                ),
                format!("{want:?}"),
                "{name}"
            );
        }
    }

    #[test]
    fn test_unbuilt_architectures_are_unsupported() {
        let unsupported = || ErrorMatcher::kind(BunsenErrorKind::Unsupported);
        unsupported()
            .message_contains("timm's resnest50d")
            .assert_err(&resnet_contract_config(&config("resnest50d", 1000)));
        unsupported()
            .message_contains("headless")
            .assert_err(&resnet_contract_config(&config("resnet50", 0)));
        let mut max_pooled = config("resnet50", 1000);
        max_pooled.global_pool = Some("max".into());
        unsupported()
            .message_contains("\"max\" pool")
            .assert_err(&resnet_contract_config(&max_pooled));
    }
}
