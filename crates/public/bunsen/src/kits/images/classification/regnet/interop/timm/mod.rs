//! # `RegNet` as timm publishes it
//!
//! A timm hub repo's `RegNet` (`timm/regnety_032.ra_in1k`): its `config.json`
//! names the architecture, one of the kit's prefabs, and its
//! `model.safetensors` holds timm's `RegNet` state dict.
//!
//! - **Conversion:** [`regnet_contract_config`], a timm `config.json` as the
//!   [`RegNetContractConfig`] it builds.
//! - **Names:** [`regnet_rekey`], onto the conv net's paths.
//! - **Loading** (feature `store`): `load_regnet_tensors`; with `cache`,
//!   `load_regnet`, a resource set's checkpoint.

use alloc::format;

use crate::{
    blocks::images::models::conv_net::GlobalPool2dConfig,
    burner::module::{
        Indices,
        Rekey,
    },
    data::pretrained::Prefabs,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::images::{
        classification::regnet::RegNetContractConfig,
        support::interop::timm::TimmModelConfigMirror,
    },
};

#[cfg(feature = "store")]
mod weights;

#[cfg(feature = "store")]
pub use weights::*;

/// The [`RegNetContractConfig`] a timm `config.json` describes: the prefab
/// of its architecture's name (`regnety_032`, …), with its class count (0
/// for a headless trunk) and its global pool.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of, or a pool the head doesn't build
/// (`''`).
pub fn regnet_contract_config(
    config: &TimmModelConfigMirror
) -> BunsenResult<RegNetContractConfig> {
    let architecture = config.architecture.as_str();
    let mut contract = RegNetContractConfig::prefab(architecture).ok_or_else(|| {
        BunsenError::unsupported(format!(
            "timm's {architecture}: the regnet kit builds {}",
            RegNetContractConfig::prefab_names().join(", ")
        ))
    })?;
    if let Some(pool) = config.global_pool.as_deref() {
        contract.global_pool = GlobalPool2dConfig::from_timm(pool).ok_or_else(|| {
            BunsenError::unsupported(format!(
                "timm's {architecture} with a {pool:?} pool: the regnet kit pools by avg, max, \
                 avgmax or catavgmax"
            ))
        })?;
    }
    contract.num_classes = config.num_classes;
    Ok(contract)
}

/// timm's `RegNet` state-dict names onto a
/// [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel)'s
/// paths.
///
/// | timm | bunsen |
/// |---|---|
/// | `stem.conv`, `stem.bn` | `stem.convs.0.conv`, `stem.convs.0.norm` |
/// | `s1` … `s4`, each block `b{j}` | `stages.0` … `stages.3`, each as `units.{j-1}` |
/// | a block's `conv{k}.conv`, `conv{k}.bn` | its `convs.{k-1}.conv`, `convs.{k-1}.norm` |
/// | a block's `se` (`fc1`, `fc2`) | its `attn` |
/// | a block's `downsample.conv`, `downsample.bn` | its `shortcut.conv`, `shortcut.norm` |
/// | an `avg` block's `downsample.1` (after the pool, `downsample.0`) | its `shortcut` |
/// | `RegNetV`'s `stem.weight`, a block's `norm1`, `norm{k}`, `conv{k}.weight`, `downsample.weight` | `stem.convs.0.conv.weight`, its `pre.norm`, `convs.{k-1}.norm`, `convs.{k-1}.conv.weight`, `shortcut.conv.weight` |
/// | `RegNetZ`'s `final_conv.conv`, `final_conv.bn` | `final_conv.conv`, `final_conv.norm` |
/// | `head.fc` | `head.fc` |
///
/// Batch-norm counters (`num_batches_tracked`) are dropped; anything else is
/// kept as it is, for the load to report.
///
/// ```rust
/// use bunsen::kits::images::classification::regnet::interop::timm::regnet_rekey;
///
/// assert_eq!(
///     regnet_rekey().apply("s2.b3.conv2.bn.running_var").as_deref(),
///     Some("stages.1.units.2.convs.1.norm.running_var")
/// );
/// ```
pub fn regnet_rekey() -> Rekey {
    // A conv block's `conv` and `bn`, or a pre-activation net's bare conv.
    let conv = Rekey::keep_rest()
        .rename("bn.", "norm.")
        .exact("weight", "conv.weight");
    let unit = Rekey::keep_rest()
        .mount_indexed_with("conv", "convs.", Indices::From(1), conv)
        .rename("norm1.", "pre.norm.")
        .rename("norm2.", "convs.1.norm.")
        .rename("norm3.", "convs.2.norm.")
        .rename("se.", "attn.")
        .rename("downsample.conv.", "shortcut.conv.")
        .rename("downsample.bn.", "shortcut.norm.")
        .rename("downsample.1.conv.", "shortcut.conv.")
        .rename("downsample.1.bn.", "shortcut.norm.")
        .exact("downsample.weight", "shortcut.conv.weight")
        .exact("downsample.1.weight", "shortcut.conv.weight");
    Rekey::keep_rest()
        .drop_suffix(".num_batches_tracked")
        .rename("stem.conv.", "stem.convs.0.conv.")
        .rename("stem.bn.", "stem.convs.0.norm.")
        .exact("stem.weight", "stem.convs.0.conv.weight")
        .rename("final_conv.bn.", "final_conv.norm.")
        .mount_indexed_with(
            "s",
            "stages.",
            Indices::From(1),
            Rekey::keep_rest().mount_indexed_with("b", "units.", Indices::From(1), unit),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        kits::images::support::interop::timm::timm_model_config_from_str,
    };

    /// A timm config of `architecture` with `num_classes` classes.
    fn config(
        architecture: &str,
        num_classes: usize,
    ) -> TimmModelConfigMirror {
        timm_model_config_from_str(&format!(
            r#"{{"architecture": "{architecture}", "num_classes": {num_classes},
                "num_features": 368, "pretrained_cfg": {{"input_size": [3, 224, 224],
                "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}}}}"#
        ))
        .unwrap()
    }

    /// Each prefab is a timm architecture of its name; the class count is
    /// the config's, 0 for a trunk.
    #[test]
    fn test_prefab_architectures_map() {
        for name in RegNetContractConfig::prefab_names() {
            for classes in [0, 11821] {
                let mut want = RegNetContractConfig::expect_prefab(name);
                want.num_classes = classes;
                assert_eq!(
                    format!(
                        "{:?}",
                        regnet_contract_config(&config(name, classes)).unwrap()
                    ),
                    format!("{want:?}"),
                    "{name}"
                );
            }
        }
        let unsupported = || ErrorMatcher::unsupported();
        unsupported()
            .message_contains("timm's resnet50")
            .assert_err(&regnet_contract_config(&config("resnet50", 1000)));
        let mut unpooled = config("regnetx_002", 1000);
        unpooled.global_pool = Some("".into());
        unsupported()
            .message_contains("\"\" pool")
            .assert_err(&regnet_contract_config(&unpooled));
    }

    /// timm's names land on the conv net's paths.
    #[test]
    fn test_regnet_rekey() {
        let rekey = regnet_rekey();
        for (timm, bunsen) in [
            ("stem.conv.weight", "stem.convs.0.conv.weight"),
            ("stem.bn.bias", "stem.convs.0.norm.bias"),
            (
                "s1.b1.conv1.conv.weight",
                "stages.0.units.0.convs.0.conv.weight",
            ),
            (
                "s4.b7.conv3.bn.weight",
                "stages.3.units.6.convs.2.norm.weight",
            ),
            ("s2.b1.se.fc1.bias", "stages.1.units.0.attn.fc1.bias"),
            (
                "s1.b1.downsample.conv.weight",
                "stages.0.units.0.shortcut.conv.weight",
            ),
            (
                "s1.b1.downsample.bn.running_mean",
                "stages.0.units.0.shortcut.norm.running_mean",
            ),
            ("head.fc.weight", "head.fc.weight"),
        ] {
            assert_eq!(rekey.apply(timm).as_deref(), Some(bunsen), "{timm}");
        }
        assert_eq!(rekey.apply("s1.b1.conv1.bn.num_batches_tracked"), None);
        // `RegNetV`'s pre-activation names, and `RegNetZ`'s final conv.
        for (timm, bunsen) in [
            ("stem.weight", "stem.convs.0.conv.weight"),
            ("s1.b1.norm1.weight", "stages.0.units.0.pre.norm.weight"),
            ("s1.b1.conv1.weight", "stages.0.units.0.convs.0.conv.weight"),
            (
                "s1.b1.norm2.running_var",
                "stages.0.units.0.convs.1.norm.running_var",
            ),
            ("s1.b1.norm3.bias", "stages.0.units.0.convs.2.norm.bias"),
            ("s3.b2.conv3.weight", "stages.2.units.1.convs.2.conv.weight"),
            (
                "s1.b1.downsample.weight",
                "stages.0.units.0.shortcut.conv.weight",
            ),
            (
                "s1.b1.downsample.1.weight",
                "stages.0.units.0.shortcut.conv.weight",
            ),
            ("final_conv.conv.weight", "final_conv.conv.weight"),
            ("final_conv.bn.weight", "final_conv.norm.weight"),
        ] {
            assert_eq!(rekey.apply(timm).as_deref(), Some(bunsen), "{timm}");
        }
    }
}
