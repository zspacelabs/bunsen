//! # `Res2Net` as timm publishes it
//!
//! A timm hub repo's `Res2Net` (`timm/res2net50_26w_4s.in1k`): its
//! `config.json` names the architecture, one of the kit's prefabs, and its
//! `model.safetensors` holds timm's `ResNet` state dict, its blocks
//! `Bottle2neck`s.
//!
//! - **Conversion:** [`res2net_contract_config`], a timm `config.json` as the
//!   [`Res2NetContractConfig`] it builds.
//! - **Names:** [`res2net_rekey`], onto the conv net's paths.
//! - **Loading** (feature `store`): `load_res2net_tensors`; with `cache`,
//!   `load_res2net`, a resource set's checkpoint.

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
        classification::{
            res2net::Res2NetContractConfig,
            resnet::interop::torchvision::{
                ResNetStateLayout,
                resnet_rekey,
            },
        },
        support::interop::timm::TimmModelConfigMirror,
    },
};

#[cfg(feature = "store")]
mod weights;

#[cfg(feature = "store")]
pub use weights::*;

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
    let architecture = config.architecture.as_str();
    let mut contract = Res2NetContractConfig::prefab(architecture).ok_or_else(|| {
        BunsenError::unsupported(format!(
            "timm's {architecture}: the res2net kit builds {}",
            Res2NetContractConfig::prefab_names().join(", ")
        ))
    })?;
    if config.num_classes == 0 {
        return Err(BunsenError::unsupported(format!(
            "timm's headless {architecture}: the res2net kit builds a classifier"
        )));
    }
    if let Some(pool) = config.global_pool.as_deref() {
        contract.resnet.global_pool = GlobalPool2dConfig::from_timm(pool).ok_or_else(|| {
            BunsenError::unsupported(format!(
                "timm's {architecture} with a {pool:?} pool: the res2net kit pools by avg, max, \
                 avgmax or catavgmax"
            ))
        })?;
    }
    contract.resnet.num_classes = config.num_classes;
    Ok(contract)
}

/// timm's `Res2Net` state-dict names, in `layout`, onto a
/// [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel)'s
/// paths: a `Bottle2neck`'s branch convs and norms onto its mixer's, then
/// [`resnet_rekey`] for the rest (the stem, `conv1`, `bn1`, `conv3`, `bn3`,
/// the shortcuts and the head).
///
/// | timm | bunsen |
/// |---|---|
/// | `layer{i}.{j}.convs.{k}` | `stages.{i-1}.units.{j}.convs.1.convs.{k}.conv` |
/// | `layer{i}.{j}.bns.{k}` | `stages.{i-1}.units.{j}.convs.1.convs.{k}.norm` |
///
/// ```rust
/// use bunsen::kits::images::classification::{
///     res2net::interop::timm::res2net_rekey,
///     resnet::interop::torchvision::ResNetStateLayout,
/// };
///
/// let rekey = res2net_rekey(ResNetStateLayout::default());
/// assert_eq!(
///     rekey.apply("layer2.0.bns.2.running_var").as_deref(),
///     Some("stages.1.units.0.convs.1.convs.2.norm.running_var")
/// );
/// assert_eq!(
///     rekey.apply("layer2.0.bn3.weight").as_deref(),
///     Some("stages.1.units.0.convs.2.norm.weight")
/// );
/// ```
pub fn res2net_rekey(layout: ResNetStateLayout) -> Rekey {
    let branch = |part: &'static str| Rekey::keep_rest().rename("", part);
    let unit = Rekey::pass_rest()
        .mount_indexed("convs.", "convs.1.convs.", branch("conv."))
        .mount_indexed("bns.", "convs.1.convs.", branch("norm."));
    Rekey::keep_rest()
        .mount_indexed_with(
            "layer",
            "stages.",
            Indices::From(1),
            Rekey::pass_rest().mount_indexed("", "units.", unit),
        )
        .then(resnet_rekey(layout))
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
                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}}}"#
        ))
        .unwrap()
    }

    /// Each prefab is a timm architecture of its name; the class count is
    /// the config's.
    #[test]
    fn test_prefab_architectures_map() {
        for name in Res2NetContractConfig::prefab_names() {
            let mut want = Res2NetContractConfig::expect_prefab(name);
            want.resnet.num_classes = 10;
            assert_eq!(
                format!("{:?}", res2net_contract_config(&config(name, 10)).unwrap()),
                format!("{want:?}"),
                "{name}"
            );
        }
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("timm's resnet50")
            .assert_err(&res2net_contract_config(&config("resnet50", 1000)));
    }

    /// The branches land on the mixer's; the rest on `ResNet`'s paths, a
    /// deep stem's and `avg_down` shortcuts' too.
    #[test]
    fn test_res2net_rekey() {
        let rekey = res2net_rekey(ResNetStateLayout {
            stem_convs: 3,
            avg_down: true,
        });
        for (timm, bunsen) in [
            ("conv1.0.weight", "stem.convs.0.conv.weight"),
            ("bn1.bias", "stem.convs.2.norm.bias"),
            (
                "layer1.0.convs.0.weight",
                "stages.0.units.0.convs.1.convs.0.conv.weight",
            ),
            (
                "layer4.2.bns.6.bias",
                "stages.3.units.2.convs.1.convs.6.norm.bias",
            ),
            (
                "layer1.0.conv1.weight",
                "stages.0.units.0.convs.0.conv.weight",
            ),
            (
                "layer3.0.downsample.2.weight",
                "stages.2.units.0.shortcut.norm.weight",
            ),
            ("fc.weight", "head.fc.weight"),
        ] {
            assert_eq!(rekey.apply(timm).as_deref(), Some(bunsen), "{timm}");
        }
        assert_eq!(rekey.apply("layer1.0.bns.0.num_batches_tracked"), None);
    }
}
