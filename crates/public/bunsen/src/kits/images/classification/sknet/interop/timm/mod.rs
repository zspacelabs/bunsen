//! # `SKNet` as timm publishes it
//!
//! A timm hub repo's `SKNet` (`timm/skresnet18.ra_in1k`): its `config.json`
//! names the architecture, one of the kit's prefabs, and its
//! `model.safetensors` holds timm's `ResNet` state dict, its blocks
//! `SelectiveKernelBasic`s or `SelectiveKernelBottleneck`s.
//!
//! - **Conversion:** [`sknet_contract_config`], a timm `config.json` as the
//!   [`SKNetContractConfig`] it builds.
//! - **Names:** [`sknet_rekey`], onto the conv net's paths.
//! - **Loading** (feature `store`): `load_sknet_tensors`; with `cache`,
//!   `load_sknet`, a resource set's checkpoint.

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
            resnet::interop::torchvision::{
                ResNetStateLayout,
                resnet_rekey,
            },
            sknet::SKNetContractConfig,
        },
        support::interop::timm::TimmModelConfigMirror,
    },
};

#[cfg(feature = "store")]
mod weights;

#[cfg(feature = "store")]
pub use weights::*;

/// The [`SKNetContractConfig`] a timm `config.json` describes: the prefab
/// of its architecture's name (`skresnet18`, …), with its class count
/// and its global pool.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of, a headless model (0 classes), or a
/// pool the head doesn't build (`''`).
pub fn sknet_contract_config(config: &TimmModelConfigMirror) -> BunsenResult<SKNetContractConfig> {
    let architecture = config.architecture.as_str();
    let mut contract = SKNetContractConfig::prefab(architecture).ok_or_else(|| {
        BunsenError::unsupported(format!(
            "timm's {architecture}: the sknet kit builds {}",
            SKNetContractConfig::prefab_names().join(", ")
        ))
    })?;
    if config.num_classes == 0 {
        return Err(BunsenError::unsupported(format!(
            "timm's headless {architecture}: the sknet kit builds a classifier"
        )));
    }
    if let Some(pool) = config.global_pool.as_deref() {
        contract.resnet.global_pool = GlobalPool2dConfig::from_timm(pool).ok_or_else(|| {
            BunsenError::unsupported(format!(
                "timm's {architecture} with a {pool:?} pool: the sknet kit pools by avg, max, \
                 avgmax or catavgmax"
            ))
        })?;
    }
    contract.resnet.num_classes = config.num_classes;
    Ok(contract)
}

/// timm's `SKNet` state-dict names, in `layout`, onto a
/// [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel)'s
/// paths: a block's convs (each a `ConvNormAct`, or the selective kernel)
/// onto its path's, then [`resnet_rekey`] for the rest (the stem, the
/// shortcuts and the head).
///
/// | timm | bunsen |
/// |---|---|
/// | `layer{i}.{j}.conv{k}.conv`, `.bn` | `stages.{i-1}.units.{j}.convs.{k-1}.conv`, `.norm` |
/// | `layer{i}.{j}.conv{k}.paths.{p}.conv`, `.bn` | `stages.{i-1}.units.{j}.convs.{k-1}.paths.{p}.conv`, `.norm` |
/// | `layer{i}.{j}.conv{k}.attn.fc_reduce`, `.bn` | `stages.{i-1}.units.{j}.convs.{k-1}.reduce.conv`, `.reduce.norm` |
/// | `layer{i}.{j}.conv{k}.attn.fc_select` | `stages.{i-1}.units.{j}.convs.{k-1}.select` |
///
/// ```rust
/// use bunsen::kits::images::classification::{
///     resnet::interop::torchvision::ResNetStateLayout,
///     sknet::interop::timm::sknet_rekey,
/// };
///
/// let rekey = sknet_rekey(ResNetStateLayout::default());
/// assert_eq!(
///     rekey
///         .apply("layer2.0.conv1.paths.1.bn.running_var")
///         .as_deref(),
///     Some("stages.1.units.0.convs.0.paths.1.norm.running_var")
/// );
/// assert_eq!(
///     rekey.apply("layer2.0.conv2.bn.weight").as_deref(),
///     Some("stages.1.units.0.convs.1.norm.weight")
/// );
/// ```
pub fn sknet_rekey(layout: ResNetStateLayout) -> Rekey {
    let conv = Rekey::keep_rest()
        .rename("bn.", "norm.")
        .mount_indexed(
            "paths.",
            "paths.",
            Rekey::keep_rest().rename("bn.", "norm."),
        )
        .rename("attn.fc_reduce.", "reduce.conv.")
        .rename("attn.bn.", "reduce.norm.")
        .rename("attn.fc_select.", "select.");
    let unit = Rekey::pass_rest().mount_indexed_with("conv", "convs.", Indices::From(1), conv);
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
        burner::module::testing::assert_rekeys,
        errors::testing::ErrorMatcher,
        kits::images::support::interop::timm::timm_model_config_from_str,
        support::testing::assert_config_eq,
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
        for name in SKNetContractConfig::prefab_names() {
            let mut want = SKNetContractConfig::expect_prefab(name);
            want.resnet.num_classes = 10;
            assert_config_eq(
                name,
                &sknet_contract_config(&config(name, 10)).unwrap(),
                &want,
            );
        }
        ErrorMatcher::unsupported()
            .message_contains("timm's resnet50")
            .assert_err(&sknet_contract_config(&config("resnet50", 1000)));
    }

    /// A basic block's selective kernel and its `ConvNormAct`, and a
    /// bottleneck's, land on the path's convs; the rest on `ResNet`'s paths.
    #[test]
    fn test_sknet_rekey() {
        let rekey = sknet_rekey(ResNetStateLayout::default());
        assert_rekeys(
            &rekey,
            [
                ("conv1.weight", Some("stem.convs.0.conv.weight")),
                ("bn1.bias", Some("stem.convs.0.norm.bias")),
                (
                    "layer1.0.conv1.paths.0.conv.weight",
                    Some("stages.0.units.0.convs.0.paths.0.conv.weight"),
                ),
                (
                    "layer1.0.conv1.attn.fc_reduce.weight",
                    Some("stages.0.units.0.convs.0.reduce.conv.weight"),
                ),
                (
                    "layer1.0.conv1.attn.bn.bias",
                    Some("stages.0.units.0.convs.0.reduce.norm.bias"),
                ),
                (
                    "layer4.1.conv2.attn.fc_select.weight",
                    Some("stages.3.units.1.convs.1.select.weight"),
                ),
                (
                    "layer3.0.conv3.conv.weight",
                    Some("stages.2.units.0.convs.2.conv.weight"),
                ),
                (
                    "layer2.0.downsample.1.running_mean",
                    Some("stages.1.units.0.shortcut.norm.running_mean"),
                ),
                ("fc.weight", Some("head.fc.weight")),
                ("layer1.0.conv1.paths.1.bn.num_batches_tracked", None),
            ],
        );
    }
}
