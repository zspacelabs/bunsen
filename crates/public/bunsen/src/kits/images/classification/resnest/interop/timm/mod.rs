//! # `ResNeSt` as timm publishes it
//!
//! A timm hub repo's `ResNeSt` (`timm/resnest50d.in1k`): its `config.json`
//! names the architecture, one of the kit's prefabs, and its
//! `model.safetensors` holds timm's `ResNet` state dict, its blocks
//! `ResNestBottleneck`s.
//!
//! - **Conversion:** [`resnest_contract_config`], a timm `config.json` as the
//!   [`ResNeStContractConfig`] it builds.
//! - **Names:** [`resnest_rekey`], onto the conv net's paths.
//! - **Loading** (feature `store`): `load_resnest_tensors`; with `cache`,
//!   `load_resnest`, a resource set's checkpoint.

use crate::{
    burner::module::{
        Indices,
        Rekey,
    },
    errors::BunsenResult,
    kits::images::{
        classification::{
            resnest::ResNeStContractConfig,
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
    use crate::kits::images::support::interop::timm::{
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

/// timm's `ResNeSt` state-dict names, in `layout`, onto a
/// [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel)'s
/// paths: a `ResNestBottleneck`'s split attention (its `conv2`) onto its
/// mixer's, then [`resnet_rekey`] for the rest (the stem, `conv1`, `bn1`,
/// `conv3`, `bn3`, the shortcuts and the head).
///
/// | timm | bunsen |
/// |---|---|
/// | `layer{i}.{j}.conv2.conv`, `.bn0` | `stages.{i-1}.units.{j}.convs.1.conv.conv`, `.conv.norm` |
/// | `layer{i}.{j}.conv2.fc1`, `.bn1` | `stages.{i-1}.units.{j}.convs.1.fc1.conv`, `.fc1.norm` |
/// | `layer{i}.{j}.conv2.fc2` | `stages.{i-1}.units.{j}.convs.1.fc2` |
///
/// ```rust
/// use bunsen::kits::images::classification::{
///     resnest::interop::timm::resnest_rekey,
///     resnet::interop::torchvision::ResNetStateLayout,
/// };
///
/// let rekey = resnest_rekey(ResNetStateLayout::default());
/// assert_eq!(
///     rekey.apply("layer2.0.conv2.bn1.running_var").as_deref(),
///     Some("stages.1.units.0.convs.1.fc1.norm.running_var")
/// );
/// assert_eq!(
///     rekey.apply("layer2.0.bn3.weight").as_deref(),
///     Some("stages.1.units.0.convs.2.norm.weight")
/// );
/// ```
pub fn resnest_rekey(layout: ResNetStateLayout) -> Rekey {
    let unit = Rekey::pass_rest()
        .rename("conv2.conv.", "convs.1.conv.conv.")
        .rename("conv2.bn0.", "convs.1.conv.norm.")
        .rename("conv2.fc1.", "convs.1.fc1.conv.")
        .rename("conv2.bn1.", "convs.1.fc1.norm.")
        .rename("conv2.fc2.", "convs.1.fc2.");
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
        data::pretrained::Prefabs,
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
        for name in ResNeStContractConfig::prefab_names() {
            let mut want = ResNeStContractConfig::expect_prefab(name);
            want.resnet.num_classes = 10;
            assert_config_eq(
                name,
                &resnest_contract_config(&config(name, 10)).unwrap(),
                &want,
            );
        }
        ErrorMatcher::unsupported()
            .message_contains("timm's resnet50")
            .assert_err(&resnest_contract_config(&config("resnet50", 1000)));
    }

    /// The split attention lands on the mixer's; the rest on `ResNet`'s
    /// paths, a deep stem's and `avg_down` shortcuts' too.
    #[test]
    fn test_resnest_rekey() {
        let rekey = resnest_rekey(ResNetStateLayout {
            stem_convs: 3,
            avg_down: true,
        });
        assert_rekeys(
            &rekey,
            [
                ("conv1.0.weight", Some("stem.convs.0.conv.weight")),
                ("bn1.bias", Some("stem.convs.2.norm.bias")),
                (
                    "layer1.0.conv2.conv.weight",
                    Some("stages.0.units.0.convs.1.conv.conv.weight"),
                ),
                (
                    "layer3.1.conv2.bn0.bias",
                    Some("stages.2.units.1.convs.1.conv.norm.bias"),
                ),
                (
                    "layer4.0.conv2.fc1.bias",
                    Some("stages.3.units.0.convs.1.fc1.conv.bias"),
                ),
                (
                    "layer4.0.conv2.fc2.weight",
                    Some("stages.3.units.0.convs.1.fc2.weight"),
                ),
                (
                    "layer1.0.conv1.weight",
                    Some("stages.0.units.0.convs.0.conv.weight"),
                ),
                (
                    "layer3.0.downsample.2.weight",
                    Some("stages.2.units.0.shortcut.norm.weight"),
                ),
                ("fc.weight", Some("head.fc.weight")),
                ("layer1.0.conv2.bn1.num_batches_tracked", None),
            ],
        );
    }
}
