//! # `ResNetV2` as timm publishes it
//!
//! A timm hub repo's `ResNetV2` (`timm/resnetv2_50x1_bit.goog_in21k_ft_in1k`):
//! its `config.json` names the architecture, one of the kit's prefabs, and its
//! `model.safetensors` holds timm's `ResNetV2` state dict.
//!
//! - **Conversion:** [`resnetv2_contract_config`], a timm `config.json` as the
//!   [`ResNetV2ContractConfig`] it builds.
//! - **Names:** [`resnetv2_rekey`], onto the conv net's paths.
//! - **Loading** (feature `store`): `load_resnetv2_tensors`; with `cache`,
//!   `load_resnetv2`, a resource set's checkpoint.

use alloc::format;

use crate::{
    blocks::images::models::conv_net::GlobalPool2dConfig,
    burner::module::{
        Indices,
        Rekey,
        TensorLoadTransform,
    },
    data::pretrained::Prefabs,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::images::{
        classification::resnetv2::ResNetV2ContractConfig,
        support::interop::timm::TimmModelConfigMirror,
    },
};

#[cfg(feature = "store")]
mod weights;

#[cfg(feature = "store")]
pub use weights::*;

/// The [`ResNetV2ContractConfig`] a timm `config.json` describes: the
/// prefab of its architecture's name (`resnetv2_50x1_bit`, …), with its class
/// count and its global pool.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of, or a pool the head doesn't build
/// (`''`).
pub fn resnetv2_contract_config(
    config: &TimmModelConfigMirror
) -> BunsenResult<ResNetV2ContractConfig> {
    let architecture = config.architecture.as_str();
    let mut contract = ResNetV2ContractConfig::prefab(architecture).ok_or_else(|| {
        BunsenError::unsupported(format!(
            "timm's {architecture}: the resnetv2 kit builds {}",
            ResNetV2ContractConfig::prefab_names().join(", ")
        ))
    })?;
    if let Some(pool) = config.global_pool.as_deref() {
        contract.global_pool = GlobalPool2dConfig::from_timm(pool).ok_or_else(|| {
            BunsenError::unsupported(format!(
                "timm's {architecture} with a {pool:?} pool: the resnetv2 kit pools by avg, max, \
                 avgmax or catavgmax"
            ))
        })?;
    }
    contract.num_classes = config.num_classes;
    Ok(contract)
}

/// timm's `ResNetV2` state-dict names onto a
/// [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel)'s
/// paths, for a head of `num_classes` over `d_features` features, of a
/// pre-activation net when `preact` (every published `ResNetV2`), or the
/// hybrid `ViT`'s backbone (timm's `preact=False`: its `Bottleneck`s).
///
/// | timm | bunsen |
/// |---|---|
/// | `stem.conv`, a normed stem's `stem.norm` | `stem.convs.0.conv`, `stem.convs.0.norm` |
/// | a deep stem's `stem.conv{k}`, `stem.norm{k}` | `stem.convs.{k-1}.conv`, `stem.convs.{k-1}.norm` |
/// | `stages.{i}.blocks.{j}` | `stages.{i}.units.{j}` |
/// | a pre-activation block's `norm1` | its `pre.norm` |
/// | a pre-activation block's `conv{k}`, `norm{k}` (`k` > 1) | its `convs.{k-1}.conv`, `convs.{k-1}.norm` |
/// | a `Bottleneck`'s `conv{k}`, `norm{k}` | its `convs.{k-1}.conv`, `convs.{k-1}.norm` |
/// | a block's `downsample.conv`, a `Bottleneck`'s `downsample.norm` | its `shortcut.conv`, `shortcut.norm` |
/// | a pre-activation net's `norm` | `final_conv.norm` |
/// | a stem alone's `conv`, `norm` (timm's `create_resnetv2_stem`, not pre-activation) | `stem.convs.0.conv`, `stem.convs.0.norm` |
/// | `head.fc`, a `1x1` conv | `head.fc`, its weight reshaped to `[num_classes, d_features]` |
///
/// Batch-norm counters (`num_batches_tracked`) are dropped; anything else is
/// kept as it is, for the load to report.
///
/// ```rust
/// use bunsen::kits::images::classification::resnetv2::interop::timm::resnetv2_rekey;
///
/// assert_eq!(
///     resnetv2_rekey(1000, 2048, true)
///         .apply("stages.1.blocks.2.norm2.weight")
///         .as_deref(),
///     Some("stages.1.units.2.convs.1.norm.weight")
/// );
/// assert_eq!(
///     resnetv2_rekey(0, 1024, false)
///         .apply("stages.1.blocks.2.norm2.weight")
///         .as_deref(),
///     Some("stages.1.units.2.convs.1.norm.weight")
/// );
/// ```
pub fn resnetv2_rekey(
    num_classes: usize,
    d_features: usize,
    preact: bool,
) -> Rekey {
    let unit = match preact {
        true => Rekey::keep_rest()
            .rename("norm1.", "pre.norm.")
            .mount_indexed_with(
                "conv",
                "convs.",
                Indices::From(1),
                Rekey::keep_rest().rename("", "conv."),
            ),
        false => Rekey::keep_rest().mount_indexed_with(
            "conv",
            "convs.",
            Indices::From(1),
            Rekey::keep_rest().rename("", "conv."),
        ),
    };
    let unit = unit
        .mount_indexed_with(
            "conv",
            "convs.",
            Indices::From(1),
            Rekey::keep_rest().rename("", "conv."),
        )
        .mount_indexed_with(
            "norm",
            "convs.",
            Indices::From(1),
            Rekey::keep_rest().rename("", "norm."),
        )
        .rename("downsample.conv.", "shortcut.conv.")
        .rename("downsample.norm.", "shortcut.norm.");
    let stem = Rekey::keep_rest()
        .rename("conv.", "convs.0.conv.")
        .rename("norm.", "convs.0.norm.")
        .mount_indexed_with(
            "conv",
            "convs.",
            Indices::From(1),
            Rekey::keep_rest().rename("", "conv."),
        )
        .mount_indexed_with(
            "norm",
            "convs.",
            Indices::From(1),
            Rekey::keep_rest().rename("", "norm."),
        );
    let rekey = Rekey::keep_rest()
        .drop_suffix(".num_batches_tracked")
        .mount("stem.", "stem.", stem);
    let rekey = match preact {
        true => rekey.rename("norm.", "final_conv.norm."),
        false => rekey
            .rename("conv.", "stem.convs.0.conv.")
            .rename("norm.", "stem.convs.0.norm."),
    };
    let mut rekey = rekey.mount_indexed(
        "stages.",
        "stages.",
        Rekey::keep_rest().mount_indexed("blocks.", "units.", unit),
    );
    if num_classes > 0 {
        rekey = rekey.transform(
            "head.fc.weight",
            TensorLoadTransform::reshape([num_classes, d_features]),
        );
    }
    rekey
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
                "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}}}}"#
        ))
        .unwrap()
    }

    /// Each prefab is a timm architecture of its name; the class count is
    /// the config's.
    #[test]
    fn test_prefab_architectures_map() {
        for name in ResNetV2ContractConfig::prefab_names() {
            let mut want = ResNetV2ContractConfig::expect_prefab(name);
            want.num_classes = 21843;
            assert_config_eq(
                name,
                &resnetv2_contract_config(&config(name, 21843)).unwrap(),
                &want,
            );
        }
        ErrorMatcher::unsupported()
            .message_contains("timm's resnet50")
            .assert_err(&resnetv2_contract_config(&config("resnet50", 1000)));
    }

    /// timm's names land on the conv net's paths.
    #[test]
    fn test_resnetv2_rekey() {
        let rekey = resnetv2_rekey(1000, 2048, true);
        assert_rekeys(
            &rekey,
            [
                ("stem.conv.weight", "stem.convs.0.conv.weight"),
                ("stem.conv1.weight", "stem.convs.0.conv.weight"),
                ("stem.norm2.bias", "stem.convs.1.norm.bias"),
                ("stem.conv3.weight", "stem.convs.2.conv.weight"),
                (
                    "stages.0.blocks.0.norm1.weight",
                    "stages.0.units.0.pre.norm.weight",
                ),
                (
                    "stages.0.blocks.0.conv1.weight",
                    "stages.0.units.0.convs.0.conv.weight",
                ),
                (
                    "stages.3.blocks.2.norm3.bias",
                    "stages.3.units.2.convs.2.norm.bias",
                ),
                (
                    "stages.1.blocks.0.downsample.conv.weight",
                    "stages.1.units.0.shortcut.conv.weight",
                ),
                ("norm.weight", "final_conv.norm.weight"),
                ("head.fc.weight", "head.fc.weight"),
            ],
        );
    }

    /// The hybrid `ViT`'s backbone's names: a normed stem, `Bottleneck`s
    /// whose `norm1` is their first conv's, normed shortcuts; a stem alone's
    /// bare `conv` and `norm`.
    #[test]
    fn test_non_preact_rekey() {
        let rekey = resnetv2_rekey(0, 1024, false);
        assert_rekeys(
            &rekey,
            [
                ("stem.conv.weight", "stem.convs.0.conv.weight"),
                ("stem.norm.weight", "stem.convs.0.norm.weight"),
                (
                    "stages.0.blocks.0.norm1.weight",
                    "stages.0.units.0.convs.0.norm.weight",
                ),
                (
                    "stages.2.blocks.8.conv3.weight",
                    "stages.2.units.8.convs.2.conv.weight",
                ),
                (
                    "stages.1.blocks.0.downsample.norm.bias",
                    "stages.1.units.0.shortcut.norm.bias",
                ),
                ("conv.weight", "stem.convs.0.conv.weight"),
                ("norm.bias", "stem.convs.0.norm.bias"),
            ],
        );
    }
}
