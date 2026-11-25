//! # `ResNet` as torchvision saves it
//!
//! torchvision's `ResNet` state dict (a `.pt`; `timm`'s checkpoints use the
//! same names) as bunsen's
//! [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel).
//! A checkpoint does not describe its own geometry: the catalog row's prefab
//! does.
//!
//! - **Names:** [`resnet_rekey`].
//! - **Loading** (feature `store`): `load_resnet_tensors`; with `cache`,
//!   `load_resnet`, a resource set's checkpoint.

use crate::burner::module::{
    Indices,
    Rekey,
};

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

/// torchvision's `ResNet` state-dict names (timm's are the same) onto a
/// [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel)'s
/// paths.
///
/// | torchvision | bunsen |
/// |---|---|
/// | `conv1`, `bn1` | `stem.convs.0.conv`, `stem.convs.0.norm` |
/// | `layer1` … `layer4` | `stages.0` … `stages.3`, each block `j` as `units.j` |
/// | a block's `conv{k}`, `bn{k}` | its `convs.{k-1}.conv`, `convs.{k-1}.norm` |
/// | a block's `downsample.0`, `downsample.1` | its `shortcut.conv`, `shortcut.norm` |
/// | `fc` | `head.fc` |
///
/// A unit's path is its block's convs in order, whether basic or
/// bottleneck, so one map serves both. Batch-norm counters
/// (`num_batches_tracked`) are dropped; anything else is kept as it is, for
/// the load to report.
///
/// ```rust
/// use bunsen::kits::images::classification::resnet::interop::torchvision::resnet_rekey;
///
/// let rekey = resnet_rekey();
/// assert_eq!(
///     rekey.apply("layer1.0.bn2.running_var").as_deref(),
///     Some("stages.0.units.0.convs.1.norm.running_var")
/// );
/// ```
pub fn resnet_rekey() -> Rekey {
    let conv_block = |part: &'static str| Rekey::keep_rest().rename("", part);
    let unit = Rekey::keep_rest()
        .mount_indexed_with("conv", "convs.", Indices::From(1), conv_block("conv."))
        .mount_indexed_with("bn", "convs.", Indices::From(1), conv_block("norm."))
        .rename("downsample.0.", "shortcut.conv.")
        .rename("downsample.1.", "shortcut.norm.");
    Rekey::keep_rest()
        .drop_suffix(".num_batches_tracked")
        .rename("conv1.", "stem.convs.0.conv.")
        .rename("bn1.", "stem.convs.0.norm.")
        .rename("fc.", "head.fc.")
        .mount_indexed_with(
            "layer",
            "stages.",
            Indices::From(1),
            Rekey::keep_rest().mount_indexed("", "units.", unit),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// torchvision's names onto bunsen's, one of each kind: the stem, a
    /// basic block's and a bottleneck's convolutions and norms, a
    /// downsample, the head; the counters dropped.
    #[test]
    fn test_resnet_rekey() {
        let rekey = resnet_rekey();
        for (torchvision, bunsen) in [
            ("conv1.weight", Some("stem.convs.0.conv.weight")),
            ("bn1.running_mean", Some("stem.convs.0.norm.running_mean")),
            ("bn1.num_batches_tracked", None),
            (
                "layer1.0.conv1.weight",
                Some("stages.0.units.0.convs.0.conv.weight"),
            ),
            (
                "layer1.1.bn2.bias",
                Some("stages.0.units.1.convs.1.norm.bias"),
            ),
            (
                "layer4.2.conv3.weight",
                Some("stages.3.units.2.convs.2.conv.weight"),
            ),
            (
                "layer2.0.downsample.0.weight",
                Some("stages.1.units.0.shortcut.conv.weight"),
            ),
            (
                "layer2.0.downsample.1.running_var",
                Some("stages.1.units.0.shortcut.norm.running_var"),
            ),
            ("layer3.5.bn3.num_batches_tracked", None),
            ("fc.weight", Some("head.fc.weight")),
        ] {
            assert_eq!(rekey.apply(torchvision).as_deref(), bunsen, "{torchvision}");
        }
    }
}
