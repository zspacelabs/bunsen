//! # `ResNet` as torchvision saves it
//!
//! torchvision's `ResNet` state dict (a `.pt`; `timm`'s checkpoints use the
//! same names) as bunsen's [`ResNet`](crate::kits::images::resnet::ResNet).
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
/// [`ResNet`](crate::kits::images::resnet::ResNet)'s module paths.
///
/// | torchvision | bunsen |
/// |---|---|
/// | `conv1`, `bn1` | `input_cb.conv`, `input_cb.norm` |
/// | `layer1` … `layer4` | `layers.0` … `layers.3`, each block `N` as `blocks.N` |
/// | a block's `conv{k}`, `bn{k}` | its `cb{k}.conv`, `cb{k}.norm` |
/// | a block's `downsample.0`, `downsample.1` | its `downsample.conv`, `downsample.norm` |
/// | `fc` | `output_fc` |
///
/// Batch-norm counters (`num_batches_tracked`) are dropped; anything else
/// is kept as it is, for the load to report.
///
/// ```rust
/// use bunsen::kits::images::resnet::interop::torchvision::resnet_rekey;
///
/// let rekey = resnet_rekey();
/// assert_eq!(
///     rekey.apply("layer1.0.bn2.running_var").as_deref(),
///     Some("layers.0.blocks.0.cb2.norm.running_var")
/// );
/// ```
pub fn resnet_rekey() -> Rekey {
    let conv_block = |part: &'static str| Rekey::keep_rest().rename("", part);
    let block = Rekey::keep_rest()
        .mount_indexed("conv", "cb", conv_block("conv."))
        .mount_indexed("bn", "cb", conv_block("norm."))
        .rename("downsample.0.", "downsample.conv.")
        .rename("downsample.1.", "downsample.norm.");
    Rekey::keep_rest()
        .drop_suffix(".num_batches_tracked")
        .rename("conv1.", "input_cb.conv.")
        .rename("bn1.", "input_cb.norm.")
        .rename("fc.", "output_fc.")
        .mount_indexed_with(
            "layer",
            "layers.",
            Indices::From(1),
            Rekey::keep_rest().mount_indexed("", "blocks.", block),
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
            ("conv1.weight", Some("input_cb.conv.weight")),
            ("bn1.running_mean", Some("input_cb.norm.running_mean")),
            ("bn1.num_batches_tracked", None),
            (
                "layer1.0.conv1.weight",
                Some("layers.0.blocks.0.cb1.conv.weight"),
            ),
            ("layer1.1.bn2.bias", Some("layers.0.blocks.1.cb2.norm.bias")),
            (
                "layer4.2.conv3.weight",
                Some("layers.3.blocks.2.cb3.conv.weight"),
            ),
            (
                "layer2.0.downsample.0.weight",
                Some("layers.1.blocks.0.downsample.conv.weight"),
            ),
            (
                "layer2.0.downsample.1.running_var",
                Some("layers.1.blocks.0.downsample.norm.running_var"),
            ),
            ("layer3.5.bn3.num_batches_tracked", None),
            ("fc.weight", Some("output_fc.weight")),
        ] {
            assert_eq!(rekey.apply(torchvision).as_deref(), bunsen, "{torchvision}");
        }
    }
}
