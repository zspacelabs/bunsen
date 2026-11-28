//! The state-dict names.

use crate::{
    burner::module::{
        Indices,
        Rekey,
    },
    kits::images::classification::resnet::interop::torchvision::{
        ResNetStateLayout,
        resnet_rekey,
    },
};

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
