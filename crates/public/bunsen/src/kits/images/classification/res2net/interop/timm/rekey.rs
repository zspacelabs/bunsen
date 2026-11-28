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
