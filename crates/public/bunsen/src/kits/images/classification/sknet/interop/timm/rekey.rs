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
