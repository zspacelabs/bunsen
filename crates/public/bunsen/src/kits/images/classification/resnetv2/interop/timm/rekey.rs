//! The state-dict names.

use crate::burner::module::{
    Indices,
    Rekey,
    TensorLoadTransform,
};

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
