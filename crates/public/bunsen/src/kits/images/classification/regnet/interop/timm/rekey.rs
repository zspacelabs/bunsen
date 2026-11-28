//! The state-dict names.

use crate::burner::module::{
    Indices,
    Rekey,
};

/// timm's `RegNet` state-dict names onto a
/// [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel)'s
/// paths.
///
/// | timm | bunsen |
/// |---|---|
/// | `stem.conv`, `stem.bn` | `stem.convs.0.conv`, `stem.convs.0.norm` |
/// | `s1` … `s4`, each block `b{j}` | `stages.0` … `stages.3`, each as `units.{j-1}` |
/// | a block's `conv{k}.conv`, `conv{k}.bn` | its `convs.{k-1}.conv`, `convs.{k-1}.norm` |
/// | a block's `se` (`fc1`, `fc2`) | its `attn` |
/// | a block's `downsample.conv`, `downsample.bn` | its `shortcut.conv`, `shortcut.norm` |
/// | an `avg` block's `downsample.1` (after the pool, `downsample.0`) | its `shortcut` |
/// | `RegNetV`'s `stem.weight`, a block's `norm1`, `norm{k}`, `conv{k}.weight`, `downsample.weight` | `stem.convs.0.conv.weight`, its `pre.norm`, `convs.{k-1}.norm`, `convs.{k-1}.conv.weight`, `shortcut.conv.weight` |
/// | `RegNetZ`'s `final_conv.conv`, `final_conv.bn` | `final_conv.conv`, `final_conv.norm` |
/// | `head.fc` | `head.fc` |
///
/// Batch-norm counters (`num_batches_tracked`) are dropped; anything else is
/// kept as it is, for the load to report.
///
/// ```rust
/// use bunsen::kits::images::classification::regnet::interop::timm::regnet_rekey;
///
/// assert_eq!(
///     regnet_rekey().apply("s2.b3.conv2.bn.running_var").as_deref(),
///     Some("stages.1.units.2.convs.1.norm.running_var")
/// );
/// ```
pub fn regnet_rekey() -> Rekey {
    // A conv block's `conv` and `bn`, or a pre-activation net's bare conv.
    let conv = Rekey::keep_rest()
        .rename("bn.", "norm.")
        .exact("weight", "conv.weight");
    let unit = Rekey::keep_rest()
        .mount_indexed_with("conv", "convs.", Indices::From(1), conv)
        .rename("norm1.", "pre.norm.")
        .rename("norm2.", "convs.1.norm.")
        .rename("norm3.", "convs.2.norm.")
        .rename("se.", "attn.")
        .rename("downsample.conv.", "shortcut.conv.")
        .rename("downsample.bn.", "shortcut.norm.")
        .rename("downsample.1.conv.", "shortcut.conv.")
        .rename("downsample.1.bn.", "shortcut.norm.")
        .exact("downsample.weight", "shortcut.conv.weight")
        .exact("downsample.1.weight", "shortcut.conv.weight");
    Rekey::keep_rest()
        .drop_suffix(".num_batches_tracked")
        .rename("stem.conv.", "stem.convs.0.conv.")
        .rename("stem.bn.", "stem.convs.0.norm.")
        .exact("stem.weight", "stem.convs.0.conv.weight")
        .rename("final_conv.bn.", "final_conv.norm.")
        .mount_indexed_with(
            "s",
            "stages.",
            Indices::From(1),
            Rekey::keep_rest().mount_indexed_with("b", "units.", Indices::From(1), unit),
        )
}
