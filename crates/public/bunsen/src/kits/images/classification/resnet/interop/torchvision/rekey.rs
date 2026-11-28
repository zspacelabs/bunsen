//! The state-dict names.

use alloc::format;

use crate::{
    blocks::images::models::conv_net::{
        ConvNetModel,
        ConvShortcut,
    },
    burner::module::{
        Indices,
        Rekey,
    },
};

/// What a `ResNet` state dict's names depend on, past its depths: how its
/// stem and its shortcuts are laid out.
///
/// The default is torchvision's (one stem conv, conv shortcuts).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResNetStateLayout {
    /// The stem's convs: 1 is `conv1` and `bn1`; timm's deep and tiered stems'
    /// 3 are a `conv1` sequence (`conv1.0`, `conv1.1`, `conv1.3`, `conv1.4`,
    /// `conv1.6`, its activations at 2 and 5) and then `bn1`.
    pub stem_convs: usize,

    /// Average-pool shortcuts: a block's `downsample.1` is the conv and
    /// `downsample.2` the norm (`downsample.0` is the parameterless pool);
    /// else `downsample.0` and `downsample.1`.
    pub avg_down: bool,
}

impl Default for ResNetStateLayout {
    fn default() -> Self {
        Self {
            stem_convs: 1,
            avg_down: false,
        }
    }
}

impl ResNetStateLayout {
    /// The layout of `model`'s checkpoints: its stem's convs, and whether its
    /// shortcuts pool. A net of `avg_down` shortcuts has one that pools (each
    /// that strides or dilates); timm's names every one of them
    /// `downsample.1` and `.2`, the unpooled ones too (their pool is
    /// `nn.Identity`).
    pub fn of(model: &ConvNetModel) -> Self {
        let avg_down = model
            .stages
            .iter()
            .flat_map(|stage| &stage.units)
            .filter_map(|unit| unit.shortcut.as_ref())
            .any(|shortcut: &ConvShortcut| shortcut.pool.is_some());
        Self {
            stem_convs: model.stem.convs.len(),
            avg_down,
        }
    }
}

/// torchvision's and timm's `ResNet` state-dict names, in `layout`, onto a
/// [`ConvNetModel`]'s
/// paths.
///
/// | torchvision, timm | bunsen |
/// |---|---|
/// | `conv1`, `bn1` | `stem.convs.0.conv`, `stem.convs.0.norm` |
/// | a deep stem's `conv1.{3k}`, `conv1.{3k+1}`, the last's `bn1` | `stem.convs.{k}.conv`, `stem.convs.{k}.norm` |
/// | `ResNet-RS`'s conv pool, `maxpool.0`, `maxpool.1` | `stem.pool.conv`, `stem.pool.norm` |
/// | `layer1` … `layer4` | `stages.0` … `stages.3`, each block `j` as `units.j` |
/// | a block's `conv{k}`, `bn{k}` | its `convs.{k-1}.conv`, `convs.{k-1}.norm` |
/// | a block's `downsample.0`, `downsample.1` | its `shortcut.conv`, `shortcut.norm` |
/// | an `avg_down` block's `downsample.1`, `downsample.2` | its `shortcut.conv`, `shortcut.norm` |
/// | a block's `se` (squeeze-excite `fc1`, `fc2`; ECA `conv`) | its `attn` |
/// | `fc` | `head.fc` |
///
/// A unit's path is its block's convs in order, whether basic or
/// bottleneck, so one map serves both. Batch-norm counters
/// (`num_batches_tracked`) are dropped; anything else is kept as it is, for
/// the load to report.
///
/// ```rust
/// use bunsen::kits::images::classification::resnet::interop::torchvision::{
///     ResNetStateLayout,
///     resnet_rekey,
/// };
///
/// let rekey = resnet_rekey(ResNetStateLayout::default());
/// assert_eq!(
///     rekey.apply("layer1.0.bn2.running_var").as_deref(),
///     Some("stages.0.units.0.convs.1.norm.running_var")
/// );
/// ```
pub fn resnet_rekey(layout: ResNetStateLayout) -> Rekey {
    let conv_block = |part: &'static str| Rekey::keep_rest().rename("", part);
    let (conv, norm) = if layout.avg_down {
        ("downsample.1.", "downsample.2.")
    } else {
        ("downsample.0.", "downsample.1.")
    };
    let unit = Rekey::keep_rest()
        .mount_indexed_with("conv", "convs.", Indices::From(1), conv_block("conv."))
        .mount_indexed_with("bn", "convs.", Indices::From(1), conv_block("norm."))
        .rename(conv, "shortcut.conv.")
        .rename(norm, "shortcut.norm.")
        .rename("se.", "attn.");
    let last = layout.stem_convs.saturating_sub(1);
    let mut rekey = Rekey::keep_rest().drop_suffix(".num_batches_tracked");
    if layout.stem_convs == 1 {
        rekey = rekey.rename("conv1.", "stem.convs.0.conv.");
    } else {
        for k in 0..layout.stem_convs {
            rekey = rekey.rename(format!("conv1.{}.", 3 * k), format!("stem.convs.{k}.conv."));
            if k < last {
                rekey = rekey.rename(
                    format!("conv1.{}.", 3 * k + 1),
                    format!("stem.convs.{k}.norm."),
                );
            }
        }
    }
    rekey
        .rename("bn1.", format!("stem.convs.{last}.norm."))
        .rename("maxpool.0.", "stem.pool.conv.")
        .rename("maxpool.1.", "stem.pool.norm.")
        .rename("fc.", "head.fc.")
        .mount_indexed_with(
            "layer",
            "stages.",
            Indices::From(1),
            Rekey::keep_rest().mount_indexed("", "units.", unit),
        )
}
