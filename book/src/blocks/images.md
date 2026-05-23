# Image Blocks

`bunsen::blocks::images` collects the building blocks used by
2-D vision models &mdash; convolutional composites, patch
tokenization, same-padding pooling, and stochastic-depth-style
regularization layers.

API: <https://docs.rs/bunsen/latest/bunsen/blocks/images/>

## Conv composites

The [`conv`](https://docs.rs/bunsen/latest/bunsen/blocks/images/conv/index.html)
submodule packages the conv-plus-something composites that show up
across ResNet, EfficientNet, and friends.

### `ConvNorm2d`

[`ConvNorm2d`](https://docs.rs/bunsen/latest/bunsen/blocks/images/conv/struct.ConvNorm2d.html)
is the standard `Conv2d` + `BatchNorm` pairing with a single
`forward`. Beyond the convenience of one module instead of two, it
carries `zero_init_norm()` &mdash; the "zero-initialize the last batch
norm in a residual branch" trick used by `ResNet` and successors to
make residual branches start as identities.

### `CNA2d`

[`CNA2d`](https://docs.rs/bunsen/latest/bunsen/blocks/images/conv/struct.CNA2d.html)
is the more general *Conv / Norm / Activation* block. Beyond the
basic forward, it provides:

- `match_norm_features()` &mdash; adapts a generic
  `NormalizationConfig` (`BatchNorm::new(0)`, `RmsNorm::new(0)`, etc.)
  to the right channel count after the conv. Lets callers pass a
  norm config without knowing the channel count yet.
- `map_forward(f)` &mdash; runs the conv and norm, then hands the
  intermediate tensor to a user closure before the activation. Useful
  for inserting attention, channel reweighting, or per-residual
  side-effects without copying out the rest of the block.

## Patching

[`patching`](https://docs.rs/bunsen/latest/bunsen/blocks/images/patching/index.html)
holds patch tokenization, the entry point for transformer-style
vision models.

### `PatchEmbed`

[`PatchEmbed`](https://docs.rs/bunsen/latest/bunsen/blocks/images/patching/struct.PatchEmbed.html)
is ViT-style patch tokenization: it takes an $H \times W \times C$
image, splits it into non-overlapping $p \times p$ patches, and
projects each patch into an embedding, producing a sequence of
$N = (H/p) \times (W/p)$ tokens of width `embed_dim`.

## Pooling

[`pool`](https://docs.rs/bunsen/latest/bunsen/blocks/images/pool/index.html)
holds the pooling layers that don't fit `burn`'s defaults.

### `AvgPool2dSame`

[`AvgPool2dSame`](https://docs.rs/bunsen/latest/bunsen/blocks/images/pool/struct.AvgPool2dSame.html)
is TensorFlow-style *same* padding for average pooling &mdash;
asymmetric where needed to keep the spatial dimensions of the output
aligned with `ceil(input / stride)`. Helpers `get_same_padding` and
`pad_same` are exposed for the underlying arithmetic, useful when
you're matching a Keras / TF-Slim reference implementation.

## Stochastic regularization

[`drop`](https://docs.rs/bunsen/latest/bunsen/blocks/images/drop/index.html)
collects regularization layers that drop *structured* pieces of the
activations rather than individual scalars.

### `DropBlock`

[`DropBlock`](https://docs.rs/bunsen/latest/bunsen/blocks/images/drop/drop_block/index.html)
is structured spatial dropout from
[Ghiasi et al., 2018](https://arxiv.org/pdf/1810.12890.pdf): instead
of dropping independent pixels, it drops contiguous blocks of
activations. For convnets this acts as a substantially stronger
regularizer than plain dropout, because adjacent pixels are highly
correlated and independent dropout barely removes information.

### `DropPath`

[`DropPath`](https://docs.rs/bunsen/latest/bunsen/blocks/images/drop/drop_path/index.html)
is stochastic depth from
[Huang et al., 2016](https://arxiv.org/abs/1603.09382): with some
probability, the *entire* residual branch is zeroed for a given
sample, so the network sees a shorter effective depth on each
training step.

### Supporting types

- **`progressive_dpr`** &mdash; rate table that linearly ramps drop
  rates over a network's depth, matching the SWIN V2 / `timm`
  convention of giving deeper blocks higher drop probabilities.
- **`SizeConfig`** &mdash; a small enum describing a size as either
  `Default`, a `Ratio(f64)`, or a `Fixed(usize)`. Used by the drop
  layers when the effective region size is relative to a wrapped
  layer's spatial dimensions.
