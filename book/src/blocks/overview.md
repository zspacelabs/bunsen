# Blocks Overview

`bunsen::blocks` is the library of reusable `burn::module::Module`
components &mdash; the *parts* you compose into larger models. Where
[`bunsen::ops`](../ops/overview.md) supplies pure functional tensor
operations, `blocks` supplies the stateful layers that own trainable
parameters. Where [`bunsen::kits`](../kits/bimm.md) supplies whole
end-to-end models, `blocks` supplies the sub-modules those kits are
assembled from.

A typical user picks a kit; an author building a new model reaches
into `blocks` and stitches it together.

API: <https://docs.rs/bunsen/latest/bunsen/blocks/>

## Map of the module

`blocks` is organized by *domain*: each major sub-chapter covers one
domain's worth of building blocks.

- **[Transformers](./transformers.md)** &mdash; attention
  (`CausalSelfAttention`, scaled-dot-product attention helpers,
  `KVCache`) and positional embedding (`RotaryEmbedding`).
- **[Images](./images.md)** &mdash; convolutional composites
  (`ConvNorm2d`, `CNA2d`), patch tokenization (`PatchEmbed`),
  same-padding pooling (`AvgPool2dSame`), and stochastic
  regularization layers (`DropBlock`, `DropPath`).

## Conventions used across `blocks`

Every block follows the patterns described in
[Building Reusable Modules](../guides/building-reusable-modules.md):

- **`{Block}Meta` trait** when other modules will need to introspect
  this one at runtime. `CausalSelfAttentionMeta` is implemented on
  both `CausalSelfAttentionConfig` and `CausalSelfAttention<B>`, so a
  parent transformer can ask `n_head` / `head_dim` of whichever form
  it's holding without copying metadata.
- **Contract &rarr; Structure config split** when the user-facing
  knobs differ from the implementation parameter list. The
  `ResidualBlock` triple in `bimm::resnet` is the in-tree reference
  example.
- **Inline shape contracts** at module boundaries via
  [`bunsen::contracts`](../contracts/overview.md). Every block's
  `forward` documents its shape in the docstring; the matching
  `unpack_shape_contract!` / `assert_shape_contract_periodically!`
  calls turn that docstring into a runtime check.

## Where blocks come from

Most blocks land here in one of three ways:

1. **Direct ports.** Implementations of well-known layers from the
   `timm` / `torchvision` / reference-paper ecosystems, kept in `burn`
   form so they're available across the entire `bunsen` stack.
2. **Extraction from kits.** When a layer in a [`kits`](../kits/bimm.md)
   model is reusable beyond that one model, it gets promoted out of
   the kit and into `blocks`.
3. **`burn`-gap fills.** Layers that aren't in `burn` core today but
   are needed by everything else (e.g., `AvgPool2dSame`).

The result is a curated catalog, not a comprehensive one &mdash; new
blocks land here when a kit or downstream user needs them.
