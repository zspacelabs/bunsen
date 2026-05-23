# Overview

This page is a tour through the major pieces of `bunsen`, in roughly
the order you'd encounter them building a model on top of `burn`. The
sections below each have a dedicated chapter elsewhere in the book;
this page is the orientation map.

## Tensor Contracts

Shape errors in tensor code are hard to diagnose: a bad reshape
produces the wrong meaning, not an exception, and the eventual
failure points at a symptom three layers downstream.
[`bunsen::contracts`](../contracts/overview.md) lets you write the
shape of a tensor the way a paper would,

$$
x \in \RR^{B \times C \times H \times W}
$$

and then check it at module boundaries, in one step that both
*asserts* the pattern and *unpacks* the named dimensions you'll
need:

```rust
# use bunsen::contracts::unpack_shape_contract;
# let tensor = [12, 3 * 4, 5 * 4, 3];
let [b, h_wins, w_wins, c] = unpack_shape_contract!(
    [
        "batch",
        "height" = "h_wins" * "window_size",
        "width"  = "w_wins" * "window_size",
        "channels",
    ],
    &tensor,
    &["batch", "h_wins", "w_wins", "channels"],
    &[("window_size", 4)],
);
```

If the shape doesn't match, the failure is loud and specific &mdash;
it names the offending dimension, the pattern it failed against, and
the parameter bindings in scope. The check is fast enough (~160 ns
per unpack on a four-dimensional shape) to stay enabled in release
builds.

Contracts are the *shared vocabulary* the rest of `bunsen` is built
on; almost every block in the crate uses them at its module
boundaries.

See [Contracts](../contracts/overview.md) for the pattern grammar,
the full macro surface, the cost-control mechanisms, and the
rationale behind the design.

## Ops &mdash; pure tensor functions

[`bunsen::ops`](../ops/overview.md) is the *functional* layer: pure
functions over tensors that extend `burn::tensor::Tensor`'s surface
without owning any trainable parameters. Range generators, clamping,
dropout, noise, RMSNorm, repeat-interleave, and a substantial
collection of convolution-shape arithmetic and functional-conv
helpers.

A typical block's `forward` is a parametric layer (a `Linear`, a
`Conv2d`) wrapped in two or three `ops` calls. Lifting the
non-parametric work out makes the block readable and makes the
underlying math testable in isolation.

```text
   contracts        validate shapes between layers
       │
       ▼
   ops              pure functions over Tensors
       │
       ▼
   blocks           stateful parameter-owning Modules
       │
       ▼
   kits             whole models built from blocks
```

The `ops` surface also includes small value-object types
(`ClampOp`, `NoiseConfig`) designed to be embedded directly into a
`#[derive(Config)]` struct of a downstream module.

## Blocks &mdash; reusable Module components

[`bunsen::blocks`](../blocks/overview.md) is the stateful layer:
`burn::module::Module` building blocks that own parameters and can be
trained. Organized by domain:

- **Transformers** &mdash; multi-head causal self-attention with
  grouped-query support, a KV cache for autoregressive decoding,
  scaled-dot-product attention helpers, and rotary positional
  embeddings.
- **Images** &mdash; conv composites (`ConvNorm2d`, `CNA2d`),
  ViT-style patch tokenization (`PatchEmbed`), TF-style same-padding
  pooling, and stochastic regularization layers (`DropBlock`,
  `DropPath`).

Every block follows the patterns documented in
[Building Reusable Modules](../guides/building-reusable-modules.md):
`Meta` traits for cross-module introspection, Contract&rarr;Structure
config splits where the user-facing knobs differ from the
implementation parameter list, and inline shape contracts at module
boundaries.

## Kits &mdash; complete domain implementations

[`bunsen::kits`](../kits/bimm.md) is for *whole* things you pick up
and use end-to-end. The current categories:

- **[`bimm`](../kits/bimm.md)** &mdash; Bunsen/Burn Image Models, an
  incremental port of the `timm` ecosystem. Currently includes the
  `ResNet` family with pretrained-weight loaders and the Swin
  Transformer V2 family.
- **[`gpts`](../kits/gpts.md)** &mdash; full GPT / LLM variants.
  Currently includes `NanoChat`, a compact GPT suitable for
  experimentation and fine-tuning.
- **[`sims`](../kits/sims.md)** &mdash; iterative tensor simulations.
  Currently includes Conway's Game of Life in 2D and 3D, and a D2Q9
  lattice-Boltzmann fluid solver.

Kits compose the lower layers: each one uses `contracts`, `ops`, and
`blocks` (and, where training matters, `burner`) to deliver a full
domain solution rather than a building block. They're also where to
look for worked examples of every convention in real code.

## Burner &mdash; `burn`-adjacent infrastructure

[`bunsen::burner`](../burner/overview.md) is the infrastructure
layer: the pieces that sit *next to* `burn` itself, not on top of
its tensor surface. Most code that uses `bunsen` won't import from
`burner` at all &mdash; you reach for it when you need to:

- **introspect a model** generically &mdash; the
  [reflection layer](../burner/module-introspection.md) turns a
  `Module` into a queryable XML document with an XPath query API,
  for "select every rank-2 weight under the transformer blocks"
  problems;
- **compose optimizers** &mdash; the
  [`GroupOptimizerAdaptor{N}`](../burner/composite-optimizers.md)
  family mounts multiple optimizers on a single module (e.g. Muon
  for matrix parameters, AdamW for the rest), each driving a disjoint
  group of parameters, with per-group learning-rate selectors;
- **carry tensor metadata** in non-generic code paths
  (`TensorParamDesc`);
- **work with `burn::record`** outside what the derive macros
  provide for free.

The reflection and group-optimizer pieces compose: the canonical
pattern is to walk a model with `XmlModuleTree`, slice it into
parameter groups with XPath, and hand the groups to a
`GroupOptimizerAdaptor`. The NanoChat training demo
(`demos/chat/examples/train`) does exactly that.
