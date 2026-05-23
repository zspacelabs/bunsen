# Burner Overview

`bunsen::burner` is the layer that sits *next to* `burn` itself. Where
[`bunsen::blocks`](../blocks/overview.md) gives you new `Module`s and
[`bunsen::kits`](../kits/bimm.md) gives you whole models, `burner`
gives you the infrastructure for working with `burn` modules,
optimizers, and records at a level `burn`'s default surface doesn't
expose.

API: <https://docs.rs/bunsen/latest/bunsen/burner/>

The module currently contains:

- **[`descriptors`](https://docs.rs/bunsen/latest/bunsen/burner/descriptors/)** —
  type-erased descriptors for tensors and parameters. A `TensorParamDesc`
  captures the metadata of any `Param<Tensor<B, R, K>>` (its `ParamId`,
  `Shape`, rank, dtype, kind) without carrying the generics that the
  underlying tensor type does. This is the lingua franca used by the
  reflection and optimizer machinery to talk about parameters
  uniformly.

- **[`module`](https://docs.rs/bunsen/latest/bunsen/burner/module/)** —
  module-side helpers: a type-mapper for `Module` field re-typing, and
  (under `features = ["reflection"]`) the XML/XPath reflection layer
  documented in [Module Introspection](./module-introspection.md).

- **[`optim`](https://docs.rs/bunsen/latest/bunsen/burner/optim/)** —
  optimizer extensions (under `features = ["train"]`). The headline
  feature is the `GroupOptimizerAdaptor{N}` family, which lets you
  mount *multiple* optimizers on a single `Module`, each driving a
  disjoint group of parameters. Covered in
  [Composite Optimizers](./composite-optimizers.md).

- **[`record`](https://docs.rs/bunsen/latest/bunsen/burner/record/)** —
  helpers for working with `burn::record` types.

- **[`tensor`](https://docs.rs/bunsen/latest/bunsen/burner/tensor/)** —
  tensor helpers that don't fit neatly in
  [`bunsen::ops`](../ops/overview.md), including a `DataView`
  abstraction.

- **[`distribution`](https://docs.rs/bunsen/latest/bunsen/burner/distribution/)** —
  distribution-related utilities.

## When to reach for `burner`

Most code that uses `bunsen` won't import from `burner` at all — the
ops, blocks, and contracts are what you write models against. You
reach for `burner` when:

- you need to **introspect a model** to drive something else: split
  parameters into groups, audit shapes, build a parameter manifest;
- you need to **compose optimizers** (e.g., Muon for matrix
  parameters, AdamW for everything else; different learning rates per
  group);
- you need to **carry tensor / parameter metadata** in non-generic
  code paths (a `TensorParamDesc` instead of a `Param<Tensor<B, R,
  K>>`);
- you need to **manipulate records** outside of what the derive macros
  give you for free.

The two largest user-facing surfaces — the reflection / XPath query
machinery and the group-optimizer family — get their own chapters in
this section.
