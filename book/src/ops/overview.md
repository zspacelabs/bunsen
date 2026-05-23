# Ops Overview

`bunsen::ops` is a *functional* tensor API: standalone functions and
small configuration value-objects that extend the surface of
`burn::tensor::Tensor` itself. It's where the utilities go that you'd
reach for inside a `Module::forward` body but that don't have a home
on `Tensor` upstream.

Nothing in `ops` owns trainable parameters &mdash; that's the job of
[`bunsen::blocks`](../blocks/overview.md). The two layers compose: a
block bundles a parametric layer (a `Linear`, a `Conv2d`, ...) with a
sequence of `ops::*` calls in its `forward`.

API: <https://docs.rs/bunsen/latest/bunsen/ops/>

## How `ops` fits in the stack

```text
   contracts        validate shapes between layers
       │
       ▼
   ops              pure functions over Tensors          ◄── you are here
       │
       ▼
   blocks           stateful parameter-owning Modules
       │
       ▼
   kits             whole models built from blocks
```

A `blocks` `forward` typically pulls in two or three `ops` calls
around its parametric core. Lifting those into named functions keeps
the block readable and makes the underlying math testable in
isolation &mdash; the `ops` are what get unit-tested, the `block`
asserts they're wired correctly.

## Map of the module

The contents are split across two sub-chapters by character:

- **[Tensor Functions](./tensor-functions.md)** &mdash; element-wise
  and shape-level helpers: range generation (`arange`), clamping
  (`ClampOp`), dropout (`drop`), noise (`NoiseConfig`), RMS
  normalization (`rms_norm`), and `repeat_interleave`.
- **[Convolution Support](./convolution.md)** &mdash; the larger
  `conv` submodule: shape arithmetic for conv outputs, functional
  convolution (`convolve_func_2d`) for kernels that aren't linear,
  and helper filters.

## Value-objects as configuration

Several of the "options" types in `ops` (notably
[`ClampOp`](https://docs.rs/bunsen/latest/bunsen/ops/clamp/struct.ClampOp.html)
and [`NoiseConfig`](https://docs.rs/bunsen/latest/bunsen/ops/noise/struct.NoiseConfig.html))
are designed to be embedded in `Config` structs elsewhere in the
crate. That's the intentional shape of the seam: when an op has
parameters worth naming, it has a value-object, and that
value-object is the unit of configuration. A `Module::Config` can
then `#[config(default = ...)]` one of these directly instead of
duplicating its fields.
