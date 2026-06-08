# Bunsen

*by [ZSpaceLabs](https://zspacelabs.ai)*

[![Crates.io Version](https://img.shields.io/crates/v/bunsen)](https://crates.io/crates/bunsen)
[![Documentation](https://img.shields.io/docsrs/bunsen)](https://docs.rs/bunsen/latest/bunsen/)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)
[![Discord](https://img.shields.io/discord/1475229838754316502?label=discord)](https://discord.gg/vBgXHWCeah)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zspacelabs/bunsen)

`bunsen` aims to be a "batteries included" complementary
community standard library for extending the [burn](https://burn.dev) tensor library.

# Book

Read the [bunsen book](https://zspacelabs.ai/bunsen/book)

## Organization

### Burn Extensions

* `bunsen::burner` - this is a library of `burn::module::Module` lifecycle
  components that extend the current functionality of burn.
    * `bunsen::burner::module::reflection` has powerful tools for dynamic `burn::module::Module` reflection.
    * `bunsen::burner::optim` has parameter-group optimizer extensions.
* `contracts` - this is a library of runtime tensor-shape contracts.

### Component Libraries

* `blocks` - this is a library of `burn::module::Module` components.
  This includes simple inner layers, recurrent utility blocks, and entire
  model families.
* `ops` - this is a library `burn::tensor::Tensor` operations.
* `kits` - this is a library of full models and simulation kits.

### App and Testing Support Libs

* `errors` - this is a library of error types and tooling.
* `support` - this is a library of support functions for bunsen, including
  testing tooling which may be useful for clients.
* `zspace` - this is a library of z-space / index utilities.

# API Examples

A "good parts" survey of some of `bunsen`'s features. See the
[docs](https://docs.rs/bunsen/latest/bunsen/) and the
[book](https://zspacelabs.ai/bunsen/book) for the full API.

## Shape Contracts

`bunsen::contracts` provides allocation-free, always-on runtime tensor-shape
contracts. A contract pairs paper-style shape notation with runtime checking:
it asserts that a tensor's shape matches a declared pattern *and* unpacks named
dimensions for downstream use, catching shape errors at their source. Single
checks run in ~160 ns; the amortized periodic variants average a few ns.

```rust
use bunsen::contracts::*;

// Assert and unpack named dimensions in one shot:
let shape = [4, 5, 3];
let [h, w, c] = unpack_shape_contract!(["h", "w", "c"], &shape);
assert_eq!((h, w, c), (4, 5, 3));

// Patterns support products, sums, and bound dimensions — e.g. a windowed
// image where height = h_wins * window_size:
let [b, h_wins, w_wins, c] = unpack_shape_contract!(
    [
        "batch",
        "height" = "h_wins" * "window_size",
        "width" = "w_wins" * "window_size",
        "channels"
    ],
    &shape,
    &["batch", "h_wins", "w_wins", "channels"],
    &[("window_size", 4)],
);
```

In hot loops, use `assert_shape_contract_periodically!` to amortize the check
via exponential backoff while still catching regressions:

```rust
use bunsen::contracts::*;

assert_shape_contract_periodically!(
    ["batch", "planes", "height", "width"],
    &x.dims(),
    &[("planes", planes), ("height", height), ("width", width)]
);
```

## TensorData Index Views

`TensorDataIndexView` and `TensorDataIndexMutView` wrap burn's low-level
`TensorData` to give ergonomic multi-dimensional element access via bracket
notation — `view[&[i, j]]` — instead of manually flattening indices. The views
deref to the underlying `TensorData`, so `.shape` and friends are right there.
Handy for inspecting or patching raw tensor data without building full tensors.

```rust
use bunsen::burner::tensor::*;
use burn::prelude::*;

let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
let view: TensorDataIndexView<f64> = TensorDataIndexView::view( & data);

// Deref exposes the underlying TensorData metadata:
assert_eq!(view.shape, [2, 2]);

assert_eq!(view[&[0, 0]], 1.0);
assert_eq!(view[&[1, 1]], 4.0);
```

The mut view supports in-place writes:

```rust
use bunsen::burner::tensor::*;
use burn::prelude::*;

let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
let mut view: TensorDataIndexMutView<f64> = TensorDataIndexMutView::view( & mut data);

view[& [0, 0]] = 10.0;
assert_eq!(view[&[0, 0]], 10.0);
```

## XML Module Reflection

`bunsen::burner::module::reflection::XmlModuleTree` turns any burn `Module`
into a queryable XML meta-description of its structure. This enables
type-erased introspection and XPath-style parameter selection — e.g. "every
rank-2 weight under the transformer blocks" — which is exactly what you need to
slice a model into parameter groups for per-group optimizers.

Take a small container module:

```rust
use burn::nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
struct Block<B: Backend> {
    linear: Linear<B>,
    norm: LayerNorm<B>,
}

let module = Block::<B> {
linear: LinearConfig::new(4, 8).init( & device),
norm: LayerNormConfig::new(8).init( & device),
};
```

Reflecting it yields a queryable XML description of the structure:

```rust
use bunsen::burner::module::reflection::XmlModuleTree;

// As XmlModuleTree holds a non-Send active query environment, it must be `mut`
// to run queries.
let mut mtree = XmlModuleTree::build( & module);

// Dump the structure to inspect it:
println!("{}", mtree.to_xml(true));

// Select parameters by XPath and collect their ParamIds — e.g. just the
// rank-2 Linear weights:
let matrix_params = mtree
.select_params("Block/Linear/*[@name='weight',@rank=2]")
.to_param_ids() ?;
```

The dumped structure mirrors the module's fields, with each `@name` taken from
the struct field and a stable `param_id` per tensor:

```xml

<XmlModuleTree version="0.22.2">
    <Structure>
        <Block id="n:1" class="struct">
            <Linear id="n:2" name="linear" class="struct">
                <Param id="n:3" name="weight" param_id="si0gu6g09smnm" class="tensor" kind="Float" dtype="F32"
                       shape="4 8" rank="2"/>
                <Param id="n:4" name="bias" param_id="sai8ouqd8krmg" class="tensor" kind="Float" dtype="F32" shape="8"
                       rank="1"/>
            </Linear>
            <LayerNorm id="n:5" name="norm" class="struct">
                <Param id="n:6" name="gamma" param_id="7ufbn5ojagumq" class="tensor" kind="Float" dtype="F32" shape="8"
                       rank="1"/>
                <Param id="n:7" name="beta" param_id="ho9nkq19bnm6i" class="tensor" kind="Float" dtype="F32" shape="8"
                       rank="1"/>
            </LayerNorm>
        </Block>
    </Structure>
</XmlModuleTree>
```

## Blocks & Ops

`bunsen::blocks` is a library of `burn::module::Module` building blocks (stateful
layers with trainable parameters), and `bunsen::ops` is a library of stateless
`Tensor` operations. A survey of what's available:

```text
blocks/
├── transformers/
│   ├── attention   — CausalSelfAttention, scaled_dot_product_attention,
│   │                 causal_mask, KVCache (autoregressive decode cache)
│   ├── embedding   — RotaryEmbedding (RoPE)
│   └── mlp          — Mlp feed-forward block, layer_norm_mlp
└── images/
    ├── conv         — ConvNorm2d, ConvBlock2d (Conv → Norm → Activation)
    ├── patching     — PatchEmbed (ViT-style patch tokenizer)
    ├── pool         — AvgPool2dSame (TF-style SAME padding)
    └── drop         — DropBlock2d, DropPath (stochastic depth)

ops/
├── arange     — float_arange, float_linspace (+ Vec variants)
├── noise      — noise, noise_like (distribution sampling + clamp)
├── clamp      — ClampOp (optional min/max bounds)
├── drop       — dropout, drop_block_2d
├── norm       — rms_norm (RMS normalization)
├── repeat     — repeat_interleave (NumPy/PyTorch semantics)
├── conv       — conv output-shape arithmetic, same-padding helpers,
│                convolve_func_2d
└── embedding  — unembed, iota_embedding, identity_embedding
```

# Examples

The `bunsen` repo includes a number of complex demos. The goal of the demos is to showcase the capabilities of the
library; while also collecting a working edge of problems which could and should be improved by further development.

See [`examples/`](https://github.com/zspacelabs/bunsen/tree/main/examples/) for the full index.

# License

`bunsen` is distributed under the terms of both the MIT license and the Apache License
(Version 2.0).
