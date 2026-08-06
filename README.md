# Bunsen

*by [ZSpaceLabs](https://zspacelabs.ai)*

[![Crates.io Version](https://img.shields.io/crates/v/bunsen)](https://crates.io/crates/bunsen)
[![Documentation](https://img.shields.io/docsrs/bunsen)](https://docs.rs/bunsen/latest/bunsen/)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)
[![Discord](https://img.shields.io/discord/1475229838754316502?label=discord)](https://discord.gg/vBgXHWCeah)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zspacelabs/bunsen)

`bunsen` aims to be a "batteries included" complementary community standard library for extending
the [burn](https://burn.dev) tensor library.

# Book

Read the [bunsen book](https://zspacelabs.ai/bunsen/book)

# Crates

## Public / API Crates

* [`bunsen-firehose`](crates/bunsen-firehose) — a columnar dataloader / processing pipeline, with a burn batcher bridge.

## Utility Crates

* [`bunsen-contracts-macros`](crates/bunsen-contracts-macros) — the
  `shape_contract![]` proc-macro backing `bunsen`'s runtime tensor-shape contracts.

## Experimental Crates

These represent complex-interface + work-in-progress, unstable interface extensions to `bunsen`; particulary those which
incur large dependencies or are not yet ready for general consumption.

* [`bunsen-firehose-image`](crates/bunsen-firehose-image) — image loading, augmentation, and tensor-conversion operators
  for `bunsen-firehose`.
* [`bunsen`](crates/bunsen) — the main "batteries included" library extending burn: model blocks, kits, ops, contracts,
  and support tooling.
* [`bunsen-preview-chat-dataloader`](crates/bunsen-preview-chat-dataloader) — *(preview)* an Arrow-backed chat
  dataloader with tokenization for LLM training.

# API Examples

A "good parts" survey of some of `bunsen`'s features. See the
[docs](https://docs.rs/bunsen/latest/bunsen/) and the
[book](https://zspacelabs.ai/bunsen/book) for the full API.

## Shape Contracts

`bunsen::contracts` provides allocation-free, always-on runtime tensor-shape contracts. A contract pairs paper-style
shape notation with runtime checking:
it asserts that a tensor's shape matches a declared pattern *and* unpacks named dimensions for downstream use, catching
shape errors at their source. Single checks run in ~160 ns; the amortized periodic variants average a few ns.

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

In hot loops, use `assert_shape_contract_periodically!` to amortize the check via exponential backoff while still
catching regressions:

```rust
use bunsen::contracts::*;

assert_shape_contract_periodically!(
    ["batch", "planes", "height", "width"],
    &x.dims(),
    &[("planes", planes), ("height", height), ("width", width)]
);
```

## TensorData Index Views

`TensorDataView` and `TensorDataViewMut` wrap burn's low-level
`TensorData` to give ergonomic multi-dimensional element access via bracket notation — `view[&[i, j]]` — instead of
manually flattening indices. The views deref to the underlying `TensorData`, so `.shape` and friends are right there.
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
into a queryable XML meta-description of its structure. This enables type-erased introspection and XPath-style parameter
selection — e.g. "every rank-2 weight under the transformer blocks" — which is exactly what you need to slice a model
into parameter groups for per-group optimizers.

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

The dumped structure mirrors the module's fields, with each `@name` taken from the struct field and a stable `param_id`
per tensor:

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

`bunsen::blocks` is a library of `burn::module::Module` building blocks (stateful layers with trainable parameters), and
`bunsen::ops` is a library of stateless
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

See [`examples/`](examples/) for the full index. At a glance:

* [`conway_benchmark`](examples/conway_benchmark) — headless Game of Life (2D/3D) throughput benchmark.
* [`conway_vis`](examples/conway_vis) — real-time OpenGL Game of Life visualization.
* [`lbm2d_vis`](examples/lbm2d_vis) — real-time 2D Lattice Boltzmann fluid-flow visualization.
* [`resnet_finetune`](examples/resnet_finetune) — fine-tune a pretrained ResNet with model surgery.
* [`resnet_tiny`](examples/resnet_tiny) — train a ResNet from scratch on CINIC-10 via a firehose pipeline.
* [`swin_tiny`](examples/swin_tiny) — train a Swin Transformer V2 Tiny on CINIC-10.
* [`train-chat`](examples/train-chat) — train a NanoChat-style GPT with per-group Muon/AdamW optimizers.
* [`whisper-dev`](examples/whisper-dev) — import an OpenAI Whisper model from a PyTorch checkpoint.
* [`zsl-data-cache`](examples/zsl-data-cache) — nanochat dataset shard download/disk cache (+ `pull_shards` CLI).

# Motivation

This library is a synthesis of the utility and extension work that I've been accumulating in:

* <https://github.com/zspacelabs/wordchipper>
* <https://github.com/zspacelabs/bimm>
* <https://github.com/zspacelabs/bimm-contracts>
* <https://github.com/zspacelabs/zsl-chat>
* <https://github.com/crutcher/clockmill>

This library is a work in progress, and I'm working to fold the various utilities and support code from these projects
into a single place; where we can closely track the burn release cycle, and minimize the dependency-hell churn problem
for writing extensions.

I plan on continuing to work on this library, and recruit community involvement for landing and publishing new operators
and blocks in a place we can lock down their testings and documentation.

## Future Components

The base libraries have significant features which haven't been polished and stabilized for bunsen yet.

* weight/data download disk cache - there are several implementations of this in my codebase so far, the most robust is
  probably in the `wordchipper` code.
* shard fetching - being able to bind a family of shards to URL template + range pattern; with information on the target
  format; and wire that smoothly into the download and cache layer. this is also currently in some of the LLM/chat
  codebases.
* LLM `DataLoader` - a high-performance burn data loader for LLM models, built on parquet/arrow; and
  `wordchipper`. This is currently in the `zsl-chat` codebase.
* `clap` tooling - I've built a lot of burn-related clap tools, and I'm pretty sure some of the arguments/setup
  machinery could be shared.

# License

`bunsen` is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details. Opening a pull request is assumed to
signal agreement with these licensing terms
