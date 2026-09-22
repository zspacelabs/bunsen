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

## Organization

### Burn Extensions

* `bunsen::burner` - this is a library of `burn::module::Module` lifecycle components that extend the current
  functionality of burn.
    * `bunsen::burner::module::reflection` has powerful tools for dynamic `burn::module::Module` reflection.
    * `bunsen::burner::optim` has parameter-group optimizer extensions.
    * `bunsen::burner::tensor` has `Tensor` extension traits (`swap`/`release`,
      `select_dim`, elementwise range masks, `bool` counting) and `TensorData`
      index views.
* `contracts` - this is a library of runtime tensor-shape contracts.

### Component Libraries

* `bunsen::blocks` - this is a library of `burn::module::Module` components. This includes simple inner layers,
  recurrent utility blocks, and entire model families.
* `bunsen::ops` - this is a library `burn::tensor::Tensor` operations.
* `bunsen::kits` - this is a library of full models and simulation kits.
    * `bimm` - image models: `resnet`, `swinn`
    * `gpts` - partial implementation of `nanochat`
    * `sims` - `conway` game of life, `lbm` 2D fluid flow.
    * `speech` - complete `silero_vad`, partial `whisper`

### App and Testing Support Libs

* `bunsen::errors` - this is a library of error types and tooling.
* `bunsen::support` - this is a library of support functions for bunsen, including testing tooling which may be useful
  for clients.
* `bunsen::zspace` - this is a library of z-space / index utilities.

# API Examples

A "good parts" survey of some of `bunsen`'s features. See the
[docs](https://docs.rs/bunsen/latest/bunsen/) and the
[book](https://zspacelabs.ai/bunsen/book) for the full API.

# Builtin Kits

`bunsen::kits` collects complete, runnable domain implementations over the crate's blocks and ops: whole models with
their pretrained loaders, and simulations. Every kit's module docs carry a worked example, generic over the backend;
the stubs below show the default way in, and link to the example that uses each kit. Pretrained weights come through
one mechanism everywhere: a `PretrainedCache`, the kit's `default_{kit}_factory()`, and a name (`openai/base`,
`torchvision/resnet18`, `hf:org/repo` for a Hugging Face repo, `bundled:...` for weights built into the binary).

## Speech

### Whisper

[docs](https://docs.rs/bunsen/latest/bunsen/kits/speech/whisper/index.html) &middot;
example: [`whisper-cli`](https://github.com/zspacelabs/bunsen/tree/main/examples/whisper-cli)

OpenAI's Whisper, with an index of pretrained checkpoints and a stream driver that turns audio pushed in chunks into
timed transcript segments.

```rust,ignore
use bunsen::{
    data::pretrained::{PretrainedCache, PretrainedCacheOptions},
    kits::speech::whisper::{
        driver::{RunningMaxClamp, StreamClock, WhisperStreamDriver, WhisperStreamDriverConfig},
        pretrained::default_whisper_factory,
    },
};

let cache = PretrainedCache::new(PretrainedCacheOptions::default())?;
let bundle = default_whisper_factory()?.load_bundle::<B>("openai/base", &cache, &device)?;
let driver: WhisperStreamDriver<B> = WhisperStreamDriverConfig::new().init_from_bundle(bundle, &device)?;

let mut ctx = driver.new_context(StreamClock::uniform(driver.sample_rate()), RunningMaxClamp::new())?;
for block in samples.chunks(driver.sample_rate() / 10) {
    for event in ctx.write_read(block)? {
        println!("{:?}", event.segment().text);
    }
}
```

### Silero VAD

[docs](https://docs.rs/bunsen/latest/bunsen/kits/speech/silero_vad/index.html) &middot;
example: [`whisper-cli`](https://github.com/zspacelabs/bunsen/tree/main/examples/whisper-cli)'s real-time presets

Voice-activity detection: the probability that each chunk of audio holds speech, with the weights built into the
binary under the `silero-weights` feature.

```rust,ignore
use bunsen::kits::speech::silero_vad::{SileroVadContextConfig, SileroVadMeta, pretrained::default_silero_factory};

let vad = default_silero_factory()?.load::<B>("bundled:silero/vad", &cache, &device)?.handle;
let vad = vad.expect_branch(16000);

let mut ctx = SileroVadContextConfig::new(vad.sample_rate()).init(vad, &device);
for chunk in samples.chunks_exact(vad.chunk_size()) {
    let chunk: Tensor<B, 2> = Tensor::<B, 1>::from_floats(chunk, &device).unsqueeze();
    let (probability, next) = vad.context_forward(chunk, ctx);
    ctx = next;
}
```

## Vision (`bimm`)

### ResNet

[docs](https://docs.rs/bunsen/latest/bunsen/kits/bimm/resnet/index.html) &middot;
examples: [`resnet_finetune`](https://github.com/zspacelabs/bunsen/tree/main/examples/resnet_finetune), [
`resnet_tiny`](https://github.com/zspacelabs/bunsen/tree/main/examples/resnet_tiny)

The ResNet family, with torchvision's and timm's pretrained rows (`torchvision/resnet50`, `timm/resnet18_a1`) and the
model surgery a fine-tune wants.

```rust,ignore
use bunsen::kits::bimm::resnet::{ResNet, default_resnet_factory};

let loaded = default_resnet_factory()?.load::<B>("torchvision/resnet18", &cache, &device)?;
let model: ResNet<B> = Arc::unwrap_or_clone(loaded.handle)
    .with_classes(10)
    .with_stochastic_drop_block(0.2);
```

## Simulations (`sims`)

### Conway's Game of Life

[docs](https://docs.rs/bunsen/latest/bunsen/kits/sims/conway/index.html) &middot;
example: [`conway`](https://github.com/zspacelabs/bunsen/tree/main/examples/conway)

```rust,ignore
use bunsen::kits::sims::conway::{life2d::{ConwayLife2DConfig, ConwayLife2DState}, util::ConwaySim};

let mut sim: ConwayLife2DState<B> = ConwayLife2DConfig { shape }.init(&device);
sim.fuzz(0.3);
sim.step();
```

### Lattice Boltzmann Fluid Flow

[docs](https://docs.rs/bunsen/latest/bunsen/kits/sims/lbm/d2q9/index.html) &middot;
example: [`lbm2d_vis`](https://github.com/zspacelabs/bunsen/tree/main/examples/lbm2d_vis)

A D2Q9 lattice Boltzmann fluid. The population lives on a `[H, W, 3, 3]` grid, one 3x3 velocity stencil per cell, and
each step streams it, relaxes it toward thermal equilibrium (BGK, with a mass-conserving correction), and bounces it
back off a solid mask.

```rust,ignore
use bunsen::{
    kits::sims::lbm::d2q9::{LBMD2Q9Config, LBMD2Q9State, RelaxationParam, SPEED_OF_SOUND, macroscopic_momentum},
    support::geometry::GridShape2D,
};

let rho = SPEED_OF_SOUND / 100.0;
let mut sim: LBMD2Q9State<B> = LBMD2Q9Config::new(GridShape2D::square(400))
    .with_relaxation(RelaxationParam::Tau(0.9))
    .init(&device, rho);

// A dense spot in the rest population, a wall, and the mass to hold.
sim.dist = sim.dist.slice_fill(s![50, 20, 1, 1], 5.0 * rho);
sim.solid_mask = sim.solid_mask.slice_fill(s![130..150, 60..200], true);
sim.save_correct_total_mass();

sim.advance_step();
// [H, W, (y, x)] momentum: the field the example draws.
let momentum = macroscopic_momentum(sim.dist.clone(), sim.lbm_tables.e_vec());
```

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

## Tensor Op Extensions

`bunsen::burner::tensor` provides extension traits that add utility methods directly to `burn::Tensor` — in scope after
`use bunsen::burner::tensor::*;`:

* `TensorOpExt` (all tensor kinds) — `swap` exchanges two tensors in place;
  `release` moves a tensor out of a field, leaving an empty tensor behind;
  `select_dim` selects one index along a dimension and squeezes it, dropping the rank by one.
* `TensorIntOpExt` (`Int` tensors) — `square`, and `bounded_elem` for elementwise `[start, end)` range checks producing
  `Bool` masks.
* `TensorBoolOpExt` (`Bool` tensors) — `count_dim` / `count_dims` count `true`
  elements along one or more dimensions, with negative indexing.

```rust
use bunsen::burner::tensor::*;
use burn::prelude::*;

// Extract row `i` of a matrix as a vector:
fn row<B: Backend>(m: Tensor<B, 2>, i: usize) -> Tensor<B, 1> {
    m.select_dim(0, i)
}

// Count the `true` cells in each row of a boolean grid:
fn row_counts<B: Backend>(grid: Tensor<B, 2, Bool>) -> Tensor<B, 2, Int> {
    grid.count_dim(-1)
}
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

See [`examples/`](https://github.com/zspacelabs/bunsen/tree/main/examples/) for the full index.

# License

`bunsen` is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
