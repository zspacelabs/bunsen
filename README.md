# Bunsen

*by [ZSpaceLabs](https://zspacelabs.ai)*

[![Crates.io Version](https://img.shields.io/crates/v/bunsen)](https://crates.io/crates/public/bunsen)
[![Documentation](https://img.shields.io/docsrs/bunsen)](https://docs.rs/bunsen/latest/bunsen/)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)
[![Discord](https://img.shields.io/discord/1475229838754316502?label=discord)](https://discord.gg/vBgXHWCeah)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zspacelabs/bunsen)

`bunsen` aims to be a "batteries included" complementary community standard library for extending
the [burn](https://burn.dev) tensor library.

# Book

Read the [bunsen book](https://zspacelabs.ai/bunsen/book)

# Builtin Kits

`bunsen::kits` collects complete, runnable domain implementations over the crate's blocks and ops: whole models with
their pretrained loaders, and simulations. Every kit's module docs carry a worked example, generic over the backend;
the stubs below show the default way in, and link to the example that uses each kit. Pretrained weights come through
one mechanism everywhere: a `PretrainedCache`, the kit's `default_{kit}_factory()`, and a name (`openai/base`,
`torchvision/resnet18`, `hf:org/repo` for a Hugging Face repo, `bundled:...` for weights built into the binary).

## Speech

### Whisper

[docs](https://docs.rs/bunsen/latest/bunsen/kits/speech/whisper/index.html) &middot;
example: [`whisper-cli`](examples/whisper-cli)

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
example: [`whisper-cli`](examples/whisper-cli)'s real-time presets

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
examples: [`resnet_finetune`](examples/resnet_finetune), [`resnet_tiny`](examples/resnet_tiny)

The ResNet family, with torchvision's and timm's pretrained rows (`torchvision/resnet50`, `timm/resnet18_a1`) and the
model surgery a fine-tune wants.

```rust,ignore
use bunsen::kits::bimm::resnet::{ResNet, default_resnet_factory};

let loaded = default_resnet_factory()?.load::<B>("torchvision/resnet18", &cache, &device)?;
let model: ResNet<B> = Arc::unwrap_or_clone(loaded.handle)
    .with_classes(10)
    .with_stochastic_drop_block(0.2);
```

### Swin Transformer V2

[docs](https://docs.rs/bunsen/latest/bunsen/kits/bimm/swin/v2/index.html) &middot;
example: [`swin_tiny`](examples/swin_tiny)

```rust,ignore
use bunsen::kits::bimm::swin::v2::{LayerConfig, SwinTransformerV2, SwinTransformerV2Config};

let swin: SwinTransformerV2<B> = SwinTransformerV2Config::new(
    image_dimensions, patch_size, image_channels, num_classes, embed_dim,
    vec![LayerConfig::new(8, 6), LayerConfig::new(8, 12)],
)
.with_window_size(window_size)
.try_init(&device)?;
```

## Language (`gpts`)

### NanoChat

[docs](https://docs.rs/bunsen/latest/bunsen/kits/gpts/nanochat/index.html) &middot;
example: [`train-chat`](examples/train-chat)

A compact GPT for experimentation and fine-tuning, with its datasets.

```rust,ignore
use bunsen::kits::gpts::nanochat::{NanoChatGpt, NanoChatGptConfig};

let gpt: NanoChatGpt<B> = NanoChatGptConfig::new()
    .with_n_embed(768)
    .with_n_layer(12)
    .with_vocab_size(vocab_size)
    .init::<B>(&device);
```

## Simulations (`sims`)

### Conway's Game of Life

[docs](https://docs.rs/bunsen/latest/bunsen/kits/sims/conway/index.html) &middot;
example: [`conway`](examples/conway)

```rust,ignore
use bunsen::kits::sims::conway::{life2d::{ConwayLife2DConfig, ConwayLife2DState}, util::ConwaySim};

let mut sim: ConwayLife2DState<B> = ConwayLife2DConfig { shape }.init(&device);
sim.fuzz(0.3);
sim.step();
```

### Lattice Boltzmann (D2Q9)

[docs](https://docs.rs/bunsen/latest/bunsen/kits/sims/lbm/index.html) &middot;
example: [`lbm2d_vis`](examples/lbm2d_vis)

```rust,ignore
use bunsen::kits::sims::lbm::d2q9::{LBMD2Q9Config, LBMD2Q9State, RelaxationParam};

let mut world: LBMD2Q9State<B> = LBMD2Q9Config::new(grid_shape)
    .with_relaxation(RelaxationParam::Tau(0.6))
    .init(&device, background_density);
world.advance_step();
```

## Tokens

[docs](https://docs.rs/bunsen/latest/bunsen/kits/tokens/index.html)

What the kits share on the token side: the `Detokenizer` seam from ids to text, and its `wordchipper` implementation
behind the `tokenizer` feature, which the Whisper bundle carries for its vocabulary.

# Crates

## Public / API Crates

* [`bunsen-firehose`](crates/public/bunsen-firehose) — a columnar dataloader / processing pipeline, with a burn batcher
  bridge.

## Utility Crates

* [`bunsen-contracts-macros`](crates/public/bunsen-contracts-macros) — the
  `shape_contract![]` proc-macro backing `bunsen`'s runtime tensor-shape contracts.

## Experimental Crates

These represent complex-interface + work-in-progress, unstable interface extensions to `bunsen`; particulary those which
incur large dependencies or are not yet ready for general consumption.

* [`bunsen-firehose-image`](crates/public/bunsen-firehose-image) — image loading, augmentation, and tensor-conversion
  operators for `bunsen-firehose`.
* [`bunsen`](crates/public/bunsen) — the main "batteries included" library extending burn: model blocks, kits, ops,
  contracts, and support tooling.
* [`bunsen-arrow-dataloaders`](crates/public/bunsen-arrow-dataloaders) — *(preview)* an Arrow-backed chat dataloader
  with tokenization for LLM training.

# API Examples

A "good parts" survey of some of `bunsen`'s features. See the
[docs](https://docs.rs/bunsen/latest/bunsen/) and the
[book](https://zspacelabs.ai/bunsen/book) for the full API.

## Shape Contracts

`bunsen::contracts` provides allocation-free, always-on runtime tensor-shape contracts. A contract pairs paper-style
shape notation with runtime checking:
it asserts that a tensor's shape matches a declared pattern *and* unpacks named dimensions for downstream use, catching
shape errors at their source. Single checks run in ~160 ns; the amortized periodic variants average a few ns.

```rust,ignore
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

```rust,ignore
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

```rust,ignore
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

```rust,ignore
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

```rust,ignore
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

See [`examples`](examples) for the full index. At a glance:

* [`conway`](examples/conway) — Conway's Game of Life : visual and benchmark.
* [`lbm2d_vis`](examples/lbm2d_vis) — real-time 2D Lattice Boltzmann fluid-flow visualization.
* [`resnet_finetune`](examples/resnet_finetune) — fine-tune a pretrained ResNet with model surgery.
* [`resnet_tiny`](examples/resnet_tiny) — train a ResNet from scratch on CINIC-10 via a firehose pipeline.
* [`swin_tiny`](examples/swin_tiny) — train a Swin Transformer V2 Tiny on CINIC-10.
* [`train-chat`](examples/train-chat) — train a NanoChat-style GPT with per-group Muon/AdamW optimizers.
* [`whisper-cli`](examples/whisper-cli) — transcribe audio through the Whisper stream driver, with the model by name
  (`openai/base`, `large`, `hf:openai/whisper-large-v3`, or a checkpoint path) and a `models` subcommand that lists,
  fetches and inspects them.

# License

`bunsen` is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details. Opening a pull request is assumed to
signal agreement with these licensing terms
