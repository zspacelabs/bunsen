# Building Reusable Modules

When you wrap a `burn::Module`, `burn` gives you the basic ingredients:
a `Config` struct, derived via `#[derive(Config)]`, that builds a
`Module` via `init()`. This is enough for a single self-contained
module, but it strains once your modules start composing into larger
ones.

Two conventions show up repeatedly in `bunsen` to manage that strain:

1. A **`{Module}Meta` trait** — a shared introspection API implemented
   by both the configs *and* the built module, so anyone holding any of
   those forms can ask the same structural questions.
2. A **`{Module}ContractConfig` &rarr; `{Module}StructureConfig` split** —
   two configs at two levels of abstraction. The contract describes
   *what the module is for*; the structure describes *how it's built*.

Neither is required by `burn`. Both pay off once a module is used
inside something else, or once its parameter surface starts evolving
faster than its callers want.

## The `{Module}Meta` trait

### Why

A parent module that owns a child needs to know structural things
about the child at inference time — its embedding dimension, its
number of heads, its sequence length. The naive answer is to copy
those numbers into the parent's own fields. That works, but now the
same number lives in two places, and updating the child means
remembering to update the parent. Worse, configuration values copied
into a `Module`'s state are awkward to keep in sync with the actual
tensor shapes that ended up inside.

### Solution

Define a trait that exposes the structural questions, and implement
it on every form that can answer them: the user-facing config, the
lowered structure config (see below), and the built module itself.

```rust,ignore
pub trait MlpMeta {
    /// Input/output embedding dimension.
    fn embed_dim(&self) -> usize;

    /// Hidden dimension inside the MLP.
    fn hidden_dim(&self) -> usize;
}
```

Now anything holding an `&impl MlpMeta` can ask the question, and the
answer comes from whichever source actually knows: a field on the
config, or a tensor `.dims()` on the module.

### Toy example

```rust,ignore
use burn::{prelude::*, nn::{Linear, LinearConfig}};

pub trait MlpMeta {
    fn embed_dim(&self) -> usize;
    fn hidden_dim(&self) -> usize;
}

#[derive(Config, Debug)]
pub struct MlpConfig {
    pub embed_dim: usize,
    #[config(default = "4")]
    pub expansion_factor: usize,
}

impl MlpMeta for MlpConfig {
    fn embed_dim(&self) -> usize { self.embed_dim }
    fn hidden_dim(&self) -> usize { self.expansion_factor * self.embed_dim }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    in_proj: Linear<B>,
    out_proj: Linear<B>,
}

impl<B: Backend> MlpMeta for Mlp<B> {
    fn embed_dim(&self) -> usize {
        // Derived from the live tensor shape — no cached field.
        self.in_proj.weight.dims()[0]
    }
    fn hidden_dim(&self) -> usize {
        self.in_proj.weight.dims()[1]
    }
}
```

A parent that contains an `Mlp` can now read `mlp.embed_dim()`
directly from the live module, instead of carrying its own
`mlp_embed_dim: usize` field. The metadata stays in *one* place per
form, and the forms agree by construction.

### Where this shows up

- **[`NanoChatGptMeta`](https://docs.rs/bunsen/latest/bunsen/kits/gpts/nanochat/trait.NanoChatGptMeta.html)**
  is implemented by *three* types: `NanoChatGptConfig` (user-facing
  knobs), `NanoChatGptStructureConfig` (lowered per-layer configs),
  and `NanoChatGpt<B>` (the built module reading dims from its actual
  layers). All three answer the same questions about `n_embed`,
  `n_head`, `head_dim`, `n_layer`, and so on.
- **[`ResidualBlockMeta`](https://docs.rs/bunsen/latest/bunsen/kits/bimm/resnet/trait.ResidualBlockMeta.html)**
  is implemented on the structure config *and* the built block. A
  `ResNet` model that holds a `Vec<ResidualBlock<B>>` can call
  `block.output_resolution([h, w])` to walk the resolution through the
  network from the live modules, with no separate "shape table"
  alongside.

## Contract &rarr; Structure config split

### Why

A module's parameter list grows in two unrelated directions:

- **Intent-level.** "How do I describe this thing at the level of what
  it's *for*?" — `embed_dim`, `n_layer`, `vocab_size`. These are the
  knobs a user actually wants to set when they say "give me a 12-layer
  GPT". Short list, stable shape, evolves slowly.
- **Implementation-level.** "What does `init` need to wire?" —
  explicit per-layer `LinearConfig`s, `EmbeddingConfig`s,
  `RotaryEmbeddingConfig`s, normalization choices per sub-block. Long
  list, evolves with the implementation.

Cramming both into one `Config` makes it tedious to instantiate (the
user has to fill in fields they don't care about) *and* hard to
evolve (every implementation change is an API break for callers who
only wanted to say "12 layers please").

### Solution

Split the config in two:

- **`{Module}ContractConfig`** — the intent-level description.
- **`{Module}StructureConfig`** — the implementation parameter list,
  one field per sub-module config.
- A `to_structure()` / `into_structure()` method on the contract that
  produces the structure config.
- The `init` that builds the module hangs off the *structure* config.

The contract is small, friendly, and stable. The structure is verbose
but maps one-to-one onto the implementation; it's the natural home for
serialization, pretrained-weight loaders, and any code that needs to
reason about the actual layers.

### Toy example

Continuing the `Mlp` from above, split the single `MlpConfig` into a
contract and a structure:

```rust,ignore
#[derive(Config, Debug)]
pub struct MlpContractConfig {
    pub embed_dim: usize,
    #[config(default = "4")]
    pub expansion_factor: usize,
}

impl MlpMeta for MlpContractConfig {
    fn embed_dim(&self) -> usize { self.embed_dim }
    fn hidden_dim(&self) -> usize { self.expansion_factor * self.embed_dim }
}

impl MlpContractConfig {
    /// Lower the contract into a concrete per-layer structure.
    pub fn to_structure(&self) -> MlpStructureConfig {
        MlpStructureConfig {
            in_proj:  LinearConfig::new(self.embed_dim, self.hidden_dim()),
            out_proj: LinearConfig::new(self.hidden_dim(), self.embed_dim),
        }
    }
}

#[derive(Config, Debug)]
pub struct MlpStructureConfig {
    pub in_proj:  LinearConfig,
    pub out_proj: LinearConfig,
}

impl MlpMeta for MlpStructureConfig {
    fn embed_dim(&self) -> usize  { self.in_proj.d_input }
    fn hidden_dim(&self) -> usize { self.in_proj.d_output }
}

impl MlpStructureConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> Mlp<B> {
        Mlp {
            in_proj:  self.in_proj.init(device),
            out_proj: self.out_proj.init(device),
        }
    }
}
```

A typical caller stays in contract-land:

```rust,ignore
let mlp: Mlp<B> = MlpContractConfig::new(768)
    .with_expansion_factor(4)
    .to_structure()
    .init(&device);
```

…but a power user or a pretrained-weight loader that *needs* to set
per-layer details can drop down to `MlpStructureConfig` directly.

### What the split buys you

1. **Multiple contracts, shared structure.** A `GatedMlpContractConfig`
   that adds a SiLU gate can lower to a slightly extended structure;
   you can also have several "kinds" of contract sharing the same
   `MlpStructureConfig` family. The user picks the contract that
   matches their intent; the implementation only knows about
   structures.
2. **Prefabs live on the contract.** Named presets ("resnet18",
   "resnet50") are small, intent-level descriptions and naturally fit
   as `ContractConfig` constructors. The big, verbose `StructureConfig`
   doesn't need a constructor per preset.
3. **Stable user API across implementation churn.** Adding a new
   sub-module to the implementation extends `StructureConfig` without
   touching `ContractConfig`. Callers who only ever wrote
   `MlpContractConfig::new(768)` don't notice.
4. **A natural seam for tooling.** Serialization, weight loading, and
   inspection tools work against `StructureConfig`, where the layers
   are spelled out. Documentation and tutorials work against
   `ContractConfig`, where the surface stays small.

### Where this shows up

- **[`ResNetContractConfig`](https://docs.rs/bunsen/latest/bunsen/kits/bimm/resnet/struct.ResNetContractConfig.html)
  / `ResNetStructureConfig`**. The contract says "a ResNet with these
  block counts, optionally bottlenecked"; the structure spells out the
  stem, layer blocks, and head. `PREFAB_RESNET_MAP` ships
  `ContractConfig` builders for the standard variants ("resnet18",
  "resnet50", &hellip;).
- **[`ResidualBlockContractConfig`](https://docs.rs/bunsen/latest/bunsen/kits/bimm/resnet/struct.ResidualBlockContractConfig.html)
  / `ResidualBlockStructureConfig`**. The contract describes
  "downsample input, use a bottleneck policy"; the structure is an
  *enum* that dispatches to either a `BasicBlock` or a
  `BottleneckBlock`. Two different concrete implementations sit behind
  one contract.
- **[`NanoChatGptConfig`](https://docs.rs/bunsen/latest/bunsen/kits/gpts/nanochat/struct.NanoChatGptConfig.html)
  / `NanoChatGptStructureConfig`**. The contract names embedding
  width, head counts, layer count; the structure spells out the
  embedding, per-block configs, LM head, rotary embedding, and final
  norm.

## When to reach for each

- **Always implement `{Module}Meta`** if anything else (a parent
  module, a builder, a test) is going to ask structural questions
  about this module at runtime. The cost is one trait and a few
  one-line methods; the payoff is no duplicated metadata.
- **Reach for the Contract &rarr; Structure split when:**
  - the user-facing knobs differ in number or shape from the
    implementation parameters,
  - you anticipate multiple intent-level "kinds" of this module
    sharing one implementation (or one kind backed by multiple
    implementations, as with `ResidualBlock`),
  - you want a clean place to land prefab / preset constructors and
    weight-loader hooks.
- **Skip the split** for tiny modules whose user-facing config *is*
  the implementation. Both conventions exist to manage growth; don't
  pay their cost up front.
