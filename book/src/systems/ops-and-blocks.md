# Ops, blocks and burn extensions

bunsen's reusable tensor code lives in three places:
[`bunsen::ops`], [`bunsen::blocks`], and the
tensor extension traits in [`bunsen::burner::tensor`].
This chapter explains how they divide the work, so that you know where to
look for something and where new code belongs. Each module's own docs carry
its map: [ops](bunsen::ops#map-of-the-module),
[blocks](bunsen::blocks#map-of-the-module).

## The rule: what a thing is for

- **Blocks** are `torch.nn`-like components, meant to be used as `Module`
  roots or as parts of a module tree.
- **Ops** are operation-focused. They may use the `Module` and `Config`
  machinery, but only to hold cached tables or state.

The rule asks what a type is *for*, not whether it owns parameters or is a
pure function. [`SlidingStft`](bunsen::ops::signal::SlidingStft) is
a `Module` that holds a fixed analysis window and learns nothing.
[`DropPath`](bunsen::blocks::images::drop::drop_path::DropPath) owns no
parameters but sits in a model as a layer.
[`KVCache`](bunsen::ops::transformers::attention::KVCache) is a `Module` over
a bare tensor that holds per-decode state. Asking "would you put this in a
module tree as a layer?" places each of these exactly once: `DropPath` is a
block, and the STFT and the cache are ops.

The dependency runs one way: blocks import ops, and ops never import blocks.
A block is typically "these parameters, plus these ops, called in order".
For example, [`DropBlock2d`](bunsen::blocks::images::drop::drop_block::DropBlock2d)
holds its settings and calls
[`drop_block_2d`](bunsen::ops::drop::drop_block_2d) while training. Keeping
the operation separate means the math is tested on its own, and the block's
tests only have to show that it is wired correctly.

The rule is defined in [`bunsen::ops`](bunsen::ops#ops-and-blocks).

## Caches and stream state are ops

Cache and stream state is *injected*. The caller builds it and passes it in,
and the model never owns it, so one model can serve several caches or
streams in one process. That is why `KVCache` is an op, not a block, even
though attention blocks use it:
[`CausalSelfAttention::forward`](bunsen::blocks::transformers::attention::csa::CausalSelfAttention::forward)
takes the cache as an argument and does not hold it. The same split, an
immutable module plus a per-stream context, runs through the signal front
ends and the speech kits. [Stateless modules and streaming
contexts](./streaming.md) covers the pattern.

## Ops or extension methods

Two places hold tensor helpers, and the split is about how the helper reads
at the call site:

- **[`burner::tensor`](bunsen::burner::tensor)** holds extension traits:
  small, general methods that read as if they were burn's own, such as
  [`count_dim`](bunsen::burner::tensor::TensorBoolOpExt::count_dim),
  [`in_range`](bunsen::burner::tensor::TensorOrderedOpExt::in_range) or
  [`replace_with`](bunsen::burner::tensor::TensorOpExt::replace_with).
  [`bunsen::prelude`] brings them into scope.
- **`ops`** holds free functions: operations you would name as a step in a
  model, often with settings of their own, such as
  [`rms_norm`](bunsen::ops::norm::rms_norm),
  [`scaled_dot_product_attention`](bunsen::ops::transformers::attention::scaled_dot_product_attention)
  or [`repeat_interleave`](bunsen::ops::repeat::repeat_interleave).

## Value objects as configuration

When an op has settings worth naming, they form one small value object:
cloneable, comparable and serializable. That value is the unit of
configuration. A config embeds it as a field rather than copying its fields,
and passes it straight to the op. The in-tree chain nests four deep:

```text
ClampOp ⊂ NoiseConfig ⊂ DropBlockOptions ⊂ DropBlock2dConfig
```

[`ClampOp`](bunsen::ops::clamp::ClampOp) is an optional min and max.
[`NoiseConfig`](bunsen::ops::noise::NoiseConfig) is a distribution plus an
optional clamp.
[`DropBlockOptions`](bunsen::ops::drop::DropBlockOptions) holds the
`DropBlock` settings plus an optional noise fill. And
[`DropBlock2dConfig`](bunsen::blocks::images::drop::drop_block::DropBlock2dConfig)'s
one field is a `DropBlockOptions`.

The payoff is that a setting added to an op reaches every config that
embeds it, with no fields to keep in sync, and the whole chain is saved with
the config. The implementation styles and names are mixed today (`*Op`,
`*Config`, `*Options`); the
[module docs](bunsen::ops#value-objects-as-configuration) list which is which.

## Where blocks come from

`blocks` is a curated catalog, not a comprehensive one. A block lands there
in one of three ways:

1. **A port of a known layer** from a paper or a reference library, kept in
   burn form so that every kit can use it. `DropBlock2d` and `DropPath` are
   the DropBlock and stochastic-depth regularizers, the latter after
   `timm`'s.
2. **A part shared by kits.** A layer that a second kit needs moves out of
   the first one. [`ConvBlock1d`](bunsen::blocks::conv::ConvBlock1d) and
   [`ConvSeq1d`](bunsen::blocks::conv::ConvSeq1d) serve both Whisper and
   Silero VAD, and [`Mlp`](bunsen::blocks::transformers::mlp::Mlp) serves
   both Whisper and NanoChat.
3. **A gap in burn.** burn ships a sequence-level `Lstm`.
   [`FusedLstm`](bunsen::blocks::rnn::lstm::FusedLstm) is what a streaming
   model needs instead: one step at a time, with the caller carrying the
   state.

New blocks arrive when a kit or a downstream user needs them, and each one
follows the [module design conventions](./conventions.md).

## Two attention stacks

bunsen has two attention stacks, and both are deliberate:

- **bunsen's own
  [`CausalSelfAttention`](bunsen::blocks::transformers::attention::csa::CausalSelfAttention)**
  block (the NanoChat GPT kit) has its own projections, QK-norm, rotary
  embeddings and grouped-query attention, over a preallocated multi-layer
  `KVCache`. Use it for a decoder-only model whose layers you define.
- **Functions over burn's `MultiHeadAttention`** (the Whisper kit) keep
  burn's module, so its weights, its checkpoint loading and any
  cross-checks are unchanged, and add KV-cached decoding with a per-layer
  [`AttnKvPair`](bunsen::ops::transformers::attention::AttnKvPair). Use them
  when the model is built on burn's module: a ported checkpoint in that
  layout, or encoder-decoder cross-attention.

[The ops attention docs](bunsen::ops::transformers::attention#two-attention-stacks)
say when to use which, and
[two caches](bunsen::ops::transformers::attention#two-caches) compares
`KVCache` with `AttnKvPair`.
