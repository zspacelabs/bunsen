# Kits

A kit is a whole thing you can pick up and use: a model family with its
configs, the weights it can load and a program that runs it, or a simulation
you can step. Where [blocks and ops](../systems/ops-and-blocks.md) are parts,
a kit is what the parts are for.

The [`bunsen::kits`](bunsen::kits#the-kits) docs list the kits, and describe
the [anatomy of a kit](bunsen::kits#anatomy-of-a-kit) piece by piece, with a
table of how the ResNet, Whisper and Silero kits fill each piece in. This
page explains why kits are shaped that way, and what building one involves.

## Why a shared shape

No trait makes a module a kit. Kits share a shape because they solve the
same problems, and a solved problem should look the same each time it
appears:

- **Naming a model.** A user should be able to say `resnet50` or
  `openai/base.en` and get a model. A prefab map names geometries; a
  pretrained table names checkpoints; a factory turns either name into a
  loaded module. The machinery is shared and lives in
  [`bunsen::data::pretrained`]. A kit contributes
  tables and a `Construct` hook, not a loader of its own. See
  [Files, caches and pretrained models](../systems/pretrained.md).
- **Configuring it.** A kit's configs follow the Simple or Stacked shape from
  [Module design conventions](../systems/conventions.md). A user-facing
  policy config is small and stable; the structure config spells out the
  implementation.
- **Running it.** A model that decodes or streams keeps no per-stream state of
  its own. The state is injected as a context or cache, so one loaded model
  serves several streams. See
  [Stateless modules and streaming contexts](../systems/streaming.md).
- **Trusting it.** A port of a published model is only as good as its
  agreement with the original. Kits that load published weights have a
  validation crate that steps them against the reference implementation. See
  [Validation against reference implementations](../development/validation.md).

## Building a kit

A new kit grows in roughly this order, and each step is useful on its own:

1. **The model and its configs**, in the kit's module, built from existing
   blocks and ops where they fit, with shape contracts at its boundaries and
   a narrow `Meta` trait.
2. **An example** under `examples/` that runs it end to end.
3. **Names**: a prefab map, then pretrained rows and a `Construct` hook, and
   a `default_{kit}_factory()`, once there are weights to load.
4. **A validation crate** under `crates/validation/`, once the kit claims
   parity with an upstream implementation.
5. **Promotion**: a layer that a second kit needs moves out to `blocks` or
   `ops`.

The kit's module docs carry its lifecycle and an example; the
[`bunsen::kits`](bunsen::kits#the-kits) table gains a row. This book doesn't
need to change unless the kit brings a new cross-cutting system.

Kits evolve faster than the rest of the crate while their domains are being
filled in; see [`bunsen::kits`](bunsen::kits#stability) on stability.
