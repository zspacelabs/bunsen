# The bunsen crate

`bunsen` is organized as layers: components higher up are built from the
ones below and never the reverse. Knowing the layers tells you where a new
piece of code belongs, and what it may depend on.

From the top of the stack down:

| Layer | Contents |
|---|---|
| `kits` | Whole models and simulations, with their configs, pretrained indexes and runtime drivers. |
| `blocks` | `torch.nn`-like components: module roots and tree parts. |
| `ops` | Operations: free functions, and `Module`/`Config` types that hold cached tables or stream state. |
| foundation | `burner` (burn extensions), `contracts` (shape checks), `errors`, `rust_ext` (Rust helpers), `support`, `zspace`. |
| `burn` | The framework underneath; not part of `bunsen`. |

Beside the stack, outside the layering:

- `data`: files, caches, pretrained models and dataset shards; kits load
  through it.
- `audit`: recording and verifying tensor checkpoints across runs and
  backends.

The split between `ops` and `blocks` is a rule, not a convenience: blocks
are `nn`-like components meant to sit in a module tree; ops are operations
that may use the `Module` and `Config` machinery only to hold cached tables
or state. The [`bunsen::ops`](bunsen::ops#ops-and-blocks) module docs state
it, and [Ops, blocks and burn extensions](../systems/ops-and-blocks.md)
explains why.

## The modules

Each module's own docs carry its map; this table is the one-line version.

| Module | What it is for |
|---|---|
| [`kits`](bunsen::kits) | Complete domain implementations: image models (`images`), GPTs (`gpts`), simulations (`sims`), speech (`speech`), and the shared token seam (`tokens`). See [Kits](./kits.md). |
| [`blocks`](bunsen::blocks) | Reusable `Module` components: conv blocks, image layers, recurrent cells, transformer parts. |
| [`ops`](bunsen::ops) | Tensor operations, including the signal-processing front ends and attention ops. |
| [`burner`](bunsen::burner) | Extensions to burn's module machinery: the config lifecycle ([`ModuleInit`](bunsen::burner::module::ModuleInit), [`ToStructureConfig`](bunsen::burner::module::ToStructureConfig)), reflection, grouped optimizers, tensor extension traits, PyTorch-load repairs. |
| [`contracts`](bunsen::contracts) | Runtime tensor-shape contracts. See [Shape contracts](../systems/contracts.md). |
| [`data`](bunsen::data) | The disk cache, pretrained-model loading, and dataset shard sets. See [Files, caches and pretrained models](../systems/pretrained.md). |
| [`audit`](bunsen::audit) | Audit probes: record a stream of tensor checkpoints from one run and verify another against it. See [Audit probes and baselines](../development/audit.md). |
| [`errors`](bunsen::errors) | `BunsenError` and its kinds, context frames and typed causes; `BunsenResult`; the `try_x` / `x` convention; and, with `testing`, the `ErrorMatcher` test matchers. |
| [`rust_ext`](bunsen::rust_ext) | Rust-language extensions with no tensor or burn dependency: `CloneBox` / `CloneRef`, array and range helpers, and `LocationDesc`, a serializable source location. |
| [`support`](bunsen::support) | Shared utilities, including the test backends and devices in [`support::testing`](bunsen::support::testing). |
| [`zspace`](bunsen::zspace) | Integer-lattice index and shape helpers. |
| [`prelude`](bunsen::prelude) | The traits most code needs in scope, for one glob import. |
| [`public`](bunsen::public) | Re-exports of public dependencies (`burn`), so downstream code uses the same versions. |

## Where new code goes

- A new **layer** or **module family** used as part of a model is a block.
- A new **operation**, even one that keeps a precomputed table or stream
  state, is an op.
- A **whole model** with its configs, weights and drivers is a kit; its
  reusable parts get promoted to `blocks` or `ops` once a second consumer
  appears.
- Anything that **extends burn's machinery** itself (module traversal,
  records, optimizers) goes in `burner`.
- A **Rust-language helper** with no tensor or burn dependency goes in
  `rust_ext`.

Whatever the layer, a new module follows the
[module design conventions](../systems/conventions.md).
