# Bunsen Style Guide

Conventions for code, documentation, and build configuration across the
`bunsen` workspace. This file is the source of truth; assertions added under
each chapter are applied across the code base.

> See also the book's [Writing documentation](book/src/development/docs.md)
> for the book's own conventions: linking into the API, code in the book, and
> building it.

## rustdoc

### Base Style

The base style for rustdoc is [rfc1574].

[rfc1574]: https://github.com/rust-lang/rfcs/blob/master/text/1574-more-api-documentation-conventions.md

### Coverage

* Every public interface requires rustdoc.
* Objects in a lifecycle ('Config'>'Module') require a short situational relationship description with cross-links.

Every public item needs rustdoc. This chapter defines the structure we expect,
the cross-links between paired types, and how tensor shapes are written.

### rustdoc is the reference

rustdoc is the primary documentation for every interface and lifecycle: the
compiler checks its links, its examples run as doctests, and it moves with
the code. The book (`book/`) explains how the workspace and the crate are
organized, the systems that cut across modules and why they are built that
way, and how the project is developed, and links into the API; it never
restates signatures, method or field lists, feature tables, or module maps.
A fact that lives only in the book is a rustdoc gap, and nothing checks it:
add it to rustdoc, then link to it.

* The module map lives once: the crate root (`lib.rs`) for top-level
  modules, and each area's `mod.rs` for its submodules.
* The book links to items by intra-doc path,
  ``[`ShapeContract`](bunsen::contracts::ShapeContract)``, and the book build
  resolves every such link with rustdoc. A rename that breaks a book link
  fails CI in the PR that made it. The book's
  [Linking into the API](book/src/development/docs.md#linking-into-the-api)
  has the link forms.

### Module docs must render

A private module's `//!` is never rendered: `mod x; pub use x::*;` shows the
items, not the file. Prose written there is invisible, and it rots unchecked.

* What a type is and how it relates to its neighbours goes on the type's
  `///`.
* Why an area exists, and the lifecycle across its files, goes in the
  public parent's `//!`.
* A private file keeps at most a one-line `//!` title.

CI enforces this with `tools/check_hidden_module_docs.py`, which fails when a
non-public, non-test module has more than one `//!` line.

`#[doc(inline)]` on a re-export of a private module is a no-op (rustdoc
inlines items that are not publicly reachable anyway); don't write it. Use
it only on a re-export from a *public* module, where it changes the output.

### Errors: `try_x` and `x`

The [`bunsen::errors`](https://docs.rs/bunsen/latest/bunsen/errors/index.html#convention-try_x-and-x)
docs define the convention: a fallible `try_x` returning `BunsenResult`, its
panicking twin `x` (or `expect_x`), which `BunsenErrorKind` fits which
failure, and how to build, add context to, and handle an error. A pair's docs
add two rules on top of it:

* `try_x` carries an `# Errors` section: each kind it returns, and when, and
  the cause type where a caller may want to read it (`LookupError`,
  `ConstraintError`, a subsystem's own cause). Its summary links the
  convention.
* `x` names `try_x` as its fallible half and carries a `# Panics` section:
  it panics with the `try_x` error's report, plus anything it checks on its
  own.

Building errors:

* Context goes on with `.context(..)` / `.with_context(..)`, never by
  formatting an error into a new message: that loses its kind and cause.
* A message is one line. Evidence (a diff, a dump, a list) goes in
  `.with_details(..)` or a frame's details.
* An `io::Error` goes through `sys_at` (naming its path) or `sys_op`. A
  foreign error is kept as the cause, not formatted into the message.

Tests check errors with `bunsen::errors::testing::ErrorMatcher`, naming the
kind and only the parts of the message or cause the test is about.

### Tensor shape notation

A tensor shape is written as a **single** backtick code span wrapping the whole
shape in **square brackets**. Dimension expressions and inline annotations live
*inside* the brackets:

```text
`[batch, time, embed]`
`[batch, in_channels, height, width]`
`[batch, height/2 * width/2, 2 * channels]`
`[batch, out=planes*expansion, h, w]`
`[VY=3, VX=3, (VY, VX)=2]`
```

Do **not** use any of these forms for a tensor shape:

```text
``[batch, time, embed]``              (double backticks)
[`batch`, `time`, `embed`]            (per-element backticks)
(`b_nw`, `num_heads`, ws*ws)          (parentheses + per-element backticks)
(batch, time, embed)                  (parentheses)
```

Prefer **shape-first** phrasing — put the shape *before* the object it
describes (`SHAPE object`) rather than trailing it (`object of shape SHAPE`).
Articles (`a`/`an`/`the`) stay first; other modifiers follow the shape:

```text
prefer:  a `[batch, time, embed]` input tensor
         a `[2*h-1, 2*w-1, 2]` 3D tensor containing the offsets
not:     an input tensor of shape `[batch, time, embed]`
         a 3D tensor of shape `[2*h-1, 2*w-1, 2]` containing the offsets
```

These are **not** tensor shapes — leave them as written:

- Coordinate / value tuples and pairs: `(y, x)`, `(k, v)`,
  `(density, velocity)`, `(height, width)` tuples.
- Rust type code spans: `&[usize; D]`, `Param<Tensor<R, K>>`.
- Half-open ranges and indexing: `[start, end)`, `env[$VAR]`.
- Intra-doc link syntax: `[text](url)`.

## Module design

How a bunsen module, its config, and its metadata fit together. What the
two config shapes are and how they work (Simple and Stacked,
`ToStructureConfig` and its blanket `ModuleInit`, the role of the `FooMeta`
trait) is defined, with compiled examples, in the rustdoc of
[`ModuleInit`](https://docs.rs/bunsen/latest/bunsen/burner/module/trait.ModuleInit.html#two-config-shapes)
and
[`ToStructureConfig`](https://docs.rs/bunsen/latest/bunsen/burner/module/trait.ToStructureConfig.html).
This section sets what the author of a module family must do.

### Config shapes: Simple and Stacked

* The config lives in the same file as the module it builds.
* A Simple family's config is `FooConfig`, and builds `Foo`.
* A Stacked family's structure config is `FooStructureConfig`. Each policy
  config is named for its policy (`FooContractConfig`, `FooApiConfig`,
  `FooSignalConfig`, ...), never bare `FooConfig`.
* A bare `FooConfig` never coexists with a `FooStructureConfig`.
* Promote a Simple family to Stacked when its user-facing knobs diverge from
  the implementation's parameters, when a second default policy appears, or
  when loaders and tooling need the unrolled tree.
* Every Stacked family has a test that `policy.init(&d)` and
  `policy.to_structure().init(&d)` build modules that agree.

### Meta traits

Every family has a narrow `FooMeta` trait; the
[`ModuleInit`](https://docs.rs/bunsen/latest/bunsen/burner/module/trait.ModuleInit.html#two-config-shapes)
docs say what it is for.

* It holds only the values a caller or a test needs to read back from
  either form. Raw fields are required methods; derived values are provided
  methods. Everything else stays on the config, reached from the module
  through an accessor (`options()`, `config()`).
* Simple: `FooMeta` is implemented by `FooConfig` and `Foo` (and a context,
  if there is one).
* Stacked: required on `FooStructureConfig` and `Foo`; optional on policies.
* A test asserts that a config and the module built from it answer every
  `FooMeta` method alike, using a config that is non-default in every field.

### Modules over bare tensors

A type that owns a `Tensor` derives `Module`, even when nothing in it is
learnable: `Module` is the traversal and device-mapping trait, and deriving it
is what lets `to_device` reach the tensors. Hold non-learnable tensors bare,
not as `Param`.

* `#[derive(Module, Debug)]`, without `Clone`: the derive provides `Clone`.
* A held config, or any other value object, is `#[module(skip)]`: the derive
  treats every field that is not a primitive as a sub-module, so a field that
  is not a `Module` must be skipped. A `Param` field cannot be.
* Bare tensors are not written to records, are skipped by `ModuleMapper`
  passes (dtype casts), and do not appear in reflection. Say so on the type.

### Injected state

Cache and stream state is injected, never owned by the model; the rule's one
definition is in the "Ops and blocks" section of the
[`bunsen::ops`](https://docs.rs/bunsen/latest/bunsen/ops/index.html#ops-and-blocks)
module docs.

### `ops` and `blocks`

* **blocks** are `torch.nn`-like components, meant to be used as `Module`
  roots or as parts of a module tree.
* **ops** are operation-focused. They may use the `Module` / `Config`
  machinery, but only to hold cached tables or state.
* `ops` never imports `blocks`.

The rule's one definition is the "Ops and blocks" section of the
[`bunsen::ops`](https://docs.rs/bunsen/latest/bunsen/ops/index.html#ops-and-blocks)
module docs.

### Variant behaviour lives on the enum

A value or transform that differs per enum variant is a method on the enum,
not a `match` at the use site. The calling pipeline then reads as a sequence
of named steps, and a new variant changes one place.

### Value objects as configuration

When an op has parameters worth naming, they form a value object that
serializes and embeds in other configs, and that value object is the unit of
configuration: `ClampOp` inside `NoiseConfig` inside `DropBlockOptions`
inside `DropBlock2dConfig`. A config embeds the value object; it does not
copy its fields.

### Test modules

Every `#[cfg(test)] mod tests` opens with `use super::*;`, so the preamble
tracks the parent's imports. After it, import only what the parent does
not: dev-dependencies and crate-private test helpers.

## cargo features

### Conventional names

* `wgpu`, `cuda`, `metal` — select an accelerator backend.
* `gpu-tests` — compile the tests that need one.
* `download` — the build may reach the network.
* `fetch` — the program may reach the network at run time (`cache` alone is local).
* `onnx_gen` — generate reference models from an ONNX graph.
* `checkpoint` — fetch pretrained weights.
* `tui` — burn's terminal training dashboard; on by default in the training
  examples.

These names are **reserved** across the workspace. No crate is obliged to
offer one; a crate that offers one means this, and nothing else. Most crates
predate the list, so this is an interim expectation — new features are named
*to* it rather than around it.

### Backend selection

`cuda`, `metal`, `vulkan` and `wgpu` each select one accelerator, and are
**never** in `default`. Precedence, when more than one is set, is `cuda` >
`metal` > `vulkan` > `wgpu` > `flex`.

A backend feature enables the backend in **`bunsen`**, not only in `burn`:

```text
prefer:  wgpu = ["bunsen/wgpu"]                (a test picks the backend)
         wgpu = ["burn/wgpu", "bunsen/wgpu"]   (and `burn` is used directly)
not:     wgpu = ["burn/wgpu"]                  (`performance_device()` is Flex)
```

Only `bunsen/<backend>` moves `bunsen::support::testing::performance_device()`.
Its `cfg_select!` falls through to `Flex`, a CPU backend, when no backend
feature reaches `bunsen` — so a test written against `performance_device()`
still compiles and still passes, having quietly measured the CPU.

`flex` names that fallback rather than an accelerator, and is exempt from the
prohibition above.

An example binary does not use `performance_device()`: it picks its device
and precision at run time, with `bunsen_app::device::DeviceArgs` (`--device`,
`--precision`, `--float-dtype`), from the preferences it states in a
`DevicePrefs`. `bunsen-app` always has the CPU, and offers an accelerator only
when it was built with that backend, so an example's backend features (and the
training examples' `tui`) forward to it and to nothing else:

```text
prefer:  wgpu = ["bunsen-app/wgpu"]
not:     wgpu = ["burn/wgpu", "bunsen-app/wgpu"]   (bunsen-app already enables it)
```

### GPU tests

`gpu-tests` gates tests too slow to run without an accelerator. It is off by
default, and maps to `[]` — it enables no dependency and pulls in no backend.
It selects only which tests compile.

It is **orthogonal** to the backend features: `gpu-tests` selects *whether*
the expensive tests are built, a backend feature selects *which* accelerator
runs them. Both are passed:

```text
cargo test --release -p whisper-model-validation --features gpu-tests,wgpu
```

Do **not** infer the intent from a backend feature:

```text
prefer:  #[cfg(all(test, feature = "gpu-tests"))]
not:     #[cfg(any(feature = "wgpu", feature = "cuda", feature = "metal"))]
```

The second conflates *an accelerator is available* with *the slow tests were
asked for*, and leaves no way to have one without the other. A
`compile_error!` on the same condition is the same conflation, louder.

### Network access

`download` marks a build that may reach the network. It is **off by default**
— a plain `cargo build --workspace` never fetches. Assets are pinned to a
digest, re-verified on every build, and kept in `OUT_DIR`. An environment
variable points the build at a local file instead. A crate that exists to
bundle an asset (`bunsen-bundled-whisper`) still fetches only under the
feature that names the asset, never by default.

### Fetched assets live in `OUT_DIR`

A fetched asset lands in the build script's `OUT_DIR`, never beside the
manifest:

```text
crates/public/bunsen-bundled-whisper/
    build.rs                                  the URLs and their pinned digests
target/<profile>/build/bunsen-bundled-whisper-<hash>/out/
                                              what the digests name
```

A crate that bundles *pretrained* files lays `OUT_DIR` out as a pretrained
cache, `pretrained/<kit>/<namespace>/<sha256>/<file>`, and exposes the root
(`cache_dir()`). A `PretrainedCache` pointed at it hits the files without
knowing they were bundled: bundling is a populated cache directory, not a
kind of source, and a deployment does the same by populating the runtime
cache ahead of time.

`OUT_DIR` is the one directory a build script may write to. `cargo publish`
builds the packaged crate and fails if the build touched anything else in the
package — a `cache/` beside the manifest is exactly that — and a crate unpacked
from crates.io must not write into `~/.cargo/registry` either. The cost is that
`cargo clean` discards the assets with everything else, and a cold CI run
fetches them again. The override variables (`WHISPER_BASE_PT` and friends) are
the way to supply a local copy instead.

Do not add a cache step for them. `Swatinem/rust-cache` prunes workspace crates
from `target/` before it saves, so the assets do not ride along; a separate
`actions/cache` over an `OUT_DIR` path goes stale the moment the hash in that
path changes, which every toolchain or lockfile update does.

### Model assets

`onnx_gen` generates Rust reference models from an ONNX graph and exposes them
in a `crate::onnx_gen` module. `checkpoint` fetches pretrained weights.

Neither implies the other — they are different assets. `download` is the
aggregate switch a consumer flips; `onnx_gen` and `checkpoint` are the
per-asset switches on the crate that owns them.

### Feature documentation

Every feature carries a `##` doc comment, in the form `document-features`
renders:

```text
## Fetch the pretrained checkpoint and expose it as [`base_pt`].
checkpoint = []
```

A crate whose features are part of its public surface renders them into its
crate docs:

```text
#![doc = document_features::document_features!()]
```

The comment is required either way — the manifest is the first place a reader
looks.

## IDE run configurations

`.idea/runConfigurations/` is tracked. The configs are shared, and they are
the local form of the merge bar: what CI runs, runnable by name from the IDE
or through its MCP gateway.

### Three tiers

**Steps** (folder `Steps`) do one thing each and have no before-launch tasks:

| config | command | note |
|---|---|---|
| `Clippy Fix (single-step)` | `cargo clippy --fix --allow-no-vcs` | rewrites in place |
| `Clippy (single-step)` | `cargo clippy` | the read-only form, for a look without edits |
| `Format (single-step)` | `cargo fmt` on the **nightly** channel | `rustfmt.toml` uses unstable options; stable `fmt` silently ignores them and reformats the tree |
| `Doc Lint (single-step)` | `cargo doc --no-deps` | the workspace lints table applies to rustdoc, so a broken intra-doc link fails here |

**`Pre-Test Group`** is a shell configuration with an empty script. Its whole
job is its before-launch chain, which runs in this order:

```text
Pre-Test Group
  ├─ 1. Clippy Fix (single-step)    fix first: its rewrites need formatting
  ├─ 2. Format (single-step)        then format what clippy wrote
  └─ 3. Doc Lint (single-step)      last: slowest, and the one failure the
                                    other two cannot cause
```

**Tests** (folders `Workspace` and `bunsen`) run `cargo test`. Every one
carries `Pre-Test Group` as a before-launch task; whether it is *enabled*
is what separates the two folders:

| config | command | Pre-Test Group |
|---|---|---|
| `Test Workspace` | `cargo test --workspace` | enabled |
| `Test Workspace (wgpu\|vulkan\|cuda\|metal)` | `cargo test --workspace --features <backend>,download,gpu-tests` | enabled |
| `` Test `bunsen` `` | `cargo test -p bunsen` | attached, **disabled** |
| `` Test `bunsen` (wgpu) `` | `cargo test -p bunsen --features wgpu` | attached, **disabled** |

The `Workspace` configs are the merge bar; the `bunsen` configs are the fast
loop, and skip the chain so an edit-test cycle is one build. Flip the
checkbox on the config, not the chain, to change that for a run.

The Cargo template (`_template__of_Cargo.xml`) carries `Pre-Test Group`
enabled, so a configuration created in the IDE — a gutter run-point
included — inherits the chain. A new fast-loop config disables it on the
config.

### Rules

* **One chain, one place.** A new step goes into `Pre-Test Group`'s
  before-launch list. Never attach a step directly to a test configuration;
  the test configs reference the group and nothing else.
* **Names say the tier.** Steps are `<Verb> (single-step)`; tests are
  `Test <scope>` with the backend in parentheses; the group is the group.
  `folderName` groups them in the IDE (`Steps`, `Workspace`, `bunsen`).
* **Features live on the command line**, not the config: `requiredFeatures`
  is `false` on every test config. Backend and switch names follow the
  [cargo features](#cargo-features) chapter — `download` for the network,
  `gpu-tests` for the slow tests, a backend never implied.
* **A test that "fails to start" is a step that failed.** The IDE reports a
  before-launch failure as `Process not started... Probably build process
  failed`, with no diagnostic. Run the steps singly, in chain order, to get
  the real one; do not fall back to a bare `cargo test`, which would test a
  different configuration than the one that failed.
* **Through the MCP gateway, the chain runs too.** `execute_run_configuration`
  on a `Workspace` config runs the group first; on a `bunsen` config it does
  not. Anything that compiles is launched with `waitForExit: false` and its
  log polled: the gateway's transport caps a call well under any `timeout`
  passed, and a timed-out run keeps going with its output lost.

<!-- Assertions go here. -->
