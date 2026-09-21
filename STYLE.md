# Bunsen Style Guide

Conventions for code, documentation, and build configuration across the
`bunsen` workspace. This file is the source of truth; assertions added under
each chapter are applied across the code base.

> See also [`book/src/contributing/style.md`](book/src/contributing/style.md)
> for prose/Book conventions. This file governs in-source `rustdoc` and the
> `Cargo.toml` manifests.

## rustdoc

### Base Style

The base style for rustdoc is [rfc1574].

[rfc1574]: https://github.com/rust-lang/rfcs/blob/master/text/1574-more-api-documentation-conventions.md

### Coverage

* Every public interface requires rustdoc.
* Objects in a lifecycle ('Config'>'Module') require a short situational relationship description with cross-links.

Every public item needs rustdoc. This chapter defines the structure we expect,
the cross-links between paired types, and how tensor shapes are written.

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
- Rust type code spans: `&[usize; D]`, `Param<Tensor<B, R, K>>`.
- Half-open ranges and indexing: `[start, end)`, `env[$VAR]`.
- Intra-doc link syntax: `[text](url)`.

## cargo features

### Conventional names

* `wgpu`, `cuda`, `metal` — select an accelerator backend.
* `gpu-tests` — compile the tests that need one.
* `download` — the build may reach the network.
* `fetch` — the program may reach the network at run time (`cache` alone is local).
* `onnx_gen` — generate reference models from an ONNX graph.
* `checkpoint` — fetch pretrained weights.

These names are **reserved** across the workspace. No crate is obliged to
offer one; a crate that offers one means this, and nothing else. Most crates
predate the list, so this is an interim expectation — new features are named
*to* it rather than around it.

### Backend selection

`cuda`, `metal` and `wgpu` each select one accelerator, and are **never** in
`default`. Precedence, when more than one is set, is `cuda` > `metal` >
`wgpu` > `flex`.

A backend feature enables the backend in **`bunsen`**, not only in `burn`:

```text
prefer:  wgpu = ["bunsen/wgpu"]                (a test picks the backend)
         wgpu = ["burn/wgpu", "bunsen/wgpu"]   (and `burn` is used directly)
not:     wgpu = ["burn/wgpu"]                  (`PerformanceBackend` is Flex)
```

Only `bunsen/<backend>` moves `bunsen::support::testing::PerformanceBackend`.
Its `cfg_select!` falls through to `Flex`, a CPU backend, when no backend
feature reaches `bunsen` — so a test written against `PerformanceBackend`
still compiles and still passes, having quietly measured the CPU.

`flex` names that fallback rather than an accelerator, and is exempt from the
prohibition above: an example that wants to run without hardware carries it in
`default`.

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
