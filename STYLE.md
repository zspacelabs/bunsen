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
digest, re-verified on every build, and cached outside `OUT_DIR`, so
`cargo clean` does not force a re-download. An environment variable points the
build at a local file instead.

`bunsen-bundled-whisper` is the one exception, enumerated rather than implied:
`checkpoint` is in its `default`, so a workspace build does fetch 145 MB on a
cold cache. Naming the exception keeps the rule absolute everywhere else.

### The `cache/` directory

A fetched asset lands in `cache/`, beside the manifest of the crate that
fetched it:

```text
crates/public/bunsen-bundled-whisper/
    build.rs                     the URLs and their pinned digests
    cache/                       what the digests name
```

Beside the manifest, not under `OUT_DIR` — `cargo clean` would otherwise cost
a re-download, and these are measured in hundreds of megabytes. Not hidden
either: a directory that large is easier to reason about when it is visible.

Its `.gitignore` entry is a **full path**, not a `**/` glob. `cache` is an
ordinary module name — `bunsen::data::cache` is source — and a glob would
quietly untrack it:

```text
prefer:  /crates/public/bunsen-bundled-whisper/cache/
not:     **/cache/                     (also matches src/data/cache)
```

A build cache does **not** substitute for this. `Swatinem/rust-cache` prunes
anything it does not recognize as a dependency from `target/`, so assets kept
there are deleted before the cache is written; CI gives `cache/` its own entry,
keyed on the `build.rs` that pins the digests.

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

<!-- Assertions go here. -->
