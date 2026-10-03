# Features, backends and the network

bunsen's cargo features fall into a few families, and each family answers a
different question about your build. The [crate docs](bunsen#crate-features)
render every feature with its own description; this page explains how to
choose among them. The naming rules behind them are in STYLE.md's
[cargo features](https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#cargo-features)
chapter.

## Which backend?

`wgpu`, `vulkan`, `cuda`, `metal` and `flex` select a burn backend for bunsen's
own use: they decide what
[`PerformanceBackend`](bunsen::support::testing::PerformanceBackend) names,
which is the backend bunsen's tests and examples run on. Your own code can use
any burn backend with bunsen whatever these say; the features only matter for
what bunsen itself instantiates.

Without a backend feature, `PerformanceBackend` falls back to the CPU. A plain
`cargo test` therefore passes while testing nothing on a GPU; see
[Testing and backends](../development/testing.md).

## Which parts of bunsen?

Some modules are optional because they pull in dependencies most users don't
need:

- `reflection`: the XML/XPath module reflection behind
  [parameter groups](../systems/param-groups.md).
- `train`: the grouped optimizers and other `burn/train` integration.
- `audio` and `tokenizer`: audio file decoding, and turning token ids into
  text, for the speech kits.
- `testing` and `audit`: the test support in
  [`support::testing`](bunsen::support::testing) and the
  [audit probes](../development/audit.md).
- `store`, `store_pytorch` and `store_safetensors`: reading PyTorch and
  safetensors checkpoints through burn-store.

The crate docs say which are on by default. `default-features = false` plus
the features you use gives the smallest build; the no-default-features
configuration is checked in CI, so it always builds.

## May the build, or the program, reach the network?

Two separate switches, never on by default:

- **`fetch`**: the program may reach the network *at run time*, to fetch
  pretrained weights or dataset shards into the
  [disk cache](bunsen::data::cache). Without it, everything works from files
  already in the cache. `indicatif` adds a progress bar.
- **Build-time network** is the `download` convention: a build script may
  fetch assets. bunsen itself has no `download` feature; the asset crates do,
  through bunsen's `*-weights` features. `silero-weights` builds Silero VAD's
  weights from a committed graph, offline. `whisper-weights` fetches
  Whisper's `base` checkpoint at build time, digest-pinned.

[Assets, bundling and the network](../development/assets.md) explains the
scheme, and why bundled weights are "a populated cache directory".

## What does docs.rs show?

docs.rs builds bunsen with an explicit feature list (`fetch`, `indicatif`,
`silero-weights`, `testing`, `audit`), not `all-features`, because
`whisper-weights` would need the network at build time. Items behind a
feature carry a badge saying which one. This book's API links are resolved
against the same list, so a link that works here works on docs.rs.
