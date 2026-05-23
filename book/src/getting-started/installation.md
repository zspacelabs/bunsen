# Installation

`bunsen` is published to [crates.io](https://crates.io/crates/bunsen) and tracks
the `burn` release cadence; the major/minor version of `bunsen` matches the
`burn` version it is built against.

## Add the dependency

```toml
[dependencies]
bunsen = "0.21"
burn   = "0.21"
```

`bunsen` is a workspace with several crates; the `bunsen` umbrella crate
re-exports the commonly used pieces from `bunsen::blocks`, `bunsen::ops`,
`bunsen::contracts`, and friends.

## Toolchain

`bunsen` targets the Rust edition listed in
[`rust-toolchain.toml`](https://github.com/zspacelabs/bunsen/blob/main/rust-toolchain.toml).
A reasonably recent stable toolchain is sufficient for downstream users;
contributors building the workspace should match the pinned toolchain.

## Feature flags

The umbrella crate exposes feature flags that line up with the underlying
crates. The most important are:

| Feature   | Enables                                            |
|-----------|----------------------------------------------------|
| `default` | The most commonly used modules.                    |
| `blocks`  | The `bunsen::blocks` module catalog.               |
| `ops`     | Additional tensor operations.                      |

See each component chapter for crate-specific features.
