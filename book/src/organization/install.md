# Installing and first use

## Add the dependency

```bash
cargo add bunsen
```

bunsen is versioned on its own semver line, released by release-plz; its
version is *not* tied to burn's. Each release builds against one burn
version, and bunsen re-exports it as [`bunsen::public::burn`](bunsen::public),
so you can use the same burn without pinning it twice. bunsen requires a
recent stable Rust; its minimum supported version is the `rust-version` on its
crates.io page.

Pick features as described in
[Features, backends and the network](./features.md). The defaults build
offline and include the common modules.

## A first program

This builds a block from its config, runs it, and checks the shape of its
output with a shape contract. It is compiled and run by `cargo test` as part
of the workspace, so it can't drift from the API:

```rust,ignore
{{#include ../../../crates/dev/book-snippets/src/first_use.rs:first_use}}
```

Three things to notice, each with its own chapter:

- `MlpConfig::init` comes from
  [`ModuleInit`](bunsen::burner::module::ModuleInit), which
  [`bunsen::prelude`] brings into scope, and the binding's type (`Mlp`)
  names the module it builds. See
  [Module design conventions](../systems/conventions.md).
- [`unpack_shape_contract!`](bunsen::contracts#macros-and-methods) both
  checks the shape and names its dimensions. See
  [Shape contracts](../systems/contracts.md).
- The device comes from bunsen's test support:
  [`cpu_device`](bunsen::support::testing::cpu_device) is burn's CPU
  backend. Your own code can create devices however burn allows, such as
  `Device::default()`, which follows the backends compiled in. See
  [Testing and backends](../development/testing.md).

## Next

- [The bunsen crate](./crate.md): what's in the library, and where things go.
- [Kits](./kits.md): whole models you can load by name.
- The [API docs](bunsen) for everything else.
