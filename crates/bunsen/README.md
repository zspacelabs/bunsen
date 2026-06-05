# Bunsen

*by [ZSpaceLabs](https://zspacelabs.ai)*

[![Crates.io Version](https://img.shields.io/crates/v/bunsen)](https://crates.io/crates/bunsen)
[![Documentation](https://img.shields.io/docsrs/bunsen)](https://docs.rs/bunsen/latest/bunsen/)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)
[![Discord](https://img.shields.io/discord/1475229838754316502?label=discord)](https://discord.gg/vBgXHWCeah)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zspacelabs/bunsen)

`bunsen` aims to be a "batteries included" complementary
community standard library for extending the [burn](https://burn.dev) tensor library.

# Book

Read the [bunsen book](https://zspacelabs.ai/bunsen/book)

## Organization

### Burn Extensions

* `bunsen::burner` - this is a library of `burn::module::Module` lifecycle
  components that extend the current functionality of burn.
    * `bunsen::burner::module::reflection` has powerful tools for dynamic `burn::module::Module` reflection.
    * `bunsen::burner::optim` has parameter-group optimizer extensions.
* `contracts` - this is a library of runtime tensor-shape contracts.

### Component Libraries

* `blocks` - this is a library of `burn::module::Module` components.
  This includes simple inner layers, recurrent utility blocks, and entire
  model families.
* `ops` - this is a library `burn::tensor::Tensor` operations.
* `kits` - this is a library of full models and simulation kits.

### App and Testing Support Libs

* `errors` - this is a library of error types and tooling.
* `support` - this is a library of support functions for bunsen, including
  testing tooling which may be useful for clients.
* `zspace` - this is a library of z-space / index utilities.

# Examples

The `bunsen` repo includes a number of complex demos. The goal of the demos is to showcase the capabilities of the
library; while also collecting a working edge of problems which could and should be improved by further development.

See [`examples/`](https://github.io/zspacelabs/bunsen/tree/main/examples/) for the full index.

# License

`bunsen` is distributed under the terms of both the MIT license and the Apache License
(Version 2.0).
