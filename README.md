# Bunsen

*by [ZSpaceLabs](https://zspacelabs.ai)*

[![Crates.io Version](https://img.shields.io/crates/v/bunsen)](https://crates.io/crates/bunsen)
[![Documentation](https://img.shields.io/docsrs/bunsen)](https://docs.rs/bunsen/latest/bunsen/)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)
[![Discord](https://img.shields.io/discord/1475229838754316502?label=discord)](https://discord.gg/vBgXHWCeah)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zspacelabs/bunsen)

`bunsen` aims to be a "batteries included" complementary
community standard library for extending the [burn](https://burn.dev) tensor library.

# Motivation

I've been working on burn extensions and libs for coming on a year and a half; and I've accumulated
a number of interesting ideas and demos:

* <https://github.com/zspacelabs/wordchipper>
* <https://github.com/zspacelabs/bimm>
* <https://github.com/zspacelabs/bimm-contracts>
* <https://github.com/zspacelabs/zsl-chat>
* <https://github.com/crutcher/clockmill>

I've also contributed a fair amount of code to the burn ecosystem,
and I've written up the over motivation and history on my website: <https://zspacelabs.ai/history/>

This library is a work in progress, and I'm working to fold the various utilities and support code
from these projects into a single place; where we can closely track the burn release cycle,
and minimize the dependency-hell churn problem for writing extensions.

I plan on continuing to work on this library, and recruit community involvement for landing
and publishing new operators and blocks in a place we can lock down their testings and
documentation.

# Components

* `bunsen::blocks` - this is a library of `burn::module::Module` components.
  This includes simple inner layers, recurrent utility blocks, and entire
  model families.
* `bunsen::burner` - this is a library of `burn::module::Module` lifecycle
  components which extend the current functionality of burn.
* `bunsen::ops` - this is a library `burn::tensor::Tensor` operations.
* `bunsen::support` - this is a library of support functions for bunsen, including
  testing tooling which may be useful for clients.
* `bunsen::zspace` - this is a library of z-space / index utilities.
* `bunsen::contracts` - this is a library of runtime tensor-shape contracts.
* `bunsen::errors` - this is a library of error types and tooling.

# Future Components

The base libraries have significant features which haven't been polished and stabilized for bunsen
yet.

* weight/data download disk cache - there are several implementations of this in my codebase so far,
  the most robust is probably in the `wordchipper` code.
* shard fetching - being able to bind a family of shards to URL template + range pattern;
  with information on the target format; and wire that smoothly into the download and cache layer.
  this is also currently in some of the LLM/chat codebases.
* LLM `DataLoader` - a high-performance burn data loader for LLM models, built on parquet/arrow; and
  `wordchipper`.
  This is currently in the `zsl-chat` codebase.
* Data transform pipeline - I did a pretty solid pass over an in-memory data transform pipeline
  for images, called `bimm-firehose`; and it is still in the `bimm` codebase. Something like
  it is needed to train image models.
* `clap` tooling - I've built a lot of burn-related clap tools, and I'm pretty sure some of the
  arguments/setup
  machinery
  could be shared.
* the rest of the `bimm` models.
* the `bimm` and `zsl-chat` training demos.

# Demos

The `bunsen` repo includes a number of complex demos. The goal of the demos is to showcase the capabilities of the
library; while also collecting a working edge of problems which could and should be improved by further development.

* [clockmill](demos/simulations/README.md) - this is a demonstration of 2D/3D simulation on constant volume grids.

# License

`bunsen` is distributed under the terms of both the MIT license and the Apache License
(Version 2.0).
See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details. Opening a pull
request is assumed to signal agreement with these licensing terms
