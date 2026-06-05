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

# Crates

## Public / API Crates

* [`bunsen-firehose`](crates/bunsen-firehose) — a columnar dataloader /
  processing pipeline, with a burn batcher bridge.

## Utility Crates

* [`bunsen-contracts-macros`](crates/bunsen-contracts-macros) — the
  `shape_contract![]` proc-macro backing `bunsen`'s runtime tensor-shape
  contracts.

## Experimental Crates

These represent complex-interface + work-in-progress, unstable interface
extensions to `bunsen`; particulary those which incur large dependencies
or are not yet ready for general consumption.

* [`bunsen-firehose-image`](crates/bunsen-firehose-image) — image loading,
  augmentation, and tensor-conversion operators for `bunsen-firehose`.
* [`bunsen`](crates/bunsen) — the main "batteries included" library extending
  burn: model blocks, kits, ops, contracts, and support tooling.
* [`bunsen-preview-chat-dataloader`](crates/bunsen-preview-chat-dataloader) —
  *(preview)* an Arrow-backed chat dataloader with tokenization for LLM
  training.

# Examples

The `bunsen` repo includes a number of complex demos. The goal of the demos is to showcase the capabilities of the
library; while also collecting a working edge of problems which could and should be improved by further development.

See [`examples/`](examples/) for the full index. At a glance:

* [`conway_benchmark`](examples/conway_benchmark) — headless Game of Life (2D/3D) throughput benchmark.
* [`conway_vis`](examples/conway_vis) — real-time OpenGL Game of Life visualization.
* [`lbm2d_vis`](examples/lbm2d_vis) — real-time 2D Lattice Boltzmann fluid-flow visualization.
* [`resnet_finetune`](examples/resnet_finetune) — fine-tune a pretrained ResNet with model surgery.
* [`resnet_tiny`](examples/resnet_tiny) — train a ResNet from scratch on CINIC-10 via a firehose pipeline.
* [`swin_tiny`](examples/swin_tiny) — train a Swin Transformer V2 Tiny on CINIC-10.
* [`train-chat`](examples/train-chat) — train a NanoChat-style GPT with per-group Muon/AdamW optimizers.
* [`whisper-dev`](examples/whisper-dev) — import an OpenAI Whisper model from a PyTorch checkpoint.
* [`zsl-data-cache`](examples/zsl-data-cache) — nanochat dataset shard download/disk cache (+ `pull_shards` CLI).

# Motivation

This library is a synthesis of the utility and extension work that
I've been accumulating in:

* <https://github.com/zspacelabs/wordchipper>
* <https://github.com/zspacelabs/bimm>
* <https://github.com/zspacelabs/bimm-contracts>
* <https://github.com/zspacelabs/zsl-chat>
* <https://github.com/crutcher/clockmill>

This library is a work in progress, and I'm working to fold the various
utilities and support code from these projects into a single place; where we
can closely track the burn release cycle, and minimize the dependency-hell
churn problem for writing extensions.

I plan on continuing to work on this library, and recruit community
involvement for landing and publishing new operators and blocks in a place
we can lock down their testings and documentation.

## Future Components

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
* `clap` tooling - I've built a lot of burn-related clap tools, and I'm pretty sure some of the
  arguments/setup
  machinery
  could be shared.

# License

`bunsen` is distributed under the terms of both the MIT license and the Apache License
(Version 2.0).
See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details. Opening a pull
request is assumed to signal agreement with these licensing terms
