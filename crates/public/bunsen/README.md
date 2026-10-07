# bunsen - tackling tomorrow's burn problems today!

*by [ZSpaceLabs](https://zspacelabs.ai)*

[![Crates.io Version](https://img.shields.io/crates/v/bunsen)](https://crates.io/crates/bunsen)
[![Documentation](https://img.shields.io/docsrs/bunsen)](https://docs.rs/bunsen/latest/bunsen/)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)
[![Discord](https://img.shields.io/discord/1475229838754316502?label=discord)](https://discord.gg/vBgXHWCeah)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zspacelabs/bunsen)

The [burn](https://burn.dev) team is working hard to give us a stable long term foundation
for developing high-performance and maintainable tensor programming in
Rust; The "in Rust" aspect requires a degree of type and runtime R&D research
and that requires a conservative posture to api extension.

`bunsen` provides a "batteries included" complementary community standard library
for research extensions to `burn`; ideas which are useful enough
to collect into one place, document, and build upon; but front-run
the `burn` interface frontier.
Every idea in `bunsen` seeks to either be refined and ported into `burn`,
or supplanted by the eventual `burn` solution to that problem.

- [The bunsen book](https://zspacelabs.ai/bunsen/book/) explains how bunsen is organized, the systems that cut across
  it, and how it is developed. Start with
  [Installing and first use](https://zspacelabs.ai/bunsen/book/organization/install.html).
- [The API docs](https://docs.rs/bunsen/latest/bunsen/) document every type and lifecycle, with worked examples, and
  list the crate's features.

## Kits

A kit is a whole model or simulation, with its configs and, for a model, the weights it loads by name
([Kits](https://zspacelabs.ai/bunsen/book/organization/kits.html)):

- [`images::resnet`](https://docs.rs/bunsen/latest/bunsen/kits/images/resnet/index.html): ResNet image classifiers,
  with torchvision's and `timm`'s checkpoints by name.
- [`images::swin::v2`](https://docs.rs/bunsen/latest/bunsen/kits/images/swin/v2/index.html): Swin Transformer V2
  image classifiers, without pretrained weights.
- [`gpts::nanochat`](https://docs.rs/bunsen/latest/bunsen/kits/gpts/nanochat/index.html): a port of karpathy's
  nanochat GPT, and the corpus it trains on. Work in progress.
- [`sims::conway`](https://docs.rs/bunsen/latest/bunsen/kits/sims/conway/index.html): Conway's Game of Life on
  toroidal 2D and 3D boards.
- [`sims::lbm`](https://docs.rs/bunsen/latest/bunsen/kits/sims/lbm/index.html): a D2Q9 lattice-Boltzmann fluid.
- [`speech::whisper`](https://docs.rs/bunsen/latest/bunsen/kits/speech/whisper/index.html): OpenAI's Whisper, with
  a pretrained index, decoders, and a stream driver.
- [`speech::silero_vad`](https://docs.rs/bunsen/latest/bunsen/kits/speech/silero_vad/index.html): the Silero
  voice-activity detector, with weights a build can bundle.
- [`tokens`](https://docs.rs/bunsen/latest/bunsen/kits/tokens/index.html): not a model; the ids-to-text seam the
  model kits share.

## The workspace

Most users depend on `bunsen` alone. The [bunsen repository](https://github.com/zspacelabs/bunsen) also publishes its
companion crates (the firehose data pipeline, a Parquet-to-tokens data loader, the shape-contract macro, and bundled
model assets), and holds the development, validation, and example crates.
[The workspace](https://zspacelabs.ai/bunsen/book/organization/workspace.html) describes each one.

## Examples

The repository's [`examples/`](https://github.com/zspacelabs/bunsen/tree/main/examples) directory holds runnable
programs, one or more per kit or system. The book's
[Examples](https://zspacelabs.ai/bunsen/book/organization/examples.html) page says what each one shows.

## License

`bunsen` is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
