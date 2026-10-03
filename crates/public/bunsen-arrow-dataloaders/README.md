# bunsen-arrow-dataloaders

[![Crates.io Version](https://img.shields.io/crates/v/bunsen-arrow-dataloaders)](https://crates.io/crates/bunsen-arrow-dataloaders)
[![docs.rs](https://img.shields.io/docsrs/bunsen-arrow-dataloaders)](https://docs.rs/bunsen-arrow-dataloaders/latest/bunsen_arrow_dataloaders/)

*Preview.* Streams a text corpus stored as Parquet shards into [burn](https://burn.dev) training batches for a
language model: Parquet shards, a text column, tokens, dense token blocks, then shuffled batches of token ids.
`ChatDataLoader` runs the whole pipeline as a burn data loader; each stage is public and composes on its own.

It was built for one consumer, the
[`train-chat`](https://github.com/zspacelabs/bunsen/tree/main/examples/train-chat) example, which trains bunsen's
NanoChat GPT, and its API follows that example's needs. Expect it to change between releases.

- [API docs](https://docs.rs/bunsen-arrow-dataloaders/latest/bunsen_arrow_dataloaders/): the pipeline, stage by
  stage.
- [Shard sets and training data](https://zspacelabs.ai/bunsen/book/systems/shards.html), in the bunsen book: where
  the shards come from, and how this crate fits.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0).
