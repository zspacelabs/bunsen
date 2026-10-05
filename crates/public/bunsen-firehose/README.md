# bunsen-firehose

[![Crates.io Version](https://img.shields.io/crates/v/bunsen-firehose)](https://crates.io/crates/bunsen-firehose)
[![docs.rs](https://img.shields.io/docsrs/bunsen-firehose)](https://docs.rs/bunsen-firehose/latest/bunsen_firehose/)

A column-oriented data pipeline for feeding [burn](https://burn.dev) models. A pipeline is a schema of typed columns,
plus a build plan for each derived column that names a registered operator; an executor runs batches of rows through
the plans in dependency order, and a batcher bridge runs the whole pipeline inside burn's data loader.

It does not depend on `bunsen`. Image operators (loading, augmentation, conversion to tensor data) are in
[`bunsen-firehose-image`](https://crates.io/crates/bunsen-firehose-image).

- [API docs](https://docs.rs/bunsen-firehose/latest/bunsen_firehose/): why a pipeline rather than a batcher function,
  and an example that defines an operator, plans a column and runs a batch.
- [Firehose](https://zspacelabs.ai/bunsen/book/systems/firehose.html), in the bunsen book: an orientation to schemas,
  plans, operators and the batcher bridge.
- Examples: [`resnet_tiny`](https://github.com/zspacelabs/bunsen/tree/main/examples/resnet_tiny) and
  [`swin_tiny`](https://github.com/zspacelabs/bunsen/tree/main/examples/swin_tiny) train image classifiers on
  CINIC-10 through a firehose image pipeline.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0).
