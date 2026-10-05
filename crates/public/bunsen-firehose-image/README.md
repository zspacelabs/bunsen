# bunsen-firehose-image

[![Crates.io Version](https://img.shields.io/crates/v/bunsen-firehose-image)](https://crates.io/crates/bunsen-firehose-image)
[![docs.rs](https://img.shields.io/docsrs/bunsen-firehose-image)](https://docs.rs/bunsen-firehose-image/latest/bunsen_firehose_image/)

Image operators for the [`bunsen-firehose`](https://crates.io/crates/bunsen-firehose) data pipeline: loading an image
from a path (with optional resizing and recoloring), seeded augmentation stages, and conversion to
[burn](https://burn.dev) tensor data. Each operator registers itself when the crate is linked, so the default
firehose operator environment already knows them.

- [API docs](https://docs.rs/bunsen-firehose-image/latest/bunsen_firehose_image/): the operators, and an example that
  loads, resizes and tensorizes an image.
- [Firehose](https://zspacelabs.ai/bunsen/book/systems/firehose.html), in the bunsen book: an orientation to the
  pipeline these operators plug into.
- Examples: [`resnet_tiny`](https://github.com/zspacelabs/bunsen/tree/main/examples/resnet_tiny) and
  [`swin_tiny`](https://github.com/zspacelabs/bunsen/tree/main/examples/swin_tiny) load and augment CINIC-10 images
  with these operators.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0).
