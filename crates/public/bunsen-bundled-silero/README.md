# bunsen-bundled-silero

[![Crates.io Version](https://img.shields.io/crates/v/bunsen-bundled-silero)](https://crates.io/crates/bunsen-bundled-silero)
[![docs.rs](https://img.shields.io/docsrs/bunsen-bundled-silero)](https://docs.rs/bunsen-bundled-silero/latest/bunsen_bundled_silero/)

Silero VAD's pretrained weights, built into the binary, for [`bunsen`](https://crates.io/crates/bunsen)'s Silero
voice-activity kit.

The crate ships Silero's ONNX graph (from [`snakers4/silero-vad`](https://github.com/snakers4/silero-vad), MIT). Its
build script converts the graph to burn weights offline, with no network and no feature needed, and records their
SHA-256. Depend on it through `bunsen`'s `silero-weights` feature, which serves the weights as the
`bundled:silero/vad` pretrained model. The `onnx_gen` feature also compiles the Rust model generated from the graph,
the reference that bunsen's Silero implementation is validated against.

- [API docs](https://docs.rs/bunsen-bundled-silero/latest/bunsen_bundled_silero/).
- [Assets, bundling and the network](https://zspacelabs.ai/bunsen/book/development/assets.html), in the bunsen book:
  why bunsen bundles weights this way.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0). The bundled ONNX graph is
Silero's, under its MIT license.
