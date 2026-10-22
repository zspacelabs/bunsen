# bunsen-bundled-silero

[![Crates.io Version](https://img.shields.io/crates/v/bunsen-bundled-silero)](https://crates.io/crates/bunsen-bundled-silero)
[![docs.rs](https://img.shields.io/docsrs/bunsen-bundled-silero)](https://docs.rs/bunsen-bundled-silero/latest/bunsen_bundled_silero/)

Silero VAD's files for [`bunsen`](https://crates.io/crates/bunsen)'s Silero voice-activity kit. **Hosts none**: its
build script fetches them from a pinned revision of [`snakers4/silero-vad`](https://github.com/snakers4/silero-vad)
(MIT), checks each against a SHA-256, and keeps them in `OUT_DIR`, each behind a feature. `checkpoint` fetches upstream's
16 kHz `silero_vad_16k.safetensors`, laid out as a pretrained cache that resolves `silero/vad-16k` offline;
`silero-model-validation`'s `weights` feature turns it on. `onnx_gen` fetches upstream's ONNX graph and generates the Rust reference model
that bunsen's Silero implementation is validated against.

- [API docs](https://docs.rs/bunsen-bundled-silero/latest/bunsen_bundled_silero/).
- [Assets, bundling and the network](https://zspacelabs.ai/bunsen/book/development/assets.html), in the bunsen book:
  why bunsen bundles weights this way.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0). The files it fetches are
Silero's, under their MIT license; none is distributed with the crate.
