# bunsen-bundled-whisper

[![Crates.io Version](https://img.shields.io/crates/v/bunsen-bundled-whisper)](https://crates.io/crates/bunsen-bundled-whisper)
[![docs.rs](https://img.shields.io/docsrs/bunsen-bundled-whisper)](https://docs.rs/bunsen-bundled-whisper/latest/bunsen_bundled_whisper/)

Whisper's pretrained assets for [`bunsen`](https://crates.io/crates/bunsen)'s Whisper kit, fetched at build time.

**This crate hosts nothing.** Its build script downloads each asset into the build directory and checks it against a
pinned SHA-256; the checkpoint and vocabularies are laid out there as a bunsen pretrained cache. Each asset is behind a
feature, and none is on by default, so a plain build fetches nothing:

- `checkpoint`: OpenAI's multilingual `base.pt` (145 MB);
- `vocab`: Whisper's two `.tiktoken` vocabularies;
- `onnx_gen`: the `onnx-community/whisper-base` ONNX export, and the Rust models generated from it, the reference
  that bunsen's Whisper implementation is validated against.

Depend on it through `bunsen`'s `whisper-weights` feature, which turns on `checkpoint` and `vocab` and serves them as
the `bundled:openai/base` pretrained model. A build without the network can point the `WHISPER_*` environment
variables at local copies; the API docs list them.

- [API docs](https://docs.rs/bunsen-bundled-whisper/latest/bunsen_bundled_whisper/).
- [Assets, bundling and the network](https://zspacelabs.ai/bunsen/book/development/assets.html), in the bunsen book:
  why bundled weights are a populated cache directory.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0). The assets it fetches are
under their publishers' licenses.
