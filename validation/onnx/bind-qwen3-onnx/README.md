# bind-qwen3-onnx

Qwen3 language-model assets for [`bunsen`](https://crates.io/crates/bunsen), fetched at build time.

**This crate hosts nothing.** Its build script downloads each asset into the build directory and checks it against a
pinned SHA-256. The asset is behind a feature, off by default, so a plain build fetches nothing:

- `onnx_gen`: `onnx-community`'s fp32 ONNX export of `Qwen3-0.6B`, its onnxruntime-only ops rewritten to standard
  ONNX, and the Rust model generated from it: the reference that bunsen's Qwen3 implementation is validated against.

The checkpoints are not fetched here: `qwen3-model-validation`'s tests load them through bunsen's own pinned rows, into
the weights cache.

A build without the network can point the per-file environment variables at local copies; the API docs list them.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0). The assets it fetches are
under their publishers' licenses: Qwen's checkpoints and the `onnx-community` export of `Qwen3-0.6B` are Apache-2.0.
