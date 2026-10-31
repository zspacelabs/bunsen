# bind-qwen3-onnx

Qwen3 language-model assets for [`bunsen`](https://crates.io/crates/bunsen), fetched at build time.

**This crate hosts nothing.** Its build script downloads each asset into the build directory and checks it against a
pinned SHA-256. Each asset is behind a feature, and none is on by default, so a plain build fetches nothing:

- `checkpoint`: Qwen's `Qwen3-0.6B`, as `transformers` ships it, laid out as a bunsen pretrained cache.
- `large_checkpoints`: `Qwen3-1.7B`, `Qwen3-4B` and `Qwen3-4B-Instruct-2507`, laid out the same way.
- `onnx_gen`: `onnx-community`'s fp32 ONNX export of `Qwen3-0.6B`, its onnxruntime-only ops rewritten to standard
  ONNX, and the Rust model generated from it: the reference that bunsen's Qwen3 implementation is validated against.

A build without the network can point the per-file environment variables at local copies; the API docs list them.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0). The assets it fetches are
under their publishers' licenses: Qwen's checkpoints and the `onnx-community` export of `Qwen3-0.6B` are Apache-2.0.
