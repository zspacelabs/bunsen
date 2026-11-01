# bind-voxtral-onnx

Voxtral Realtime validation assets for [`bunsen`](https://crates.io/crates/bunsen), fetched at build time. Not
published: only `voxtral-model-validation` depends on it.

**This crate hosts nothing.** Its build script downloads each asset into the build directory and checks it against a
pinned SHA-256. The asset is behind a feature, off by default, so a plain build fetches nothing:

- `onnx_gen`: `onnx-community`'s fp32 ONNX export of `mistralai/Voxtral-Mini-4B-Realtime-2602` (the audio encoder,
  the decoder and the token embedding), its onnxruntime-only ops rewritten to standard ONNX by
  [`rewrite-genai-onnx`](../rewrite-genai-onnx), and the Rust models generated from it: the reference that bunsen's
  Voxtral Realtime implementation is validated against.

The checkpoint is not fetched here: `voxtral-model-validation`'s tests load it through bunsen's own pinned row, into
the weights cache.

A build without the network can point the per-file environment variables at local copies; the API docs list them.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0). The assets it fetches are
under their publishers' licenses: Mistral's checkpoint and the `onnx-community` export of it are Apache-2.0.
