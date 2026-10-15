# bunsen-bundled-parakeet

Parakeet TDT assets for [`bunsen`](https://crates.io/crates/bunsen), fetched at build time.

**This crate hosts nothing.** Its build script downloads each asset into the build directory and checks it against a
pinned SHA-256. Each asset is behind a feature, and none is on by default, so a plain build fetches nothing:

- `onnx_gen`: the `istupakov/parakeet-tdt-0.6b-v3-onnx` export of NVIDIA's `parakeet-tdt-0.6b-v3` (the
  FastConformer encoder, and the TDT prediction and joint networks), and the Rust models
  generated from it: the reference that bunsen's Parakeet implementation is validated against.

A build without the network can point the `PARAKEET_*` environment variables at local copies; the API docs list them.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0). The assets it fetches are
under their publishers' licenses: NVIDIA's `parakeet-tdt-0.6b-v3` and its ONNX export are CC-BY-4.0.
