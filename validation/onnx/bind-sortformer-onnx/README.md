# bind-sortformer-onnx

Sortformer speaker-diarization assets for [`bunsen`](https://crates.io/crates/bunsen), fetched at build time.

**This crate hosts nothing.** Its build script downloads each asset into the build directory and checks it against a
pinned SHA-256. Each asset is behind a feature, and none is on by default, so a plain build fetches nothing:

- `onnx_gen`: `altunenes/parakeet-rs`'s ONNX export of `Nemotron-3-Diarization` (one streaming step, at 80 ms and
  10 ms), rewritten where `burn-onnx` cannot import it, and the Rust model generated from it: the reference that
  bunsen's Sortformer implementation is validated against.
- `onnx_gen_v2_1`: `Scrybl`'s ONNX export of NVIDIA's streaming Sortformer v2.1, and the Rust model generated from it:
  the reference for the v2 family.

The checkpoints are not fetched here: `sortformer-model-validation`'s tests load them through bunsen's own pinned rows,
at run time, into the weights cache.

A build without the network can point the `SORTFORMER_ONNX_*` environment variables at local copies; the API docs list
them.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0). The assets it fetches are
under their publishers' licenses: NVIDIA's `Nemotron-3-Diarization` and the `parakeet-rs` export of it are
OpenMDW-1.1.
