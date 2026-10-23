# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.41.0](https://github.com/zspacelabs/bunsen/compare/bunsen-bundled-silero-v0.40.0...bunsen-bundled-silero-v0.41.0) - 2026-10-23

### Other

- release v0.41.0 ([#393](https://github.com/zspacelabs/bunsen/pull/393))

## [0.40.0](https://github.com/zspacelabs/bunsen/compare/bunsen-bundled-silero-v0.39.0...bunsen-bundled-silero-v0.40.0) - 2026-10-23

### Other

- *(speech)* [**breaking**] no build fetches by default; the bundled-weight checks move to the validation crates ([#345](https://github.com/zspacelabs/bunsen/pull/345))

## [0.39.0](https://github.com/zspacelabs/bunsen/compare/bunsen-bundled-silero-v0.38.0...bunsen-bundled-silero-v0.39.0) - 2026-10-17

### Added

- *(silero-vad)* load upstream's silero_vad_16k.safetensors by name ([#264](https://github.com/zspacelabs/bunsen/pull/264))

### Other

- *(silero-vad)* [**breaking**] ship no Silero weights; upstream's safetensors is the model ([#265](https://github.com/zspacelabs/bunsen/pull/265))

## [0.38.0](https://github.com/zspacelabs/bunsen/compare/bunsen-bundled-silero-v0.37.0...bunsen-bundled-silero-v0.38.0) - 2026-10-07

### Added

- [**breaking**] migrate to burn 0.22 ([#239](https://github.com/zspacelabs/bunsen/pull/239))

## [0.37.0](https://github.com/zspacelabs/bunsen/compare/bunsen-bundled-silero-v0.36.1...bunsen-bundled-silero-v0.37.0) - 2026-10-06

### Added

- [**breaking**] reconcile the book and rustdoc; restructure modules; fix defects ([#237](https://github.com/zspacelabs/bunsen/pull/237))

## [0.34.0](https://github.com/zspacelabs/bunsen/compare/bunsen-bundled-silero-v0.33.0...bunsen-bundled-silero-v0.34.0) - 2026-09-22

### Other

- Resource maps, the pretrained cache and the construction hook: a pretrained is as many files as it is ([#217](https://github.com/zspacelabs/bunsen/pull/217))

### Added

- `BURNPACK_FILE` and `BURNPACK_SHA256`: the burnpack's name and its digest, computed when the crate is built, so bunsen's Silero kit can list it as the pinned `bundled:silero/vad`.

## [0.31.0](https://github.com/zspacelabs/bunsen/compare/bunsen-bundled-silero-v0.30.1...bunsen-bundled-silero-v0.31.0) - 2026-09-03

### Added

- *(whisper)* the stream driver, all eleven phases, and whisper-cli ([#167](https://github.com/zspacelabs/bunsen/pull/167))
- [**breaking**] Whisper reference parity, and split bundled assets from model validation ([#163](https://github.com/zspacelabs/bunsen/pull/163))

### Other

- Refactor beam search decoder: rename variables for clarity, consolidate tail clipping, and optimize test structure. Update device handling to `PerformanceBackend`. ([#173](https://github.com/zspacelabs/bunsen/pull/173))

## [0.29.0](https://github.com/zspacelabs/bunsen/compare/bunsen-bundled-silero-v0.28.0...bunsen-bundled-silero-v0.29.0) - 2026-07-17

### Other

- Refactor and modularize TenVad implementation ([#126](https://github.com/zspacelabs/bunsen/pull/126))
- add ten-vad to onnx gen ([#124](https://github.com/zspacelabs/bunsen/pull/124))
