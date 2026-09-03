# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
