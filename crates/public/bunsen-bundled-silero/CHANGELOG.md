# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- The crate builds without `onnx_gen`. Its build script always generates the
  burnpack weights from the committed ONNX graph, so `burn-onnx` is now an
  unconditional build dependency; `onnx_gen` only decides whether the
  generated Rust reference model is compiled in. Before this,
  `bunsen/silero-weights` alone did not build — it only worked inside a
  workspace build where `silero-model-validation` had enabled `onnx_gen`.

### Changed

- Renamed from `bunsen-onnx-gen` to `bunsen-bundled-silero`. The crate is one
  model's bundled assets, not general ONNX tooling: it ships the Silero VAD
  weights, and exposes the ONNX-generated reference behind `onnx_gen`.

### Changed

- Moved from `..` to the new `../../validation` crate group, which collects reference implementations and the
  cross-checks that step bunsen against them. The package name, contents and public API are unchanged; `bunsen` still
  depends on it for the Silero pretrained weights.

## [0.29.0](https://github.com/zspacelabs/bunsen/compare/bunsen-bundled-silero-v0.28.0...bunsen-bundled-silero-v0.29.0) - 2026-07-17

### Other

- Refactor and modularize TenVad implementation ([#126](https://github.com/zspacelabs/bunsen/pull/126))
- add ten-vad to onnx gen ([#124](https://github.com/zspacelabs/bunsen/pull/124))
