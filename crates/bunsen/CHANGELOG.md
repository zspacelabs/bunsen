# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.24.1](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.24.0...bunsen-v0.24.1) - 2026-06-09

### Added

- add `with_act` and `with_norm` methods to `ConvSeq1dConfig` and `ConvSeq2dConfig` ([#90](https://github.com/zspacelabs/bunsen/pull/90))

## [0.24.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.23.0...bunsen-v0.24.0) - 2026-06-09

### Fixed

- conditionally import `contracts` APIs only in debug builds ([#84](https://github.com/zspacelabs/bunsen/pull/84))

### Other

- crutcher/convseq ([#88](https://github.com/zspacelabs/bunsen/pull/88))
- consolidate module organization under `blocks` directories across kits ([#87](https://github.com/zspacelabs/bunsen/pull/87))
- remove `ResNetDownsample`, migrate to `ConvBlock2d` for downsampling ([#86](https://github.com/zspacelabs/bunsen/pull/86)) ([#86](https://github.com/zspacelabs/bunsen/pull/86))
- replace `ResNetDownsample` with `ConvBlock2d` for residual connections ([#85](https://github.com/zspacelabs/bunsen/pull/85))

## [0.23.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.22.2...bunsen-v0.23.0) - 2026-06-08

### Added

- unify around reusable Conv blocks. ([#81](https://github.com/zspacelabs/bunsen/pull/81))
- add `shape` deref tests and examples for TensorDataIndexView and TensorDataIndexMutView ([#76](https://github.com/zspacelabs/bunsen/pull/76))

### Fixed

- update module path for `unpack_shape_contract` in debug/test builds ([#80](https://github.com/zspacelabs/bunsen/pull/80))

### Other

- make `AudioEncoder` fields public with detailed rustdoc comments ([#82](https://github.com/zspacelabs/bunsen/pull/82))
- add API examples showcasing key `bunsen` features ([#77](https://github.com/zspacelabs/bunsen/pull/77))

## [0.22.2](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.22.1...bunsen-v0.22.2) - 2026-06-06

### Fixed

- fix a README link. ([#74](https://github.com/zspacelabs/bunsen/pull/74))

## [0.22.1](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.22.0...bunsen-v0.22.1) - 2026-06-05

### Added

- restructure READMEs. ([#72](https://github.com/zspacelabs/bunsen/pull/72))

### Other

- Expanded TensorDataIndexView, added TensorDataIndexMutView. ([#70](https://github.com/zspacelabs/bunsen/pull/70))
- Add STYLE.md and enforce tensor shape conventions across rustdoc ([#68](https://github.com/zspacelabs/bunsen/pull/68))

## [0.22.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.21.3...bunsen-v0.22.0) - 2026-06-05

### Fixed

- Update the README. ([#66](https://github.com/zspacelabs/bunsen/pull/66))

### Other

- Refactor and optimize DropBlock, Whisper, and SwinTransformer modules ([#61](https://github.com/zspacelabs/bunsen/pull/61))
- Modularize and implement Dropout mechanisms ([#60](https://github.com/zspacelabs/bunsen/pull/60)) ([#60](https://github.com/zspacelabs/bunsen/pull/60))
- Introduce `ModuleInit` trait for standardized module initialization ([#59](https://github.com/zspacelabs/bunsen/pull/59))
- Define and reuse constant for Whisper head dimensionality ([#58](https://github.com/zspacelabs/bunsen/pull/58))
- Refactor MLP and Whisper modules for improved configuration clarity ([#57](https://github.com/zspacelabs/bunsen/pull/57))
- Update embedding utilities and refactor Whisper modules ([#56](https://github.com/zspacelabs/bunsen/pull/56))
- Modularize pretrained Whisper model support by adding `pretrained` module ([#55](https://github.com/zspacelabs/bunsen/pull/55)) ([#55](https://github.com/zspacelabs/bunsen/pull/55))
- Refactor and add `whisper-dev` example for PyTorch Whisper models ([#54](https://github.com/zspacelabs/bunsen/pull/54))
- Fix Inconsistency ([#42](https://github.com/zspacelabs/bunsen/pull/42))
- crutcher/wblocks ([#53](https://github.com/zspacelabs/bunsen/pull/53))
- Add `n_heads` and `n_layers` methods to `TextDecoder` and `AudioEncoder`; refactor `WhisperApiConfig` with `PassConfig` for improved structural clarity ([#51](https://github.com/zspacelabs/bunsen/pull/51))
- Improve utility methods in Whisper modules and boost test coverage ([#49](https://github.com/zspacelabs/bunsen/pull/49))
- Rename `n_states` to `d_model` across Whisper modules ([#48](https://github.com/zspacelabs/bunsen/pull/48))
- Standardize naming conventions in Whisper modules ([#47](https://github.com/zspacelabs/bunsen/pull/47))
- Refactor Whisper model and enhance testing infrastructure ([#46](https://github.com/zspacelabs/bunsen/pull/46))
- Add Whisper model and refactor TextDecoder and AudioEncoder ([#45](https://github.com/zspacelabs/bunsen/pull/45))
- Add Whisper TextDecoder module and refine audio context handling ([#44](https://github.com/zspacelabs/bunsen/pull/44))
- crutcher/whisper ([#43](https://github.com/zspacelabs/bunsen/pull/43))
- Merge sub-workspaces ([#41](https://github.com/zspacelabs/bunsen/pull/41))
- crutcher/chatloader ([#38](https://github.com/zspacelabs/bunsen/pull/38))
- Migrate cache components ([#37](https://github.com/zspacelabs/bunsen/pull/37))
- Partial move of prefab ([#36](https://github.com/zspacelabs/bunsen/pull/36))
- Replace `anyhow` with `BunsenError` in validators and bounds modules ([#35](https://github.com/zspacelabs/bunsen/pull/35))
- Partial impl of bunsen::data::cache ([#34](https://github.com/zspacelabs/bunsen/pull/34))
