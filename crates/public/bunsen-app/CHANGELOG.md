# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.42.0](https://github.com/zspacelabs/bunsen/compare/bunsen-app-v0.41.0...bunsen-app-v0.42.0) - 2026-10-31

### Other

- *(bunsen-app)* [**breaking**] the device arguments report `BunsenError`s; `bunsen-asr`'s own `DeviceArgs` goes, and `bunsen-chat` takes the precision flags ([#413](https://github.com/zspacelabs/bunsen/pull/413))

## [0.40.0](https://github.com/zspacelabs/bunsen/compare/bunsen-app-v0.39.0...bunsen-app-v0.40.0) - 2026-10-23

### Fixed

- *(asr)* exit with the device idle: `burner::device::DeviceSession`, opened through `bunsen-app` ([#363](https://github.com/zspacelabs/bunsen/pull/363))

### Other

- *(logging)* replace `stderrlog` with `env_logger`, add custom logging policies ([#303](https://github.com/zspacelabs/bunsen/pull/303))

## [0.39.0](https://github.com/zspacelabs/bunsen/compare/bunsen-app-v0.38.0...bunsen-app-v0.39.0) - 2026-10-17

### Added

- [**breaking**] publish bunsen-app and bunsen-asr ([#291](https://github.com/zspacelabs/bunsen/pull/291))

### Other

- *(device)* simplify backend selection and feature gating ([#298](https://github.com/zspacelabs/bunsen/pull/298))
