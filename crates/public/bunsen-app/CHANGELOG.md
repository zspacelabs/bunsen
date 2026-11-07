# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.42.0](https://github.com/zspacelabs/bunsen/compare/bunsen-app-v0.41.0...bunsen-app-v0.42.0) - 2026-11-07

### Added

- *(bunsen-tool)* [**breaking**] `asr` is the `speech` feature, on by default; bunsen-tool and bunsen-app no longer take bunsen's defaults ([#646](https://github.com/zspacelabs/bunsen/pull/646))
- *(validation)* [**breaking**] Qwen3-ASR by profile; the cache records when a file was last used, and `cache clear --age` ([#453](https://github.com/zspacelabs/bunsen/pull/453))
- *(bunsen-tool)* [**breaking**] `bunsen-tool`, bunsen's models on the command line; `bunsen-asr` is `bunsen-tool asr`, and its cache `bunsen-tool cache` ([#446](https://github.com/zspacelabs/bunsen/pull/446))

### Other

- test support chains through the repository's libraries, as dev-dependencies ([#639](https://github.com/zspacelabs/bunsen/pull/639))
- docs.rs builds bunsen's `nemo` and `fermion`, and `bunsen-fetch`'s `fetch` and `testing`; `bunsen-app` lists its features ([#638](https://github.com/zspacelabs/bunsen/pull/638))
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
