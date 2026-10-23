# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.40.0](https://github.com/zspacelabs/bunsen/compare/bunsen-fetch-v0.39.0...bunsen-fetch-v0.40.0) - 2026-10-23

### Added

- *(sortformer)* `hf:` repos that hold a `.nemo`, their config read first ([#339](https://github.com/zspacelabs/bunsen/pull/339))

### Other

- *(validation)* Silero's and Whisper's whole suites on the CPU, when asked, over a digest-keyed asset cache ([#346](https://github.com/zspacelabs/bunsen/pull/346))

## [0.39.0](https://github.com/zspacelabs/bunsen/compare/bunsen-fetch-v0.38.0...bunsen-fetch-v0.39.0) - 2026-10-17

### Fixed

- *(bunsen-fetch)* retry a build-time download that fails on the way ([#268](https://github.com/zspacelabs/bunsen/pull/268))

### Other

- *(fetch)* one fetch implementation for the cache and the build scripts ([#263](https://github.com/zspacelabs/bunsen/pull/263))
