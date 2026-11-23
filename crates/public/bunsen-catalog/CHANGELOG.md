# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.44.0](https://github.com/zspacelabs/bunsen/compare/bunsen-catalog-v0.43.0...bunsen-catalog-v0.44.0) - 2026-11-23

### Added

- *(zonos3)* the construct loads the speaker encoder; validated on the release ([#891](https://github.com/zspacelabs/bunsen/pull/891))

### Other

- *(voxtral)* [**breaking**] the caches' pre-sizes are config fields ([#928](https://github.com/zspacelabs/bunsen/pull/928))

## [0.43.0](https://github.com/zspacelabs/bunsen/compare/bunsen-catalog-v0.42.0...bunsen-catalog-v0.43.0) - 2026-11-22

### Added

- *(qwen3_asr)* [**breaking**] the audio encoder's stem, biases, norm and positions are config fields ([#848](https://github.com/zspacelabs/bunsen/pull/848))
- *(qwen3_asr)* [**breaking**] the base models read their geometry from config.json; no silent fallback ([#847](https://github.com/zspacelabs/bunsen/pull/847))
- *(qwen3_asr)* [**breaking**] the front end is read from the model's feature-extractor file ([#846](https://github.com/zspacelabs/bunsen/pull/846))
- *(qwen3)* [**breaking**] Qwen3's attention conventions and cache pre-size are config fields ([#842](https://github.com/zspacelabs/bunsen/pull/842))
- *(qwen3_tts)* the Qwen3-TTS catalog rows and their loader ([#769](https://github.com/zspacelabs/bunsen/pull/769))
- *(zonos3)* the `zyphra/zonos3/zonos3-mini` catalog row and its loader ([#709](https://github.com/zspacelabs/bunsen/pull/709))

### Other

- [**breaking**] the rotary base is an `f64` ([#792](https://github.com/zspacelabs/bunsen/pull/792))
- crutcher/tts voice info ([#729](https://github.com/zspacelabs/bunsen/pull/729))
- [**breaking**] the attention softmax widens through `Widening`, as `softmax_widening` ([#715](https://github.com/zspacelabs/bunsen/pull/715))

## [0.42.0](https://github.com/zspacelabs/bunsen/compare/bunsen-catalog-v0.41.0...bunsen-catalog-v0.42.0) - 2026-11-13

### Added

- *(resources)* provider descriptions, Hub mirrors, digest-filed downloads, and the cards rebuilt ([#686](https://github.com/zspacelabs/bunsen/pull/686))
- *(datasets)* [**breaking**] datasets are catalog builds ([#684](https://github.com/zspacelabs/bunsen/pull/684))
- *(pretrained)* [**breaking**] the kits load from the built-in catalog ([#678](https://github.com/zspacelabs/bunsen/pull/678))
- *(catalog)* the built-in catalog crate, and ryl YAML across the tree ([#677](https://github.com/zspacelabs/bunsen/pull/677))

### Other

- *(data)* [**breaking**] the catalog, resource and build names ([#681](https://github.com/zspacelabs/bunsen/pull/681))
- *(data)* [**breaking**] the `data` tree: resources, catalog, formats, builds, datasets ([#680](https://github.com/zspacelabs/bunsen/pull/680))
