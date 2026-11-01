# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`bunsen-tool` continues `bunsen-asr`, whose speech to text is `bunsen-tool asr`; `bunsen-asr`'s history, to its last
release (0.41.0), is in [its changelog](https://github.com/zspacelabs/bunsen/blob/bunsen-asr-v0.41.0/crates/public/bunsen-asr/CHANGELOG.md).

## [Unreleased]

## [0.42.0](https://github.com/zspacelabs/bunsen/compare/bunsen-tool-v0.41.0...bunsen-tool-v0.42.0) - 2026-11-01

### Added

- *(bunsen-tool)* `bunsen-tool card`, the build, its cache and the models it knows, as YAML or JSON; the READMEs carry its models card ([#456](https://github.com/zspacelabs/bunsen/pull/456))
- *(validation)* [**breaking**] Qwen3-ASR by profile; the cache records when a file was last used, and `cache clear --age` ([#453](https://github.com/zspacelabs/bunsen/pull/453))
- *(bunsen-tool)* [**breaking**] `bunsen-tool chat`; bunsen-chat's library is `bunsen_tool::chat` ([#447](https://github.com/zspacelabs/bunsen/pull/447))
- *(bunsen-tool)* [**breaking**] `bunsen-tool`, bunsen's models on the command line; `bunsen-asr` is `bunsen-tool asr`, and its cache `bunsen-tool cache` ([#446](https://github.com/zspacelabs/bunsen/pull/446))
