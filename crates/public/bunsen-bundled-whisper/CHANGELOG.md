# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.31.0](https://github.com/zspacelabs/bunsen/compare/bunsen-bundled-whisper-v0.30.1...bunsen-bundled-whisper-v0.31.0) - 2026-09-03

### Added

- *(whisper)* the stream driver, all eleven phases, and whisper-cli ([#167](https://github.com/zspacelabs/bunsen/pull/167))
- [**breaking**] Whisper reference parity, and split bundled assets from model validation ([#163](https://github.com/zspacelabs/bunsen/pull/163))

### Other

- Refactor `MelFilterbank` into `MelFilterbankConfig` for improved configuration and validation. Rewrite tests for API updates and streamline related logic for cleaner organization. ([#165](https://github.com/zspacelabs/bunsen/pull/165))

### Added

- `vocab` fetches, digest-pins and caches Whisper's two `.tiktoken` rank
  files and exposes them as `multilingual_tiktoken()` and `gpt2_tiktoken()`.
  The URLs are pinned to the `openai/whisper` commit that last changed them,
  not to `main`. `WHISPER_MULTILINGUAL_TIKTOKEN` and `WHISPER_GPT2_TIKTOKEN`
  point the build at local copies. Off by default and independent of
  `checkpoint`, like `onnx_gen`; `bunsen/whisper-weights` turns it on.
- Initial release, mirroring `bunsen-bundled-silero`. Two independent halves,
  each behind its own off-by-default feature:
  - `checkpoint` fetches, digest-pins and caches OpenAI's multilingual Whisper
    `base.pt` and exposes it as `base_pt()`. This is what
    `bunsen::kits::speech::whisper::Whisper::load_pretrained` reads.
  - `onnx_gen` fetches the `onnx-community/whisper-base` export and generates
    `onnx_gen::{encoder, decoder}` from it — the reference implementation
    `whisper-model-validation` compares against.

  Both were previously inside `whisper-model-validation`'s own `build.rs`. The
  checkpoint is what `bunsen` loads, so it belongs beside the library; the
  reference moved with it so that one crate owns every Whisper asset and the
  validation crate is only the comparison.

  Unlike the Silero bundle, nothing here is committed: the assets total ~435 MB,
  so they are fetched and cached, and the weights are a path rather than bytes.
