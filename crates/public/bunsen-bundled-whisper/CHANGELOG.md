# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
