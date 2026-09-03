# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This cycle consolidated the repository into a single Cargo workspace, promoted two demo-local crates into first-class
components, and landed a full Whisper speech model with pretrained-weight loading, alongside cleanups to dropout,
attention, embeddings, and module initialization.

### Added

- **Whisper speech model** — a complete Whisper implementation under
  `bunsen::kits::speech::whisper`, including the audio encoder, text decoder, encoder/decoder blocks, and a top-level
  `whisper_model`. A `pretrained` module loads real Whisper checkpoints from PyTorch weights (`pytorch_utils`). A new
  `crates/dev/whisper-dev` harness exercises the model (#43–#61).
- **Data cache subsystem** — `bunsen::data::cache` with on-disk caching (`disk_cache`), path resolution
  (`path_resolver`), and path utilities (`path_utils`), plus a new `bunsen::data` module root (#34, #36, #37).
- **Chat dataloader crate** — `bunsen-arrow-dataloaders`, providing Arrow-backed iteration and tokenization (`arrow`,
  `iterators`, `tokens`) for chat training data, with crate-level documentation (#38, #39, #40).
- **`ModuleInit` trait** — a standardized module-initialization trait (`bunsen::burner::module::module_init`), adopted
  across the MLP, Whisper, and Swin configs (#59).
- **Attention helpers** — `masks` and `multihead_utils` under
  `bunsen::blocks::transformers::attention` (#43, #57).
- **Embedding ops** — `bunsen::ops::embedding` module with `inverse` and
  `trivial_builders` (#56).

### Changed

- **Single unified workspace** — removed the nested per-demo sub-workspaces (`demos/bimm`, `demos/chat`, `demos/sims`),
  each of which carried its own
  `Cargo.toml` / `Cargo.lock` / `Makefile.toml`. All runnable binaries now live under a single top-level `examples/`
  directory: `conway_vis`, `conway_benchmark`, `lbm2d_vis`,
  `resnet_tiny`, `resnet_finetune`, `swin_tiny`, `train-chat`, `zsl-data-cache`, and
  `whisper-dev` (#33, #41).
- **Dropout / DropBlock modularized** — `ops::drop` became a module with a dedicated
  `drop_block` implementation, mirrored by `blocks::images::drop`; DropBlock and the SwinTransformer were optimized in a
  follow-up pass (#60, #61).
- **Whisper naming and configuration** — standardized naming (`n_states` → `d_model`), added `n_heads` / `n_layers`
  accessors to `TextDecoder` and `AudioEncoder`, introduced
  `WhisperApiConfig` with `PassConfig`, and factored out a head-dimensionality constant (#47, #48, #51, #57, #58).
- **Error handling** — replaced `anyhow` with `BunsenError` in the validators and bounds modules (#35).

### Removed

- **`bunsen-cache` crate** — its disk-cache implementation was migrated into the library as `bunsen::data::cache` (#37).
- Per-workspace `Cargo.lock` and demo scaffolding files removed as part of the workspace consolidation (#41).

[Unreleased]: https://github.com/zspacelabs/bunsen/compare/0b0f359c...HEAD
