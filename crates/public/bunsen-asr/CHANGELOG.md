# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.40.0](https://github.com/zspacelabs/bunsen/compare/bunsen-asr-v0.39.0...bunsen-asr-v0.40.0) - 2026-10-23

### Added

- *(bunsen-asr)* [**breaking**] profiles, speech to text by what it is for; the families under `family` ([#344](https://github.com/zspacelabs/bunsen/pull/344))
- *(sortformer)* `hf:` repos that hold a `.nemo`, their config read first ([#339](https://github.com/zspacelabs/bunsen/pull/339))
- *(sortformer)* the streaming Sortformer v2 and v2.1 as well-known rows ([#338](https://github.com/zspacelabs/bunsen/pull/338))
- *(sortformer)* streaming presets by family and latency ([#335](https://github.com/zspacelabs/bunsen/pull/335))
- *(bunsen-asr)* `--diarize`, who said what, on every family's transcribe and live ([#324](https://github.com/zspacelabs/bunsen/pull/324))
- *(bunsen-asr)* `diarize`, who spoke when, through Sortformer ([#322](https://github.com/zspacelabs/bunsen/pull/322))
- *(speech)* attach a diarizer to the ASR stream driver ([#308](https://github.com/zspacelabs/bunsen/pull/308))

### Fixed

- *(asr)* exit with the device idle: `burner::device::DeviceSession`, opened through `bunsen-app` ([#363](https://github.com/zspacelabs/bunsen/pull/363))

### Other

- *(bunsen-asr)* microphone capture in its own module ([#321](https://github.com/zspacelabs/bunsen/pull/321))
- *(logging)* replace `stderrlog` with `env_logger`, add custom logging policies ([#303](https://github.com/zspacelabs/bunsen/pull/303))

## [0.39.0](https://github.com/zspacelabs/bunsen/compare/bunsen-asr-v0.38.0...bunsen-asr-v0.39.0) - 2026-10-17

### Added

- [**breaking**] whisper-cli is retired; bunsen-asr covers it ([#295](https://github.com/zspacelabs/bunsen/pull/295))
- *(bunsen-asr)* whisper-cli's Whisper flags, checkpoint paths, `whisper inspect` and `whisper prefabs` ([#294](https://github.com/zspacelabs/bunsen/pull/294))
- *(bunsen-asr)* `<family> fetch [--verify] NAMES...` ([#293](https://github.com/zspacelabs/bunsen/pull/293))
- *(bunsen-asr)* `<family> live`, the microphone through every family's stream driver ([#292](https://github.com/zspacelabs/bunsen/pull/292))
- [**breaking**] publish bunsen-app and bunsen-asr ([#291](https://github.com/zspacelabs/bunsen/pull/291))
