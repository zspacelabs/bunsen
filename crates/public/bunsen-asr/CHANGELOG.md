# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.42.0](https://github.com/zspacelabs/bunsen/compare/bunsen-asr-v0.41.0...bunsen-asr-v0.42.0) - 2026-10-31

### Added

- *(bunsen-asr)* `family voxtral`, Mistral's Voxtral Realtime: transcribe, live, fetch and list ([#428](https://github.com/zspacelabs/bunsen/pull/428))

### Fixed

- *(bunsen-asr)* [**breaking**] `--language` implies `--speech`; English translated to English is transcribed on every path; `--json` records the need served ([#396](https://github.com/zspacelabs/bunsen/pull/396))

### Other

- *(bunsen-app)* [**breaking**] the device arguments report `BunsenError`s; `bunsen-asr`'s own `DeviceArgs` goes, and `bunsen-chat` takes the precision flags ([#413](https://github.com/zspacelabs/bunsen/pull/413))
- *(decoding)* [**breaking**] `kits::decoding`, the search machinery Whisper's decode used, shared: logit filters, greedy and beam search, sampling, ranking, and the temperature ladder ([#404](https://github.com/zspacelabs/bunsen/pull/404))

## [0.41.0](https://github.com/zspacelabs/bunsen/compare/bunsen-asr-v0.40.0...bunsen-asr-v0.41.0) - 2026-10-23

### Added

- *(bunsen-asr)* [**breaking**] `--publish`, a transcript subsystem `transcribe` and `live` share: text, JSON lines, RTTM, log, webhook and MQTT targets at once ([#392](https://github.com/zspacelabs/bunsen/pull/392))

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
