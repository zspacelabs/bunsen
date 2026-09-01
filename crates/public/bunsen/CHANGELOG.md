# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- *(whisper)* `kits::speech::whisper::tokens` — the token layer, with no
  dependency. `WhisperSpecialIds` derives every special id (`eot`, `sot`, the
  language block, task and control tokens, the 1501 timestamps) from the
  base rank count and language count, or from a checkpoint's vocabulary size
  alone via `from_vocab_size` — so a multilingual model can no longer be
  driven with English-only ids by accident. `TokenPolicy` is the decode
  loop's view of it: `is_text` / `is_special` / `is_timestamp`, timestamp
  index and seconds, `text_ids`, and `sot_sequence` for the prompt. Also
  `LANGUAGES`, `Task`, and the special-token spellings as a generator, all
  pinned against `whisper.tokenizer` for both vocabularies and both language
  counts.
- *(whisper)* `kits::speech::whisper::vocab::TiktokenRanks` parses a
  `.tiktoken` rank file. Written here rather than borrowed because
  `multilingual.tiktoken` ends with `= 50256` — base64 of nothing, a
  genuinely empty token that strict decoders reject and Python accepts.
- *(kits)* `kits::tokens` — what kits share on the token side.
  `Detokenizer` is the one-method ids-to-text seam a kit holds as
  `Option<Arc<dyn _>>`. `WordchipperDetokenizer<T>` implements it behind the
  new `tokenizer` feature over any `wordchipper` decoder, with `from_spans`
  as the decode-only path (`TokenDictDecoder`, never the slab decoder, which
  reads an empty token as absent); `i64 -> T` narrowing and error mapping
  are confined to that adapter. `tokenizer` pulls `wordchipper` with
  `default-features = false`, which adds 17 crates to a `std` build (115 to
  132); `testing` enables it, as it does `audio`.
- *(whisper)* `kits::speech::whisper::text` — `token_spans` builds Whisper's
  full `{ id -> bytes }` table from parsed ranks and the special-id layout,
  and `detokenizer` / `load_detokenizer` hand it to `WordchipperDetokenizer`.
- *(whisper)* `kits::speech::whisper::clamp` — `ClampPolicy`, the injected
  object that decides the reference maximum a window is floored against:
  `observe(&mut self)` on the arrival path, `reference(&self)` immediately
  before packaging, so a provisional decode can package without mutating
  anything. `PerWindow` is today's behaviour; `MaxSeen` is the running (or,
  fed everything first, global) maximum, per batch row.
- *(whisper)* `package_mels` is now the composition of `trim_stream_tail`
  (drop the end-padding frame, once per stream) and `package_window` (floor 8
  dB below a per-row reference, affine, transpose), and is unchanged in
  behaviour — pinned by a bit-equality test against the split.
- *(whisper)* `kits::speech::whisper::driver` — the stream driver, offline
  slice. `WhisperDriverConfig::init` builds a `WhisperDriver` over a model,
  deriving the token layout from its vocabulary size; `new_context(clock,
  clamp)` opens a `WhisperStreamContext` that takes samples of any length
  through `push` / `push_at` / `flush` and hands back `Emission`s. Windows
  are decoded as they fill and committed whole; a single push of a clip
  reproduces `decode_chunked` exactly, and random-sized pushes reproduce a
  single push exactly. Voice activity, timestamps and drafts are refused at
  `init` until their phases land.
- *(whisper)* `kits::speech::whisper::clock::TimestampHistory` — a stream's
  sample-to-media-time map: anchors plus a rate, with `uniform`, `anchor`,
  `time_at` and `slice`. A bare stream is `uniform(16_000)`.
- *(whisper)* `kits::speech::whisper::emission` — `Triggers`, `CommitRule`
  and `EmissionPolicy` with its `offline` / `conservative` / `responsive`
  presets, and `Emission::{Committed, Draft}` over a `Segment`.
- *(whisper)* `ClampPolicy` gained `CloneClampPolicy` as a supertrait
  (implemented for every `ClampPolicy + Clone`), so a boxed policy can live in
  a `Module`.
- *(whisper)* `kits::speech::whisper::gate` — the speech gate: `SpeechGate`,
  the hysteresis machine over Silero's per-chunk probabilities as a streaming
  fold, and `speech_regions`, the whole-clip form, which reproduces
  `faster-whisper`'s `get_speech_timestamps` exactly (its tests are that
  function's own answers over synthetic tracks, including both `max_speech`
  split variants). `SpeechGateConfig` carries `faster_whisper()` and
  `fast_whisper_burn()` presets.
- *(whisper)* `kits::speech::whisper::regions` — `SpeechRegion` (a span of
  samples) with `snap_outward` onto the 320-sample encoder grid and `clock`
  (the parent stream's clock, sliced), plus `pad_regions` and `merge_gaps`.
- *(testdata)* `testdata/audio/jfk_moon_4s.mp3`, four seconds of speech
  around a one-second pause, for the gate's golden test against Silero.
- *(whisper)* `whisper-weights` now also fetches the two `.tiktoken`
  vocabularies through `bunsen-bundled-whisper/vocab`, reachable as
  `pretrained::bundled::{multilingual_tiktoken, gpt2_tiktoken}`.

- *(whisper)* `Whisper::load_pretrained` — loads OpenAI's multilingual Whisper
  *base* checkpoint and reports the config scanned from it. Behind the new
  `whisper-weights` feature, which pulls in `bunsen-bundled-whisper`; that
  crate fetches the 145 MB checkpoint at build time rather than shipping it, so
  **enabling the feature makes the build reach the network** on a cold cache.
  The Silero bundle ships its weights inline, because they are small enough to.
- *(whisper)* `kits::speech::whisper::pretrained::bundled` re-exports
  `bunsen-bundled-whisper`, matching
  `kits::speech::silero_vad::pretrained::bundled`. Behind `whisper-weights`.

- *(whisper)* `kits::speech::whisper::mel` — `mel_options` names the encoder's
  geometry, and `package_mels` applies the packaging its input needs: drop the
  trailing frame, floor the dynamic range 8 dB below the maximum, apply the
  `(log + 4) / 4` tail, and transpose to channels-first. Previously these
  constants existed only inside a dev example.
- *(ops)* `ops::split::split_padded` — split a (negatively indexable) dimension
  into equal chunks, zero-padding the last one so every chunk has the same
  width.
- *(support)* `support::audio::load_audio_mono_sr` reads compressed audio.
  `.wav` still goes through `hound`; every other extension is handed to
  `symphonia` (mp3), with gapless decoding on so an mp3's encoder delay does
  not shift every frame of a spectrogram computed from it. The `audio` feature
  gained `symphonia` as a dependency.
- *(support)* `support::testing::asr` — `word_error_rate`,
  `normalize_transcript` and `text_error_rate`. Behind the `testing`
  feature. These are what the `whisper-model-validation` crate judges a
  transcription with; ids become text through `kits::tokens`.

### Fixed

- *(silero)* `SileroVadCollection::load_pretrained` was not gated on
  `silero-weights` although the per-branch loaders it calls are, so any build
  with `store` but without `silero-weights` failed to compile.

### Changed

- *(burner)* move `repair_pytorch_strided_weight` from `burner::module` to a new
  `burner::store` module, which collects helpers for what crosses a module-store
  boundary.
- **breaking** *(support)* `support::audio::load_audio_mono_sr` returns
  `Vec<f32>` rather than `(hound::WavSpec, Vec<f32>)`. The spec carried nothing
  a caller did not already assert — the function rejects any file that is not
  mono at the requested rate — and it named a WAV type from a function that now
  also reads mp3. Callers that wrote `let (_, wav) = ...` drop the tuple.

### Removed

- **breaking** *(burner)* remove `burner::repro`. A reproduction is a statement
  about somebody else's code: it carries a fixture, it wants a real accelerator
  to say anything, and its whole purpose is to stop being true — none of which
  belongs on a published library's public surface. It is now the
  `burn_bug_repro` dev crate. `burner::store::repair_pytorch_strided_weight`,
  the workaround one of those defects is pinned against, is unchanged.

- **breaking** *(silero)* remove `kits::speech::silero_vad::reference`. A
  generated transliteration of the upstream ONNX graph is validation machinery
  — it exists to be disagreed with — and shipping it put a second, redundant
  Silero implementation on the public surface. It now lives in the
  `silero-model-validation` crate. The *weights* did not move:
  `SileroVadCollection::load_pretrained` is unchanged.

- *(signal)* remove `MelConverterOptions::range_clamp` and `::affine`. Dynamic-range
  packaging reduces over whatever it is handed, so carrying it inside a streaming
  converter made a chunked run differ from a whole-signal one. `MelConversionContext`
  is now unconditionally a homomorphism over chunking. `RangeClamp` and
  `AffineCompress` remain public; apply them once to a finished spectrogram, as the
  Whisper front end already did.
- *(blocks)* remove `MlpConfig::repair_strided_weights`. Repairing a checkpoint
  against `burn-store`'s stride-blind `PyTorch` read is a property of the
  checkpoint being loaded, not of the block's architecture. Callers that need it
  apply `burner::store::repair_pytorch_strided_weight` to the built module's
  `linear1`/`linear2` weights, as the Whisper kit already did for its attention
  projections.

## [0.30.1](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.30.0...bunsen-v0.30.1) - 2026-07-24

### Fixed

- *(tensor)* correct range handling in `bounded_elem` by replacing `bool_or` with `bool_and`, add test coverage ([#146](https://github.com/zspacelabs/bunsen/pull/146))

### Other

- *(tensor)* remove redundant clone in `bool_or` operation within `bounded_elem` method ([#145](https://github.com/zspacelabs/bunsen/pull/145))

## [0.30.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.29.1...bunsen-v0.30.0) - 2026-07-23

### Added

- *(tensor)* add comprehensive tensor operation extensions, including traits for `swap`, `release`, `select_dim`, range masks, counting, and more ([#143](https://github.com/zspacelabs/bunsen/pull/143))
- *(tensor)* add `select_dim` method to `TensorOpExt` for slicing and squeezing dimensions, and update related implementations ([#142](https://github.com/zspacelabs/bunsen/pull/142))
- *(tensor)* add `bounded_elem` method to `TensorIntOpExt`, refactor Conway simulations to use it for range checks ([#138](https://github.com/zspacelabs/bunsen/pull/138))
- *(stft)* initial stft sliding support. ([#134](https://github.com/zspacelabs/bunsen/pull/134))

### Other

- *(signal)* remove redundant intermediate variables in cosine window implementation ([#140](https://github.com/zspacelabs/bunsen/pull/140))
- *(life3d)* extract range utilities into `range_util` module and simplify range handling with utility methods ([#141](https://github.com/zspacelabs/bunsen/pull/141))
- Refactor life3d state update logic and range handling ([#139](https://github.com/zspacelabs/bunsen/pull/139))
- Add boolean and integer tensor operation extensions ([#137](https://github.com/zspacelabs/bunsen/pull/137))
- crutcher/l3d ([#136](https://github.com/zspacelabs/bunsen/pull/136))
- *(bunsen)* optimize state updates with inplace operations in simulations ([#135](https://github.com/zspacelabs/bunsen/pull/135))

## [0.29.1](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.29.0...bunsen-v0.29.1) - 2026-07-19

### Other

- *(builds)* fix no_default_features builds. ([#132](https://github.com/zspacelabs/bunsen/pull/132))

## [0.29.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.28.0...bunsen-v0.29.0) - 2026-07-17

### Added

- *(ten-vad)* initial shape contracts for ten-vad. ([#128](https://github.com/zspacelabs/bunsen/pull/128))

### Fixed

- fix docs ([#122](https://github.com/zspacelabs/bunsen/pull/122))

### Other

- crutcher/key index ([#130](https://github.com/zspacelabs/bunsen/pull/130))
- *(ten)* minor shape refactor ([#129](https://github.com/zspacelabs/bunsen/pull/129))
- Refactor and modularize TenVad implementation ([#126](https://github.com/zspacelabs/bunsen/pull/126))
- Backport LstmState updates ([#127](https://github.com/zspacelabs/bunsen/pull/127))

## [0.28.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.27.0...bunsen-v0.28.0) - 2026-07-13

### Other

- Remove Unused util ([#120](https://github.com/zspacelabs/bunsen/pull/120))

## [0.27.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.26.0...bunsen-v0.27.0) - 2026-07-13

### Other

- *(silero)* move SileroVadContext to a dif file ([#118](https://github.com/zspacelabs/bunsen/pull/118))
- *(silero-vad)* modularize pretrained model loading with 16kHz and 8kHz support ([#117](https://github.com/zspacelabs/bunsen/pull/117))

## [0.26.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.25.0...bunsen-v0.26.0) - 2026-07-13

### Added

- *(cuda, silero-vad)* add CUDA test support, document features, and update configurations ([#110](https://github.com/zspacelabs/bunsen/pull/110))
- *(silero-vad)* add golden context cross-test, expected output checks, and refine benchmark logic ([#107](https://github.com/zspacelabs/bunsen/pull/107))
- *(silero-vad bench)* benchmark tool ([#105](https://github.com/zspacelabs/bunsen/pull/105))

### Fixed

- make downloader and directories-next optional, activated by the cache feature ([#112](https://github.com/zspacelabs/bunsen/pull/112))

### Other

- *(silero-vad)* modularize pretrained model loading with 16kHz and 8kHz support ([#115](https://github.com/zspacelabs/bunsen/pull/115))
- *(bunsen)* update README component library names and document `silero_vad` and `whisper` modules in speech kits ([#111](https://github.com/zspacelabs/bunsen/pull/111))
- *(life2d)* optimize tensor updates, improve backend feature handling, and simplify `wrap_state_2d` logic ([#109](https://github.com/zspacelabs/bunsen/pull/109))
- *(life2d)* update data casting to `i32`, simplify slice handling, and enhance boolean conversion logic ([#108](https://github.com/zspacelabs/bunsen/pull/108))
- *(silero-vad)* replace `VadRunningContext` with `SileroVadContext`, update context handling methods, and streamline API usage in benchmarks ([#106](https://github.com/zspacelabs/bunsen/pull/106))
- Refactor Silero VAD for improved immutability and clarity ([#104](https://github.com/zspacelabs/bunsen/pull/104))
- Refactor Silero VAD for consistency, readability, and streaming support ([#102](https://github.com/zspacelabs/bunsen/pull/102))
- Refactor Silero VAD module for clarity and optimization ([#101](https://github.com/zspacelabs/bunsen/pull/101))
- *(silero)* improve test coverage. ([#99](https://github.com/zspacelabs/bunsen/pull/99))
- *(silero)* Silero VAD for improved clarity and structure ([#98](https://github.com/zspacelabs/bunsen/pull/98))

## [0.25.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.24.2...bunsen-v0.25.0) - 2026-06-24

### Added

- add Silero VAD model implementation ([#94](https://github.com/zspacelabs/bunsen/pull/94))

### Other

- cleanup unused imports and add conditional imports for feature-specific modules ([#96](https://github.com/zspacelabs/bunsen/pull/96))

## [0.24.2](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.24.1...bunsen-v0.24.2) - 2026-06-10

### Other

- usage docs. ([#92](https://github.com/zspacelabs/bunsen/pull/92))

## [0.24.1](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.24.0...bunsen-v0.24.1) - 2026-06-09

### Added

- add `with_act` and `with_norm` methods to `ConvSeq1dConfig` and `ConvSeq2dConfig` ([#90](https://github.com/zspacelabs/bunsen/pull/90))

## [0.24.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.23.0...bunsen-v0.24.0) - 2026-06-09

### Fixed

- conditionally import `contracts` APIs only in debug builds ([#84](https://github.com/zspacelabs/bunsen/pull/84))

### Other

- crutcher/convseq ([#88](https://github.com/zspacelabs/bunsen/pull/88))
- consolidate module organization under `blocks` directories across kits ([#87](https://github.com/zspacelabs/bunsen/pull/87))
- remove `ResNetDownsample`, migrate to `ConvBlock2d` for downsampling ([#86](https://github.com/zspacelabs/bunsen/pull/86)) ([#86](https://github.com/zspacelabs/bunsen/pull/86))
- replace `ResNetDownsample` with `ConvBlock2d` for residual connections ([#85](https://github.com/zspacelabs/bunsen/pull/85))

## [0.23.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.22.2...bunsen-v0.23.0) - 2026-06-08

### Added

- unify around reusable Conv blocks. ([#81](https://github.com/zspacelabs/bunsen/pull/81))
- add `shape` deref tests and examples for TensorDataIndexView and TensorDataIndexMutView ([#76](https://github.com/zspacelabs/bunsen/pull/76))

### Fixed

- update module path for `unpack_shape_contract` in debug/test builds ([#80](https://github.com/zspacelabs/bunsen/pull/80))

### Other

- make `AudioEncoder` fields public with detailed rustdoc comments ([#82](https://github.com/zspacelabs/bunsen/pull/82))
- add API examples showcasing key `bunsen` features ([#77](https://github.com/zspacelabs/bunsen/pull/77))

## [0.22.2](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.22.1...bunsen-v0.22.2) - 2026-06-06

### Fixed

- fix a README link. ([#74](https://github.com/zspacelabs/bunsen/pull/74))

## [0.22.1](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.22.0...bunsen-v0.22.1) - 2026-06-05

### Added

- restructure READMEs. ([#72](https://github.com/zspacelabs/bunsen/pull/72))

### Other

- Expanded TensorDataIndexView, added TensorDataIndexMutView. ([#70](https://github.com/zspacelabs/bunsen/pull/70))
- Add STYLE.md and enforce tensor shape conventions across rustdoc ([#68](https://github.com/zspacelabs/bunsen/pull/68))

## [0.22.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.21.3...bunsen-v0.22.0) - 2026-06-05

### Fixed

- Update the README. ([#66](https://github.com/zspacelabs/bunsen/pull/66))

### Other

- Refactor and optimize DropBlock, Whisper, and SwinTransformer modules ([#61](https://github.com/zspacelabs/bunsen/pull/61))
- Modularize and implement Dropout mechanisms ([#60](https://github.com/zspacelabs/bunsen/pull/60)) ([#60](https://github.com/zspacelabs/bunsen/pull/60))
- Introduce `ModuleInit` trait for standardized module initialization ([#59](https://github.com/zspacelabs/bunsen/pull/59))
- Define and reuse constant for Whisper head dimensionality ([#58](https://github.com/zspacelabs/bunsen/pull/58))
- Refactor MLP and Whisper modules for improved configuration clarity ([#57](https://github.com/zspacelabs/bunsen/pull/57))
- Update embedding utilities and refactor Whisper modules ([#56](https://github.com/zspacelabs/bunsen/pull/56))
- Modularize pretrained Whisper model support by adding `pretrained` module ([#55](https://github.com/zspacelabs/bunsen/pull/55)) ([#55](https://github.com/zspacelabs/bunsen/pull/55))
- Refactor and add `whisper-dev` example for PyTorch Whisper models ([#54](https://github.com/zspacelabs/bunsen/pull/54))
- Fix Inconsistency ([#42](https://github.com/zspacelabs/bunsen/pull/42))
- crutcher/wblocks ([#53](https://github.com/zspacelabs/bunsen/pull/53))
- Add `n_heads` and `n_layers` methods to `TextDecoder` and `AudioEncoder`; refactor `WhisperApiConfig` with `PassConfig` for improved structural clarity ([#51](https://github.com/zspacelabs/bunsen/pull/51))
- Improve utility methods in Whisper modules and boost test coverage ([#49](https://github.com/zspacelabs/bunsen/pull/49))
- Rename `n_states` to `d_model` across Whisper modules ([#48](https://github.com/zspacelabs/bunsen/pull/48))
- Standardize naming conventions in Whisper modules ([#47](https://github.com/zspacelabs/bunsen/pull/47))
- Refactor Whisper model and enhance testing infrastructure ([#46](https://github.com/zspacelabs/bunsen/pull/46))
- Add Whisper model and refactor TextDecoder and AudioEncoder ([#45](https://github.com/zspacelabs/bunsen/pull/45))
- Add Whisper TextDecoder module and refine audio context handling ([#44](https://github.com/zspacelabs/bunsen/pull/44))
- crutcher/whisper ([#43](https://github.com/zspacelabs/bunsen/pull/43))
- Merge sub-workspaces ([#41](https://github.com/zspacelabs/bunsen/pull/41))
- crutcher/chatloader ([#38](https://github.com/zspacelabs/bunsen/pull/38))
- Migrate cache components ([#37](https://github.com/zspacelabs/bunsen/pull/37))
- Partial move of prefab ([#36](https://github.com/zspacelabs/bunsen/pull/36))
- Replace `anyhow` with `BunsenError` in validators and bounds modules ([#35](https://github.com/zspacelabs/bunsen/pull/35))
- Partial impl of bunsen::data::cache ([#34](https://github.com/zspacelabs/bunsen/pull/34))
