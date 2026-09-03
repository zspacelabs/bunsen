# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- *(whisper)* `pretrained::bundled_vocabulary(ids)`, under `whisper-weights`:
  the bundled `.tiktoken` rank file that matches a token layout (`multilingual` or `gpt2`), so the pairing with a
  checkpoint follows from its vocabulary size rather than the caller's memory. `examples/whisper-cli`
  transcribes an audio file with the bundled checkpoint and vocabulary through the stream driver, with the presets,
  timestamps, beams, the fallback ladder, and the bundled Silero VAD behind flags.
- *(whisper)* The shared cross-attention cache for beams, §5.8's first tower level.
  `TextDecoder::new_cache_grouped(xa, group)` projects the cross-attention keys and values once per audio, and a forward
  over
  `group` rows per audio (`row = audio * group + member`) folds the group into the query's sequence axis around the
  cross-attention call (`layer_norm_cross_attn_w_kv_cache_grouped`,
  `ResidualDecoderAttentionBlock::forward_w_kv_cache_grouped`): two reshapes on a few kilobytes of queries, and the
  cache — the bulk of what a decode holds, five times over at beam 5 — is never repeated. The search uses it for beams
  by default; `DecodeConfig::shared_cross_kv`
  set false keeps the materialized ground as the oracle. Pinned (I10) at the cache on logits, at the search on ids and
  score, and by the beam-5 gate against upstream. Measurement declined the other levels: the per-step cost is flat from
  32 to 224 tokens, so §5.7's preallocated self-attention cache and a copy-free reorder have nothing to remove yet.
- *(whisper)* `benches/whisper_decode.rs`: the decode loop at `base.en`'s shapes with random weights and a stop token
  never emitted, so every case runs to its cap — the encoder on its own, then greedy at 32 and 224 tokens, beam 5, and a
  batch of four on encoder output already in hand. The first call of each case is timed once and printed as `cold`,
  since this backend's shape-keyed autotune makes cold and warm differ by more than any copy an optimization could
  remove; criterion's numbers are warm. The tower levels of §5.7 and §5.8 are gated on these numbers.
- *(whisper)* The responsive preset. `EmissionPolicy::responsive()` is accepted: under the `interval` trigger, while
  speech is in progress and an interval of media time has passed since the last draft or commit, a stream decodes
  everything past its seek pointer and emits it as
  `Emission::Draft` — covering all audio since the last commit and superseding the previous draft whole, touching
  nothing but the pacing.
  `advance_ready` batches drafts with commits. The commits are exactly
  `conservative()`'s (I9), pinned on the speech clip with a scripted decode. A zero interval is refused.
- *(whisper)* Fallback. `kits::speech::whisper::decode::fallback`:
  `WhisperFallbackConfig` (the temperature ladder, the compression-ratio / log-probability / no-speech thresholds,
  `best_of`), its clauses as pure functions (`needs_fallback`, `should_skip`, `resets_prompt`),
  `compression_ratio` (zlib, as upstream), and `decode_with_fallback`, the ladder as an orchestration over a decode
  closure. `DecodedTokens` is what a decode now says about an audio beyond its ids: the cumulative log probability
  (`avg_logprob`) and the `<|nospeech|>` probability probed at the sot position of the first forward, through
  `decode_windows_full` /
  `decode_features_full`; `DecodeConfig` takes `temperature` and
  `best_of`, and `WhisperGreedyDecoder` samples above zero by Gumbel-max on the backend's own random numbers, `best_of`
  trajectories per audio. The driver takes `fallback`; a window the policy calls silence is skipped whole; a rung above
  0.5 resets the prompt carry; the batched path hands its temperature-zero result in as the first rung. bunsen's default
  ladder is temperature zero alone (`FallbackConfig::upstream()` is
  `transcribe()`'s), a deliberate choice for a stream driver.
- *(whisper)* Timestamps. `ApplyTimestampRules` is upstream's timestamp grammar as a `LogitFilter`, its history clauses
  a pure function (`forbidden`) tested clause by clause; `RestrictToLanguages` and
  `Whisper::detect_language` are upstream's one-step language detection;
  `Whisper::decode_features` decodes from encoder output already in hand.
  `kits::speech::whisper::segments::split_window` is the seek loop's splitting as a pure function: consecutive
  timestamps close segments, a lone final timestamp takes the whole window, otherwise the seek advances to the last
  closed timestamp and the unfinished tail is decoded again. The driver accepts `timestamps` (with
  `max_initial_timestamp`, default one second) and, on a multilingual checkpoint, `language: None`, detecting the
  language per stream from its first window. Under timestamps a decode commits one segment per closed pair on the
  stream's clock and the seek advances by timestamps; `CommitRule::LastTimestamp` emits the unfinished tail as a draft
  first. Pinned through the validation crate: fixed windows with timestamps and the whole-clip seek loop against
  `transcribe()`'s segments and times, and detection saying `en`.
- *(whisper)* `kits::speech::whisper::decode` is now a directory module with the search behind two seams. `TokenDecoder`
  is the search (`WhisperGreedyDecoder`, and `WhisperBeamSearchDecoder`, upstream's beam search: candidates deduplicated
  by full sequence, a finished set capped by `patience`, and the self-attention cache permuted through the new
  `TextDecoderCache::reorder`
  while the cross-attention cache stays put); `LogitFilter` is what it may not pick (`SuppressTokens`, `SuppressBlank`,
  and `default_filters`, which derives upstream's default suppress list from the rank file alone, pinned against
  `whisper.tokenizer` for both vocabularies); `SequenceRanker` and
  `MaximumLikelihoodRanker` pick the winner. `DecodeConfig` and
  `Whisper::decode_windows` drive it; `GreedyDecodeConfig` and the
  `decode_window*` methods are unchanged and decode exactly as before. The driver takes `beam_size`, `patience` and
  `length_penalty`, and
  `WhisperDriver::with_logit_filters`; it is no longer a `Module` (it holds backend-typed policy objects, which the
  derive cannot carry) but stays
  `Clone`.
- *(whisper)* `kits::speech::whisper::tokens` — the token layer, with no dependency. `WhisperSpecialIds` derives every
  special id (`eot`, `sot`, the language block, task and control tokens, the 1501 timestamps) from the base rank count
  and language count, or from a checkpoint's vocabulary size alone via `from_vocab_size` — so a multilingual model can
  no longer be driven with English-only ids by accident. `WhisperTokenLayout` is the decode loop's view of it:
  `is_text` /
  `is_special` / `is_timestamp`, timestamp index and seconds, `text_ids`, and `sot_sequence` for the prompt. Also
  `LANGUAGES`, `WhisperTask`, and the special-token spellings as a generator, all pinned against `whisper.tokenizer` for
  both vocabularies and both language counts.
- *(whisper)* `kits::speech::whisper::vocab::TiktokenRanks` parses a
  `.tiktoken` rank file. Written here rather than borrowed because
  `multilingual.tiktoken` ends with `= 50256` — base64 of nothing, a genuinely empty token that strict decoders reject
  and Python accepts.
- *(kits)* `kits::tokens` — what kits share on the token side.
  `Detokenizer` is the one-method ids-to-text seam a kit holds as
  `Option<Arc<dyn _>>`. `WordchipperDetokenizer<T>` implements it behind the new `tokenizer` feature over any
  `wordchipper` decoder, with `from_spans`
  as the decode-only path (`TokenDictDecoder`, never the slab decoder, which reads an empty token as absent); `i64 -> T`
  narrowing and error mapping are confined to that adapter. `tokenizer` pulls `wordchipper` with
  `default-features = false`, which adds 17 crates to a `std` build (115 to 132); `testing` enables it, as it does
  `audio`.
- *(whisper)* `kits::speech::whisper::text` — `token_spans` builds Whisper's full `{ id -> bytes }` table from parsed
  ranks and the special-id layout, and `detokenizer` / `load_detokenizer` hand it to `WordchipperDetokenizer`.
- *(whisper)* `kits::speech::whisper::clamp` — `StreamClampPolicy`, the injected object that decides the reference
  maximum a window is floored against:
  `observe(&mut self)` on the arrival path, `reference(&self)` immediately before packaging, so a provisional decode can
  package without mutating anything. `PerWindow` is today's behaviour; `RunningMaxClamp` is the running (or, fed
  everything first, global) maximum, per batch row.
- *(whisper)* `package_mels` is now the composition of `drop_last_frame`
  (drop the end-padding frame, once per stream) and `package_window` (floor 8 dB below a per-row reference, affine,
  transpose), and is unchanged in behaviour — pinned by a bit-equality test against the split.
- *(whisper)* `kits::speech::whisper::driver` — the stream driver, offline slice. `WhisperDriverConfig::init` builds a
  `WhisperStreamDriver` over a model, deriving the token layout from its vocabulary size; `new_context(clock,
  clamp)` opens a `WhisperStreamContext` that takes samples of any length through `push` / `anchor_write_read` / `flush`
  and hands back `WhisperEmission`s. Windows are decoded as they fill and committed whole; a single push of a clip
  reproduces
  `decode_chunked` exactly, and random-sized pushes reproduce a single push exactly. Voice activity, timestamps and
  drafts are refused at
  `init` until their phases land.
- *(whisper)* `kits::speech::whisper::clock::TimestampHistory` — a stream's sample-to-media-time map: anchors plus a
  rate, with `uniform`, `anchor`,
  `time_at` and `slice`. A bare stream is `uniform(16_000)`.
- *(whisper)* `kits::speech::whisper::emission` — `DecodeTriggers`, `CommitRule`
  and `EmissionPolicy` with its `offline` / `conservative` / `responsive`
  presets, and `Emission::{Committed, Draft}` over a `TranscriptSegment`.
- *(whisper)* `StreamClampPolicy` gained `CloneClampPolicy` as a supertrait (implemented for every
  `ClampPolicy + Clone`), so a boxed policy can live in a `Module`.
- *(whisper)* `kits::speech::whisper::gate` — the speech gate: `VoiceActivityFilter`, the hysteresis machine over
  Silero's per-chunk probabilities as a streaming fold, and `SpeechGateConfig::speech_regions`, the whole-clip form,
  which reproduces
  `faster-whisper`'s `get_speech_timestamps` exactly (its tests are that function's own answers over synthetic tracks,
  including both `max_speech`
  split variants). `VoiceActivityFilterConfig` carries `faster_whisper()` and
  `fast_whisper_burn()` presets.
- *(whisper)* `kits::speech::whisper::regions` — `SpeechRegion` (a span of samples) with `snap_outward` onto the
  320-sample encoder grid and `clock`
  (the parent stream's clock, sliced), plus `pad_regions` and `merge_gaps`.
- *(testdata)* `testdata/audio/jfk_moon_4s.mp3`, four seconds of speech around a one-second pause, for the gate's golden
  test against Silero.
- *(whisper)* The driver's second deployment, conservative real time.
  `WhisperDriver::with_vad` attaches a `SileroVad` and a `VoiceActivityFilterConfig`; under
  `EmissionPolicy::conservative()` each speech region the gate closes is decoded as its own unit and committed with
  times off the parent stream's clock, and a full window of silence is skipped rather than decoded. `feed` / `advance`
  split `push` in two, and
  `driver::advance_ready(&driver, &mut [context])` advances many streams with one decode per prompt group — server-batch
  mode as a function, not a type. `end_input` is the input half of `flush`.
- *(whisper)* `whisper-weights` now also fetches the two `.tiktoken`
  vocabularies through `bunsen-bundled-whisper/vocab`, reachable as
  `pretrained::bundled::{multilingual_tiktoken, gpt2_tiktoken}`.

- *(whisper)* `Whisper::load_pretrained` — loads OpenAI's multilingual Whisper *base* checkpoint and reports the config
  scanned from it. Behind the new
  `whisper-weights` feature, which pulls in `bunsen-bundled-whisper`; that crate fetches the 145 MB checkpoint at build
  time rather than shipping it, so **enabling the feature makes the build reach the network** on a cold cache. The
  Silero bundle ships its weights inline, because they are small enough to.
- *(whisper)* `kits::speech::whisper::pretrained::bundled` re-exports
  `bunsen-bundled-whisper`, matching
  `kits::speech::silero_vad::pretrained::bundled`. Behind `whisper-weights`.

- *(whisper)* `kits::speech::whisper::mel` — `mel_converter_options` names the encoder's geometry, and `package_mels`
  applies the packaging its input needs: drop the trailing frame, floor the dynamic range 8 dB below the maximum, apply
  the
  `(log + 4) / 4` tail, and transpose to channels-first. Previously these constants existed only inside a dev example.
- *(ops)* `ops::split::split_padded` — split a (negatively indexable) dimension into equal chunks, zero-padding the last
  one so every chunk has the same width.
- *(support)* `support::audio::load_audio_mono_sr` reads compressed audio.
  `.wav` still goes through `hound`; every other extension is handed to
  `symphonia` (mp3), with gapless decoding on so an mp3's encoder delay does not shift every frame of a spectrogram
  computed from it. The `audio` feature gained `symphonia` as a dependency.
- *(support)* `support::testing::asr` — `word_error_rate`,
  `normalize_transcript` and `text_error_rate`. Behind the `testing`
  feature. These are what the `whisper-model-validation` crate judges a transcription with; ids become text through
  `kits::tokens`.

### Fixed

- *(silero)* `SileroVadCollection::load_pretrained` was not gated on
  `silero-weights` although the per-branch loaders it calls are, so any build with `store` but without `silero-weights`
  failed to compile.

### Changed

- *(whisper)* Everything a checkpoint takes on convention now lives on the model as two defaulted configs, and nothing
  is a constant. `WhisperApiConfig::front_end` (`WhisperFrontEndConfig`: sample rate, hop and window in ms, clamp range
  in dB) and `WhisperApiConfig::tokens` (`WhisperTokenLayoutConfig`: the language table, the two base-vocabulary sizes,
  the special spellings, the timestamp count and step) ride into `Whisper<B>` and are read through
  `WhisperMeta::front_end()` / `token_layout()`; `PytorchWhisperScanner` stamps both. The driver derives from them:
  `front_end.mel_options(n_mels)` (fallible: the grid must fall on whole samples), `package_window` /
  `package_mels` as methods on the front end, and `WhisperDriver::sample_rate()`, `front_end()` and
  `encoder_grid()` in place of the removed `SAMPLE_RATE`, `TIMESTAMP_STEP_SAMPLES`, `ENCODER_GRID` and
  `RANGE_CLAMP_DB`; `AUDIO_ENCODER_STRIDE` names the conv head's stride. `WhisperSpecialIds` stays a `Copy` value of
  numbers (gaining `timestamp_tokens` and `multilingual`); `WhisperTokenLayout` carries the layout, is `Clone` rather
  than
  `Copy`, and owns the name lookups (`language_token`, `language_code`, `languages`, `special_names`,
  `timestamp_seconds`); `detokenizer`, `load_detokenizer` and `token_spans` take a `&TokenPolicy`.
  `WhisperDriver::with_vad` and `configure_vad_filter` return a `BunsenResult`, rejecting a VAD or a filter at another
  rate, or a filter chunk that is not the model's.
- *(whisper)* The driver's API surface is `kits::speech::whisper::driver::` — `WhisperEmission`, `EmissionPolicy`,
  `StreamClock`, the clamp policies, `VoiceActivityFilterConfig`, `WhisperTokenLayout`, the detokenizer helpers — and
  `driver::support::` holds only internals (regions, segments). Imports of those items through
  `driver::support::` move up one level.
- *(burner)* move `repair_pytorch_strided_weight` from `burner::module` to a new
  `burner::store` module, which collects helpers for what crosses a module-store boundary.
- **breaking** *(support)* `support::audio::load_audio_mono_sr` returns
  `Vec<f32>` rather than `(hound::WavSpec, Vec<f32>)`. The spec carried nothing a caller did not already assert — the
  function rejects any file that is not mono at the requested rate — and it named a WAV type from a function that now
  also reads mp3. Callers that wrote `let (_, wav) = ...` drop the tuple.

### Removed

- **breaking** *(burner)* remove `burner::repro`. A reproduction is a statement about somebody else's code: it carries a
  fixture, it wants a real accelerator to say anything, and its whole purpose is to stop being true — none of which
  belongs on a published library's public surface. It is now the
  `burn_bug_repro` dev crate. `burner::store::repair_pytorch_strided_weight`, the workaround one of those defects is
  pinned against, is unchanged.

- **breaking** *(silero)* remove `kits::speech::silero_vad::reference`. A generated transliteration of the upstream ONNX
  graph is validation machinery — it exists to be disagreed with — and shipping it put a second, redundant Silero
  implementation on the public surface. It now lives in the
  `silero-model-validation` crate. The *weights* did not move:
  `SileroVadCollection::load_pretrained` is unchanged.

- *(signal)* remove `MelConverterOptions::range_clamp` and `::affine`. Dynamic-range packaging reduces over whatever it
  is handed, so carrying it inside a streaming converter made a chunked run differ from a whole-signal one.
  `PerceptiveAudioConversionContext`
  is now unconditionally a homomorphism over chunking. `RangeClamp` and
  `AffineCompress` remain public; apply them once to a finished spectrogram, as the Whisper front end already did.
- *(blocks)* remove `MlpConfig::repair_strided_weights`. Repairing a checkpoint against `burn-store`'s stride-blind
  `PyTorch` read is a property of the checkpoint being loaded, not of the block's architecture. Callers that need it
  apply `burner::store::repair_pytorch_strided_weight` to the built module's
  `linear1`/`linear2` weights, as the Whisper kit already did for its attention projections.

## [0.30.1](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.30.0...bunsen-v0.30.1) - 2026-07-24

### Fixed

- *(tensor)* correct range handling in `bounded_elem` by replacing `bool_or` with `bool_and`, add test coverage
  ([#146](https://github.com/zspacelabs/bunsen/pull/146))

### Other

- *(tensor)* remove redundant clone in `bool_or` operation within `bounded_elem` method
  ([#145](https://github.com/zspacelabs/bunsen/pull/145))

## [0.30.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.29.1...bunsen-v0.30.0) - 2026-07-23

### Added

- *(tensor)* add comprehensive tensor operation extensions, including traits for `swap`, `release`, `select_dim`, range
  masks, counting, and more ([#143](https://github.com/zspacelabs/bunsen/pull/143))
- *(tensor)* add `select_dim` method to `TensorOpExt` for slicing and squeezing dimensions, and update related
  implementations ([#142](https://github.com/zspacelabs/bunsen/pull/142))
- *(tensor)* add `bounded_elem` method to `TensorIntOpExt`, refactor Conway simulations to use it for range checks
  ([#138](https://github.com/zspacelabs/bunsen/pull/138))
- *(stft)* initial stft sliding support. ([#134](https://github.com/zspacelabs/bunsen/pull/134))

### Other

- *(signal)* remove redundant intermediate variables in cosine window implementation
  ([#140](https://github.com/zspacelabs/bunsen/pull/140))
- *(life3d)* extract range utilities into `range_util` module and simplify range handling with utility methods
  ([#141](https://github.com/zspacelabs/bunsen/pull/141))
- Refactor life3d state update logic and range handling ([#139](https://github.com/zspacelabs/bunsen/pull/139))
- Add boolean and integer tensor operation extensions ([#137](https://github.com/zspacelabs/bunsen/pull/137))
- crutcher/l3d ([#136](https://github.com/zspacelabs/bunsen/pull/136))
- *(bunsen)* optimize state updates with inplace operations in simulations
  ([#135](https://github.com/zspacelabs/bunsen/pull/135))

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
- *(silero-vad)* modularize pretrained model loading with 16kHz and 8kHz support
  ([#117](https://github.com/zspacelabs/bunsen/pull/117))

## [0.26.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.25.0...bunsen-v0.26.0) - 2026-07-13

### Added

- *(cuda, silero-vad)* add CUDA test support, document features, and update configurations
  ([#110](https://github.com/zspacelabs/bunsen/pull/110))
- *(silero-vad)* add golden context cross-test, expected output checks, and refine benchmark logic
  ([#107](https://github.com/zspacelabs/bunsen/pull/107))
- *(silero-vad bench)* benchmark tool ([#105](https://github.com/zspacelabs/bunsen/pull/105))

### Fixed

- make downloader and directories-next optional, activated by the cache feature
  ([#112](https://github.com/zspacelabs/bunsen/pull/112))

### Other

- *(silero-vad)* modularize pretrained model loading with 16kHz and 8kHz support
  ([#115](https://github.com/zspacelabs/bunsen/pull/115))
- *(bunsen)* update README component library names and document `silero_vad` and `whisper` modules in speech kits
  ([#111](https://github.com/zspacelabs/bunsen/pull/111))
- *(life2d)* optimize tensor updates, improve backend feature handling, and simplify `wrap_state_2d` logic
  ([#109](https://github.com/zspacelabs/bunsen/pull/109))
- *(life2d)* update data casting to `i32`, simplify slice handling, and enhance boolean conversion logic
  ([#108](https://github.com/zspacelabs/bunsen/pull/108))
- *(silero-vad)* replace `VadRunningContext` with `SileroVadContext`, update context handling methods, and streamline
  API usage in benchmarks ([#106](https://github.com/zspacelabs/bunsen/pull/106))
- Refactor Silero VAD for improved immutability and clarity ([#104](https://github.com/zspacelabs/bunsen/pull/104))
- Refactor Silero VAD for consistency, readability, and streaming support
  ([#102](https://github.com/zspacelabs/bunsen/pull/102))
- Refactor Silero VAD module for clarity and optimization ([#101](https://github.com/zspacelabs/bunsen/pull/101))
- *(silero)* improve test coverage. ([#99](https://github.com/zspacelabs/bunsen/pull/99))
- *(silero)* Silero VAD for improved clarity and structure ([#98](https://github.com/zspacelabs/bunsen/pull/98))

## [0.25.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.24.2...bunsen-v0.25.0) - 2026-06-24

### Added

- add Silero VAD model implementation ([#94](https://github.com/zspacelabs/bunsen/pull/94))

### Other

- cleanup unused imports and add conditional imports for feature-specific modules
  ([#96](https://github.com/zspacelabs/bunsen/pull/96))

## [0.24.2](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.24.1...bunsen-v0.24.2) - 2026-06-10

### Other

- usage docs. ([#92](https://github.com/zspacelabs/bunsen/pull/92))

## [0.24.1](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.24.0...bunsen-v0.24.1) - 2026-06-09

### Added

- add `with_act` and `with_norm` methods to `ConvSeq1dConfig` and `ConvSeq2dConfig`
  ([#90](https://github.com/zspacelabs/bunsen/pull/90))

## [0.24.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.23.0...bunsen-v0.24.0) - 2026-06-09

### Fixed

- conditionally import `contracts` APIs only in debug builds ([#84](https://github.com/zspacelabs/bunsen/pull/84))

### Other

- crutcher/convseq ([#88](https://github.com/zspacelabs/bunsen/pull/88))
- consolidate module organization under `blocks` directories across kits
  ([#87](https://github.com/zspacelabs/bunsen/pull/87))
- remove `ResNetDownsample`, migrate to `ConvBlock2d` for downsampling
  ([#86](https://github.com/zspacelabs/bunsen/pull/86)) ([#86](https://github.com/zspacelabs/bunsen/pull/86))
- replace `ResNetDownsample` with `ConvBlock2d` for residual connections
  ([#85](https://github.com/zspacelabs/bunsen/pull/85))

## [0.23.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.22.2...bunsen-v0.23.0) - 2026-06-08

### Added

- unify around reusable Conv blocks. ([#81](https://github.com/zspacelabs/bunsen/pull/81))
- add `shape` deref tests and examples for TensorDataIndexView and TensorDataIndexMutView
  ([#76](https://github.com/zspacelabs/bunsen/pull/76))

### Fixed

- update module path for `unpack_shape_contract` in debug/test builds
  ([#80](https://github.com/zspacelabs/bunsen/pull/80))

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

- Refactor and optimize DropBlock, Whisper, and SwinTransformer modules
  ([#61](https://github.com/zspacelabs/bunsen/pull/61))
- Modularize and implement Dropout mechanisms ([#60](https://github.com/zspacelabs/bunsen/pull/60))
  ([#60](https://github.com/zspacelabs/bunsen/pull/60))
- Introduce `ModuleInit` trait for standardized module initialization
  ([#59](https://github.com/zspacelabs/bunsen/pull/59))
- Define and reuse constant for Whisper head dimensionality ([#58](https://github.com/zspacelabs/bunsen/pull/58))
- Refactor MLP and Whisper modules for improved configuration clarity
  ([#57](https://github.com/zspacelabs/bunsen/pull/57))
- Update embedding utilities and refactor Whisper modules ([#56](https://github.com/zspacelabs/bunsen/pull/56))
- Modularize pretrained Whisper model support by adding `pretrained` module
  ([#55](https://github.com/zspacelabs/bunsen/pull/55)) ([#55](https://github.com/zspacelabs/bunsen/pull/55))
- Refactor and add `whisper-dev` example for PyTorch Whisper models
  ([#54](https://github.com/zspacelabs/bunsen/pull/54))
- Fix Inconsistency ([#42](https://github.com/zspacelabs/bunsen/pull/42))
- crutcher/wblocks ([#53](https://github.com/zspacelabs/bunsen/pull/53))
- Add `n_heads` and `n_layers` methods to `TextDecoder` and `AudioEncoder`; refactor `WhisperApiConfig` with
  `PassConfig` for improved structural clarity ([#51](https://github.com/zspacelabs/bunsen/pull/51))
- Improve utility methods in Whisper modules and boost test coverage
  ([#49](https://github.com/zspacelabs/bunsen/pull/49))
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
- Replace `anyhow` with `BunsenError` in validators and bounds modules
  ([#35](https://github.com/zspacelabs/bunsen/pull/35))
- Partial impl of bunsen::data::cache ([#34](https://github.com/zspacelabs/bunsen/pull/34))
