# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`bunsen-tool` continues `bunsen-asr`, whose speech to text is `bunsen-tool asr`; `bunsen-asr`'s history, to its last
release (0.41.0), is in [its changelog](https://github.com/zspacelabs/bunsen/blob/bunsen-asr-v0.41.0/crates/public/bunsen-asr/CHANGELOG.md).

## [Unreleased]

## [0.42.0](https://github.com/zspacelabs/bunsen/compare/bunsen-tool-v0.41.0...bunsen-tool-v0.42.0) - 2026-11-07

### Added

- *(bunsen)* [**breaking**] one `store` feature reads every checkpoint format ([#647](https://github.com/zspacelabs/bunsen/pull/647))
- *(bunsen-tool)* [**breaking**] `asr` is the `speech` feature, on by default; bunsen-tool and bunsen-app no longer take bunsen's defaults ([#646](https://github.com/zspacelabs/bunsen/pull/646))
- *(tool)* [**breaking**] turn-boundary snapshots, so a hybrid model's chat prefills from the last user turn ([#632](https://github.com/zspacelabs/bunsen/pull/632))
- *(tool)* [**breaking**] Qwen3.5 chat, `bunsen-tool chat family qwen3.5`; a hybrid decode cache rewinds by resetting ([#631](https://github.com/zspacelabs/bunsen/pull/631))
- *(tool)* `HfTokenizer` adds a `tokenizer_config.json`'s added tokens, as `transformers` does ([#618](https://github.com/zspacelabs/bunsen/pull/618))
- *(tool)* `bunsen-tool decide`, typed questions answered by a Laya decision model ([#603](https://github.com/zspacelabs/bunsen/pull/603))
- *(tokens)* [**breaking**] `kits::support::tokens::Tokenizer`, the encoding seam beside `Detokenizer` ([#599](https://github.com/zspacelabs/bunsen/pull/599))
- [**breaking**] the chat template is a preset field; Gemma's family states one for the rows without ([#559](https://github.com/zspacelabs/bunsen/pull/559))
- *(gemma)* the turn's end is probed into Gemma's stop tokens; probes accumulate ([#557](https://github.com/zspacelabs/bunsen/pull/557))
- *(bunsen-tool)* chat with Gemma: `ChatTokenizer`, `HfTokenizer`, and one chat backend for stacked language models ([#555](https://github.com/zspacelabs/bunsen/pull/555))
- [**breaking**] `DecoderCache` and `StackedLm`, the generic cached decode; Qwen3 on them ([#537](https://github.com/zspacelabs/bunsen/pull/537))
- *(bunsen-tool)* [**breaking**] `chat` drops `--precision`, `--float-dtype` and `--int-dtype`; the load policy picks the precision ([#524](https://github.com/zspacelabs/bunsen/pull/524))
- *(bunsen-tool)* `chat --stats` names the model: its lookup expression, on a line of its own ([#523](https://github.com/zspacelabs/bunsen/pull/523))
- [**breaking**] `LoadPolicy`: a load's dtype policy, named by intent at the call site; Qwen3's hook resolves it ([#520](https://github.com/zspacelabs/bunsen/pull/520))
- what a device does with each float dtype: `burner::dtypes` and `bunsen-tool devices` ([#515](https://github.com/zspacelabs/bunsen/pull/515))
- *(bunsen-tool)* a profile command's `--model` runs in the family that lists it ([#507](https://github.com/zspacelabs/bunsen/pull/507))
- *(bunsen-tool)* the listings and the card show each model's presets ([#502](https://github.com/zspacelabs/bunsen/pull/502))
- *(bunsen-tool)* "did you mean?" on a name no table has ([#501](https://github.com/zspacelabs/bunsen/pull/501))
- *(bunsen-tool)* [**breaking**] the policy flags and `--opt` are gone: a model's settings are its expression's ([#500](https://github.com/zspacelabs/bunsen/pull/500))
- *(bunsen-tool)* [**breaking**] the VAD is a slot: `--vad <expr>`, an error where no VAD runs; `VAD_MODEL` gone ([#499](https://github.com/zspacelabs/bunsen/pull/499))
- *(bunsen-tool)* [**breaking**] `--preset` everywhere: a model's presets on every command, and `asr presets` for the profiles ([#497](https://github.com/zspacelabs/bunsen/pull/497))
- *(bunsen-tool)* [**breaking**] `bunsen-tool chat chat` is `chat repl` ([#493](https://github.com/zspacelabs/bunsen/pull/493))
- *(qwen3)* [**breaking**] Qwen3's generation is presets: probed from `generation_config.json`, a `thinking-mode` axis, held to the chat template ([#492](https://github.com/zspacelabs/bunsen/pull/492))
- *(speech)* [**breaking**] Parakeet's, Qwen3-ASR's and Voxtral's stream policies are what their presets resolve ([#490](https://github.com/zspacelabs/bunsen/pull/490))
- *(whisper)* [**breaking**] Whisper's decode policy is presets: a `task` axis on the multilingual checkpoints ([#489](https://github.com/zspacelabs/bunsen/pull/489))
- *(sortformer)* [**breaking**] Sortformer's latencies and post-processing are family presets ([#488](https://github.com/zspacelabs/bunsen/pull/488))
- *(silero)* [**breaking**] Silero's VAD policy is presets: `silero/vad-16k[silero]` and `[faster-whisper]` ([#486](https://github.com/zspacelabs/bunsen/pull/486))
- *(pretrained)* [**breaking**] sizes and dtypes are fields, not prose: the card and listings render them ([#479](https://github.com/zspacelabs/bunsen/pull/479))
- *(qwen3)* `qwen/qwen3-4b-saferl` and `qwen/qwen3guard-gen-{0.6b,4b,8b}` by name ([#463](https://github.com/zspacelabs/bunsen/pull/463))
- *(qwen3)* `qwen/qwen3-8b` and `qwen/qwen3-8b-base` by name, the `qwen3-8b` prefab; held to `transformers` in bf16 on CUDA ([#462](https://github.com/zspacelabs/bunsen/pull/462))
- *(bunsen-tool)* `bunsen-tool card`, the build, its cache and the models it knows, as YAML or JSON; the READMEs carry its models card ([#456](https://github.com/zspacelabs/bunsen/pull/456))
- *(validation)* [**breaking**] Qwen3-ASR by profile; the cache records when a file was last used, and `cache clear --age` ([#453](https://github.com/zspacelabs/bunsen/pull/453))
- *(bunsen-tool)* [**breaking**] `bunsen-tool chat`; bunsen-chat's library is `bunsen_tool::chat` ([#447](https://github.com/zspacelabs/bunsen/pull/447))
- *(bunsen-tool)* [**breaking**] `bunsen-tool`, bunsen's models on the command line; `bunsen-asr` is `bunsen-tool asr`, and its cache `bunsen-tool cache` ([#446](https://github.com/zspacelabs/bunsen/pull/446))

### Fixed

- *(bunsen-tool)* "did you mean" matches a bare name without its group, as the table answers it ([#530](https://github.com/zspacelabs/bunsen/pull/530))
- *(bunsen-tool)* `card --policies` lists a preset set's presets under `modes` ([#512](https://github.com/zspacelabs/bunsen/pull/512))
- *(bunsen-tool)* `card --policies` writes a preset's name before its description ([#511](https://github.com/zspacelabs/bunsen/pull/511))
- *(pretrained)* a lookup expression writes its patch as YAML: `{beam_size: 5, language: de}` ([#506](https://github.com/zspacelabs/bunsen/pull/506))
- *(bunsen-tool)* the card strips nulls and empties from its data ([#505](https://github.com/zspacelabs/bunsen/pull/505))

### Other

- *(bunsen)* [**breaking**] the error core without `std`; maps on `hashbrown` ([#651](https://github.com/zspacelabs/bunsen/pull/651))
- *(bunsen-tool)* the library is the whole tool, one module per command, each with its `cli` ([#644](https://github.com/zspacelabs/bunsen/pull/644))
- *(tool)* [**breaking**] Qwen3's chat tokenizes through `HfTokenizer`; `Qwen3Tokenizer` is gone ([#619](https://github.com/zspacelabs/bunsen/pull/619))
- [**breaking**] `kits::speech` is grouped by task: `asr`, `diarization`, `vad` ([#581](https://github.com/zspacelabs/bunsen/pull/581))
- [**breaking**] `kits::gpts` is `kits::language`, and `driver_support` is `language::driver` ([#580](https://github.com/zspacelabs/bunsen/pull/580))
- *(blocks)* [**breaking**] family-named blocks: `Transformer*` feed-forwards, `DecoderStack*` parts, `EncoderSubsampling`, owner-named orders, `BoundaryPermutes` ([#571](https://github.com/zspacelabs/bunsen/pull/571))
- *(bunsen-tool)* cpal 0.18 ([#528](https://github.com/zspacelabs/bunsen/pull/528))
- *(bunsen-tool)* an unset axis's default is the construct's data ([#513](https://github.com/zspacelabs/bunsen/pull/513))
- *(bunsen-tool)* what a family's engine takes from the command line is the construct's data ([#510](https://github.com/zspacelabs/bunsen/pull/510))
- *(bunsen-tool)* `--language`'s implication is the construct's data ([#509](https://github.com/zspacelabs/bunsen/pull/509))
- *(bunsen-tool)* [**breaking**] the ASR construct's names: `AsrConstructConfig`, `AsrPreset`, `AsrPoint`, `AsrRecipe`, `AsrConstruct` ([#508](https://github.com/zspacelabs/bunsen/pull/508))
- *(bunsen-tool)* the ASR construct's table is data: `asr_construct.yml`, its engines lookup expressions ([#498](https://github.com/zspacelabs/bunsen/pull/498))
- [**breaking**] the emission policy is not a preset: `--emission <offline|conservative|drafting>`, `EmissionMode` ([#495](https://github.com/zspacelabs/bunsen/pull/495))
- *(kits)* [**breaking**] Whisper's pretrained table is its embedded catalog ([#477](https://github.com/zspacelabs/bunsen/pull/477))
- [**breaking**] each kit's in-memory assemblage is its construct: `XBundle` is `XConstruct` ([#469](https://github.com/zspacelabs/bunsen/pull/469))
- [**breaking**] each kit's construct config is its hook: `ConstructInit`, built by `{Kit}ConstructConfig` ([#468](https://github.com/zspacelabs/bunsen/pull/468))
- [**breaking**] the construct side of the pretrained pathway moves to `kits::support::construct` ([#467](https://github.com/zspacelabs/bunsen/pull/467))
- *(kits)* [**breaking**] `kits::support` holds what kits of more than one family share: `tokens` and `decoding` move under it ([#466](https://github.com/zspacelabs/bunsen/pull/466))
- *(pretrained)* [**breaking**] `ResourceMap` is `ResourceBundle`, and what speaks of it is named for it ([#465](https://github.com/zspacelabs/bunsen/pull/465))
