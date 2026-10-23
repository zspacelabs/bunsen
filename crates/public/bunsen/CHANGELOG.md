# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.41.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.40.0...bunsen-v0.41.0) - 2026-10-23

### Other

- release v0.41.0 ([#393](https://github.com/zspacelabs/bunsen/pull/393))

## [0.40.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.39.0...bunsen-v0.40.0) - 2026-10-23

### Added

- *(rekey)* [**breaking**] `Indices`, how `mount_indexed_with` numbers a stack; `Rekey::plan_for` sees the module; ResNet loads through `ResNet::torchvision_rekey` ([#380](https://github.com/zspacelabs/bunsen/pull/380))
- *(rekey)* `Rekey::plan` resolves a whole checkpoint's names; `apply_rekeyed_tensors` refuses a plan that cannot load as written ([#378](https://github.com/zspacelabs/bunsen/pull/378))
- *(conv)* `ConvSeq{1,2}d::one_based_rekey`, a checkpoint's `conv1`, `conv2`, ... onto a sequence's blocks ([#377](https://github.com/zspacelabs/bunsen/pull/377))
- *(attention)* `KVCache` holds cross-attention keys and values, shared by a `group`, and `reorder`s its history ([#368](https://github.com/zspacelabs/bunsen/pull/368))
- *(attention)* [**breaking**] `KVCacheStorage`, the KV cache stores its history as its config's policy says ([#365](https://github.com/zspacelabs/bunsen/pull/365))
- *(attention)* [**breaking**] `KVCacheModel`, a model hands out its own `KVCacheConfig` ([#364](https://github.com/zspacelabs/bunsen/pull/364))
- *(blocks)* [**breaking**] `CausalDecoderBlock`, the generic decoder-only layer; NanoChat's and Qwen3's layers are one ([#358](https://github.com/zspacelabs/bunsen/pull/358))
- *(blocks)* [**breaking**] layout orders are block config, and a block's key map permutes a weight to match ([#357](https://github.com/zspacelabs/bunsen/pull/357))
- *(blocks)* [**breaking**] `SubsampledEncoder`, a front end then an encoder stack, with `AxisOrder` at its boundary; Sortformer's encoder is one ([#354](https://github.com/zspacelabs/bunsen/pull/354))
- *(blocks)* [**breaking**] `EncoderTransformer`, any encoder stack chosen by config; the Conformer's stack is its own type ([#352](https://github.com/zspacelabs/bunsen/pull/352))
- *(bunsen-asr)* [**breaking**] profiles, speech to text by what it is for; the families under `family` ([#344](https://github.com/zspacelabs/bunsen/pull/344))
- *(sortformer)* `hf:` repos that hold a `.nemo`, their config read first ([#339](https://github.com/zspacelabs/bunsen/pull/339))
- *(sortformer)* the streaming Sortformer v2 and v2.1 as well-known rows ([#338](https://github.com/zspacelabs/bunsen/pull/338))
- *(sortformer)* a Sortformer `.nemo`'s config, as bunsen's ([#336](https://github.com/zspacelabs/bunsen/pull/336))
- *(sortformer)* streaming presets by family and latency ([#335](https://github.com/zspacelabs/bunsen/pull/335))
- *(sortformer)* stream Sortformer v2, held to NeMo's sync streaming ([#333](https://github.com/zspacelabs/bunsen/pull/333))
- *(sortformer)* the Sortformer v2 model, held to NeMo on a toy ([#332](https://github.com/zspacelabs/bunsen/pull/332))
- *(blocks)* the residual encoder block's norm is an optional NormalizationConfig ([#331](https://github.com/zspacelabs/bunsen/pull/331))
- *(speech)* read `.nemo` checkpoints, their configs, and NeMo's names ([#329](https://github.com/zspacelabs/bunsen/pull/329))
- *(blocks)* post-norm and a configurable activation in the residual encoder block ([#327](https://github.com/zspacelabs/bunsen/pull/327))
- *(blocks)* ConformerEncoder splits at its subsampling ([#326](https://github.com/zspacelabs/bunsen/pull/326))
- *(speech)* engines cut windows at speaker turns; speech scoring in kits::speech::testing ([#318](https://github.com/zspacelabs/bunsen/pull/318))
- *(sortformer)* pretrained Nemotron-3-Diarization, the bundle, and the diarizer ([#315](https://github.com/zspacelabs/bunsen/pull/315))
- *(sortformer)* stream a Sortformer through the Arrival-Order Speaker Cache ([#314](https://github.com/zspacelabs/bunsen/pull/314))
- *(sortformer)* the Sortformer kit's model, held to transformers' Nemotron-3-Diarization ([#313](https://github.com/zspacelabs/bunsen/pull/313))
- *(blocks)* `blocks::frame_rate`, FeatureStacking and a sub-pixel upsample ([#312](https://github.com/zspacelabs/bunsen/pull/312))
- *(blocks)* bidirectional rotary self-attention and a pre-norm RoPE encoder ([#311](https://github.com/zspacelabs/bunsen/pull/311))
- *(speech)* `kits::speech::nemo`, NeMo's mel front end, shared by Parakeet and Sortformer ([#310](https://github.com/zspacelabs/bunsen/pull/310))
- *(speech)* attach a diarizer to the ASR stream driver ([#308](https://github.com/zspacelabs/bunsen/pull/308))
- *(speech)* diarization's model-independent parts: activity, turns, RTTM, DER ([#306](https://github.com/zspacelabs/bunsen/pull/306))
- *(speech)* add `ToneConfig` for test tone generation and refactor usage across engines ([#304](https://github.com/zspacelabs/bunsen/pull/304))

### Fixed

- *(rekey)* `mount_indexed` claims only a decimal index; `Rekey::pass_rest`, a map that claims only what its rules claim ([#376](https://github.com/zspacelabs/bunsen/pull/376))
- *(asr)* exit with the device idle: `burner::device::DeviceSession`, opened through `bunsen-app` ([#363](https://github.com/zspacelabs/bunsen/pull/363))
- *(descriptors)* unsigned dtypes are Int tensors in TensorKindDesc::from ([#316](https://github.com/zspacelabs/bunsen/pull/316))

### Other

- *(transformers)* [**breaking**] the remaining layers take `Normalization`; `norm_self_attn` and friends take any norm ([#390](https://github.com/zspacelabs/bunsen/pull/390))
- *(parakeet)* streamed features match the one-shot front end within `Tolerance::permissive` ([#375](https://github.com/zspacelabs/bunsen/pull/375))
- *(whisper)* [**breaking**] Whisper's `transformers` loader is a `Rekey` over its blocks' maps ([#373](https://github.com/zspacelabs/bunsen/pull/373))
- *(silero)* [**breaking**] Silero's upstream safetensors loader is a `Rekey`; `FusedLstm::torch_cell_rekey` ([#372](https://github.com/zspacelabs/bunsen/pull/372))
- *(qwen3)* the Qwen3 model's blocks move into `qwen3::blocks` ([#371](https://github.com/zspacelabs/bunsen/pull/371))
- *(blocks)* [**breaking**] `decoders::cross::CrossDecoderBlock`, the encoder-decoder layer; Whisper's `ResidualDecoderAttentionBlock` retires ([#370](https://github.com/zspacelabs/bunsen/pull/370))
- *(whisper)* [**breaking**] Whisper decodes through the shared `KVCache`; `TextDecoderCache` and `AttnKvPair` retire ([#369](https://github.com/zspacelabs/bunsen/pull/369))
- *(kits)* [**breaking**] a family's checkpoint names map in its kit; blocks keep the shared key-map machinery ([#362](https://github.com/zspacelabs/bunsen/pull/362))
- *(speech)* [**breaking**] Qwen3-ASR's loader is a `Rekey`; the residual encoder block owns its `transformers` names ([#360](https://github.com/zspacelabs/bunsen/pull/360))
- *(blocks)* [**breaking**] the encoder and decoder layers name their fields by role ([#359](https://github.com/zspacelabs/bunsen/pull/359))
- *(blocks)* [**breaking**] `blocks::frame_rate` is `blocks::signals`, home of the encoder front ends ([#356](https://github.com/zspacelabs/bunsen/pull/356))
- *(blocks)* [**breaking**] the FastConformer is a policy for a `SubsampledEncoder`; `ConformerEncoder` is no longer a type ([#355](https://github.com/zspacelabs/bunsen/pull/355))
- *(blocks)* [**breaking**] `Rekey`, checkpoint names onto module paths by composed rules; blocks own their maps ([#351](https://github.com/zspacelabs/bunsen/pull/351))
- *(blocks)* [**breaking**] Sortformer's transformer is the generic `ResidualEncoderTransformer` ([#350](https://github.com/zspacelabs/bunsen/pull/350))
- *(blocks)* [**breaking**] the Conformer is a transformer encoder: `blocks::conformer` moves to `blocks::transformers::encoders::conformer` ([#349](https://github.com/zspacelabs/bunsen/pull/349))
- *(speech)* [**breaking**] no build fetches by default; the bundled-weight checks move to the validation crates ([#345](https://github.com/zspacelabs/bunsen/pull/345))
- *(blocks)* attention without a key bias through `with_key_bias(false)` ([#342](https://github.com/zspacelabs/bunsen/pull/342))
- *(speech)* the Sortformer v2 rows across the kits index, the book, and the dev crates ([#341](https://github.com/zspacelabs/bunsen/pull/341))
- *(sortformer)* Sortformer v2 and v2.1 diarize AMI as NeMo does ([#340](https://github.com/zspacelabs/bunsen/pull/340))
- *(sortformer)* a Sortformer v2.1 ONNX reference, held to onnxruntime ([#330](https://github.com/zspacelabs/bunsen/pull/330))
- *(blocks)* [**breaking**] encoder layers under `blocks::transformers::encoders` ([#325](https://github.com/zspacelabs/bunsen/pull/325))
- *(burner)* bf16 rounding lives in burner::tensor ([#320](https://github.com/zspacelabs/bunsen/pull/320))
- collect the test audio into the bunsen-test-audio dev crate ([#305](https://github.com/zspacelabs/bunsen/pull/305))
- *(logging)* replace `stderrlog` with `env_logger`, add custom logging policies ([#303](https://github.com/zspacelabs/bunsen/pull/303))

## [0.39.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.38.0...bunsen-v0.39.0) - 2026-10-17

### Added

- [**breaking**] whisper-cli is retired; bunsen-asr covers it ([#295](https://github.com/zspacelabs/bunsen/pull/295))
- [**breaking**] publish bunsen-app and bunsen-asr ([#291](https://github.com/zspacelabs/bunsen/pull/291))
- *(pretrained)* list and remove what a pretrained cache holds ([#287](https://github.com/zspacelabs/bunsen/pull/287))
- *(qwen3_asr)* a full Qwen3-ASR window is cut at its quietest stretch before decoding ([#283](https://github.com/zspacelabs/bunsen/pull/283))
- *(qwen3_asr)* a reply's attention onto its audio, and the heads that time words ([#282](https://github.com/zspacelabs/bunsen/pull/282))
- *(asr_driver)* Parakeet commits a segment per word; Qwen3-ASR's window is settable ([#281](https://github.com/zspacelabs/bunsen/pull/281))
- *(parakeet)* Parakeet streams through the asr_driver core ([#279](https://github.com/zspacelabs/bunsen/pull/279))
- *(qwen3_asr)* Qwen3-ASR streams through the asr_driver core ([#278](https://github.com/zspacelabs/bunsen/pull/278))
- *(asr_driver)* the stream driver core and the engine trait ([#275](https://github.com/zspacelabs/bunsen/pull/275))
- *(asr_driver)* a level policy shared by every model's front end ([#274](https://github.com/zspacelabs/bunsen/pull/274))
- *(contracts)* shape contracts on the Conformer, Parakeet and Qwen3 bodies ([#269](https://github.com/zspacelabs/bunsen/pull/269))
- *(bunsen-asr)* one speech-to-text CLI, a subcommand per model family ([#267](https://github.com/zspacelabs/bunsen/pull/267))
- *(asr)* `hf:` loaders for Parakeet and Qwen3-ASR, geometry from config.json ([#266](https://github.com/zspacelabs/bunsen/pull/266))
- *(silero-vad)* load upstream's silero_vad_16k.safetensors by name ([#264](https://github.com/zspacelabs/bunsen/pull/264))
- *(qwen3-asr)* pretrained Phonon-1 and Qwen3-ASR-0.6B, held to transformers ([#260](https://github.com/zspacelabs/bunsen/pull/260))
- *(qwen3-asr)* the prompt, transcript parsing and Qwen3AsrBundle ([#259](https://github.com/zspacelabs/bunsen/pull/259))
- *(qwen3-asr)* the Qwen3 decoder and the Qwen3-ASR model ([#257](https://github.com/zspacelabs/bunsen/pull/257))
- *(blocks)* share attention, encoder and feed-forward blocks across kits ([#256](https://github.com/zspacelabs/bunsen/pull/256))
- *(pretrained)* shared readers for FermionResearch's formats and MLX affine tensors ([#255](https://github.com/zspacelabs/bunsen/pull/255))
- *(qwen3-asr)* ONNX reference for Qwen3-ASR, held to onnxruntime ([#254](https://github.com/zspacelabs/bunsen/pull/254))
- *(phonon-cli)* `phonon`, speech to text with Parakeet TDT models ([#252](https://github.com/zspacelabs/bunsen/pull/252))
- *(parakeet)* Phonon-2 and parakeet-tdt-0.6b-v3 pretrained, held to their references ([#251](https://github.com/zspacelabs/bunsen/pull/251))
- *(parakeet)* greedy TDT decoding and the piece vocabulary ([#250](https://github.com/zspacelabs/bunsen/pull/250))
- *(parakeet)* the Parakeet feature extractor ([#249](https://github.com/zspacelabs/bunsen/pull/249))
- *(parakeet)* FastConformer blocks and the Parakeet TDT model ([#248](https://github.com/zspacelabs/bunsen/pull/248))

### Other

- *(lstm)* remove ExtLstmState and related utilities, pending upstream adoption ([#297](https://github.com/zspacelabs/bunsen/pull/297))
- *(asr_driver)* Whisper's streaming log-mels are asr_driver::LogMelFeatures ([#277](https://github.com/zspacelabs/bunsen/pull/277))
- *(whisper)* [**breaking**] the Whisper stream driver runs on the asr_driver core ([#276](https://github.com/zspacelabs/bunsen/pull/276))
- *(speech)* [**breaking**] lift the model-independent driver parts into kits::speech::asr_driver ([#272](https://github.com/zspacelabs/bunsen/pull/272))
- *(qwen3-asr)* the audio encoder's convolution stem is a ConvSeq2d ([#270](https://github.com/zspacelabs/bunsen/pull/270))
- *(silero-vad)* [**breaking**] ship no Silero weights; upstream's safetensors is the model ([#265](https://github.com/zspacelabs/bunsen/pull/265))
- *(fetch)* one fetch implementation for the cache and the build scripts ([#263](https://github.com/zspacelabs/bunsen/pull/263))
- *(kits)* list the Qwen3 and Qwen3-ASR kits in the kit index and the anatomy table ([#262](https://github.com/zspacelabs/bunsen/pull/262))
- *(blocks)* promote shared attention, mask and long-audio pieces ([#258](https://github.com/zspacelabs/bunsen/pull/258))
- *(whisper)* keep the stream's log-mels in a frame ring ([#246](https://github.com/zspacelabs/bunsen/pull/246))
- Intro Blurb ([#244](https://github.com/zspacelabs/bunsen/pull/244))

## [0.38.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.37.0...bunsen-v0.38.0) - 2026-10-07

### Added

- [**breaking**] migrate to burn 0.22 ([#239](https://github.com/zspacelabs/bunsen/pull/239))

## [0.37.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.36.1...bunsen-v0.37.0) - 2026-10-06

### Added

- *(errors)* [**breaking**] kinds, context frames and typed causes for BunsenError ([#238](https://github.com/zspacelabs/bunsen/pull/238))
- [**breaking**] reconcile the book and rustdoc; restructure modules; fix defects ([#237](https://github.com/zspacelabs/bunsen/pull/237))
- *(audit)* [**breaking**] move audit probes to `bunsen::audit`; add audit streams and baselines ([#236](https://github.com/zspacelabs/bunsen/pull/236))

## [0.36.1](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.36.0...bunsen-v0.36.1) - 2026-09-25

### Added

- *(dyn_tensor)* add `downcast_ref` and `unwrap` methods for enhanced tensor conversion options ([#234](https://github.com/zspacelabs/bunsen/pull/234))

### Other

- *(dyn_tensor)* change `to_data` to borrow `self` instead of taking ownership ([#233](https://github.com/zspacelabs/bunsen/pull/233))

## [0.36.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.35.1...bunsen-v0.36.0) - 2026-09-24

### Other

- Refactor and enhance tensor descriptor functionality ([#231](https://github.com/zspacelabs/bunsen/pull/231))

## [0.35.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.34.2...bunsen-v0.35.0) - 2026-09-22

### Other

- Fix LBM mass correction and ensure state validity ([#225](https://github.com/zspacelabs/bunsen/pull/225))

## [0.34.2](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.34.1...bunsen-v0.34.2) - 2026-09-22

### Other

- improve bunsen readme ([#223](https://github.com/zspacelabs/bunsen/pull/223))

## [0.34.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.33.0...bunsen-v0.34.0) - 2026-09-22

### Other

- Resource maps, the pretrained cache and the construction hook: a pretrained is as many files as it is ([#217](https://github.com/zspacelabs/bunsen/pull/217))

## [0.33.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.32.0...bunsen-v0.33.0) - 2026-09-21

### Added

- *(bunsen)* add `Display` impl for `LocationDesc`, enhance `AuditProbeEventParams`, and improve miscomparison error reporting ([#203](https://github.com/zspacelabs/bunsen/pull/203))
- *(tensor)* add `copy_slice` method to `TensorOpExt` and refactor Conway's wrapping logic to use it ([#191](https://github.com/zspacelabs/bunsen/pull/191))
- *(vulkan, update)* vulkan + cargo update ([#188](https://github.com/zspacelabs/bunsen/pull/188))
- *(bunsen)* introduce dtype inspection and boundary casting for Whi… ([#186](https://github.com/zspacelabs/bunsen/pull/186))
- *(TensorDataTestStream)* initial work on TensorData record/verifier stream. ([#184](https://github.com/zspacelabs/bunsen/pull/184))

### Fixed

- *(bunsen)* format tensor name in `DynTensorEnv::expect_*` panics ([#206](https://github.com/zspacelabs/bunsen/pull/206))
- *(bunsen)* return errors instead of panicking in DynTensor::slice_assign* ([#215](https://github.com/zspacelabs/bunsen/pull/215))
- *(bunsen)* convert test read-outs to f64 on the host, not the device ([#208](https://github.com/zspacelabs/bunsen/pull/208))

### Other

- Bundled Whisper loading retired, the bundle crate fetches nothing by default, and the prefab-map small adds ([#213](https://github.com/zspacelabs/bunsen/pull/213))
- Pretrained providers, the weights cache, the name-to-model pathway, and the Whisper vocabularies as sources ([#212](https://github.com/zspacelabs/bunsen/pull/212))
- Disk cache transfers, shard sets, and the pretrained descriptor lift ([#211](https://github.com/zspacelabs/bunsen/pull/211))
- *(trailing_comma)* clean up some trailing commas. ([#210](https://github.com/zspacelabs/bunsen/pull/210))
- Implement GPU test serialization and memory pool cleanup ([#204](https://github.com/zspacelabs/bunsen/pull/204))
- *(bunsen)* consolidate tensor comparison APIs, update `AuditProbeEventParams`, and enhance approximate equality handling ([#202](https://github.com/zspacelabs/bunsen/pull/202))
- *(bunsen)* rename variables and methods for consistency in Whisper driver and context ([#201](https://github.com/zspacelabs/bunsen/pull/201))
- *(cargo)* update dependencies and align with the latest versions of the `arrow`, `parquet`, `symphonia`, `clap`, and other related crates ([#200](https://github.com/zspacelabs/bunsen/pull/200))
- Refactor testing utilities into modular components ([#197](https://github.com/zspacelabs/bunsen/pull/197))
- Refactor tests: replace `Default::default()` with `default_device()` or `backend_device()` across multiple modules for device initialization consistency. ([#196](https://github.com/zspacelabs/bunsen/pull/196))
- Refactor audit stream handling and modularize event logic ([#195](https://github.com/zspacelabs/bunsen/pull/195))
- *(conway)* unify 2D and 3D wrapping logic for toroidal simulations by introducing `project_wrapped_toroidal_boarders` utility ([#193](https://github.com/zspacelabs/bunsen/pull/193))
- *(cargo)* update dependencies in `Cargo.lock` to latest versions. ([#190](https://github.com/zspacelabs/bunsen/pull/190))
- iterate on stream design. ([#187](https://github.com/zspacelabs/bunsen/pull/187))
- Refactor and unify Conway examples ([#189](https://github.com/zspacelabs/bunsen/pull/189))
- crutcher/ts ([#185](https://github.com/zspacelabs/bunsen/pull/185))

## [0.32.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.31.0...bunsen-v0.32.0) - 2026-09-07

### Added

- *(cli)* modularize `whisper-cli`, add `clap-common` crate, and introduce `transcribe` command
  ([#182](https://github.com/zspacelabs/bunsen/pull/182))
- *(bunsen)* enhance `DynTensorEnv` with deep copy support and improved tests
  ([#180](https://github.com/zspacelabs/bunsen/pull/180))
- *(bunsen)* introduce `DynTensorEnv` for dynamic tensor binding and management
  ([#179](https://github.com/zspacelabs/bunsen/pull/179))

### Other

- crutcher/dte iter ([#181](https://github.com/zspacelabs/bunsen/pull/181))
- crutcher/dyn tensor ([#177](https://github.com/zspacelabs/bunsen/pull/177))

## [0.31.0](https://github.com/zspacelabs/bunsen/compare/bunsen-v0.30.1...bunsen-v0.31.0) - 2026-09-03

### Added

- *(whisper)* the stream driver, all eleven phases, and whisper-cli
  ([#167](https://github.com/zspacelabs/bunsen/pull/167))
- [**breaking**] Whisper reference parity, and split bundled assets from model validation
  ([#163](https://github.com/zspacelabs/bunsen/pull/163))

### Other

- Refactor beam search decoder: rename variables for clarity, consolidate tail clipping, and optimize test structure.
  Update device handling to `PerformanceBackend`. ([#173](https://github.com/zspacelabs/bunsen/pull/173))
- Modularize LSTM components by relocating and reorganizing `fused_lstm` implementation. Integrate `fused_lstm` into
  core RNN module and refactor dependencies in `silero_vad` accordingly.
  ([#172](https://github.com/zspacelabs/bunsen/pull/172))
- More whisper refactoring. ([#171](https://github.com/zspacelabs/bunsen/pull/171))
- More whisper cleanup. ([#170](https://github.com/zspacelabs/bunsen/pull/170))
- Crutcher/whisper rf ([#169](https://github.com/zspacelabs/bunsen/pull/169))
- VAD filter cleanup, driver layout, and model-owned front end + token layout
  ([#168](https://github.com/zspacelabs/bunsen/pull/168))
- Add `window_padded` operation for zero-padded dimensional windowing with accompanying tests. Extend test coverage for
  `split_padded` and update module documentation to describe new functionality.
  ([#166](https://github.com/zspacelabs/bunsen/pull/166))
- Refactor `MelFilterbank` into `MelFilterbankConfig` for improved configuration and validation. Rewrite tests for API
  updates and streamline related logic for cleaner organization. ([#165](https://github.com/zspacelabs/bunsen/pull/165))
- Switch to `PerformanceBackend` and enhance testing configurations
  ([#164](https://github.com/zspacelabs/bunsen/pull/164))

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
- *(life2d)* optimize tensor updates, improve backend feature handling, and simplify wrap_state_2d logic
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
