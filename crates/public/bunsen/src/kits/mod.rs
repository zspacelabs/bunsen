//! # Kits: complete models
//!
//! A kit is a whole thing you can pick up and use: a model family with its
//! configs, the weights it can load and a program that runs it. The
//! simulations, built the same way, are [`crate::sims`]. [`crate::blocks`] is a
//! library of parts; a kit is what the parts are for. Kits are built on the
//! rest of the crate: [`crate::blocks`] and [`crate::ops`] supply the layers
//! and kernels, [`crate::contracts`] checks tensor shapes, [`crate::burner`]
//! holds the module conventions, and [`crate::data`] fetches, caches and
//! verifies the bytes a kit loads.
//!
//! # The kits
//!
//! | Kit | What it is |
//! |---|---|
//! | [`images::resnet`] | `ResNet` image classifiers, with 14 torchvision and `timm` checkpoints by name. |
//! | [`images::swin::v2`] | Swin Transformer V2 image classifiers: configs and model, no pretrained weights. |
//! | [`language::nanochat`] | karpathy's nanochat GPT, upstream's 2025 and 2026 models: a pretrained index of community checkpoints (`nanochat-students/rl-d20` by default), upstream's own checkpoint files loaded as they are, cached decoding, and the corpus it trains on. |
//! | [`language::qwen3`] | The Qwen3 decoder: a pretrained index of the dense Qwen3 checkpoints up to 8B (`qwen/qwen3-0.6b` by default), and the text model of [`speech::qwen3_asr`]. |
//! | [`speech::whisper`] | `OpenAI`'s Whisper: the model, a pretrained index, decoders, and a stream driver. |
//! | [`speech::parakeet`] | NVIDIA's Parakeet TDT: the model, a pretrained index (`FermionResearch`'s Phonon-2 by default), and greedy decoding. |
//! | [`speech::sortformer`] | NVIDIA's Sortformer speaker diarizers (`Nemotron-3-Diarization`, the streaming Sortformer v2 and v2.1): the model, its streaming through a speaker cache, and a pretrained index of `transformers` and `.nemo` checkpoints. |
//! | [`speech::qwen3_asr`] | Qwen's Qwen3-ASR: the model, a pretrained index (`FermionResearch`'s Phonon-1 by default), the prompt, and greedy decoding. |
//! | [`speech::voxtral`] | Mistral's Voxtral Realtime: the model, a pretrained index (`mistralai/voxtral-mini-4b-realtime-2602` by default), its stream layout and delay, greedy decoding, and a stream driver. |
//! | [`speech::silero_vad`] | The Silero voice-activity detector, with weights a build can bundle. |
//! | [`support`] | Not a model: what kits of more than one family share: [`support::tokens`], the ids-to-text seam, and [`support::decoding`], the search machinery (logit filters, greedy and beam search, sampling, ranking, and the temperature ladder). |
//!
//! The area modules, [`images`], [`language`] and [`speech`], group the
//! kits by domain and say what the kits in each have in common.
//!
//! # Anatomy of a kit
//!
//! No trait makes a module a kit. A kit is a module with a shared shape,
//! and it has the pieces of this list that its domain needs:
//!
//! - **The model and its configs.** A `Module`, and the config that builds it,
//!   in one of the two [config
//!   shapes](crate::burner::module::ModuleInit#two-config-shapes). A Simple
//!   config builds its module directly. A Stacked family has a policy config
//!   (`ResNetContractConfig`, `WhisperApiConfig`) that lowers through
//!   [`ToStructureConfig`](crate::burner::module::ToStructureConfig) to a
//!   structure config, the unrolled tree, which builds the module through
//!   [`ModuleInit`](crate::burner::module::ModuleInit). A narrow `{Name}Meta`
//!   trait reads back what the forms share.
//! - **A prefab map.** Geometries by name, before any bytes exist: a
//!   [`StaticPreFabMap`](crate::data::pretrained::StaticPreFabMap) of policy
//!   configs (`resnet50`, `base.en`). A prefab has no weights.
//! - **A pretrained table and providers.** The rows, named checkpoints, each
//!   naming the prefab it instantiates where the kit keeps prefabs, compiled
//!   into a table behind the `well-known:` provider.
//!   `default_{kit}_providers()` lists the kit's providers in search order,
//!   `bundled:` and `hf:` among them where the kit has them.
//! - **A construct config.** `{Kit}ConstructConfig` is the kit's hook, its
//!   [`ConstructInit`](crate::kits::support::construct::ConstructInit): the
//!   cache segment its files land under (`{KIT}_KIT`), and how the loaded files
//!   become its construct, `{Kit}Construct`.
//! - **`default_{kit}_factory()`.** A
//!   [`PretrainedFactory`](crate::kits::support::construct::PretrainedFactory)
//!   over the default providers, behind the construct config. A name to a model
//!   is one call: `default_{kit}_factory()?.load(name, &cache, &device)`. There
//!   is [no registry](support::construct::PretrainedFactory#no-registry): a
//!   caller who wants another provider adds it to the factory.
//! - **The built handle.** What `load` returns: a
//!   [`Loaded`](crate::kits::support::construct::Loaded) whose `handle` is an
//!   `Arc` of the config's `Construct` type, the model or a construct around
//!   it.
//! - **A driver or context.** Optional. The state a stream or a decode carries
//!   is injected, never owned by the model ([the
//!   rule](crate::ops#ops-and-blocks)): the model is immutable, and the caller
//!   creates a context or cache and passes it back in, so one loaded model
//!   serves several streams at once.
//! - **A bundled-asset crate.** Optional. `validation/onnx/bind-{model}-onnx`,
//!   in the validation workspace, unpublished: the checkpoints and reference
//!   graphs its validation crate compares, fetched at build time and laid out
//!   as a pretrained cache, and the reference model generated from the graphs.
//! - **A validation crate.** Optional. A crate under `validation/models`, in
//!   the validation workspace, that compares the kit's outputs with its
//!   upstream reference implementation.
//! - **An example.** A runnable program under `examples/`.
//!
//! The machinery behind the prefab, the rows, the hook and the factory is
//! [`data::pretrained`](crate::data::pretrained): what prefabs, rows and
//! resources are, how a factory dispatches a spec, and how the cache
//! fetches and pins files. A kit contributes tables and a hook.
//!
//! The kits that load weights, piece by piece:
//!
//! | Piece | `ResNet` | Whisper | Silero VAD | Parakeet | Qwen3-ASR | Sortformer | Qwen3 | Gemma | Voxtral | `NanoChat` |
//! |---|---|---|---|---|---|---|---|---|---|---|
//! | Configs | [`ResNetContractConfig`](images::resnet::ResNetContractConfig) -> [`ResNetStructureConfig`](images::resnet::ResNetStructureConfig) | [`WhisperApiConfig`](speech::whisper::WhisperApiConfig) -> [`WhisperStructureConfig`](speech::whisper::WhisperStructureConfig) | [`SileroVadSignalConfig`](speech::silero_vad::SileroVadSignalConfig) -> [`SileroVadStftConfig`](speech::silero_vad::SileroVadStftConfig) -> [`SileroVadStructureConfig`](speech::silero_vad::SileroVadStructureConfig) | [`ParakeetTdtConfig`](speech::parakeet::ParakeetTdtConfig) | [`Qwen3AsrConfig`](speech::qwen3_asr::Qwen3AsrConfig) | [`NemotronDiarizationConfig`](speech::sortformer::NemotronDiarizationConfig) or [`SortformerV2Config`](speech::sortformer::SortformerV2Config) -> [`SortformerStructureConfig`](speech::sortformer::SortformerStructureConfig) | [`Qwen3Config`](language::qwen3::Qwen3Config) | [`GemmaConfig`](language::gemma::GemmaConfig) | [`VoxtralRealtimeConfig`](speech::voxtral::VoxtralRealtimeConfig) | [`NanoChatGptContractConfig`](language::nanochat::NanoChatGptContractConfig) -> [`NanoChatGptStructureConfig`](language::nanochat::NanoChatGptStructureConfig) |
//! | Prefab map | [`RESNET_PREFABS`](images::resnet::pretrained::RESNET_PREFABS) | [`WHISPER_PREFABS`](speech::whisper::pretrained::WHISPER_PREFABS) | none | none | [`QWEN3_ASR_PREFABS`](speech::qwen3_asr::pretrained::QWEN3_ASR_PREFABS) | none | [`QWEN3_PREFABS`](language::qwen3::pretrained::QWEN3_PREFABS) | [`GEMMA_PREFABS`](language::gemma::pretrained::GEMMA_PREFABS) | [`VOXTRAL_PREFABS`](speech::voxtral::pretrained::VOXTRAL_PREFABS) | [`NANOCHAT_PREFABS`](language::nanochat::pretrained::NANOCHAT_PREFABS) |
//! | Table | [`RESNET_CATALOG`](images::resnet::pretrained::RESNET_CATALOG) | [`WHISPER_CATALOG`](speech::whisper::pretrained::WHISPER_CATALOG) and `hf:` | [`SILERO_CATALOG`](speech::silero_vad::pretrained::SILERO_CATALOG) | [`PARAKEET_CATALOG`](speech::parakeet::pretrained::PARAKEET_CATALOG) | [`QWEN3_ASR_CATALOG`](speech::qwen3_asr::pretrained::QWEN3_ASR_CATALOG) | [`SORTFORMER_CATALOG`](speech::sortformer::pretrained::SORTFORMER_CATALOG) and `hf:` | [`QWEN3_CATALOG`](language::qwen3::pretrained::QWEN3_CATALOG) and `hf:` | [`GEMMA_CATALOG`](language::gemma::pretrained::GEMMA_CATALOG) and `hf:` | [`VOXTRAL_CATALOG`](speech::voxtral::pretrained::VOXTRAL_CATALOG) and `hf:` | [`NANOCHAT_CATALOG`](language::nanochat::pretrained::NANOCHAT_CATALOG) |
//! | Providers | [`default_resnet_providers`](images::resnet::pretrained::default_resnet_providers) | [`default_whisper_providers`](speech::whisper::pretrained::default_whisper_providers) | [`default_silero_providers`](speech::silero_vad::pretrained::default_silero_providers) | [`default_parakeet_providers`](speech::parakeet::pretrained::default_parakeet_providers) | [`default_qwen3_asr_providers`](speech::qwen3_asr::pretrained::default_qwen3_asr_providers) | [`default_sortformer_providers`](speech::sortformer::pretrained::default_sortformer_providers) | [`default_qwen3_providers`](language::qwen3::pretrained::default_qwen3_providers) | [`default_gemma_providers`](language::gemma::pretrained::default_gemma_providers) | [`default_voxtral_providers`](speech::voxtral::pretrained::default_voxtral_providers) | [`default_nanochat_providers`](language::nanochat::pretrained::default_nanochat_providers) |
//! | Construct config | [`ResNetConstructConfig`](images::resnet::pretrained::ResNetConstructConfig) | [`WhisperConstructConfig`](speech::whisper::pretrained::WhisperConstructConfig) | [`SileroConstructConfig`](speech::silero_vad::pretrained::SileroConstructConfig) | [`ParakeetConstructConfig`](speech::parakeet::pretrained::ParakeetConstructConfig) | [`Qwen3AsrConstructConfig`](speech::qwen3_asr::pretrained::Qwen3AsrConstructConfig) | [`SortformerConstructConfig`](speech::sortformer::pretrained::SortformerConstructConfig) | [`Qwen3ConstructConfig`](language::qwen3::pretrained::Qwen3ConstructConfig) | [`GemmaConstructConfig`](language::gemma::pretrained::GemmaConstructConfig) | [`VoxtralConstructConfig`](speech::voxtral::pretrained::VoxtralConstructConfig) | [`NanoChatConstructConfig`](language::nanochat::pretrained::NanoChatConstructConfig) |
//! | Factory | [`default_resnet_factory`](images::resnet::pretrained::default_resnet_factory) | [`default_whisper_factory`](speech::whisper::pretrained::default_whisper_factory) | [`default_silero_factory`](speech::silero_vad::pretrained::default_silero_factory) | [`default_parakeet_factory`](speech::parakeet::pretrained::default_parakeet_factory) | [`default_qwen3_asr_factory`](speech::qwen3_asr::pretrained::default_qwen3_asr_factory) | [`default_sortformer_factory`](speech::sortformer::pretrained::default_sortformer_factory) | [`default_qwen3_factory`](language::qwen3::pretrained::default_qwen3_factory) | [`default_gemma_factory`](language::gemma::pretrained::default_gemma_factory) | [`default_voxtral_factory`](speech::voxtral::pretrained::default_voxtral_factory) | [`default_nanochat_factory`](language::nanochat::pretrained::default_nanochat_factory) |
//! | Construct | [`ResNet`](images::resnet::ResNet) | [`WhisperConstruct`](speech::whisper::driver::WhisperConstruct): model, token layout, vocabulary | [`SileroVadCollection`](speech::silero_vad::SileroVadCollection): one model per sample rate | [`ParakeetConstruct`](speech::parakeet::ParakeetConstruct): model, front end, vocabulary | [`Qwen3AsrConstruct`](speech::qwen3_asr::Qwen3AsrConstruct): model, front end, vocabulary, prompt, decoder | [`SortformerConstruct`](speech::sortformer::SortformerConstruct): model, front end | [`Qwen3Construct`](language::qwen3::Qwen3Construct): model, trained length, tokenizer and generation files | [`GemmaConstruct`](language::gemma::GemmaConstruct): model, trained length, tokenizer, generation and template files | [`VoxtralConstruct`](speech::voxtral::VoxtralConstruct): model, front end, vocabulary, layout, decoder | [`NanoChatConstruct`](language::nanochat::NanoChatConstruct): model, generation, meta, tokenizer files |
//! | Driver, context | none | [`WhisperStreamDriver`](speech::whisper::driver::WhisperStreamDriver), [`WhisperStreamContext`](speech::whisper::driver::WhisperStreamContext) | [`SileroVadContext`](speech::silero_vad::SileroVadContext) | [`ParakeetStreamDriver`](speech::parakeet::ParakeetStreamDriver), [`ParakeetStreamContext`](speech::parakeet::ParakeetStreamContext) | [`Qwen3AsrStreamDriver`](speech::qwen3_asr::Qwen3AsrStreamDriver), [`Qwen3AsrStreamContext`](speech::qwen3_asr::Qwen3AsrStreamContext) | [`SortformerDiarizer`](speech::sortformer::SortformerDiarizer), [`SortformerStream`](speech::sortformer::SortformerStream): a diarizer for any ASR stream driver | a [`StackedLm`](support::decoding::StackedLm)'s [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache) through [`decode_cached`](support::decoding::decode_cached); `bunsen_tool::chat`'s `ChatSession` | a [`StackedLm`](support::decoding::StackedLm)'s [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache) through [`decode_cached`](support::decoding::decode_cached) | [`VoxtralStreamDriver`](speech::voxtral::driver::VoxtralStreamDriver), [`VoxtralStreamContext`](speech::voxtral::driver::VoxtralStreamContext) | [`NanoChatDecodeState`](language::nanochat::NanoChatDecodeState) through [`decode_cached`](support::decoding::decode_cached) |
//! | Bundled crate (validation only) | none | `bind-whisper-onnx` | `bind-silero-onnx` | `bind-parakeet-onnx` | `bind-qwen3-asr-onnx` | `bind-sortformer-onnx` | `bind-qwen3-onnx` | none | `bind-voxtral-onnx` | `fetch-nanochat` |
//! | Validation crate | none | `whisper-model-validation` | `silero-model-validation` | `parakeet-model-validation` | `qwen3-asr-model-validation` | `sortformer-model-validation` | `qwen3-model-validation` | `gemma-model-validation` | `voxtral-model-validation` | `nanochat-model-validation` |
//! | Example | `examples/resnet_finetune`, `examples/resnet_tiny` | `bunsen-tool asr` | `bunsen-tool asr`, as the VAD gate | `bunsen-tool asr` | `bunsen-tool asr` | none | `bunsen-tool chat` | `bunsen-tool chat` | `bunsen-tool asr` | `examples/train-chat` |
//!
//! The other kits have a model, its configs, and an example, and no prefabs
//! or weights. Swin is Stacked
//! (`examples/swin_tiny`). `NanoChat` is Stacked too, and adds a table of
//! the corpus it trains on, a [`data::shards`](crate::data::shards) shard
//! set (`examples/train-chat`).
//!
//! Three seams are shared across kits:
//! [`data::pretrained`](crate::data::pretrained) for loading,
//! [`support::tokens::Detokenizer`] for turning ids into text, and
//! [`support::decoding::TokenSearch`] for choosing them.
//!
//! # Stability
//!
//! Kits are added incrementally and their surfaces may evolve faster
//! than the rest of the crate while their respective domains are being
//! filled in. Pin a specific `bunsen` version when you need a stable
//! API.

pub mod images;
pub mod language;
pub mod speech;
pub mod support;
