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
//! | [`language::decision::laya`] | Laya, an open decision model: a `ModernBERT` encoder and a head that answers typed questions with calibrated probabilities over their options. |
//! | [`images::classification::resnet`] | `ResNet` image classifiers, with 140 torchvision and `timm` checkpoints by name. |
//! | [`images::classification::swin::v2`] | Swin Transformer V2 image classifiers, the generic staged vision encoder, with 12 `timm` checkpoints by name. |
//! | [`language::generation::nanochat`] | karpathy's nanochat GPT, upstream's 2025 and 2026 models: a pretrained index of community checkpoints (`nanochat-students/rl-d20` by default), upstream's own checkpoint files loaded as they are, cached decoding, and the corpus it trains on. |
//! | [`language::generation::qwen3_5`] | The dense Qwen3.5 decoder, a hybrid of gated `DeltaNet` and gated full-attention layers: a pretrained index of its 0.8B to 27B checkpoints' text models (`qwen/qwen3.5-0.8b` by default), and cached decoding through a KV cache and the linear layers' states. |
//! | [`language::generation::qwen3`] | The Qwen3 decoder: a pretrained index of the dense Qwen3 checkpoints up to 8B (`qwen/qwen3-0.6b` by default), and the text model of [`speech::asr::qwen3_asr`]. |
//! | [`speech::asr::whisper`] | `OpenAI`'s Whisper: the model, a pretrained index, decoders, and a stream driver. |
//! | [`speech::asr::parakeet`] | NVIDIA's Parakeet TDT: the model, a pretrained index (`FermionResearch`'s Phonon-2 by default), and greedy decoding. |
//! | [`speech::diarization::sortformer`] | NVIDIA's Sortformer speaker diarizers (`Nemotron-3-Diarization`, the streaming Sortformer v2 and v2.1): the model, its streaming through a speaker cache, and a pretrained index of `transformers` and `.nemo` checkpoints. |
//! | [`speech::asr::qwen3_asr`] | Qwen's Qwen3-ASR: the model, a pretrained index (`FermionResearch`'s Phonon-1 by default), the prompt, and greedy decoding. |
//! | [`speech::asr::voxtral`] | Mistral's Voxtral Realtime: the model, a pretrained index (`mistralai/voxtral-mini-4b-realtime-2602` by default), its stream layout and delay, greedy decoding, and a stream driver. |
//! | [`speech::vad::silero_vad`] | The Silero voice-activity detector, with weights a build can bundle. |
//! | [`speech::tts::qwen3_tts`] | Qwen's Qwen3-TTS text to speech (the 12 Hz line): a Qwen3 talker over summed text and codec embeddings, its code predictor (a depth decoder), the dual-track prompt and the take loop, the 12 Hz codec's decoder, and a pretrained index (`qwen/qwen3-tts/qwen3-tts-12hz-0.6b-customvoice` by default). |
//! | [`speech::tts::zonos3`] | Zyphra's ZONOS3-Mini text to speech: a language model over rows of audio codec codes and text, without a residual stream, its prompt layout and stock voices, the codec decoder, a sampler and a driver (text and a voice to speech), and a pretrained index (`zyphra/zonos3/zonos3-mini`). |
//! | [`language::encoders::modernbert`] | Answer.AI's `ModernBERT` text encoder: the model, loaded from a `transformers` `ModernBertModel` state dict and its `config.json`. |
//! | [`support`] | Not a model: what kits of more than one family share: [`support::tokens`], the ids-to-text seam, and [`support::decoding`], the search machinery (logit filters, greedy and beam search, sampling, ranking, and the temperature ladder). |
//!
//! The area modules, [`images`], [`language`] and [`speech`], group the kits
//! by domain, and within it by task: a kit's path is `{area}::{task}::{kit}`
//! (`speech::asr::whisper`, `language::generation::qwen3`). Each says what
//! the kits in it have in common.
//!
//! - **A task module holds its kits and the task's model-generic parts:** a
//!   stream driver (`speech::asr::driver`, `language::generation::driver`), a
//!   seam (`speech::diarization`'s diarizer).
//! - **Shared parts sit at the lowest level that covers every user:** what two
//!   tasks of one area share is the area's `support` ([`speech::support`]);
//!   what two areas share is [`support`].
//! - **Dependencies run one way:** a speech kit may build on a language kit
//!   (Qwen3-ASR's text model is `language::generation::qwen3`); a language kit
//!   never depends on a speech one, and the image kits on neither.
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
//! - **Prefabs.** Geometries by name, before any bytes exist: constructors of
//!   the policy config (`resnet50`, `base.en`), listed by
//!   [`Prefabs`](crate::data::pretrained::Prefabs). A prefab has no weights.
//! - **Builds and providers.** The builds of the kit's contract in the built-in
//!   catalog, each stating the geometry it is built in, behind the `catalog:`
//!   provider. `default_{kit}_providers()` lists the kit's providers in search
//!   order, `hf:` among them where the kit has it.
//! - **A catalog contract.** `contract.yml`, beside the providers
//!   (`{KIT}_CONTRACT`): the signature its loader implements, the resource
//!   slots and arguments a build catalog's builds bind
//!   ([`catalog_contracts`](support::pretrained::catalog_contracts)).
//! - **A construct config.** `{Kit}ConstructConfig` is the kit's hook, its
//!   [`ConstructInit`](crate::data::builds::ConstructInit): the kit's name
//!   (`{KIT}_KIT`), and how the loaded files become its construct,
//!   `{Kit}Construct`.
//! - **`default_{kit}_factory()`.** A
//!   [`BuildFactory`](crate::data::builds::BuildFactory) over the default
//!   providers, behind the construct config. A name to a model is one call:
//!   `default_{kit}_factory()?.load(name, &cache, &device)`. There is [no
//!   registry](crate::data::builds::BuildFactory#no-registry): a caller who
//!   wants another provider adds it to the factory.
//! - **The built handle.** What `load` returns: a
//!   [`Loaded`](crate::data::builds::Loaded) whose `handle` is an `Arc` of the
//!   config's `Construct` type, the model or a construct around it.
//! - **A driver or context.** Optional. The state a stream or a decode carries
//!   is injected, never owned by the model ([the
//!   rule](crate::ops#ops-and-blocks)): the model is immutable, and the caller
//!   creates a context or cache and passes it back in, so one loaded model
//!   serves several streams at once.
//! - **A bundled-asset crate.** Optional. `validation/onnx/bind-{model}-onnx`,
//!   in the validation workspace, unpublished: the checkpoints and reference
//!   graphs its validation crate compares, fetched at build time and laid out
//!   as a resource cache, and the reference model generated from the graphs.
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
//! | Piece | `ResNet` | Whisper | Silero VAD | Parakeet | Qwen3-ASR | Sortformer | Qwen3 | Gemma | Qwen3.5 | Voxtral | `NanoChat` | ZONOS3 | Qwen3-TTS |
//! |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
//! | Configs | [`ResNetContractConfig`](images::classification::resnet::ResNetContractConfig) -> [`ConvNetContractConfig`](crate::blocks::images::models::conv_net::ConvNetContractConfig) -> [`ConvNetStructureConfig`](crate::blocks::images::models::conv_net::ConvNetStructureConfig) | [`WhisperApiConfig`](speech::asr::whisper::WhisperApiConfig) -> [`WhisperStructureConfig`](speech::asr::whisper::WhisperStructureConfig) | [`SileroVadSignalConfig`](speech::vad::silero_vad::SileroVadSignalConfig) -> [`SileroVadStftConfig`](speech::vad::silero_vad::SileroVadStftConfig) -> [`SileroVadStructureConfig`](speech::vad::silero_vad::SileroVadStructureConfig) | [`ParakeetTdtConfig`](speech::asr::parakeet::ParakeetTdtConfig) | [`Qwen3AsrConfig`](speech::asr::qwen3_asr::Qwen3AsrConfig) | [`NemotronDiarizationConfig`](speech::diarization::sortformer::NemotronDiarizationConfig) or [`SortformerV2Config`](speech::diarization::sortformer::SortformerV2Config) -> [`SortformerStructureConfig`](speech::diarization::sortformer::SortformerStructureConfig) | [`Qwen3Config`](language::generation::qwen3::Qwen3Config) | [`GemmaConfig`](language::generation::gemma::GemmaConfig) | [`Qwen3_5Config`](language::generation::qwen3_5::Qwen3_5Config) | [`VoxtralRealtimeConfig`](speech::asr::voxtral::VoxtralRealtimeConfig) | [`NanoChatGptContractConfig`](language::generation::nanochat::NanoChatGptContractConfig) -> [`NanoChatGptStructureConfig`](language::generation::nanochat::NanoChatGptStructureConfig) | [`Zonos3LmConfig`](speech::tts::zonos3::blocks::Zonos3LmConfig), [`Zonos3CodecConfig`](speech::tts::zonos3::blocks::Zonos3CodecConfig), from [`Zonos3Mirror`](speech::tts::zonos3::interop::zonos::Zonos3Mirror) | [`Qwen3TtsModelConfig`](speech::tts::qwen3_tts::blocks::Qwen3TtsModelConfig) (talker, code predictor), [`Qwen3TtsCodecDecoderContractConfig`](speech::tts::qwen3_tts::blocks::Qwen3TtsCodecDecoderContractConfig) -> [`Qwen3TtsCodecDecoderStructureConfig`](speech::tts::qwen3_tts::blocks::Qwen3TtsCodecDecoderStructureConfig), from their `config.json`s |
//! | Prefabs | [`ResNetContractConfig`](crate::kits::images::classification::resnet::ResNetContractConfig)'s prefabs | [`WhisperApiConfig`](crate::kits::speech::asr::whisper::WhisperApiConfig)'s prefabs | none | none | [`Qwen3AsrConfig`](crate::kits::speech::asr::qwen3_asr::Qwen3AsrConfig)'s prefabs | none | [`Qwen3Config`](crate::kits::language::generation::qwen3::Qwen3Config)'s prefabs | [`GemmaConfig`](crate::kits::language::generation::gemma::GemmaConfig)'s prefabs | [`Qwen3_5Config`](crate::kits::language::generation::qwen3_5::Qwen3_5Config)'s prefabs | [`VoxtralRealtimeConfig`](crate::kits::speech::asr::voxtral::VoxtralRealtimeConfig)'s prefabs | [`NanoChatGptContractConfig`](crate::kits::language::generation::nanochat::NanoChatGptContractConfig)'s prefabs | none | none |
//! | Contract | [`RESNET_CATALOG_CONTRACT`](images::classification::resnet::pretrained::RESNET_CATALOG_CONTRACT) | [`WHISPER_CATALOG_CONTRACT`](speech::asr::whisper::pretrained::WHISPER_CATALOG_CONTRACT) and `hf:` | [`SILERO_CATALOG_CONTRACT`](speech::vad::silero_vad::pretrained::SILERO_CATALOG_CONTRACT) | [`PARAKEET_CATALOG_CONTRACT`](speech::asr::parakeet::pretrained::PARAKEET_CATALOG_CONTRACT) | [`QWEN3_ASR_CATALOG_CONTRACT`](speech::asr::qwen3_asr::pretrained::QWEN3_ASR_CATALOG_CONTRACT) | [`SORTFORMER_CATALOG_CONTRACT`](speech::diarization::sortformer::pretrained::SORTFORMER_CATALOG_CONTRACT) and `hf:` | [`QWEN3_CATALOG_CONTRACT`](language::generation::qwen3::pretrained::QWEN3_CATALOG_CONTRACT) and `hf:` | [`GEMMA_CATALOG_CONTRACT`](language::generation::gemma::pretrained::GEMMA_CATALOG_CONTRACT) and `hf:` | [`QWEN3_5_CATALOG_CONTRACT`](language::generation::qwen3_5::pretrained::QWEN3_5_CATALOG_CONTRACT) and `hf:` | [`VOXTRAL_CATALOG_CONTRACT`](speech::asr::voxtral::pretrained::VOXTRAL_CATALOG_CONTRACT) and `hf:` | [`NANOCHAT_CATALOG_CONTRACT`](language::generation::nanochat::pretrained::NANOCHAT_CATALOG_CONTRACT) | [`ZONOS3_CATALOG_CONTRACT`](speech::tts::zonos3::pretrained::ZONOS3_CATALOG_CONTRACT) | [`QWEN3_TTS_CATALOG_CONTRACT`](speech::tts::qwen3_tts::pretrained::QWEN3_TTS_CATALOG_CONTRACT) |
//! | Providers | [`default_resnet_providers`](images::classification::resnet::pretrained::default_resnet_providers) | [`default_whisper_providers`](speech::asr::whisper::pretrained::default_whisper_providers) | [`default_silero_providers`](speech::vad::silero_vad::pretrained::default_silero_providers) | [`default_parakeet_providers`](speech::asr::parakeet::pretrained::default_parakeet_providers) | [`default_qwen3_asr_providers`](speech::asr::qwen3_asr::pretrained::default_qwen3_asr_providers) | [`default_sortformer_providers`](speech::diarization::sortformer::pretrained::default_sortformer_providers) | [`default_qwen3_providers`](language::generation::qwen3::pretrained::default_qwen3_providers) | [`default_gemma_providers`](language::generation::gemma::pretrained::default_gemma_providers) | [`default_qwen3_5_providers`](language::generation::qwen3_5::pretrained::default_qwen3_5_providers) | [`default_voxtral_providers`](speech::asr::voxtral::pretrained::default_voxtral_providers) | [`default_nanochat_providers`](language::generation::nanochat::pretrained::default_nanochat_providers) | [`default_zonos3_providers`](speech::tts::zonos3::pretrained::default_zonos3_providers) | [`default_qwen3_tts_providers`](speech::tts::qwen3_tts::pretrained::default_qwen3_tts_providers) |
//! | Construct config | [`ResNetConstructConfig`](images::classification::resnet::pretrained::ResNetConstructConfig) | [`WhisperConstructConfig`](speech::asr::whisper::pretrained::WhisperConstructConfig) | [`SileroConstructConfig`](speech::vad::silero_vad::pretrained::SileroConstructConfig) | [`ParakeetConstructConfig`](speech::asr::parakeet::pretrained::ParakeetConstructConfig) | [`Qwen3AsrConstructConfig`](speech::asr::qwen3_asr::pretrained::Qwen3AsrConstructConfig) | [`SortformerConstructConfig`](speech::diarization::sortformer::pretrained::SortformerConstructConfig) | [`Qwen3ConstructConfig`](language::generation::qwen3::pretrained::Qwen3ConstructConfig) | [`GemmaConstructConfig`](language::generation::gemma::pretrained::GemmaConstructConfig) | [`Qwen3_5ConstructConfig`](language::generation::qwen3_5::pretrained::Qwen3_5ConstructConfig) | [`VoxtralConstructConfig`](speech::asr::voxtral::pretrained::VoxtralConstructConfig) | [`NanoChatConstructConfig`](language::generation::nanochat::pretrained::NanoChatConstructConfig) | [`Zonos3ConstructConfig`](speech::tts::zonos3::pretrained::Zonos3ConstructConfig) | [`Qwen3TtsConstructConfig`](speech::tts::qwen3_tts::pretrained::Qwen3TtsConstructConfig) |
//! | Factory | [`default_resnet_factory`](images::classification::resnet::pretrained::default_resnet_factory) | [`default_whisper_factory`](speech::asr::whisper::pretrained::default_whisper_factory) | [`default_silero_factory`](speech::vad::silero_vad::pretrained::default_silero_factory) | [`default_parakeet_factory`](speech::asr::parakeet::pretrained::default_parakeet_factory) | [`default_qwen3_asr_factory`](speech::asr::qwen3_asr::pretrained::default_qwen3_asr_factory) | [`default_sortformer_factory`](speech::diarization::sortformer::pretrained::default_sortformer_factory) | [`default_qwen3_factory`](language::generation::qwen3::pretrained::default_qwen3_factory) | [`default_gemma_factory`](language::generation::gemma::pretrained::default_gemma_factory) | [`default_qwen3_5_factory`](language::generation::qwen3_5::pretrained::default_qwen3_5_factory) | [`default_voxtral_factory`](speech::asr::voxtral::pretrained::default_voxtral_factory) | [`default_nanochat_factory`](language::generation::nanochat::pretrained::default_nanochat_factory) | [`default_zonos3_factory`](speech::tts::zonos3::pretrained::default_zonos3_factory) | [`default_qwen3_tts_factory`](speech::tts::qwen3_tts::pretrained::default_qwen3_tts_factory) |
//! | Construct | [`ImageModelConstruct`](images::support::card::ImageModelConstruct): model, eval transform, labels | [`WhisperConstruct`](speech::asr::whisper::driver::WhisperConstruct): model, token layout, vocabulary | [`SileroVadCollection`](speech::vad::silero_vad::SileroVadCollection): one model per sample rate | [`ParakeetConstruct`](speech::asr::parakeet::ParakeetConstruct): model, front end, vocabulary | [`Qwen3AsrConstruct`](speech::asr::qwen3_asr::Qwen3AsrConstruct): model, front end, vocabulary, prompt, decoder | [`SortformerConstruct`](speech::diarization::sortformer::SortformerConstruct): model, front end | [`Qwen3Construct`](language::generation::qwen3::Qwen3Construct): model, trained length, tokenizer and generation files | [`GemmaConstruct`](language::generation::gemma::GemmaConstruct): model, trained length, tokenizer, generation and template files | [`Qwen3_5Construct`](language::generation::qwen3_5::Qwen3_5Construct): model, trained length, tokenizer and template files | [`VoxtralConstruct`](speech::asr::voxtral::VoxtralConstruct): model, front end, vocabulary, layout, decoder | [`NanoChatConstruct`](language::generation::nanochat::NanoChatConstruct): model, generation, meta, tokenizer files | [`Zonos3Construct`](speech::tts::zonos3::Zonos3Construct): language model, codec, prompt layout, generation settings, voices, tokenizer file | [`Qwen3TtsConstruct`](speech::tts::qwen3_tts::Qwen3TtsConstruct): model, codec, prompt ids, take settings, tokenizer files |
//! | Driver, context | none | [`WhisperStreamDriver`](speech::asr::whisper::driver::WhisperStreamDriver), [`WhisperStreamContext`](speech::asr::whisper::driver::WhisperStreamContext) | [`SileroVadContext`](speech::vad::silero_vad::SileroVadContext) | [`ParakeetStreamDriver`](speech::asr::parakeet::ParakeetStreamDriver), [`ParakeetStreamContext`](speech::asr::parakeet::ParakeetStreamContext) | [`Qwen3AsrStreamDriver`](speech::asr::qwen3_asr::Qwen3AsrStreamDriver), [`Qwen3AsrStreamContext`](speech::asr::qwen3_asr::Qwen3AsrStreamContext) | [`SortformerDiarizer`](speech::diarization::sortformer::SortformerDiarizer), [`SortformerStream`](speech::diarization::sortformer::SortformerStream): a diarizer for any ASR stream driver | a [`StackedLm`](support::decoding::StackedLm)'s [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache) through [`decode_cached`](support::decoding::decode_cached); `bunsen_tool::chat`'s `ChatSession` | a [`StackedLm`](support::decoding::StackedLm)'s [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache) through [`decode_cached`](support::decoding::decode_cached) | a [`StackedLm`](support::decoding::StackedLm)'s [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache), its KV cache and its linear layers' states, through [`decode_cached`](support::decoding::decode_cached); `bunsen_tool::chat`'s `ChatSession` | [`VoxtralStreamDriver`](speech::asr::voxtral::driver::VoxtralStreamDriver), [`VoxtralStreamContext`](speech::asr::voxtral::driver::VoxtralStreamContext) | [`NanoChatDecodeState`](language::generation::nanochat::NanoChatDecodeState) through [`decode_cached`](support::decoding::decode_cached) | [`Zonos3Driver`](speech::tts::zonos3::driver::Zonos3Driver), the [TTS driver](speech::tts::driver) over [`Zonos3Engine`](speech::tts::zonos3::driver::Zonos3Engine): a take through the language model's [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache), sampled by [`Zonos3Sampler`](speech::tts::zonos3::sampler::Zonos3Sampler), decoded by the codec; `bunsen_tool::tts` | [`Qwen3TtsDriver`](speech::tts::qwen3_tts::driver::Qwen3TtsDriver), the [TTS driver](speech::tts::driver) over [`Qwen3TtsEngine`](speech::tts::qwen3_tts::driver::Qwen3TtsEngine): a take through [`generate_codes`](speech::tts::qwen3_tts::blocks::Qwen3TtsModel::generate_codes), the talker's KV cache and a fresh one per frame for the code predictor |
//! | Bundled crate (validation only) | none | `bind-whisper-onnx` | `bind-silero-onnx` | `bind-parakeet-onnx` | `bind-qwen3-asr-onnx` | `bind-sortformer-onnx` | `bind-qwen3-onnx` | none | none | `bind-voxtral-onnx` | `fetch-nanochat` | none | none |
//! | Validation crate | none | `whisper-model-validation` | `silero-model-validation` | `parakeet-model-validation` | `qwen3-asr-model-validation` | `sortformer-model-validation` | `qwen3-model-validation` | `gemma-model-validation` | `qwen3-5-model-validation` | `voxtral-model-validation` | `nanochat-model-validation` | `zonos3-model-validation` | `qwen3-tts-model-validation` |
//! | Example | `examples/resnet_finetune`, `examples/resnet_tiny` | `bunsen-tool asr` | `bunsen-tool asr`, as the VAD gate | `bunsen-tool asr` | `bunsen-tool asr` | none | `bunsen-tool chat` | `bunsen-tool chat` | `bunsen-tool chat` | `bunsen-tool asr` | `examples/train-chat` | none | none |
//!
//! The other kits have a model, its configs, and an example, and no prefabs
//! or weights. Swin V2 is Stacked, and trains in `examples/swin_tiny`; it
//! has timm's weights too, but no table of its own here, as `ViT` and `DeiT`
//! have none. `NanoChat` is Stacked too, and adds a table of
//! the corpus it trains on, a [`data::datasets`](crate::data::datasets) shard
//! set (`examples/train-chat`).
//!
//! Three seams are shared across kits:
//! [`data::pretrained`](crate::data::pretrained) for loading,
//! [`support::tokens::Detokenizer`] for turning ids into text, and
//! [`support::decoding::TokenSearch`] for choosing them.
//!
//! # Design Rules
//!
//! Also binding here: the config lifecycle
//! ([`burner::module`'s Design Rules](crate::burner::module#design-rules)) and
//! the crate root's [names and layers](crate#design-rules).
//! [Anatomy of a kit](#anatomy-of-a-kit) is the reference for each piece
//! named below.
//!
//! - **A kit that loads weights has the named pieces:** its configs, in one of
//!   the two shapes; `{KIT}_KIT`; a `contract.yml` beside its providers
//!   (`{KIT}_CATALOG_CONTRACT`); `default_{kit}_providers()`; a
//!   `{Kit}ConstructConfig` hook that builds a `{Kit}Construct`; and
//!   `default_{kit}_factory()`. There is no registry. A name to a model is one
//!   call, and the catalog builder and tools find every kit the same way
//!   ([`images::classification::resnet::pretrained`],
//!   [`language::generation::qwen3::pretrained`]).
//! - **Kits group by domain, then by task where a domain's kits serve different
//!   tasks**: `speech::{asr, diarization, tts, vad}`. A task's
//!   model-independent parts sit beside its models ([`speech::asr::driver`]); a
//!   family layer that crosses tasks ([`speech::support::nemo`]) and the
//!   domain's `testing` sit beside the tasks. [`support`] holds what kits of
//!   more than one domain share, and simulations are [`crate::sims`], not kits.
//!   A kit's module is named for its model, as close to the model's own name as
//!   a Rust identifier allows, even when that repeats its group
//!   ([`speech::asr::qwen3_asr`]). An op that only one kit uses lives in that
//!   kit's `ops`. A kit is found by what it is, and kit-only code stays out of
//!   the shared layers.
//! - **A loaded model is immutable.** Decode and stream state is a context or a
//!   cache that the caller creates and passes back in
//!   ([`WhisperStreamContext`](speech::asr::whisper::driver::WhisperStreamContext)),
//!   so one loaded model serves several streams at once.
//! - **Everything a kit knows about an external ecosystem lives in an interop
//!   module**: the upstream mirrors, their conversions to the kit's configs,
//!   the rekeys and the loaders. What any kit would share about an ecosystem is
//!   [`support::interop`]`::{ecosystem}`; a kit's own mapping of its model onto
//!   one is `{kit}::interop::{ecosystem}`
//!   ([`speech::tts::qwen3_tts::interop::qwen_tts`]). An ecosystem is named for
//!   the schema's owner (`transformers`, `nemo`, `torch`, `qwen_tts`,
//!   `phonon`), never the organization, and a module's path names the source,
//!   so its maps and loaders need no prefix. A kit's model modules hold only
//!   bunsen types and carry no `store` gate; its `pretrained` resolves files
//!   and the catalog's facts and calls its interop. A second source or an
//!   export direction then adds a module, not files under every component.
//!   Every kit is on this layout, and `cargo run -p source-checks -- interop`
//!   holds it: any mirror, rekey, loader or `store` gate outside one fails.
//! - **A mirror is named `*Mirror`, and names every field exactly as the
//!   upstream schema does**, renaming only what Rust cannot spell (`_target_`);
//!   bunsen's vocabulary starts in the config the mirror converts to. Each
//!   field then greps straight to the upstream source
//!   ([`NemoTransformerEncoderMirror`](support::interop::nemo::NemoTransformerEncoderMirror)'
//!   s `pre_ln`). A generic parsed document
//!   ([`TransformersConfigJson`](support::interop::transformers::TransformersConfigJson))
//!   is not a mirror.
//! - **A conversion lives on the upstream side.** When it reads one mirror,
//!   it is a method on that mirror, named for its target
//!   ([`codec_decoder_config`](speech::tts::qwen3_tts::interop::qwen_tts::Qwen3TtsCodecDecoderMirror::codec_decoder_config));
//!   otherwise (several sources, or a parsed document) a free function in the
//!   interop module, named the same way. It is never a method on the bunsen
//!   config nor a `From` / `TryFrom` impl, and it takes the parsed form, never
//!   raw text. The bunsen config then knows nothing about the upstream.
//! - **A kit's interop entry points follow one ladder of names**:
//!   `{target}_config` (a conversion), `{target}_config_from_str` and
//!   `{target}_config_from_path` (for a config a caller starts from),
//!   `load_{target}_tensors` (a rekey and a load) and `load_{target}(loaded,
//!   &config, device, policy)` (a loaded component; a self-describing
//!   container, a `.nemo`, reads its own config). The kit's own root config's
//!   target is the kit's name (`qwen3_config`, `load_qwen3_tensors`). A reader
//!   of one kit can then find every other's.
//! - **An interop module is ungated; its loaders sit in a `store`-gated
//!   child**, so a build without `store` still reads a `config.json` into a
//!   bunsen config. A kit's toy tests load through its interop's rungs, so a
//!   toy catches a broken rekey or config read.
//! - **A kit's tests read no fetched asset.** They run toy fixtures and
//!   committed recordings, and a validation crate holds the kit's checkpoints
//!   to their upstream reference, on an accelerator in a half dtype as well as
//!   in f32. A plain build never reaches the network, and mixed-dtype faults
//!   show only in half precision.
//! - **A driver drops or rewrites user input only when told to.** By default,
//!   input the model cannot use is an error that names the opt-in, and when
//!   input is dropped, the result says so. A silent strip hides the failure
//!   from the user.
//!
//! # Stability
//!
//! Kits are added incrementally and their surfaces may evolve faster
//! than the rest of the crate while their respective domains are being
//! filled in. Pin a specific `bunsen` version when you need a stable
//! API.

pub mod images;
pub mod language;
#[cfg(feature = "speech")]
pub mod speech;
pub mod support;
