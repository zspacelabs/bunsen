//! # Kits: complete models and simulations
//!
//! A kit is a whole thing you can pick up and use: a model family with its
//! configs, the weights it can load and a program that runs it, or a
//! simulation you can step. [`crate::blocks`] is a library of parts; a kit
//! is what the parts are for. Kits are built on the rest of the crate:
//! [`crate::blocks`] and [`crate::ops`] supply the layers and kernels,
//! [`crate::contracts`] checks tensor shapes, [`crate::burner`] holds the
//! module conventions, and [`crate::data`] fetches, caches and verifies the
//! bytes a kit loads.
//!
//! # The kits
//!
//! | Kit | What it is |
//! |---|---|
//! | [`images::resnet`] | `ResNet` image classifiers, with 14 torchvision and `timm` checkpoints by name. |
//! | [`images::swin::v2`] | Swin Transformer V2 image classifiers: configs and model, no pretrained weights. |
//! | [`gpts::nanochat`] | A port of karpathy's nanochat GPT, and the corpus it trains on. Work in progress. |
//! | [`sims::conway`] | Conway's Game of Life on toroidal 2D and 3D boards. |
//! | [`sims::lbm`] | A D2Q9 lattice-Boltzmann fluid, with a mass correction. |
//! | [`speech::whisper`] | `OpenAI`'s Whisper: the model, a pretrained index, decoders, and a stream driver. |
//! | [`speech::silero_vad`] | The Silero voice-activity detector, with weights a build can bundle. |
//! | [`tokens`] | Not a model: the ids-to-text seam the model kits share. |
//!
//! The area modules, [`images`], [`gpts`], [`sims`] and [`speech`], group the
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
//! - **A `Construct` hook.** `{Kit}Construct` is the kit's
//!   [`Construct`](crate::data::pretrained::Construct): the cache segment its
//!   files land under (`{KIT}_KIT`), and how the loaded files become the model.
//! - **`default_{kit}_factory()`.** A
//!   [`PretrainedFactory`](crate::data::pretrained::PretrainedFactory) over the
//!   default providers, behind the hook. A name to a model is one call:
//!   `default_{kit}_factory()?.load::<B>(name, &cache, &device)`. There is [no
//!   registry](crate::data::pretrained::PretrainedFactory#no-registry): a
//!   caller who wants another provider adds it to the factory.
//! - **The built handle.** What `load` returns: a
//!   [`Loaded`](crate::data::pretrained::Loaded) whose `handle` is an `Arc` of
//!   the hook's `Built` type, the model or a bundle around it.
//! - **A driver or context.** Optional. The state a stream or a decode carries
//!   is injected, never owned by the model: the model is immutable, and the
//!   caller creates a context or cache and passes it back in, so one loaded
//!   model serves several streams at once.
//! - **A bundled-asset crate.** Optional. Weights a build ships with, behind a
//!   `*-weights` feature, served by the `bundled:` provider.
//! - **A validation crate.** Optional. A crate under `crates/validation` that
//!   compares the kit's outputs with its upstream reference implementation.
//! - **An example.** A runnable program under `examples/`.
//!
//! The machinery behind the prefab, the rows, the hook and the factory is
//! [`data::pretrained`](crate::data::pretrained): what prefabs, rows and
//! resources are, how a factory dispatches a spec, and how the cache
//! fetches and pins files. A kit contributes tables and a hook.
//!
//! The three kits that load weights, piece by piece:
//!
//! | Piece | `ResNet` | Whisper | Silero VAD |
//! |---|---|---|---|
//! | Configs | [`ResNetContractConfig`](images::resnet::ResNetContractConfig) -> [`ResNetStructureConfig`](images::resnet::ResNetStructureConfig) | [`WhisperApiConfig`](speech::whisper::WhisperApiConfig) -> [`WhisperStructureConfig`](speech::whisper::WhisperStructureConfig) | [`SileroVadSignalConfig`](speech::silero_vad::SileroVadSignalConfig) -> [`SileroVadStftConfig`](speech::silero_vad::SileroVadStftConfig) -> [`SileroVadStructureConfig`](speech::silero_vad::SileroVadStructureConfig) |
//! | Prefab map | [`RESNET_PREFABS`](images::resnet::RESNET_PREFABS) | [`WHISPER_PREFABS`](speech::whisper::pretrained::WHISPER_PREFABS) | none |
//! | Table | [`WELL_KNOWN_TABLE`](images::resnet::WELL_KNOWN_TABLE) | [`WELL_KNOWN_TABLE`](speech::whisper::pretrained::WELL_KNOWN_TABLE), a bundled table, and `hf:` | a bundled table |
//! | Providers | [`default_resnet_providers`](images::resnet::default_resnet_providers) | [`default_whisper_providers`](speech::whisper::pretrained::default_whisper_providers) | [`default_silero_providers`](speech::silero_vad::pretrained::default_silero_providers) |
//! | Hook | [`ResNetConstruct`](images::resnet::ResNetConstruct) | [`WhisperConstruct`](speech::whisper::pretrained::WhisperConstruct) | [`SileroConstruct`](speech::silero_vad::pretrained::SileroConstruct) |
//! | Factory | [`default_resnet_factory`](images::resnet::default_resnet_factory) | [`default_whisper_factory`](speech::whisper::pretrained::default_whisper_factory) | [`default_silero_factory`](speech::silero_vad::pretrained::default_silero_factory) |
//! | Built | [`ResNet`](images::resnet::ResNet) | [`WhisperBundle`](speech::whisper::driver::WhisperBundle): model, token layout, vocabulary | [`SileroVadCollection`](speech::silero_vad::SileroVadCollection): one model per sample rate |
//! | Driver, context | none | [`WhisperStreamDriver`](speech::whisper::driver::WhisperStreamDriver), [`WhisperStreamContext`](speech::whisper::driver::WhisperStreamContext) | [`SileroVadContext`](speech::silero_vad::SileroVadContext) |
//! | Bundled crate | none | `bunsen-bundled-whisper` (`whisper-weights`) | `bunsen-bundled-silero` (`silero-weights`) |
//! | Validation crate | none | `whisper-model-validation` | `silero-model-validation` |
//! | Example | `examples/resnet_finetune`, `examples/resnet_tiny` | `examples/whisper-cli` | `examples/whisper-cli`, as the VAD gate |
//!
//! The other kits have a model or a simulation state, its configs, and an
//! example, and no prefabs or weights. Swin is Stacked
//! (`examples/swin_tiny`). `NanoChat` is Stacked, takes an injected
//! [`KVCache`](crate::ops::transformers::attention::KVCache), and adds a
//! table of the corpus it trains on, a
//! [`data::shards`](crate::data::shards) shard set
//! (`examples/train-chat`). The simulations step a state in place
//! (`examples/conway`, `examples/lbm2d_vis`).
//!
//! Two seams are shared across kits:
//! [`data::pretrained`](crate::data::pretrained) for loading, and
//! [`tokens::Detokenizer`] for turning ids into text.
//!
//! # Stability
//!
//! Kits are added incrementally and their surfaces may evolve faster
//! than the rest of the crate while their respective domains are being
//! filled in. Pin a specific `bunsen` version when you need a stable
//! API.

pub mod gpts;
pub mod images;
pub mod sims;
pub mod speech;
pub mod tokens;
