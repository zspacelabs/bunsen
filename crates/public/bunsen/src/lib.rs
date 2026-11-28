#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "wgpu", recursion_limit = "512")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
// The paths that name `core` or `alloc` where those hold the item, so the
// `std::` paths left are the uses the `no_std` work has to gate.
#![warn(
    clippy::std_instead_of_core,
    clippy::std_instead_of_alloc,
    clippy::alloc_instead_of_core
)]
//! # bunsen
//!
//! A "batteries included" community standard library for the
//! [burn](https://burn.dev) tensor framework: reusable modules, tensor
//! operations, shape contracts, pretrained-model machinery, and the
//! module-lifecycle utilities that fall outside burn's core.
//!
//! These API docs are the reference for every item. The
//! [bunsen book](https://zspacelabs.ai/bunsen/book/) explains how the crate is
//! organized, the systems that cut across modules, and how bunsen is
//! developed.
//!
//! ## Organization
//!
//! The crate is layered: a module builds on the layers below it, never on
//! those above.
//!
//! ```text
//!   kits, sims  whole models; simulations
//!   blocks      torch.nn-like components: module roots and tree parts
//!   ops         operations; Module/Config only to hold cached tables or state
//!   foundation  burner, contracts, errors, rust_ext, support, zspace
//!   ───────────────────────────────────────────────────────────────────────
//!   burn
//!
//!   beside the stack: data (kits load through it), audit
//! ```
//!
//! The `ops` / `blocks` split is a rule, stated in [`ops`](ops#ops-and-blocks).
//!
//! ### Components
//!
//! * [`kits`]: complete models: image models, language models, speech, and the
//!   token seam the speech kits share.
//! * [`sims`]: simulations written as tensor programs: Conway's Game of Life
//!   and a lattice-Boltzmann fluid.
//! * [`blocks`]: reusable [`Module`](burn::module::Module) components: conv
//!   blocks, image layers, recurrent cells, transformer parts.
//! * [`ops`]: tensor operations, including the signal-processing front ends,
//!   attention ops, and the KV cache.
//!
//! ### Foundation
//!
//! * [`burner`]: extensions to burn's module machinery: the config lifecycle
//!   ([`ModuleInit`](burner::module::ModuleInit),
//!   [`ToStructureConfig`](burner::module::ToStructureConfig)), module
//!   [`reflection`](burner::module::reflection), grouped optimizers
//!   ([`optim`](burner::optim)), and `Tensor` extension traits
//!   ([`tensor`](burner::tensor)).
//! * [`contracts`]: runtime tensor-shape contracts.
//! * [`errors`]: `BunsenError`, `BunsenResult`, and the `try_x` / `x`
//!   convention.
//! * [`rust_ext`]: Rust-language extensions with no tensor or burn dependency:
//!   [`CloneBox`](rust_ext::CloneBox) / [`CloneRef`](rust_ext::CloneRef), array
//!   and range helpers, and
//!   [`LocationDesc`](rust_ext::reflection::LocationDesc), a serializable
//!   source location.
//! * [`support`]: shared utilities, including the test backends and devices
//!   (`support::testing`, feature `testing`).
//! * [`zspace`]: integer-lattice index and shape helpers.
//!
//! ### Beside the stack
//!
//! * [`data`]: the disk cache, pretrained-model loading, and dataset shard
//!   sets.
#![cfg_attr(
    feature = "audit",
    doc = "* [`audit`]: audit probes: record a stream of tensor checkpoints from one run,"
)]
#![cfg_attr(
    not(feature = "audit"),
    doc = "* `audit` (feature `audit`): audit probes: record a stream of tensor checkpoints from one run,"
)]
//!   and verify another run (on another backend, or against a stored
//!   baseline) against it.
//!
//! ### Entry points
//!
//! * [`prelude`]: the traits most code needs in scope, for one glob import.
//! * [`public`]: re-exports of public dependencies (`burn`), so downstream code
//!   uses the same versions.
//!
//! ### Sibling crates
//!
//! `bunsen-firehose` and `bunsen-firehose-image` (a data-processing pipeline),
//! `bunsen-arrow-dataloaders` (preview: Parquet shards to training batches),
//! `bunsen-fetch` (the fetching the cache shares with the build scripts that
//! fetch the validation crates' assets), and `bunsen-contracts-macros` (the
//! proc macro behind [`contracts`]). The book's
//! [workspace](https://zspacelabs.ai/bunsen/book/organization/workspace.html)
//! page maps them.
//!
//! ## Design Rules
//!
//! The crate-wide rules for names and layers. The config lifecycle's rules
//! are [`burner::module`'s](burner::module#design-rules); [`blocks`],
//! [`ops`] and `kits` each state their own.
//!
//! - **A module builds on the layers below it, never on those above**
//!   ([Organization](#organization)). Blocks import ops, and ops never import
//!   blocks; the foundation imports none of them. Code behind the `testing`
//!   feature is library code and keeps the layering; a `#[cfg(test)]` module
//!   may reach up for a fixture. One-way dependencies keep each layer usable
//!   alone.
//! - **Every module re-exports its children with `pub use m::*;`, so a public
//!   name must be safe beside burn's under a glob import.**
//!   - A bunsen version of a burn type, in the same role, is
//!     `Bunsen{BurnName}`:
//!     [`BunsenActivation`](blocks::activation::BunsenActivation),
//!     [`BunsenNormalization`](blocks::norm::BunsenNormalization),
//!     [`BunsenLayerNormConfig`](blocks::norm::BunsenLayerNormConfig). The
//!     exception is a type that burn-store recognizes by its name, which keeps
//!     burn's: [`LayerNorm`](blocks::norm::LayerNorm), whose checkpoint
//!     `weight` and `bias` burn-store's adapters map onto `gamma` and `beta` by
//!     matching `Struct:LayerNorm`, and leave out of the half-precision cast.
//!   - A public name carries its family or its owner, never a bare general
//!     word, and the qualifier must not narrow the meaning:
//!     [`DecoderStackCache`](blocks::transformers::decoders::stack::DecoderStackCache),
//!     not `DecoderCache`; `BunsenMlp`, not `Mlp`; `DwStridingFlattenOrder`,
//!     not `FlattenOrder`.
//!   - A name says what the value is, in the field's standard terms. The
//!     residual is the branch, `F(x)`; `x + F(x)` is an update to the residual
//!     stream, so a function returning it is `*_residual_update`. "Connection"
//!     names a link, not a value.
//!
//!   The name spreads into every namespace above it, and an ambiguity
//!   surfaces in the user's code.
//! - **Behaviour lives on its type.** A function that works on one type's
//!   internals is a method or an associated function of that type. A value or
//!   transform that differs per enum variant is a method on the enum, not a
//!   `match` at the use site ([`LogBase::apply`](ops::math::LogBase::apply)). A
//!   transform over tensors is generic over rank. The pipeline then reads as
//!   named steps, and a new variant changes one place.
//! - **A function lives at its natural home.** A tensor operation or a table
//!   lives in [`ops`]; dtype, tensor and module machinery in [`burner`]; a
//!   method-style extension to `Tensor` in [`burner::tensor`]; an operation
//!   that one kit uses in that kit's own `ops` submodule.
//! - **Two names for one concept become one** when the concept is unambiguously
//!   the same. When the parts do not line up, they stay apart, each named for
//!   its owner.
//! - **Code that works on tensors states its shapes as [`contracts`]**:
//!   [`blocks`], [`ops`] and the kits' own modules alike, each as
//!   [`blocks`'](blocks#design-rules) and [`ops`'](ops#design-rules) rules say
//!   (an always-run unpack where a body reads dims, cfg-gated checks on what it
//!   produces). New code carries them from its first layer, and code that moves
//!   or folds into a shared body carries its contracts with it: a lift that
//!   drops them is a regression, whatever its tests say. The contracts serve
//!   three ends:
//!   - **They speed R&D.** A shape mistake fails where it is made, naming its
//!     dims and their sizes, not as a panic deep in burn or as wrong numbers
//!     that a parity check has to trace back.
//!   - **They detect malformed assemblies.** A model put together from configs
//!     whose parts disagree (widths, grids, heads, a seam's variant fed another
//!     layout) is refused at its first forward: by every check in debug and
//!     test builds, and by the always-run unpacks in release.
//!   - **They aid human and agent reading of the algorithms.** Each tensor's
//!     shape is stated in named dims beside the code that makes it, in the
//!     rustdoc's notation, so a reader, a person or an agent, follows the
//!     algorithm without rebuilding the shapes from its reshapes.
//! - **Before defining a type or a reader that burn or an established crate
//!   has, use theirs**: burn's `FloatDType`, burn-store's `PytorchStore`,
//!   `serde_pickle`. A duplicate competes with the upstream vocabulary, and
//!   blocks moving code upstream.
//! - **Interop is by layer, not a module.** Below `kits`, a module or reader
//!   per format or source *is* that ecosystem's interop, named for it
//!   (`data::formats::nemo`, `data::formats::fermion`, the Hub's
//!   `data::builds::HfProvider`); `kits`, organized by model, gives an
//!   ecosystem's code an `interop` namespace: a shared tier is ecosystem first
//!   (`kits::support::{ecosystem}::interop` for any kit,
//!   `kits::{area}::support::{ecosystem}::interop` for one area's), a kit's own
//!   is interop first (`{kit}::interop::{ecosystem}`). There is no crate-level
//!   `interop` and no `data::interop`: one ecosystem has a piece at every
//!   layer, and a single module would span layers or give every item two paths.
//!   [Interop by ecosystem](#interop-by-ecosystem) lists each one's modules.
//! - **A module with submodules is laid out as `m/mod.rs` plus `m/sub.rs`**,
//!   never as `m.rs` plus `m/`, so a module's files are always where you look.
//!   A `mod.rs` holds the module's docs, its `mod`s and its `pub use`s, and no
//!   code: each item lives in a sibling file named for what it holds
//!   (`config.rs`, `rekey.rs`, `tests.rs`), and `pub use` keeps its path. The
//!   one exception, in the firehose crates, is a definition whose value is its
//!   module's path (`define_firehose_operator!`): schemas store that id.
//!
//! ## Interop by ecosystem
//!
//! Each ecosystem's pieces, by layer: its container (how bytes become named
//! tensors, `data::formats`), its shared schema and names (what any kit
//! reads alike, its shared tier's `interop`), and the kits whose own `interop`
//! maps their model onto it. Where the files come from is `data::builds`
//! (the Hub, `data::builds::HfProvider`), and a tensor's layout (torch's `[out,
//! in]` linear, a norm's `weight`) is `burner::module`'s load transforms.
//!
//! | Ecosystem | Container | Shared schema and names | Kits |
//! |---|---|---|---|
//! | `transformers` | `data::formats::SafetensorsCheckpoint` | `kits::support::transformers::interop`: a `config.json` by field, the `generation_config.json` and `tokenizer_config.json` the catalog's probes read, the Llama and Whisper layer keys | `qwen3`, `qwen3_5`, `gemma`, `modernbert`, `voxtral`, `qwen3_asr`, `parakeet`, `whisper`, `sortformer` (Nemotron) |
//! | `nemo` | `data::formats::nemo` (the `.nemo` tar) | `kits::speech::support::nemo::interop`: the YAML config sections, the `FastConformer` and encoder keys | `sortformer` (v2) |
//! | `torch` | `data::formats::read_pytorch_tensors` (`.pt`) | `kits::support::torch::interop`: `nn.TransformerEncoderLayer`'s and `nn.LSTM`'s keys | |
//! | `openai_whisper` | burn's `PytorchStore` (`.pt`) | | `whisper` |
//! | `qwen_tts` | `data::formats::SafetensorsCheckpoint` | | `qwen3_tts` |
//! | `zonos` | `data::formats::SafetensorsCheckpoint` | | `zonos3` |
//! | `mistral_common` | | `kits::support::mistral_common::interop`: `tekken.json`, the vocabulary and an audio model's stream geometry | `voxtral` |
//! | `nanochat` | `data::formats::read_pytorch_tensors` (`.pt`) | | `nanochat` |
//! | `laya` | `data::formats::SafetensorsCheckpoint` | | `laya` (its encoder's `config.json` is `modernbert`'s) |
//! | `torchvision` | `data::formats::read_pytorch_tensors` (`.pt`) | | `resnet` (`timm`'s checkpoints share the names) |
//! | `silero_vad` | burn's `SafetensorsStore` | | `silero_vad` |
//! | `burn_onnx` | burn's `BurnpackStore` (a burnpack of an ONNX graph's nodes) | | `silero_vad` |
//! | `phonon` | `data::formats::fermion` | | `qwen3_asr`, `parakeet` |
//! | MLX | `data::formats::mlx` (affine-quantized tensors) | | |
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

// Without `std`, `alloc` brings `vec!` and `format!`, which `std`'s
// prelude has.
#[cfg_attr(not(feature = "std"), macro_use)]
extern crate alloc;
extern crate core;

// Make the macro targets public.
// TODO: re-examine contracts publication.
#[doc(hidden)]
pub use bunsen_contracts_macros::shape_contract as __proc_shape_contract;

/// Re-exports public dependencies.
#[allow(unused_imports)]
#[allow(missing_docs)]
pub mod public {
    pub use burn;
    pub use hashbrown;
}

#[cfg(feature = "audit")]
pub mod audit;
pub mod blocks;
pub mod burner;
pub mod contracts;
#[cfg(feature = "std")]
pub mod data;
pub mod errors;
#[cfg(feature = "kits")]
pub mod kits;
pub mod ops;
pub mod prelude;
pub mod rust_ext;
#[cfg(feature = "sims")]
pub mod sims;
pub mod support;
pub mod zspace;
