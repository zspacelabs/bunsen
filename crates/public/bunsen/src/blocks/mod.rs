//! # Neural-network building blocks
//!
//! `bunsen::blocks` is a library of reusable `burn::module::Module`
//! components: the *parts* you compose into larger models.
//!
//! ## Blocks and ops
//!
//! A block is a component meant to be used as a Module root or as a part of
//! a module tree. [Ops and blocks](crate::ops#ops-and-blocks) defines the
//! line between blocks and [`crate::ops`], and the rule that cache and stream
//! state is injected.
//!
//! A block is typically "these parameters, plus these `ops` calls in order";
//! blocks import ops, never the reverse. A block need not own parameters:
//! `DropPath` and `DropBlock2d` own none, and are blocks
//! because they sit in a module tree as layers. A block that decodes takes
//! its cache as an argument rather than holding it:
//! [`CausalSelfAttention::forward`] takes a [`KVCache`].
//!
//! Where [`crate::kits`] supplies named models (`ResNet`, `NanoChatGpt`,
//! Whisper, ...), with their configs, weights and quirks, `blocks` supplies
//! the sub-modules those kits are assembled from, and the generic models
//! several kits share (the conv net, the language decoder). A typical user
//! picks an existing kit; an author building a new model reaches into
//! `blocks`.
//!
//! ## Lifecycle
//!
//! Each block ships as:
//!
//! - a `Config` struct (`#[derive(Config)]`);
//! - a `Module` struct (`#[derive(Module)]`), built by the config;
//! - a `{Name}Meta` trait implemented by both, so the block's geometry reads
//!   the same before and after init.
//!
//! Configs build their module through [`ModuleInit`]
//! (`init(&device)` / `try_init(&device)`); a parameter-free one, such as
//! `DropPathConfig`, ignores the device. Only a constructor that needs more
//! than a device keeps an inherent `init`: `CausalSelfAttentionConfig` also
//! takes the layer's index in its stack (see `ModuleInit`'s
//! [hand-written `init`](crate::burner::module::ModuleInit#hand-written-init)).
//!
//! ## Design Rules
//!
//! Also binding here: the config lifecycle
//! ([`burner::module`'s Design Rules](crate::burner::module#design-rules)) and
//! the crate root's [names and layers](crate#design-rules).
//!
//! - **A block is a `Module` meant to be a root or a part of a module tree:**
//!   its parameters, plus `ops` calls in order. A tensor operation or table
//!   that blocks share lives in [`crate::ops`], and a block that only one kit
//!   uses lives in that kit. Blocks stay composable and ops reusable.
//! - **A block that decodes or streams takes its state as an argument and never
//!   holds it**, as [Ops and blocks](crate::ops#ops-and-blocks) defines:
//!   [`CausalSelfAttention::forward`] takes a [`KVCache`]. One model then
//!   serves several caches or streams at once.
//! - **A forward that works on tensors checks its shapes with
//!   [`crate::contracts`]:** an always-run unpack where the body reads dims,
//!   and cfg-gated asserts, written by full path, on outputs and intermediates.
//!   An output contract binds the dims it shares with the input's unpack; where
//!   only the gated check reads a binding, a `#[cfg(not(any(debug_assertions,
//!   test)))] let _ = d;` keeps it. The rustdoc writes each tensor argument and
//!   result in the notation the contract checks. A forward that only delegates
//!   inherits its callee's checks. A block folded into a shared one keeps its
//!   checks in the shared body. The crate root's [contracts
//!   rule](crate#design-rules) gives the reasons: faster R&D, malformed
//!   assemblies caught, algorithms readable by people and agents.
//! - **A seam is an enum config plus an enum module, and each kind is a module
//!   type of its own**, never a bare value: weight loading walks each variant's
//!   module, as [`activation`]'s `Pow` kind is a
//!   [`PowActivation`](activation::PowActivation).
//! - **Fold a model into a shared block when what it needs is an option**, a
//!   field or a branch. A forward path, a state or an invariant that only one
//!   model takes is a new block family instead. Unification stops before
//!   options make the shared block unreadable:
//!   [`CausalSelfAttentionConfig`](transformers::attention::csa::CausalSelfAttentionConfig)
//!   carries the encoders' two-sided window as an option.
//! - **A two-layer position-wise network is a [`BunsenMlp`](mlp::BunsenMlp) or
//!   a [`BunsenGatedMlp`](mlp::BunsenGatedMlp)**, never a hand-rolled pair of
//!   linears: a transformer's feed-forward, a projector between towers, a
//!   conditioning scale, a position-bias table. Its widths, activation, biases
//!   and dropout are its config's; a checkpoint's names for its parts are its
//!   kit's interop's rekey. One block then carries the MLP's shape contract and
//!   dtype traits for every model.
//! - **A wrapper module never implements `Deref` or `DerefMut` to its inner
//!   module**; callers reach through the field
//!   ([`ResidualSublayer`](transformers::sublayer::ResidualSublayer)'s
//!   `branch`). The wrapper and the inner module share `Module`'s methods, so
//!   an elided field silently changes meaning.
//! - **A block publishes its own weight names through
//!   [`CanonicalKeys`](crate::burner::module::CanonicalKeys).** Key-mapping
//!   machinery has neutral names, an upstream's spelling is data, and nothing
//!   outside the Hub code is named `hf_`: the Hub is one source of checkpoints,
//!   not the namespace for key mapping.
//! - **A block provides key-mapping machinery, never an ecosystem's key
//!   set.** The machinery is generic
//!   ([`CausalDecoderBlock::layer_rekey`](transformers::decoders::causal::CausalDecoderBlock::layer_rekey)
//!   over a [`DecoderLayerKeys`](transformers::decoders::causal::DecoderLayerKeys),
//!   `ConvSeq2d::one_based_rekey`); the names an ecosystem gives a block's
//!   parts (Llama's decoder layer, Whisper's, `PyTorch`'s
//!   `nn.TransformerEncoderLayer`) are that ecosystem's interop's, in
//!   `kits::support::interop`. Blocks stay format-free, and an ecosystem's
//!   names live in one place.
//! - **A change that moves a block onto other combinators is a correct
//!   rewrite**, held to every kit built on the block: each kit is checked
//!   against its reference on a small real row, or, where none exists, the old
//!   form against the new on a toy model before the old form goes. Toy tests
//!   pass on rewrites that change a model's behavior.
//!
//! ## Map of the module
//!
//! - [`activation`]: the activation seam, `BunsenActivationConfig` and
//!   `BunsenActivation`: burn's activations, and bunsen's (`Silu`, `Pow`,
//!   `Snake`, `Chain`), behind one seam bunsen owns.
//! - [`conv`]: conv / norm / activation blocks, in 1D and 2D (`ConvBlock1d`,
//!   `ConvBlock2d`), and sequences of them (`ConvSeq1d`, `ConvSeq2d`); causal
//!   1-d convolutions and upsamplers (`CausalConv1d`, `CausalConvTranspose1d`)
//!   and a causal `ConvNeXt` block (`CausalConvNext1d`).
//! - [`signals`]: learned layers over a signal's time axis, the counterpart of
//!   [`ops::signal`](crate::ops::signal)'s fixed front ends: the frame-rate
//!   changes `FeatureStacking` (frames stacked and projected),
//!   `DwStridingSubsampling` (the `FastConformer`'s strided convolutions) and
//!   `SubpixelUpsample1d` (a sub-pixel convolution), and `EncoderSubsampling`,
//!   an encoder front end chosen by config.
//! - [`images`]: vision blocks.
//!   - [`images::anti_alias`]: anti-aliased downsampling, `BlurPool2d` and an
//!     average pool, behind the `AntiAlias` seam.
//!   - [`images::attention`]: channel attention, `SqueezeExcite` and `Eca`,
//!     behind the `ChannelAttention` seam.
//!   - [`images::drop`]: structured drop layers, `DropBlock2d` and `DropPath`,
//!     and the drop-path rate tables.
//!   - [`images::patching`]: `PatchEmbed`, the Swin Transformer V2 patch
//!     embedding.
//! - [`mlp`]: position-wise MLPs over any rank: `BunsenMlp`, the general form
//!   of burn's `PositionWiseFeedForward`.
//! - [`layout`]: `BoundaryPermutes`, the axis permutes a block applies at its
//!   boundary, so a caller's activation layout and the block's need not agree.
//! - [`pooling`]: a sequence pooled over time: `AttentiveStatsPooling`, each
//!   channel's attention-weighted mean and deviation (speaker encoders').
//! - [`quantization`]: a residual vector quantizer's decode, codes to their
//!   latent: `EuclideanCodebook`, `ResidualVectorQuantizer` and
//!   `SplitResidualVectorQuantizer` (Mimi's and Qwen3-TTS's codecs').
//! - [`rnn`]: recurrent blocks: `FusedLstm`, a single-step LSTM with fused
//!   gates, and its state type.
//! - [`transformers`]: transformer blocks: the parts, and the layers and stacks
//!   assembled from them.
//!   - [`transformers::attention`]: `CausalSelfAttention`, with QK-norm, rotary
//!     embeddings, grouped-query attention and an optional KV cache; and the
//!     encoders' bidirectional attentions, relative-position and rotary.
//!   - [`transformers::embedding`]: `RotaryEmbedding`, and fixture embeddings
//!     for tests.
//!   - [`transformers::mlp`]: `TransformerFeedForward`, a transformer layer's
//!     feed-forward (a `BunsenMlp`, a `BunsenGatedMlp` or a mixture of
//!     experts).
//!   - [`transformers::encoders`]: bidirectional encoder layers and stacks: the
//!     residual layer, the rotary encoder, and the `FastConformer` encoder with
//!     its relative-position attention, convolution and feed-forward modules
//!     and depthwise-striding subsampling.
//!
//! See the per-item rustdoc for shape contracts, defaults, and the papers each
//! block implements.
//!
//! [`CausalSelfAttention::forward`]: transformers::attention::csa::CausalSelfAttention::forward
//! [`KVCache`]: crate::ops::transformers::attention::KVCache
//! [`ModuleInit`]: crate::burner::module::ModuleInit

pub mod activation;
pub mod conv;
pub mod images;
pub mod layout;
pub mod mlp;
pub mod norm;
pub mod pooling;
pub mod quantization;
pub mod rnn;
pub mod signals;
pub mod transformers;
