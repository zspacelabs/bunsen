//! # The decoder stack
//!
//! [`DecoderStack`]: a stack of [`CausalDecoderBlock`]s and a final norm,
//! with the rotary embedding they share and their KV cache's geometry: the
//! decoder of a causal language model, without its embedding or head, or a
//! causal encoder (Voxtral's audio tower) as it stands.
//!
//! It has the Stacked config shape. [`DecoderStackStructureConfig`] is the
//! unrolled tree, one block config per layer; two policies lower to it:
//!
//! - [`UniformStackConfig`]: one block config, repeated;
//! - [`PatternedStackConfig`]: a few block configs, the stack's *kinds*, and
//!   each layer's kind (a sliding-window layer among full ones, say).
//!
//! [`DecoderStackVariantConfig`] holds any of the three, for a model config
//! that leaves its stack's shape to the model.
//!
//! A hybrid stack mixes its layers' token mixers ([`TokenMixer`]): attention
//! layers, which keep a history in the decode's [`KVCache`], and
//! linear-attention layers, which carry a state in its [`DeltaNetCache`].
//! Each layer's mixer takes the next slot of its own kind's cache, so a KV
//! cache holds the attention layers alone.

use alloc::{
    string::ToString,
    vec::Vec,
};
use core::ops::Range;

use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::{
        Bool,
        s,
    },
    tensor::Device,
};

use crate::{
    blocks::{
        norm::{
            BunsenNormalization,
            BunsenNormalizationConfig,
        },
        transformers::{
            decoders::causal::{
                CausalDecoderBlock,
                CausalDecoderBlockConfig,
                CausalDecoderBlockMeta,
                DecoderSideInputs,
            },
            embedding::{
                RotaryEmbeddingConfig,
                RotarySet,
            },
            mixers::{
                GatedDeltaNetMeta,
                TokenMixer,
                TokenMixerConfig,
            },
        },
    },
    burner::module::{
        CanonicalKeys,
        ModuleInit,
        Rekey,
        ToStructureConfig,
    },
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::transformers::{
        attention::{
            KVCache,
            KVCacheConfig,
            KVCacheStorage,
            length_mask,
        },
        linear_attention::DeltaNetCache,
    },
};

/// Common meta for [`DecoderStack`] and its configs.
pub trait DecoderStackMeta {
    /// The model width.
    fn d_model(&self) -> usize;

    /// The layer count.
    fn n_layers(&self) -> usize;

    /// The key/value head count of the first attention layer, which a
    /// shared KV cache is shaped for.
    ///
    /// # Panics
    /// If no layer attends.
    fn n_kv_heads(&self) -> usize;

    /// The head width of the first attention layer.
    ///
    /// # Panics
    /// If no layer attends.
    fn head_dim(&self) -> usize;

    /// Each layer's sliding attention window, if it has one; `None` for a
    /// layer that does not attend.
    fn layer_windows(&self) -> Vec<Option<usize>>;

    /// Whether each layer's token mixer attends (else, it is linear
    /// attention).
    fn layer_attends(&self) -> Vec<bool>;

    /// The [`KVCacheConfig`] the stack decodes with: its key/value heads,
    /// head width and attention-layer count, at batch 1, `pre_size`
    /// positions up front, uncapped; a windowed layer keeping only its
    /// window.
    fn kv_cache_config(
        &self,
        pre_size: usize,
    ) -> KVCacheConfig {
        let windows: Vec<_> = self
            .layer_windows()
            .into_iter()
            .zip(self.layer_attends())
            .filter_map(|(w, attends)| attends.then_some(w))
            .collect();
        KVCacheConfig::for_decode(self.n_kv_heads(), pre_size, self.head_dim(), windows.len())
            .with_layer_windows(if windows.iter().all(Option::is_none) {
                Vec::new()
            } else {
                windows
            })
    }
}

/// Each layer's slot in the decode's cache for its mixer's kind: the count
/// of the layers of its kind before it.
fn mixer_slots(attends: &[bool]) -> Vec<usize> {
    let mut counts = [0, 0];
    attends
        .iter()
        .map(|&a| {
            let count = &mut counts[usize::from(a)];
            *count += 1;
            *count - 1
        })
        .collect()
}

/// Config for [`DecoderStack`]: the unrolled tree, one block config per
/// layer.
///
/// Implements [`DecoderStackMeta`] (it has at least one layer once
/// [`try_init`](ModuleInit::try_init) has accepted it).
#[derive(Config, Debug)]
pub struct DecoderStackStructureConfig {
    /// The layers, in order; each one's index is its slot in a shared
    /// [`KVCache`].
    pub layers: Vec<CausalDecoderBlockConfig>,

    /// The final normalization; its feature count is the stack's width.
    pub norm: BunsenNormalizationConfig,

    /// A normalization of the stack's input, before the first layer (an
    /// encoder's over its front end's output); `None`, none.
    #[config(default = "None")]
    pub input_norm: Option<BunsenNormalizationConfig>,

    /// The rotary embedding the layers share, but those
    /// [`layer_rotary`](Self::layer_rotary) points elsewhere; its `seq_len`
    /// is set per call ([`DecoderStack::rotary`]).
    pub rotary: RotaryEmbeddingConfig,

    /// Rotary embeddings beside [`rotary`](Self::rotary), for the layers
    /// [`layer_rotary`](Self::layer_rotary) points at them (Gemma 3's global
    /// table, beside its local one).
    #[config(default = "Vec::new()")]
    pub rotary_alts: Vec<RotaryEmbeddingConfig>,

    /// Each layer's rotary embedding: `None` for [`rotary`](Self::rotary),
    /// `Some(k)` for `rotary_alts[k]`; empty, every layer on `rotary`.
    #[config(default = "Vec::new()")]
    pub layer_rotary: Vec<Option<usize>>,
}

impl DecoderStackStructureConfig {
    /// The first layer, whose width is the stack's.
    fn first_layer(&self) -> &CausalDecoderBlockConfig {
        &self.layers[0]
    }

    /// The first attention layer's attention, whose geometry the stack's KV
    /// cache takes.
    fn first_attention(
        &self
    ) -> &crate::blocks::transformers::attention::csa::CausalSelfAttentionConfig {
        self.layers
            .iter()
            .find_map(|l| l.attn.as_attention())
            .expect("a stack with an attention layer")
    }
}

impl DecoderStackMeta for DecoderStackStructureConfig {
    fn d_model(&self) -> usize {
        self.first_layer().n_embed()
    }

    fn n_layers(&self) -> usize {
        self.layers.len()
    }

    fn n_kv_heads(&self) -> usize {
        use crate::blocks::transformers::attention::csa::CausalSelfAttentionMeta;
        self.first_attention().n_kv_head()
    }

    fn head_dim(&self) -> usize {
        use crate::blocks::transformers::attention::csa::CausalSelfAttentionMeta;
        self.first_attention().head_dim()
    }

    fn layer_windows(&self) -> Vec<Option<usize>> {
        self.layers
            .iter()
            .map(|l| l.attn.as_attention().and_then(|a| a.sliding_window))
            .collect()
    }

    fn layer_attends(&self) -> Vec<bool> {
        self.layers
            .iter()
            .map(|l| matches!(l.attn, TokenMixerConfig::Attention(_)))
            .collect()
    }
}

impl ModuleInit<DecoderStack> for DecoderStackStructureConfig {
    /// Initializes each layer at its slot ([`DecoderStackMeta`]'s kinds: the
    /// count of the layers of its mixer's kind before it, its index in a
    /// stack that only attends), and the final norm at the stack's width.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, for a stack of no layers or of layers of
    /// different widths; as each block's own `try_init`.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<DecoderStack> {
        if self.layers.is_empty() {
            return Err(ConstraintError::new(
                "DecoderStackStructureConfig",
                "layers",
                Rule::Relation {
                    lhs: ("layers.len".into(), "0".into()),
                    op: ">",
                    rhs: ("0".into(), "0".into()),
                },
            )
            .into());
        }
        let width = self.d_model();
        if let Some((i, layer)) = self
            .layers
            .iter()
            .enumerate()
            .find(|(_, l)| l.n_embed() != width)
        {
            return Err(ConstraintError::new(
                "DecoderStackStructureConfig",
                "layers",
                Rule::Relation {
                    lhs: (format!("layers[{i}].n_embed"), layer.n_embed().to_string()),
                    op: "==",
                    rhs: ("layers[0].n_embed".into(), width.to_string()),
                },
            )
            .into());
        }
        if !self.layer_rotary.is_empty() && self.layer_rotary.len() != self.layers.len() {
            return Err(ConstraintError::new(
                "DecoderStackStructureConfig",
                "layer_rotary",
                Rule::Relation {
                    lhs: (
                        "layer_rotary.len".into(),
                        self.layer_rotary.len().to_string(),
                    ),
                    op: "==",
                    rhs: ("layers.len".into(), self.layers.len().to_string()),
                },
            )
            .into());
        }
        if let Some((i, k)) = self
            .layer_rotary
            .iter()
            .enumerate()
            .find_map(|(i, k)| k.filter(|&k| k >= self.rotary_alts.len()).map(|k| (i, k)))
        {
            return Err(ConstraintError::new(
                "DecoderStackStructureConfig",
                "layer_rotary",
                Rule::Relation {
                    lhs: (format!("layer_rotary[{i}]"), k.to_string()),
                    op: "<",
                    rhs: ("rotary_alts.len".into(), self.rotary_alts.len().to_string()),
                },
            )
            .into());
        }
        let slots = mixer_slots(&self.layer_attends());
        Ok(DecoderStack {
            layers: self
                .layers
                .iter()
                .zip(slots)
                .map(|(layer, slot)| layer.try_init(slot, device))
                .collect::<BunsenResult<_>>()?,
            norm: self.norm.clone().with_num_features(width).init(device),
            input_norm: self
                .input_norm
                .as_ref()
                .map(|n| n.clone().with_num_features(width).init(device)),
            rotary: self.rotary.clone(),
            rotary_alts: self.rotary_alts.clone(),
            layer_rotary: self.layer_rotary.clone(),
        })
    }
}

/// A [`DecoderStack`] of `n_layers` copies of one block config.
///
/// Lowers to [`DecoderStackStructureConfig`].
#[derive(Config, Debug)]
pub struct UniformStackConfig {
    /// The layer count.
    pub n_layers: usize,

    /// Every layer's config.
    pub layer: CausalDecoderBlockConfig,

    /// The final normalization.
    pub norm: BunsenNormalizationConfig,

    /// The rotary embedding the layers share.
    pub rotary: RotaryEmbeddingConfig,

    /// A normalization of the stack's input
    /// ([`DecoderStackStructureConfig::input_norm`]).
    #[config(default = "None")]
    pub input_norm: Option<BunsenNormalizationConfig>,
}

impl ToStructureConfig for UniformStackConfig {
    type Structure = DecoderStackStructureConfig;

    fn try_to_structure(&self) -> BunsenResult<DecoderStackStructureConfig> {
        Ok(DecoderStackStructureConfig::new(
            vec![self.layer.clone(); self.n_layers],
            self.norm.clone(),
            self.rotary.clone(),
        )
        .with_input_norm(self.input_norm.clone()))
    }
}

/// A [`DecoderStack`] of a few block configs, its *kinds*, and each layer's
/// kind, by index: a sliding-window kind among full-attention ones, say.
///
/// Lowers to [`DecoderStackStructureConfig`].
#[derive(Config, Debug)]
pub struct PatternedStackConfig {
    /// The kinds of layer.
    pub kinds: Vec<CausalDecoderBlockConfig>,

    /// Each layer's kind: an index into [`kinds`](Self::kinds).
    pub layer_kinds: Vec<usize>,

    /// The final normalization.
    pub norm: BunsenNormalizationConfig,

    /// The rotary embedding the layers share, but those of a kind
    /// [`kind_rotary`](Self::kind_rotary) points elsewhere.
    pub rotary: RotaryEmbeddingConfig,

    /// Rotary embeddings beside [`rotary`](Self::rotary)
    /// ([`DecoderStackStructureConfig::rotary_alts`]).
    #[config(default = "Vec::new()")]
    pub rotary_alts: Vec<RotaryEmbeddingConfig>,

    /// Each kind's rotary embedding: `None` for [`rotary`](Self::rotary),
    /// `Some(k)` for `rotary_alts[k]`; empty, or short, `None`.
    #[config(default = "Vec::new()")]
    pub kind_rotary: Vec<Option<usize>>,

    /// A normalization of the stack's input
    /// ([`DecoderStackStructureConfig::input_norm`]).
    #[config(default = "None")]
    pub input_norm: Option<BunsenNormalizationConfig>,
}

impl ToStructureConfig for PatternedStackConfig {
    type Structure = DecoderStackStructureConfig;

    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause naming the layer, for a layer whose kind is
    /// not one of [`kinds`](Self::kinds).
    fn try_to_structure(&self) -> BunsenResult<DecoderStackStructureConfig> {
        let layers = self
            .layer_kinds
            .iter()
            .enumerate()
            .map(|(i, &kind)| {
                self.kinds.get(kind).cloned().ok_or_else(|| {
                    BunsenError::from(ConstraintError::new(
                        "PatternedStackConfig",
                        "layer_kinds",
                        Rule::Relation {
                            lhs: (format!("layer_kinds[{i}]"), kind.to_string()),
                            op: "<",
                            rhs: ("kinds.len".into(), self.kinds.len().to_string()),
                        },
                    ))
                })
            })
            .collect::<BunsenResult<_>>()?;
        let layer_rotary = if self.kind_rotary.iter().all(Option::is_none) {
            Vec::new()
        } else {
            self.layer_kinds
                .iter()
                .map(|&kind| self.kind_rotary.get(kind).copied().flatten())
                .collect()
        };
        Ok(
            DecoderStackStructureConfig::new(layers, self.norm.clone(), self.rotary.clone())
                .with_rotary_alts(self.rotary_alts.clone())
                .with_layer_rotary(layer_rotary)
                .with_input_norm(self.input_norm.clone()),
        )
    }
}

/// Any of a [`DecoderStack`]'s configs: a policy, or the structure itself.
///
/// What a model config that holds a stack holds, so each model picks the
/// shape that says its stack best, and lowers through one step to
/// [`DecoderStackStructureConfig`]. Each config converts into it with
/// `.into()`.
#[derive(Config, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum DecoderStackVariantConfig {
    /// One block config, repeated ([`UniformStackConfig`]).
    Uniform(UniformStackConfig),

    /// A few kinds of block, and each layer's kind
    /// ([`PatternedStackConfig`]).
    Patterned(PatternedStackConfig),

    /// The unrolled tree ([`DecoderStackStructureConfig`]), as it is.
    Structure(DecoderStackStructureConfig),
}

impl From<UniformStackConfig> for DecoderStackVariantConfig {
    fn from(config: UniformStackConfig) -> Self {
        Self::Uniform(config)
    }
}

impl From<PatternedStackConfig> for DecoderStackVariantConfig {
    fn from(config: PatternedStackConfig) -> Self {
        Self::Patterned(config)
    }
}

impl From<DecoderStackStructureConfig> for DecoderStackVariantConfig {
    fn from(config: DecoderStackStructureConfig) -> Self {
        Self::Structure(config)
    }
}

impl ToStructureConfig for DecoderStackVariantConfig {
    type Structure = DecoderStackStructureConfig;

    /// # Errors
    /// As the variant's own lowering; a
    /// [`Structure`](DecoderStackVariantConfig::Structure) never fails.
    fn try_to_structure(&self) -> BunsenResult<DecoderStackStructureConfig> {
        match self {
            Self::Uniform(config) => config.try_to_structure(),
            Self::Patterned(config) => config.try_to_structure(),
            Self::Structure(config) => Ok(config.clone()),
        }
    }
}

/// A stack of [`CausalDecoderBlock`]s and a final norm: the decoder of a
/// causal language model, between its embedding and its head, or a causal
/// encoder.
///
/// Built by [`DecoderStackStructureConfig`], or a policy that lowers to it
/// ([`UniformStackConfig`], [`PatternedStackConfig`]).
/// Implements [`DecoderStackMeta`].
#[derive(Module, Debug)]
pub struct DecoderStack {
    /// The layers, in order; each one's index is its slot in a shared
    /// [`KVCache`].
    pub layers: Vec<CausalDecoderBlock>,

    /// The final normalization.
    pub norm: BunsenNormalization,

    /// The input's normalization, if the stack has one.
    pub input_norm: Option<BunsenNormalization>,

    /// The rotary embedding the layers share, but its length
    /// ([`rotary`](Self::rotary)).
    #[module(skip)]
    pub rotary: RotaryEmbeddingConfig,

    /// The rotary embeddings beside [`rotary`](Self::rotary)
    /// ([`DecoderStackStructureConfig::rotary_alts`]).
    #[module(skip)]
    pub rotary_alts: Vec<RotaryEmbeddingConfig>,

    /// Each layer's rotary embedding
    /// ([`DecoderStackStructureConfig::layer_rotary`]).
    #[module(skip)]
    pub layer_rotary: Vec<Option<usize>>,
}

impl DecoderStackMeta for DecoderStack {
    fn d_model(&self) -> usize {
        self.layers[0].n_embed()
    }

    fn n_layers(&self) -> usize {
        self.layers.len()
    }

    fn n_kv_heads(&self) -> usize {
        use crate::blocks::transformers::attention::csa::CausalSelfAttentionMeta;
        self.first_attention().n_kv_head()
    }

    fn head_dim(&self) -> usize {
        self.first_attention().head_dim
    }

    fn layer_windows(&self) -> Vec<Option<usize>> {
        self.layers
            .iter()
            .map(|l| l.attn.branch.as_attention().and_then(|a| a.sliding_window))
            .collect()
    }

    fn layer_attends(&self) -> Vec<bool> {
        self.layers
            .iter()
            .map(|l| matches!(l.attn.branch, TokenMixer::Attention(_)))
            .collect()
    }
}

/// A residual-stream mix around a [`DecoderStack`]'s layers (nanochat's):
/// before layer `i`, `x = x * resid[i] + x0 * x0_lambdas[i]`, with `x0` the
/// stack's input; and the residual after layer `backout.0`, scaled by
/// `backout.1`, taken from the stream after the last layer.
#[derive(Clone, Debug, Default)]
pub struct DecoderStackResidualMix {
    /// Per-layer `[n_layers]` scales of the stream and of the stack's input.
    pub resid_x0: Option<(Tensor<1>, Tensor<1>)>,

    /// The layer whose output is backed out, and the `[1]` scale it is
    /// backed out by.
    pub backout: Option<(usize, Tensor<1>)>,
}

/// What a [`DecoderStack`]'s [`forward_with`](DecoderStack::forward_with)
/// takes beside its input; each is only for layers configured for it.
#[derive(Debug, Default)]
pub struct DecoderStackSides {
    /// The condition every layer with an
    /// [`mlp_scale`](CausalDecoderBlock::mlp_scale) takes (Voxtral's delay):
    /// `[B, T, D]`, or one that broadcasts to it.
    pub cond: Option<Tensor<3>>,

    /// Each layer's `[B, T, H_kv * head_dim]` value embedding, by layer
    /// index, for the layers whose attention has a value residual; a layer
    /// past the end takes none.
    pub value_embeds: Vec<Option<Tensor<3>>>,

    /// The residual-stream mix.
    pub mix: DecoderStackResidualMix,

    /// `[B, T]`, true at the positions that are not padding, which alone
    /// every layer's queries attend to (an encoder's padded batch).
    pub key_valid: Option<Tensor<2, Bool>>,
}

/// What a cached decode through a [`DecoderStack`] carries between steps:
/// its KV cache, its linear-attention layers' states (none in a stack that
/// only attends), and the rotary table for every position it may reach.
///
/// Made by [`DecoderStack::decode_cache`], run through by
/// [`DecoderStack::forward_cached`]. The stack never owns one, so one loaded
/// model serves any number of decodes.
#[derive(Module, Debug)]
pub struct DecoderStackCache {
    /// The KV cache, one batch row per decode row: the attention layers'.
    pub cache: KVCache,

    /// The linear-attention layers' states, one row per decode row.
    pub linear: DeltaNetCache,

    /// The rotary tables, covering the cache's capacity.
    pub rotary: RotarySet,
}

impl DecoderStackCache {
    /// Permutes the rows: row `r` becomes row `sources[r]` (beams
    /// branching).
    pub fn reorder(
        &mut self,
        sources: &[usize],
    ) {
        self.cache.reorder(sources);
        self.linear.reorder(sources);
    }

    /// The position the next step writes at: the KV cache's.
    pub fn pos(&self) -> usize {
        self.cache.pos()
    }

    /// The furthest position at or before `pos` that
    /// [`truncate`](Self::truncate) can rewind to: `pos` itself where the KV
    /// cache can rewind there ([`KVCache::can_truncate`]) and the stack has
    /// no linear-attention layers, else 0. A linear layer's state summarizes
    /// the whole prefix: it can stay where it is, go back to its
    /// [snapshot](Self::snapshot), or start over.
    pub fn rewind_point(
        &self,
        pos: usize,
    ) -> usize {
        let kv = if self.cache.can_truncate(pos) { pos } else { 0 };
        if self.linear.is_empty() {
            return kv;
        }
        let here = self.cache.pos();
        [Some(here), self.linear.snapshot_pos()]
            .into_iter()
            .flatten()
            .filter(|&p| p <= kv && self.cache.can_truncate(p))
            .max()
            .unwrap_or(0)
    }

    /// Whether [`truncate`](Self::truncate) can rewind to `pos`
    /// ([`rewind_point`](Self::rewind_point)).
    pub fn can_truncate(
        &self,
        pos: usize,
    ) -> bool {
        self.rewind_point(pos) == pos
    }

    /// Rewinds to position `pos`, as [`KVCache::truncate`]: a rewind with
    /// linear-attention layers to their snapshot's position restores it,
    /// and one to 0 resets them.
    ///
    /// # Panics
    /// If it [cannot](Self::can_truncate) rewind to `pos`.
    pub fn truncate(
        &mut self,
        pos: usize,
    ) {
        assert!(
            self.can_truncate(pos),
            "cannot truncate a decode cache at {} to {pos}: its linear-attention states \
             go back only to their snapshot ({:?}) or to 0",
            self.cache.pos(),
            self.linear.snapshot_pos()
        );
        if pos < self.cache.pos() && !self.linear.is_empty() {
            if pos == 0 {
                self.linear.reset();
            } else {
                self.linear.restore_snapshot();
            }
        }
        self.cache.truncate(pos);
    }

    /// Keeps the linear-attention layers' states as they are now, at the
    /// cache's position, for a later [`truncate`](Self::truncate) to go back
    /// to (a chat's turn boundary); a no-op for a stack that only attends.
    pub fn snapshot(&mut self) {
        if !self.linear.is_empty() {
            self.linear.take_snapshot(self.cache.pos());
        }
    }

    /// Back to the start: the KV cache's position 0, and the linear states
    /// zeros.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.linear.reset();
    }
}

impl DecoderStack {
    /// The first attention layer's attention, whose geometry the stack's KV
    /// cache takes.
    fn first_attention(&self) -> &crate::blocks::transformers::attention::csa::CausalSelfAttention {
        self.layers
            .iter()
            .find_map(|l| l.attn.branch.as_attention())
            .expect("a stack with an attention layer")
    }

    /// A decode cache for `rows` rows of at most `positions` positions each,
    /// under `storage`: the KV cache allocated whole and capped there, as
    /// the rotary table is; and a zero state for each linear-attention
    /// layer.
    pub fn decode_cache(
        &self,
        rows: usize,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> DecoderStackCache {
        DecoderStackCache {
            cache: self
                .kv_cache_config(positions)
                .with_batch_size(rows)
                .with_max_seq_len(Some(positions))
                .with_storage(storage)
                .init(),
            linear: DeltaNetCache::new(
                self.layers
                    .iter()
                    .filter_map(|l| l.attn.branch.as_gated_delta_net())
                    .map(|net| net.state_config(rows).init(device))
                    .collect(),
            ),
            rotary: self.rotary(positions, device),
        }
    }

    /// [`forward_hidden`](Self::forward_hidden) through `cache`, at its
    /// position, advancing it.
    pub fn forward_cached(
        &self,
        x: Tensor<3>,
        cache: &mut DecoderStackCache,
    ) -> Tensor<3> {
        self.forward_with_caches(
            x,
            &cache.rotary,
            Some(&mut cache.cache),
            Some(&mut cache.linear),
            DecoderStackSides::default(),
        )
    }

    /// The rotary tables for positions `0..seq_len`: [`rotary`](Self::rotary)'s
    /// first, then each of [`rotary_alts`](Self::rotary_alts), and which one
    /// each layer applies.
    pub fn rotary(
        &self,
        seq_len: usize,
        device: &Device,
    ) -> RotarySet {
        let table = |config: &RotaryEmbeddingConfig| {
            let mut config = config.clone();
            config.seq_len = seq_len;
            config.init(device)
        };
        RotarySet {
            tables: core::iter::once(&self.rotary)
                .chain(&self.rotary_alts)
                .map(table)
                .collect(),
            layer_tables: self
                .layer_rotary
                .iter()
                .map(|k| k.map_or(0, |k| k + 1))
                .collect(),
        }
    }

    /// Runs the layers and the final norm over `x`, at positions
    /// `start..start + seq`, through the cache when there is one.
    ///
    /// # Arguments
    /// * `x` - `[batch, seq, d_model]`.
    /// * `r_emb` - rotary tables covering `start + seq` positions
    ///   ([`rotary`](Self::rotary)).
    /// * `kv_cache` - the cache, at position `start`; `None` means `start` is 0
    ///   and nothing is kept.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`, normalized.
    pub fn forward_hidden(
        &self,
        x: Tensor<3>,
        r_emb: &RotarySet,
        kv_cache: Option<&mut KVCache>,
    ) -> Tensor<3> {
        self.forward_with(x, r_emb, kv_cache, DecoderStackSides::default())
    }

    /// The stack as an encoder: its rotary table over the whole input, and
    /// each item's padding keys masked by its length.
    ///
    /// # Arguments
    /// * `x` - `[batch, time, d_model]`.
    /// * `lengths` - each item's frames; `None`, no padding.
    ///
    /// # Returns
    /// `[batch, time, d_model]`.
    pub fn encoder_forward(
        &self,
        x: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> Tensor<3> {
        let time = x.dims()[1];
        let device = x.device();
        let key_valid = lengths.map(|lengths| length_mask(lengths, time, &device));
        self.forward_with(
            x,
            &self.rotary(time, &device),
            None,
            DecoderStackSides {
                key_valid,
                ..Default::default()
            },
        )
    }

    /// [`forward_hidden`](Self::forward_hidden), with `sides`: a condition
    /// and value embeddings for the layers that take them, and a residual
    /// mix around the layers.
    ///
    /// # Panics
    /// If the stack has linear-attention layers and a KV cache is given: a
    /// hybrid stack decodes through
    /// [`forward_with_caches`](Self::forward_with_caches) (or
    /// [`forward_cached`](Self::forward_cached)), which carries their state.
    pub fn forward_with(
        &self,
        x: Tensor<3>,
        r_emb: &RotarySet,
        kv_cache: Option<&mut KVCache>,
        sides: DecoderStackSides,
    ) -> Tensor<3> {
        self.forward_with_caches(x, r_emb, kv_cache, None, sides)
    }

    /// [`forward_with`](Self::forward_with), through the decode's caches of
    /// both kinds: the attention layers' KV cache, and the linear-attention
    /// layers' [`DeltaNetCache`]. Without them, each layer starts the
    /// sequence at `x` and keeps nothing.
    pub fn forward_with_caches(
        &self,
        x: Tensor<3>,
        r_emb: &RotarySet,
        mut kv_cache: Option<&mut KVCache>,
        mut linear: Option<&mut DeltaNetCache>,
        sides: DecoderStackSides,
    ) -> Tensor<3> {
        let [batch, seq] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "d_model"],
            &x.dims(),
            &["batch", "seq"],
            &[("d_model", self.d_model())]
        );
        let start = kv_cache.as_ref().map(|c| c.pos()).unwrap_or(0);
        let r_emb = r_emb.clip_range(start..start + seq);
        let DecoderStackSides {
            cond,
            mut value_embeds,
            mix,
            key_valid,
        } = sides;
        let x = match &self.input_norm {
            Some(norm) => norm.forward(x),
            None => x,
        };
        let x0 = mix.resid_x0.is_some().then(|| x.clone());
        let mut backed_out = None;
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            if let (Some((resid, x0_lambdas)), Some(x0)) = (&mix.resid_x0, &x0) {
                let at = |t: &Tensor<1>| t.clone().slice(s![i..i + 1]).unsqueeze::<3>();
                x = x * at(resid) + x0.clone() * at(x0_lambdas);
            }
            let value_embed = value_embeds.get_mut(i).and_then(Option::take);
            x = layer.forward_with_caches(
                x,
                r_emb.for_layer(i),
                &mut kv_cache,
                linear.as_deref_mut(),
                DecoderSideInputs {
                    cond: cond.clone(),
                    value_embed,
                    key_valid: key_valid.clone(),
                },
            );
            if mix.backout.as_ref().is_some_and(|(at, _)| *at == i) {
                backed_out = Some(x.clone());
            }
        }
        if let (Some((_, lambda)), Some(mid)) = (&mix.backout, backed_out) {
            x = x - mid * lambda.clone().unsqueeze::<3>();
        }
        let y = self.norm.forward(x);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "d_model"],
            &y.dims(),
            &[("batch", batch), ("seq", seq), ("d_model", self.d_model())],
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = batch;

        y
    }

    /// The attention weights of an uncached pass over `x`, at `layers`, each
    /// cut to the queries `rows` and the keys `cols`.
    ///
    /// # Arguments
    /// * `x` - `[batch, seq, d_model]`, at positions `0..seq`.
    /// * `r_emb` - rotary tables covering `seq` positions.
    /// * `layers` - the layers, by index, in the order returned.
    /// * `rows`, `cols` - the query and key positions kept.
    ///
    /// # Returns
    /// For each of `layers`, `[batch, n_heads, rows.len(), cols.len()]`.
    ///
    /// # Panics
    /// If a layer index is past the last layer, or names a layer that does
    /// not attend.
    pub fn attention_weights(
        &self,
        x: Tensor<3>,
        r_emb: &RotarySet,
        layers: &[usize],
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Vec<Tensor<4>> {
        let [_, seq] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "d_model"],
            &x.dims(),
            &["batch", "seq"],
            &[("d_model", self.d_model())]
        );
        assert!(
            layers.iter().all(|&l| l < self.layers.len()),
            "layers {layers:?} of {}",
            self.layers.len()
        );
        let r_emb = r_emb.clip_range(0..seq);
        let x = match &self.input_norm {
            Some(norm) => norm.forward(x),
            None => x,
        };
        let deepest = layers.iter().copied().max().map_or(0, |l| l + 1);
        let mut weights: Vec<Option<Tensor<4>>> = vec![None; layers.len()];
        let mut x = x;
        for (index, layer) in self.layers.iter().enumerate().take(deepest) {
            for (slot, _) in layers.iter().enumerate().filter(|&(_, &l)| l == index) {
                let w = layer
                    .attn
                    .branch
                    .attention()
                    .attention_weights(layer.attn_input(x.clone()), r_emb.for_layer(index));
                weights[slot] = Some(w.slice(s![.., .., rows.clone(), cols.clone()]));
            }
            x = layer.forward(x, r_emb.for_layer(index), &mut None);
        }
        weights
            .into_iter()
            .map(|w| w.expect("every layer visited"))
            .collect()
    }
}

impl CanonicalKeys for DecoderStack {
    fn canonical_rekey(&self) -> Rekey {
        Rekey::keep_rest().mount_canonical("layers.", &self.layers)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::norm::RmsNormConfig,
        tensor::{
            Distribution,
            Tolerance,
        },
    };
    use serial_test::serial;

    use super::*;
    use crate::{
        blocks::transformers::{
            attention::csa::CausalSelfAttentionConfig,
            mlp::TransformerGatedMlpConfig,
        },
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        support::testing::{
            DeviceMemoryGuard,
            assert_tensors_close,
            performance_device,
            seeded_tensor,
        },
    };

    fn layer(window: Option<usize>) -> CausalDecoderBlockConfig {
        CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(4, 2, 16)
                .with_head_dim(Some(8))
                .with_sliding_window(window)
                .into(),
            TransformerGatedMlpConfig::new(16, 32).into(),
        )
    }

    fn uniform(n: usize) -> UniformStackConfig {
        UniformStackConfig::new(
            n,
            layer(None),
            BunsenNormalizationConfig::Rms(RmsNormConfig::new(0)),
            RotaryEmbeddingConfig::new(0, 8),
        )
    }

    /// The policy and its structure build the same stack, and the meta
    /// agrees across all three.
    #[test]
    fn test_uniform_pathways_agree() {
        let device = performance_device();
        let policy = uniform(3);
        let structure = policy.to_structure();
        let a: DecoderStack = policy.init(&device);
        let b: DecoderStack = structure.init(&device);
        for meta in [&a as &dyn DecoderStackMeta, &b, &structure] {
            assert_eq!(
                (
                    meta.d_model(),
                    meta.n_layers(),
                    meta.n_kv_heads(),
                    meta.head_dim()
                ),
                (16, 3, 2, 8)
            );
        }
        assert_eq!(a.num_params(), b.num_params());
        assert_eq!(
            a.kv_cache_config(64).seq_len,
            64,
            "the cache's pre-size is the caller's"
        );
    }

    /// Each layer is its kind's; a kind that is not one is refused, naming
    /// the layer.
    #[test]
    fn test_patterned_lowering() {
        let kinds = vec![layer(Some(4)), layer(None)];
        let policy = PatternedStackConfig::new(
            kinds,
            vec![0, 0, 1, 0],
            BunsenNormalizationConfig::Rms(RmsNormConfig::new(0)),
            RotaryEmbeddingConfig::new(0, 8),
        );
        let windows: Vec<_> = policy
            .to_structure()
            .layers
            .iter()
            .map(|l| l.attn.attention().sliding_window)
            .collect();
        assert_eq!(windows, [Some(4), Some(4), None, Some(4)]);

        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("layer_kinds[1]")
            .assert_err(
                &PatternedStackConfig {
                    layer_kinds: vec![1, 2],
                    ..policy
                }
                .try_to_structure(),
            );
    }

    /// Each variant lowers as its config does, and builds the same stack;
    /// a policy's refusal comes through the variant.
    #[test]
    fn test_variant_lowering() {
        let device = crate::support::testing::cpu_device();
        let uniform = uniform(2);
        let patterned = PatternedStackConfig::new(
            vec![layer(Some(4)), layer(None)],
            vec![0, 1],
            BunsenNormalizationConfig::Rms(RmsNormConfig::new(0)),
            RotaryEmbeddingConfig::new(0, 8),
        );
        let structure = patterned.to_structure();
        for (variant, expected) in [
            (
                DecoderStackVariantConfig::from(uniform.clone()),
                uniform.to_structure(),
            ),
            (patterned.clone().into(), structure.clone()),
            (structure.clone().into(), structure),
        ] {
            let lowered = variant.to_structure();
            assert_eq!(lowered.layer_windows(), expected.layer_windows());
            assert_eq!(lowered.n_layers(), expected.n_layers());
            let a: DecoderStack = variant.init(&device);
            let b: DecoderStack = expected.init(&device);
            assert_eq!(a.num_params(), b.num_params());
            assert_eq!(a.layer_windows(), b.layer_windows());
        }

        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("layer_kinds[1]")
            .assert_err(
                &DecoderStackVariantConfig::from(PatternedStackConfig {
                    layer_kinds: vec![0, 2],
                    ..patterned
                })
                .try_to_structure(),
            );
    }

    /// A kind's rotary embedding is its layers': the set holds the default
    /// table first, then the alternates, and each layer applies its kind's;
    /// the stack's forward is each layer's forward with its own table.
    #[test]
    fn test_kind_rotary() {
        let device = crate::support::testing::cpu_device();
        let global = RotaryEmbeddingConfig::new(0, 8)
            .with_base(1_000_000)
            .with_scaling(
                crate::blocks::transformers::embedding::RopeScaling::Linear { factor: 2.0 },
            );
        let policy = PatternedStackConfig::new(
            vec![layer(Some(4)), layer(None)],
            vec![0, 1, 0],
            BunsenNormalizationConfig::Rms(RmsNormConfig::new(0)),
            RotaryEmbeddingConfig::new(0, 8),
        )
        .with_rotary_alts(vec![global])
        .with_kind_rotary(vec![None, Some(0)]);
        let structure = policy.to_structure();
        assert_eq!(structure.layer_rotary, [None, Some(0), None]);
        let stack: DecoderStack = structure.init(&device);

        let n = 6;
        let set = stack.rotary(n, &device);
        assert_eq!(set.tables.len(), 2);
        assert_eq!(set.layer_tables, [0, 1, 0]);
        assert_eq!(set.seq_len(), n);
        assert!(
            !set.tables[0]
                .cos
                .clone()
                .all_close(set.tables[1].cos.clone(), None, None),
            "two tables"
        );

        let x = seeded_tensor::<3>(1, [1, n, 16], Distribution::Default, &device);
        let mut y = x.clone();
        for (i, layer) in stack.layers.iter().enumerate() {
            y = layer.forward(y, set.for_layer(i), &mut None);
        }
        assert_tensors_close(
            &stack.forward_hidden(x, &set, None),
            &stack.norm.forward(y),
            Tolerance::default(),
        );

        // Without alternates, no layer points anywhere.
        assert!(
            PatternedStackConfig::new(
                vec![layer(Some(4)), layer(None)],
                vec![0, 1],
                BunsenNormalizationConfig::Rms(RmsNormConfig::new(0)),
                RotaryEmbeddingConfig::new(0, 8),
            )
            .to_structure()
            .layer_rotary
            .is_empty()
        );
    }

    /// A layer table of the wrong length, or one naming no alternate, is
    /// refused.
    #[test]
    fn test_layer_rotary_refusals() {
        let device = crate::support::testing::cpu_device();
        let short = uniform(2).to_structure().with_layer_rotary(vec![None]);
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("layer_rotary.len")
            .assert_err(&ModuleInit::<DecoderStack>::try_init(&short, &device));
        let missing = uniform(2)
            .to_structure()
            .with_layer_rotary(vec![None, Some(0)]);
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("layer_rotary[1]")
            .assert_err(&ModuleInit::<DecoderStack>::try_init(&missing, &device));
    }

    /// No layers, or layers of two widths, are refused.
    #[test]
    fn test_structure_refusals() {
        let device = performance_device();
        let empty = uniform(0).to_structure();
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("layers")
            .assert_err(&ModuleInit::<DecoderStack>::try_init(&empty, &device));
        let mut mixed = uniform(2).to_structure();
        mixed.layers[1] = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(4, 2, 24)
                .with_head_dim(Some(8))
                .into(),
            TransformerGatedMlpConfig::new(24, 32).into(),
        );
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("layers[1].n_embed")
            .assert_err(&ModuleInit::<DecoderStack>::try_init(&mixed, &device));
    }

    /// A prefill and then one position per step, through a cache, is the
    /// uncached pass, for a uniform stack and a patterned one whose window is
    /// shorter than the sequence.
    #[test]
    #[serial]
    fn test_cached_matches_uncached() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let patterned = PatternedStackConfig::new(
            vec![layer(Some(3)), layer(None)],
            vec![0, 1, 0],
            BunsenNormalizationConfig::Rms(RmsNormConfig::new(0)),
            RotaryEmbeddingConfig::new(0, 8),
        )
        .to_structure();
        for structure in [uniform(2).to_structure(), patterned] {
            device.seed(5);
            let stack: DecoderStack = structure.init(&device);
            let n = 7;
            let x: Tensor<3> = seeded_tensor(11, [1, n, 16], Distribution::Default, &device);
            let r_emb = stack.rotary(n, &device);
            let whole = stack.forward_hidden(x.clone(), &r_emb, None);

            let mut cache = stack.kv_cache_config(2).with_chunk_size(2).init();
            let mut parts =
                vec![stack.forward_hidden(x.clone().slice_dim(1, 0..3), &r_emb, Some(&mut cache))];
            for pos in 3..n {
                parts.push(stack.forward_hidden(
                    x.clone().slice_dim(1, pos..pos + 1),
                    &r_emb,
                    Some(&mut cache),
                ));
            }
            assert_tensors_close(
                &Tensor::cat(parts, 1),
                &whole,
                Tolerance::rel_abs(1e-4, 1e-4),
            );
        }
    }

    fn linear_layer() -> CausalDecoderBlockConfig {
        CausalDecoderBlockConfig::new(
            crate::blocks::transformers::mixers::GatedDeltaNetConfig::new(16, 2, 4, 4, 3).into(),
            TransformerGatedMlpConfig::new(16, 32).into(),
        )
    }

    /// Linear, linear, attention, linear, attention.
    fn hybrid() -> DecoderStackStructureConfig {
        PatternedStackConfig::new(
            vec![linear_layer(), layer(None)],
            vec![0, 0, 1, 0, 1],
            BunsenNormalizationConfig::Rms(RmsNormConfig::new(0)),
            RotaryEmbeddingConfig::new(0, 8),
        )
        .to_structure()
    }

    /// Each layer's mixer takes the next slot of its kind's cache: the KV
    /// cache holds the attention layers alone, and a decode cache a state
    /// for each linear layer.
    #[test]
    fn test_hybrid_slots_and_caches() {
        let device = crate::support::testing::cpu_device();
        let config = hybrid();
        assert_eq!(config.layer_attends(), [false, false, true, false, true]);
        assert_eq!(config.layer_windows(), [None; 5]);
        assert_eq!((config.n_kv_heads(), config.head_dim()), (2, 8));
        let stack: DecoderStack = config.init(&device);
        assert_eq!(stack.layer_attends(), config.layer_attends());
        let slots: Vec<_> = stack
            .layers
            .iter()
            .map(|l| match &l.attn.branch {
                TokenMixer::Attention(a) => (true, a.layer_index),
                TokenMixer::GatedDeltaNet(n) => (false, n.layer_index),
            })
            .collect();
        assert_eq!(
            slots,
            [(false, 0), (false, 1), (true, 0), (false, 2), (true, 1)]
        );
        use crate::ops::transformers::attention::KVCacheMeta;
        assert_eq!(stack.kv_cache_config(4).init().num_layers(), 2);
        let cache = stack.decode_cache(3, 8, KVCacheStorage::Auto, &device);
        assert_eq!(cache.linear.layers.len(), 3);
        assert_eq!(cache.linear.layers[0].rows(), 3);
        assert!(
            uniform(2)
                .to_structure()
                .init(&device)
                .decode_cache(1, 4, KVCacheStorage::Auto, &device)
                .linear
                .is_empty()
        );
    }

    /// A hybrid stack's cached decode, a prefill and then a token at a time,
    /// is its uncached pass; with its rows reordered mid-decode (beams
    /// branching), the uncached pass of the reordered rows.
    #[test]
    #[serial]
    fn test_hybrid_cached_matches_uncached() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        device.seed(9);
        let stack: DecoderStack = hybrid().init(&device);
        let n = 7;
        let x: Tensor<3> = seeded_tensor(12, [2, n, 16], Distribution::Default, &device);
        let swapped = Tensor::cat(
            vec![x.clone().slice_dim(0, 1..2), x.clone().slice_dim(0, 0..1)],
            0,
        );
        let r_emb = stack.rotary(n, &device);
        let whole = stack.forward_hidden(x.clone(), &r_emb, None);
        let whole_swapped = stack.forward_hidden(swapped.clone(), &r_emb, None);

        let mut cache = stack.decode_cache(2, n, KVCacheStorage::Auto, &device);
        let mut parts = vec![stack.forward_cached(x.clone().slice_dim(1, 0..3), &mut cache)];
        for pos in 3..n {
            parts.push(stack.forward_cached(x.clone().slice_dim(1, pos..pos + 1), &mut cache));
        }
        let tolerance = Tolerance::rel_abs(1e-4, 1e-4);
        assert_tensors_close(&Tensor::cat(parts, 1), &whole, tolerance);

        let mut cache = stack.decode_cache(2, n, KVCacheStorage::Auto, &device);
        let _ = stack.forward_cached(x.clone().slice_dim(1, 0..3), &mut cache);
        cache.reorder(&[1, 0]);
        let rest: Vec<_> = (3..n)
            .map(|pos| stack.forward_cached(swapped.clone().slice_dim(1, pos..pos + 1), &mut cache))
            .collect();
        assert_tensors_close(
            &Tensor::cat(rest, 1),
            &whole_swapped.slice_dim(1, 3..n),
            tolerance,
        );
    }

    /// A hybrid decode rewound to its snapshot goes on as though the
    /// positions after it were never run: fed another continuation, it is
    /// the uncached pass of that conversation. Without a snapshot it can
    /// rewind only to 0, and past one only to it.
    #[test]
    #[serial]
    fn test_hybrid_rewinds_to_its_snapshot() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        device.seed(10);
        let stack: DecoderStack = hybrid().init(&device);
        let n = 7;
        let x: Tensor<3> = seeded_tensor(13, [1, n, 16], Distribution::Default, &device);
        let y: Tensor<3> = seeded_tensor(14, [1, n, 16], Distribution::Default, &device);
        let joined = Tensor::cat(
            vec![x.clone().slice_dim(1, 0..3), y.clone().slice_dim(1, 3..n)],
            1,
        );
        let r_emb = stack.rotary(n, &device);
        let expected = stack
            .forward_hidden(joined, &r_emb, None)
            .slice_dim(1, 3..n);

        let mut cache = stack.decode_cache(1, n, KVCacheStorage::Auto, &device);
        let _ = stack.forward_cached(x.clone().slice_dim(1, 0..2), &mut cache);
        assert_eq!(cache.rewind_point(1), 0, "no snapshot: only a reset");
        let _ = stack.forward_cached(x.clone().slice_dim(1, 2..3), &mut cache);
        cache.snapshot();
        let _ = stack.forward_cached(x.slice_dim(1, 3..5), &mut cache);
        assert_eq!(cache.rewind_point(4), 3, "back to the snapshot at most");
        assert_eq!(cache.rewind_point(2), 0, "before the snapshot: a reset");
        assert!(cache.can_truncate(5) && cache.can_truncate(3));
        cache.truncate(3);
        assert_eq!(cache.pos(), 3);
        let got = stack.forward_cached(y.slice_dim(1, 3..n), &mut cache);
        assert_tensors_close(&got, &expected, Tolerance::rel_abs(1e-4, 1e-4));
    }

    /// A hybrid stack refuses a KV cache without its linear layers' states:
    /// that decode would drop them.
    #[test]
    #[should_panic(expected = "decodes through a DeltaNetCache")]
    fn test_hybrid_refuses_a_kv_cache_alone() {
        let device = crate::support::testing::cpu_device();
        let stack: DecoderStack = hybrid().init(&device);
        let mut cache = stack.kv_cache_config(4).init();
        let _ = stack.forward_hidden(
            Tensor::zeros([1, 2, 16], &device),
            &stack.rotary(4, &device),
            Some(&mut cache),
        );
    }

    /// The weights at a layer are its attention's over that layer's input,
    /// cut to the rows and columns asked for, in the order asked for.
    #[test]
    fn test_attention_weights() {
        let device = performance_device();
        let stack: DecoderStack = uniform(3).init(&device);
        let x: Tensor<3> = seeded_tensor(3, [1, 5, 16], Distribution::Default, &device);
        let r_emb = stack.rotary(5, &device);

        let weights = stack.attention_weights(x.clone(), &r_emb, &[1, 0], 2..5, 0..3);
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].dims(), [1, 4, 3, 3]);

        let first = &stack.layers[0];
        let expected0 = first
            .attn
            .branch
            .attention()
            .attention_weights(first.attn_input(x.clone()), r_emb.for_layer(0))
            .slice(s![.., .., 2..5, 0..3]);
        let second = &stack.layers[1];
        let x1 = first.forward(x, r_emb.for_layer(0), &mut None);
        let expected1 = second
            .attn
            .branch
            .attention()
            .attention_weights(second.attn_input(x1), r_emb.for_layer(0))
            .slice(s![.., .., 2..5, 0..3]);
        let tolerance = Tolerance::default();
        assert_tensors_close(&weights[0], &expected1, tolerance);
        assert_tensors_close(&weights[1], &expected0, tolerance);
    }

    /// With no sides it is the plain pass; a mix is the stream mixed before
    /// each layer, the backout after the last; a condition reaches every
    /// layer that takes one.
    #[test]
    fn test_forward_with_sides() {
        let device = performance_device();
        device.seed(9);
        let stack: DecoderStack = uniform(3).init(&device);
        let x: Tensor<3> = seeded_tensor(4, [1, 5, 16], Distribution::Default, &device);
        let r_emb = stack.rotary(5, &device);
        let tolerance = Tolerance::rel_abs(1e-4, 1e-5);

        assert_tensors_close(
            &stack.forward_with(x.clone(), &r_emb, None, DecoderStackSides::default()),
            &stack.forward_hidden(x.clone(), &r_emb, None),
            tolerance,
        );

        let resid = Tensor::<1>::from_floats([0.9, 1.1, 0.8], &device);
        let x0_lambdas = Tensor::<1>::from_floats([0.1, 0.0, 0.3], &device);
        let backout = Tensor::<1>::from_floats([0.25], &device);
        let mixed = stack.forward_with(
            x.clone(),
            &r_emb,
            None,
            DecoderStackSides {
                mix: DecoderStackResidualMix {
                    resid_x0: Some((resid.clone(), x0_lambdas.clone())),
                    backout: Some((1, backout.clone())),
                },
                ..Default::default()
            },
        );
        let mut y = x.clone();
        let mut mid = None;
        for (i, layer) in stack.layers.iter().enumerate() {
            let at = |t: &Tensor<1>| t.clone().slice(s![i..i + 1]).unsqueeze::<3>();
            y = y * at(&resid) + x.clone() * at(&x0_lambdas);
            y = layer.forward(y, r_emb.for_layer(0), &mut None);
            if i == 1 {
                mid = Some(y.clone());
            }
        }
        let expected = stack
            .norm
            .forward(y - mid.unwrap() * backout.unsqueeze::<3>());
        assert_tensors_close(&mixed, &expected, tolerance);

        // A conditioned stack: every layer scales by the condition.
        let scaled = UniformStackConfig {
            layer: layer(None).with_mlp_scale(Some(
                crate::blocks::transformers::conditioning::AdaptiveScaleConfig::new(16, 8),
            )),
            ..uniform(2)
        };
        let stack: DecoderStack = scaled.init(&device);
        let cond: Tensor<3> = seeded_tensor(6, [1, 1, 16], Distribution::Default, &device);
        let got = stack.forward_with(
            x.clone(),
            &r_emb,
            None,
            DecoderStackSides {
                cond: Some(cond.clone()),
                ..Default::default()
            },
        );
        let mut y = x;
        for layer in &stack.layers {
            y = layer.forward_conditioned(y, r_emb.for_layer(0), &mut None, Some(cond.clone()));
        }
        assert_tensors_close(&got, &stack.norm.forward(y), tolerance);
    }
}
