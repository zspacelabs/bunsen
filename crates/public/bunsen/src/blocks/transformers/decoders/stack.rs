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

use std::ops::Range;

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
            attention::csa::CausalSelfAttentionMeta,
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
    ops::transformers::attention::{
        KVCache,
        KVCacheConfig,
        KVCacheStorage,
        length_mask,
    },
};

/// Common meta for [`DecoderStack`] and its configs.
pub trait DecoderStackMeta {
    /// The model width.
    fn d_model(&self) -> usize;

    /// The layer count.
    fn n_layers(&self) -> usize;

    /// The key/value head count of the first layer, which a shared KV cache
    /// is shaped for.
    fn n_kv_heads(&self) -> usize;

    /// The head width of the first layer.
    fn head_dim(&self) -> usize;

    /// Each layer's sliding attention window, if it has one.
    fn layer_windows(&self) -> Vec<Option<usize>>;

    /// The [`KVCacheConfig`] the stack decodes with: its key/value heads,
    /// head width and layer count, at batch 1, `pre_size` positions up
    /// front, uncapped; a windowed layer keeping only its window.
    fn kv_cache_config(
        &self,
        pre_size: usize,
    ) -> KVCacheConfig {
        let windows = self.layer_windows();
        KVCacheConfig::for_decode(
            self.n_kv_heads(),
            pre_size,
            self.head_dim(),
            self.n_layers(),
        )
        .with_layer_windows(if windows.iter().all(Option::is_none) {
            Vec::new()
        } else {
            windows
        })
    }
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
    /// The first layer, whose geometry is the stack's.
    fn first_layer(&self) -> &CausalDecoderBlockConfig {
        &self.layers[0]
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
        self.first_layer().attn.n_kv_head()
    }

    fn head_dim(&self) -> usize {
        self.first_layer().attn.head_dim()
    }

    fn layer_windows(&self) -> Vec<Option<usize>> {
        self.layers.iter().map(|l| l.attn.sliding_window).collect()
    }
}

impl ModuleInit<DecoderStack> for DecoderStackStructureConfig {
    /// Initializes each layer at its index, and the final norm at the
    /// stack's width.
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
        Ok(DecoderStack {
            layers: self
                .layers
                .iter()
                .enumerate()
                .map(|(i, layer)| layer.try_init(i, device))
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
        self.layers[0].attn.branch.n_kv_head()
    }

    fn head_dim(&self) -> usize {
        self.layers[0].attn.branch.head_dim
    }

    fn layer_windows(&self) -> Vec<Option<usize>> {
        self.layers
            .iter()
            .map(|l| l.attn.branch.sliding_window)
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
/// its KV cache, and the rotary table for every position it may reach.
///
/// Made by [`DecoderStack::decode_cache`], run through by
/// [`DecoderStack::forward_cached`]. The stack never owns one, so one loaded
/// model serves any number of decodes.
#[derive(Module, Debug)]
pub struct DecoderStackCache {
    /// The KV cache, one batch row per decode row.
    pub cache: KVCache,

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
    }
}

impl DecoderStack {
    /// A decode cache for `rows` rows of at most `positions` positions each,
    /// under `storage`: the KV cache allocated whole and capped there, as
    /// the rotary table is.
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
        self.forward_hidden(x, &cache.rotary, Some(&mut cache.cache))
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
            tables: std::iter::once(&self.rotary)
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
    pub fn forward_with(
        &self,
        x: Tensor<3>,
        r_emb: &RotarySet,
        mut kv_cache: Option<&mut KVCache>,
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
            x = layer.forward_with(
                x,
                r_emb.for_layer(i),
                &mut kv_cache,
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
    /// If a layer index is past the last layer.
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
                .with_sliding_window(window),
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
            .map(|l| l.attn.sliding_window)
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
            CausalSelfAttentionConfig::new(4, 2, 24).with_head_dim(Some(8)),
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
            .attention_weights(first.attn_input(x.clone()), r_emb.for_layer(0))
            .slice(s![.., .., 2..5, 0..3]);
        let second = &stack.layers[1];
        let x1 = first.forward(x, r_emb.for_layer(0), &mut None);
        let expected1 = second
            .attn
            .branch
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
