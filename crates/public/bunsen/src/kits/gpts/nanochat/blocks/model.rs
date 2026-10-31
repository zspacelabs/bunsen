//! # GPT Module

use burn::{
    Tensor,
    module::{
        Module,
        Param,
    },
    nn::{
        Embedding,
        EmbeddingConfig,
        Initializer,
        Linear,
        LinearConfig,
        activation::ActivationConfig,
        norm::{
            Normalization,
            NormalizationConfig,
            RmsNormConfig,
        },
    },
    prelude::{
        Config,
        Int,
        s,
    },
    tensor::{
        Device,
        FloatDType,
        activation::sigmoid,
    },
};

use crate::{
    blocks::transformers::{
        attention::csa::{
            CausalSelfAttentionConfig,
            CausalSelfAttentionMeta,
            ValueResidualConfig,
        },
        decoders::causal::{
            CausalDecoderBlock,
            CausalDecoderBlockConfig,
            DecoderSideInputs,
        },
        embedding::{
            RotaryEmbedding,
            RotaryEmbeddingConfig,
            RotaryEmbeddingMeta,
        },
        mlp::MlpConfig,
    },
    burner::module::{
        ModuleInit,
        ToStructureConfig,
    },
    contracts::{
        assert_shape_contract_periodically,
        unpack_shape_contract,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    ops::{
        norm::{
            RmsNormOptions,
            rms_norm,
        },
        transformers::attention::{
            KVCache,
            KVCacheConfig,
            KVCacheModel,
        },
    },
};

/// The [`KVCacheConfig`] a `NanoChat` model of `meta`'s geometry decodes with:
/// its KV heads, head width and layer count, `init_seq_len` positions up
/// front, capped at `max_seq_len` (the rotary table's length).
fn kv_cache_config(meta: &impl NanoChatGptMeta) -> KVCacheConfig {
    KVCacheConfig::new(
        1,
        meta.n_kv_head(),
        meta.init_seq_len(),
        meta.head_dim(),
        meta.n_layer(),
    )
    .with_max_seq_len(Some(meta.max_seq_len()))
}

impl KVCacheModel for NanoChatGptContractConfig {
    fn kv_cache_config(&self) -> KVCacheConfig {
        kv_cache_config(self)
    }
}

impl KVCacheModel for NanoChatGptStructureConfig {
    fn kv_cache_config(&self) -> KVCacheConfig {
        kv_cache_config(self)
    }
}

/// A decode's cache is injected state: the caller holds the one
/// [`new_kv_cache`](KVCacheModel::new_kv_cache) builds and passes it to each
/// [`forward`](NanoChatGpt::forward), and the model never owns one.
impl KVCacheModel for NanoChatGpt {
    fn kv_cache_config(&self) -> KVCacheConfig {
        kv_cache_config(self)
    }
}

/// Common meta for [`NanoChatGpt`], [`NanoChatGptStructureConfig`], and
/// [`NanoChatGptContractConfig`].
pub trait NanoChatGptMeta {
    /// Returns the size of the input and output.
    fn n_embed(&self) -> usize;

    /// Returns the number of heads.
    fn n_head(&self) -> usize;

    /// Returns the number of KV heads.
    fn n_kv_head(&self) -> usize;

    /// Returns the size of each head.
    fn head_dim(&self) -> usize {
        self.n_embed() / self.n_head()
    }

    /// Returns the initial sequence length.
    fn init_seq_len(&self) -> usize;

    /// Returns the maximum sequence length.
    fn max_seq_len(&self) -> usize;

    /// Returns the number of layers.
    fn n_layer(&self) -> usize;

    /// Returns the vocabulary size: the logits' width, whatever the
    /// embedding is padded to.
    fn vocab_size(&self) -> usize;
}

/// Whether nanochat's layer `layer` of `n_layer` has a value embedding:
/// every other layer, the last always (upstream's `has_ve`).
pub fn has_value_embed(
    layer: usize,
    n_layer: usize,
) -> bool {
    layer % 2 == (n_layer - 1) % 2
}

/// Each layer's sliding window, in keys (a query and the positions before it
/// it may attend to), from a nanochat window pattern: the pattern's letters,
/// tiled across the layers, `L` the long window and `S` the short, the last
/// layer always long. Upstream's long window is `seq_len` positions back,
/// its short one a quarter of that rounded up to 128; a window of `w`
/// positions back spans `w + 1` keys.
///
/// # Errors
/// If the pattern is empty or holds anything but `S` and `L`.
pub fn window_pattern_windows(
    pattern: &str,
    seq_len: usize,
    n_layer: usize,
) -> BunsenResult<Vec<usize>> {
    let pattern = pattern.to_ascii_uppercase();
    if pattern.is_empty() || !pattern.chars().all(|c| c == 'S' || c == 'L') {
        return Err(BunsenError::illegal(format!(
            "a window pattern is letters S and L; got {pattern:?}"
        )));
    }
    let long = seq_len;
    let short = seq_len.div_ceil(4).div_ceil(128) * 128;
    let letters: Vec<char> = pattern.chars().collect();
    let mut windows: Vec<usize> = (0..n_layer)
        .map(|i| match letters[i % letters.len()] {
            'S' => short + 1,
            _ => long + 1,
        })
        .collect();
    if let Some(last) = windows.last_mut() {
        *last = long + 1;
    }
    Ok(windows)
}

/// High-level GPT Config.
///
/// User-facing configuration for the nanoChat GPT model, exposing the common
/// hyperparameters (sizes, layer/head counts, vocabulary, softcap): the
/// policy of `NanoChatGpt`'s Stacked Config. It implements
/// [`ToStructureConfig`], expanding into a [`NanoChatGptStructureConfig`], and
/// gets [`ModuleInit`] from that trait's blanket impl, so `.init(&device)`
/// builds the [`NanoChatGpt`] module directly.
///
/// The defaults are upstream nanochat's 2025 model. Upstream's 2026 model is
/// the same with the options under "2026" on
/// ([`with_upstream_2026`](Self::with_upstream_2026)), each of which is off
/// by default.
#[derive(Config, Debug)]
pub struct NanoChatGptContractConfig {
    /// Initial sequence Length.
    #[config(default = "1024")]
    pub init_seq_len: usize,

    /// Max `seq_len` factor.
    #[config(default = "10")]
    pub max_seq_len_factor: usize,

    /// Vocabulary Size.
    #[config(default = "50304")]
    pub vocab_size: usize,

    /// Number of Blocks.
    #[config(default = "12")]
    pub n_layer: usize,

    /// Number of Query Heads.
    #[config(default = "6")]
    pub n_head: usize,

    /// Number of KV Heads.
    #[config(default = "6")]
    pub n_kv_head: usize,

    /// Embedding Size.
    #[config(default = "768")]
    pub n_embed: usize,

    /// Softcap for the logits.
    #[config(default = "15.0")]
    pub softcap: f64,

    /// MLP Expansion Factor.
    #[config(default = "4")]
    pub expansion_factor: usize,

    /// MLP Activation Config.
    #[config(default = "ActivationConfig::Relu")]
    pub activation: ActivationConfig,

    /// Normalization.
    /// This normalization will be adapted to the appropriate feature count.
    #[config(default = "NormalizationConfig::Rms(RmsNormConfig::new(0).with_epsilon(1e-6))")]
    pub norm: NormalizationConfig,

    /// The epsilon of the parameter-free RMS norm on the token embedding;
    /// the default [`norm`](Self::norm) uses the same.
    ///
    /// Upstream's norms take fp32's epsilon (≈1.19e-7); 1e-6 is the
    /// `transformers` port's, and differs from upstream's by under 5e-5,
    /// relatively, at a mean square of 0.01.
    #[config(default = "1e-6")]
    pub norm_eps: f64,

    /// The rotary embedding's base: 10000 in upstream's 2025 model, 100000
    /// in its 2026 one.
    #[config(default = "10000")]
    pub rope_base: usize,

    /// Whether the rotary tables are rounded to bf16 values, as upstream's
    /// 2025 model keeps them whatever it computes in (its checkpoints were
    /// trained with them); its 2026 model builds them in the compute dtype.
    #[config(default = "true")]
    pub rope_bf16_tables: bool,

    /// Whether the head carries a bias. Upstream's has none.
    #[config(default = "false")]
    pub head_bias: bool,

    /// 2026: pad the embedding and the head to a multiple of this many
    /// tokens (upstream's 64); the logits are sliced back to `vocab_size`.
    #[config(default = "None")]
    pub pad_vocab_to: Option<usize>,

    /// 2026: before each block `i`, `x = resid_lambdas[i] * x +
    /// x0_lambdas[i] * x0`, `x0` the normalized (and smeared) embedding.
    #[config(default = "false")]
    pub residual_lambdas: bool,

    /// 2026: a value embedding on every other layer, the last always
    /// ([`has_value_embed`]), added to the attention's values through a gate
    /// ([`ValueResidualConfig`]).
    #[config(default = "false")]
    pub value_embeds: bool,

    /// 2026: the smear: each position adds the previous position's
    /// embedding, gated by `smear_lambda * sigmoid(smear_gate(x[..,
    /// :channels]))`; `Some(channels)` (upstream's 24).
    #[config(default = "None")]
    pub smear_channels: Option<usize>,

    /// 2026: the residual stream after block `n_layer / 2`, times
    /// `backout_lambda`, is subtracted before the final norm.
    #[config(default = "false")]
    pub backout: bool,

    /// 2026: the queries' and keys' gain after their norm (upstream's 1.2).
    #[config(default = "None")]
    pub qk_gain: Option<f64>,

    /// 2026: per-layer sliding windows from a pattern of `S` and `L`
    /// ([`window_pattern_windows`]), over a long window of `init_seq_len`;
    /// `None` attends to every earlier position.
    #[config(default = "None")]
    pub window_pattern: Option<String>,
}

impl NanoChatGptMeta for NanoChatGptContractConfig {
    fn n_embed(&self) -> usize {
        self.n_embed
    }

    fn n_head(&self) -> usize {
        self.n_head
    }

    fn n_kv_head(&self) -> usize {
        self.n_kv_head
    }

    fn init_seq_len(&self) -> usize {
        self.init_seq_len
    }

    fn max_seq_len(&self) -> usize {
        self.init_seq_len() * self.max_seq_len_factor
    }

    fn n_layer(&self) -> usize {
        self.n_layer
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl NanoChatGptContractConfig {
    /// Upstream nanochat's 2026 model (`karpathy/nanochat` at `a3ca42a6`):
    /// this config with rotary base 100000 and exact rotary tables, the
    /// vocabulary padded to 64,
    /// the residual lambdas, the value embeddings, the smear over 24
    /// channels, the backout, a query/key gain of 1.2, and the windows of
    /// `window_pattern`.
    pub fn with_upstream_2026(
        self,
        window_pattern: impl Into<String>,
    ) -> Self {
        self.with_rope_base(100000)
            .with_rope_bf16_tables(false)
            .with_pad_vocab_to(Some(64))
            .with_residual_lambdas(true)
            .with_value_embeds(true)
            .with_smear_channels(Some(24))
            .with_backout(true)
            .with_qk_gain(Some(1.2))
            .with_window_pattern(Some(window_pattern.into()))
    }

    /// The embedding and head's vocabulary: `vocab_size`, padded to a
    /// multiple of `pad_vocab_to` when that is set.
    pub fn padded_vocab_size(&self) -> usize {
        match self.pad_vocab_to {
            Some(m) if m > 0 => self.vocab_size.div_ceil(m) * m,
            _ => self.vocab_size,
        }
    }

    /// Builds the [`CausalDecoderBlockConfig`] for this config, with no
    /// sliding window and no value residual: the layer-independent part.
    pub fn block_config(&self) -> CausalDecoderBlockConfig {
        CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(self.n_head, self.n_kv_head, self.n_embed)
                .with_norm(Some(self.norm.clone()))
                .with_qk_gain(self.qk_gain),
            MlpConfig::new(self.n_embed)
                .with_expansion_factor(self.expansion_factor)
                .with_activation(self.activation.clone())
                .with_act_exponent(Some(2.0))
                .into(),
        )
        .with_norm(self.norm.clone())
    }
}

impl ToStructureConfig for NanoChatGptContractConfig {
    type Structure = NanoChatGptStructureConfig;

    fn try_to_structure(&self) -> BunsenResult<NanoChatGptStructureConfig> {
        let windows = self
            .window_pattern
            .as_deref()
            .map(|p| window_pattern_windows(p, self.init_seq_len, self.n_layer))
            .transpose()?;
        let h = (0..self.n_layer)
            .map(|i| {
                let mut block = self.block_config();
                block.attn.sliding_window = windows.as_ref().map(|w| w[i]);
                if self.value_embeds && has_value_embed(i, self.n_layer) {
                    block.attn.value_residual = Some(ValueResidualConfig::new());
                }
                block
            })
            .collect();
        let vocab = self.padded_vocab_size();
        let kv_width = self.n_kv_head * self.head_dim();
        Ok(NanoChatGptStructureConfig {
            wte: EmbeddingConfig::new(vocab, self.n_embed),
            h,
            lm_head: LinearConfig::new(self.n_embed, vocab).with_bias(self.head_bias),
            r_emb: RotaryEmbeddingConfig::new(self.max_seq_len(), self.head_dim())
                .with_base(self.rope_base)
                .with_round_tables_to(self.rope_bf16_tables.then_some(FloatDType::BF16)),
            norm: self.norm.clone(),
            init_seq_len: self.init_seq_len,
            softcap: self.softcap,
            norm_eps: self.norm_eps,
            vocab_size: self.vocab_size,
            residual_lambdas: self.residual_lambdas,
            value_embeds: (0..self.n_layer)
                .map(|i| {
                    (self.value_embeds && has_value_embed(i, self.n_layer))
                        .then(|| EmbeddingConfig::new(vocab, kv_width))
                })
                .collect(),
            smear_channels: self.smear_channels,
            backout: self.backout,
        })
    }
}

/// Low-level GPT Structure Config.
///
/// The fully-expanded structural configuration for the [`NanoChatGpt`] module,
/// holding the explicit sub-configs (embedding, per-layer blocks, head, rotary
/// embedding). [`NanoChatGptContractConfig`] lowers to it. Directly builds the
/// [`NanoChatGpt`] module via [`ModuleInit`].
///
/// This config has a lot of duplicate information.
#[derive(Config, Debug)]
pub struct NanoChatGptStructureConfig {
    /// The embedding config.
    pub wte: EmbeddingConfig,

    /// The main transformer block sequence.
    pub h: Vec<CausalDecoderBlockConfig>,

    /// The config for the final linear layer.
    pub lm_head: LinearConfig,

    /// The config for the rotary embedding.
    pub r_emb: RotaryEmbeddingConfig,

    /// Initial sequence Length.
    #[config(default = "1024")]
    pub init_seq_len: usize,

    /// Softcap for the logits.
    #[config(default = "15.0")]
    pub softcap: f64,

    /// Normalization.
    /// This normalization will be adapted to the appropriate feature count.
    #[config(default = "NormalizationConfig::Rms(RmsNormConfig::new(0).with_epsilon(1e-6))")]
    pub norm: NormalizationConfig,

    /// The epsilon of the parameter-free RMS norm on the token embedding.
    #[config(default = "1e-6")]
    pub norm_eps: f64,

    /// The logits' width: the embedding's, unpadded.
    pub vocab_size: usize,

    /// Whether the residual stream is rescaled and mixed with the
    /// embedding before each block.
    #[config(default = "false")]
    pub residual_lambdas: bool,

    /// Each layer's value embedding, if it has one.
    #[config(default = "Vec::new()")]
    pub value_embeds: Vec<Option<EmbeddingConfig>>,

    /// The smear's gate width, if the embedding is smeared.
    #[config(default = "None")]
    pub smear_channels: Option<usize>,

    /// Whether the mid-stack residual is backed out before the final norm.
    #[config(default = "false")]
    pub backout: bool,
}

impl NanoChatGptMeta for NanoChatGptStructureConfig {
    fn n_embed(&self) -> usize {
        self.wte.d_model
    }

    fn n_head(&self) -> usize {
        self.h[0].attn.n_head()
    }

    fn n_kv_head(&self) -> usize {
        self.h[0].attn.n_kv_head()
    }

    fn head_dim(&self) -> usize {
        self.h[0].attn.head_dim()
    }

    fn init_seq_len(&self) -> usize {
        self.init_seq_len
    }

    fn max_seq_len(&self) -> usize {
        self.r_emb.seq_len()
    }

    fn n_layer(&self) -> usize {
        self.h.len()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl ModuleInit<NanoChatGpt> for NanoChatGptStructureConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<NanoChatGpt> {
        let n_embed = self.n_embed();
        let n_layer = self.h.len();
        if !self.value_embeds.is_empty() && self.value_embeds.len() != n_layer {
            return Err(BunsenError::illegal(format!(
                "value_embeds has {} entries for {n_layer} layers",
                self.value_embeds.len()
            )));
        }
        // Upstream's initial values: a stronger residual and more of the
        // embedding at early layers.
        let ramp = |start: f64, drop: f64| {
            let values: Vec<f32> = (0..n_layer)
                .map(|i| (start - drop * i as f64 / (n_layer.max(2) - 1) as f64) as f32)
                .collect();
            Param::from_tensor(Tensor::<1>::from_floats(values.as_slice(), device))
        };
        let scalar = |v: f32| Param::from_tensor(Tensor::<1>::from_floats([v], device));
        Ok(NanoChatGpt {
            wte: self.wte.init(device),
            h: self
                .h
                .iter()
                .enumerate()
                .map(|(layer_idx, c)| c.try_init(layer_idx, device))
                .collect::<BunsenResult<Vec<CausalDecoderBlock>>>()?,
            h_norm: self.norm.clone().with_num_features(n_embed).init(device),
            lm_head: self.lm_head.init(device),
            r_emb: self.r_emb.try_init(device)?,
            init_seq_len: self.init_seq_len,
            softcap: self.softcap,
            norm_eps: self.norm_eps,
            vocab_size: self.vocab_size,
            resid_lambdas: self.residual_lambdas.then(|| ramp(1.15, 0.10)),
            x0_lambdas: self.residual_lambdas.then(|| ramp(0.20, 0.15)),
            value_embeds: self
                .value_embeds
                .iter()
                .map(|c| c.as_ref().map(|c| c.init(device)))
                .collect(),
            smear_gate: self.smear_channels.map(|channels| {
                LinearConfig::new(channels, 1)
                    .with_bias(false)
                    .with_initializer(Initializer::Uniform {
                        min: 0.0,
                        max: 0.02,
                    })
                    .init(device)
            }),
            smear_lambda: self.smear_channels.map(|_| scalar(0.0)),
            backout_lambda: self.backout.then(|| scalar(0.2)),
        })
    }
}

/// nanoChat GPT language model.
///
/// A decoder-only transformer: token embedding, normalized by a
/// parameter-free [`rms_norm`] as upstream's is, a stack of
/// [`CausalDecoderBlock`] layers, a final normalization, and a linear head,
/// not tied to the embedding, producing softcapped vocabulary logits.
/// The normalized embedding starts the residual stream, and each block adds
/// its attention and MLP updates to it.
///
/// Upstream's 2026 additions, each present only when its config option is
/// on: the smear (the previous position's embedding mixed into each
/// position's), per-layer residual and embedding lambdas before each block,
/// value embeddings on every other layer, the mid-stack backout, a
/// query/key gain, sliding windows, and a padded vocabulary.
///
/// Incremental decoding takes a [`KVCache`] the caller holds, from
/// [`new_kv_cache`](KVCacheModel::new_kv_cache) (it implements
/// [`KVCacheModel`], as its configs do).
///
/// Built by [`NanoChatGptContractConfig`] (high-level) or
/// [`NanoChatGptStructureConfig`].
#[derive(Module, Debug)]
pub struct NanoChatGpt {
    wte: Embedding,
    h: Vec<CausalDecoderBlock>,
    h_norm: Normalization,
    lm_head: Linear,
    r_emb: RotaryEmbedding,

    init_seq_len: usize,
    softcap: f64,
    norm_eps: f64,
    vocab_size: usize,

    /// `[n_layer]`: the residual stream's scale before each block.
    resid_lambdas: Option<Param<Tensor<1>>>,
    /// `[n_layer]`: the embedding's weight mixed in before each block.
    x0_lambdas: Option<Param<Tensor<1>>>,
    /// Each layer's value embedding, `[vocab, n_kv_head * head_dim]`, if it
    /// has one.
    value_embeds: Vec<Option<Embedding>>,
    /// The smear's gate, from the embedding's first channels to one scalar.
    smear_gate: Option<Linear>,
    /// `[1]`: the smear's strength.
    smear_lambda: Option<Param<Tensor<1>>>,
    /// `[1]`: how much of the mid-stack residual is backed out.
    backout_lambda: Option<Param<Tensor<1>>>,
}

impl NanoChatGptMeta for NanoChatGpt {
    fn n_embed(&self) -> usize {
        // burn's `Embedding` weight is `[n_embedding, d_model]`.
        self.wte.weight.dims()[1]
    }

    fn n_head(&self) -> usize {
        self.h[0].attn.n_head()
    }

    fn n_kv_head(&self) -> usize {
        self.h[0].attn.n_kv_head()
    }

    fn head_dim(&self) -> usize {
        self.h[0].attn.head_dim()
    }

    fn init_seq_len(&self) -> usize {
        self.init_seq_len
    }

    fn max_seq_len(&self) -> usize {
        self.r_emb.seq_len()
    }

    fn n_layer(&self) -> usize {
        self.h.len()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl NanoChatGpt {
    /// The token embedding: its parameter's dtype is the model's.
    #[cfg(feature = "store_pytorch")]
    pub(crate) fn wte(&self) -> &Embedding {
        &self.wte
    }

    /// Whether the embedding is smeared: then a cached step past the first
    /// position needs the token before it
    /// ([`forward_after`](Self::forward_after)).
    pub fn is_smeared(&self) -> bool {
        self.smear_gate.is_some()
    }

    /// The parameter-free norm on the embedding (and nowhere else).
    fn embed_norm(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        rms_norm(
            x,
            &RmsNormOptions {
                eps: self.norm_eps as f32,
            },
        )
    }

    /// Forward Pass.
    ///
    /// # Arguments
    /// - `idx`: a `[B, T]` tensor of token ids.
    /// - `kv_cache`: the decode's cache, or `None` to run the sequence on its
    ///   own. With a cache, the step's positions start at the cache's position,
    ///   and every layer appends its keys and values.
    ///
    /// # Returns
    /// The `[B, T, vocab_size]` logits, softcapped:
    /// `softcap * tanh(logits / softcap)`.
    ///
    /// # Panics
    /// When the step runs past the rotary table: when the cache's position
    /// (0 without a cache) plus `T` exceeds `max_seq_len`. For a smeared
    /// model, when the cache's position is past 0: that needs
    /// [`forward_after`](Self::forward_after).
    pub fn forward(
        &self,
        idx: Tensor<2, Int>,
        kv_cache: &mut Option<&mut KVCache>,
    ) -> Tensor<3> {
        self.forward_after(idx, None, kv_cache)
    }

    /// Forward Pass, continuing after `prev`: the tokens just before `idx`,
    /// which a smeared model mixes into `idx`'s first position.
    ///
    /// # Arguments
    /// - `idx`: a `[B, T]` tensor of token ids.
    /// - `prev`: the `[B, 1]` token before each row's `idx`, or `None` at the
    ///   start of the sequence. Only a smeared model reads it.
    /// - `kv_cache`: the decode's cache, as for [`forward`](Self::forward).
    ///
    /// # Returns
    /// The `[B, T, vocab_size]` softcapped logits.
    ///
    /// # Panics
    /// As [`forward`](Self::forward); and for a smeared model, when the
    /// cache's position is past 0 and `prev` is `None`.
    pub fn forward_after(
        &self,
        idx: Tensor<2, Int>,
        prev: Option<Tensor<2, Int>>,
        kv_cache: &mut Option<&mut KVCache>,
    ) -> Tensor<3> {
        self.logits(self.hidden_after(idx, prev, kv_cache))
    }

    /// The head's input: the residual stream after the last block (the
    /// backout taken), normalized. As
    /// [`forward_after`](Self::forward_after) without the head.
    ///
    /// # Arguments
    /// As [`forward_after`](Self::forward_after).
    ///
    /// # Returns
    /// The `[B, T, D]` hidden states.
    ///
    /// # Panics
    /// As [`forward_after`](Self::forward_after).
    pub fn hidden_after(
        &self,
        idx: Tensor<2, Int>,
        prev: Option<Tensor<2, Int>>,
        kv_cache: &mut Option<&mut KVCache>,
    ) -> Tensor<3> {
        let [b, t] = unpack_shape_contract!(["B", "T"], &idx.dims());

        let t0 = match kv_cache {
            Some(kv_cache) => kv_cache.pos(),
            None => 0,
        };
        assert!(
            t0 + t <= self.r_emb.seq_len(),
            "Sequence position grew beyond the rotary embeddings table: {t0} + {t} > {}",
            self.r_emb.seq_len()
        );
        let r_emb = self.r_emb.clip_range(t0..t0 + t);

        // As upstream (`x = norm(wte(idx))`): a parameter-free norm, so the
        // residual stream starts normalized.
        let mut x = self.embed_norm(self.wte.forward(idx.clone()));

        if let (Some(gate), Some(lambda)) = (&self.smear_gate, &self.smear_lambda) {
            x = self.smear(x, prev, t0, gate, lambda);
        }

        let x0 = x.clone();
        let backout_layer = self.h.len() / 2;
        let mut x_backout = None;
        for (i, block) in self.h.iter().enumerate() {
            if let (Some(resid), Some(x0_lambdas)) = (&self.resid_lambdas, &self.x0_lambdas) {
                let r = resid.val().slice(s![i..i + 1]).unsqueeze::<3>();
                let l = x0_lambdas.val().slice(s![i..i + 1]).unsqueeze::<3>();
                x = x * r + x0.clone() * l;
            }
            let value_embed = self
                .value_embeds
                .get(i)
                .and_then(Option::as_ref)
                .map(|ve| ve.forward(idx.clone()));
            x = block.forward_with(
                x,
                &r_emb,
                kv_cache,
                DecoderSideInputs {
                    value_embed,
                    ..Default::default()
                },
            );
            if i == backout_layer && self.backout_lambda.is_some() {
                x_backout = Some(x.clone());
            }
        }
        if let (Some(lambda), Some(mid)) = (&self.backout_lambda, x_backout) {
            x = x - mid * lambda.val().unsqueeze::<3>();
        }
        let hidden = self.h_norm.forward(x);
        assert_shape_contract_periodically!(
            ["B", "T", "D"],
            &hidden.dims(),
            &[("B", b), ("T", t), ("D", self.n_embed())]
        );
        hidden
    }

    /// The softcapped logits of `hidden`, the head's input
    /// ([`hidden_after`](Self::hidden_after)): the head, sliced to
    /// `vocab_size`, then `softcap * tanh(logits / softcap)`.
    ///
    /// # Arguments
    /// - `hidden`: `[B, T, D]` hidden states.
    ///
    /// # Returns
    /// The `[B, T, vocab_size]` logits.
    pub fn logits(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        let [b, t] = unpack_shape_contract!(
            ["B", "T", "D"],
            &hidden.dims(),
            &["B", "T"],
            &[("D", self.n_embed())]
        );
        let logits = self.lm_head.forward(hidden);
        let logits = if logits.dims()[2] == self.vocab_size {
            logits
        } else {
            logits.slice(s![.., .., ..self.vocab_size])
        };
        let logits = logits
            .div_scalar(self.softcap)
            .tanh()
            .mul_scalar(self.softcap);

        assert_shape_contract_periodically!(
            ["B", "T", "V"],
            &logits.dims(),
            &[("B", b), ("T", t), ("V", self.vocab_size)]
        );
        logits
    }

    /// The smear: each position adds the one before it, gated by
    /// `lambda * sigmoid(gate(x[.., :channels]))`, the gate read from the
    /// position itself. The sequence's first position has nothing before it;
    /// a chunk past it takes its first position's predecessor from `prev`.
    fn smear(
        &self,
        x: Tensor<3>,
        prev: Option<Tensor<2, Int>>,
        t0: usize,
        gate: &Linear,
        lambda: &Param<Tensor<1>>,
    ) -> Tensor<3> {
        let [b, t, d] = x.dims();
        let channels = gate.weight.dims()[0];
        let g = sigmoid(gate.forward(x.clone().slice(s![.., .., ..channels])))
            * lambda.val().unsqueeze::<3>();
        let before = match (t0, prev) {
            (_, Some(prev)) => {
                let p = self.embed_norm(self.wte.forward(prev));
                Some(Tensor::cat(vec![p, x.clone().slice(s![.., ..t - 1])], 1))
            }
            (0, None) => None,
            (_, None) => panic!(
                "a smeared model continuing a decode at position {t0} needs the token before it"
            ),
        };
        match before {
            Some(before) => x + g * before,
            None if t == 1 => x,
            None => {
                let rest = x.clone().slice(s![.., 1..])
                    + g.slice(s![.., 1..]) * x.clone().slice(s![.., ..t - 1]);
                let out = Tensor::cat(vec![x.slice(s![.., ..1]), rest], 1);
                debug_assert_eq!(out.dims(), [b, t, d]);
                out
            }
        }
    }

    /// Calculates the estimated FLOPs per token for the model.
    ///
    /// Ref: <https://arxiv.org/abs/2204.02311>
    pub fn estimate_flops_per_token(&self) -> usize {
        let nparams = self.num_params();
        let nparams_embedding = self.wte.num_params()
            + self
                .value_embeds
                .iter()
                .flatten()
                .map(|ve| ve.num_params())
                .sum::<usize>();
        let nparams = nparams - nparams_embedding;

        let l = self.n_layer();
        let h = self.n_head();
        let q = self.head_dim();
        let t = self.init_seq_len;

        6 * nparams + 12 * l * h * q * t
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        module::Param,
        tensor::{
            Distribution,
            Tolerance,
        },
    };
    use serial_test::serial;

    use super::*;
    use crate::{
        contracts::assert_shape_contract,
        support::testing::{
            DeviceMemoryGuard,
            cpu_device,
            performance_device,
            seeded_tensor,
        },
    };

    #[test]
    fn test_gpt_config() {
        let cfg = NanoChatGptContractConfig::new();
        assert_eq!(cfg.init_seq_len, 1024);
        assert_eq!(cfg.vocab_size, 50304);
        assert_eq!(cfg.n_layer, 12);
        assert_eq!(cfg.n_head, 6);
        assert_eq!(cfg.n_kv_head, 6);
        assert_eq!(cfg.n_embed, 768);
        assert_eq!(cfg.expansion_factor, 4);

        assert_eq!(cfg.n_embed(), 768);
    }

    /// The defaults are upstream's 2025 model: an unbiased head, rotary base
    /// 10000, an explicit norm epsilon, and none of the 2026 options.
    #[test]
    fn test_defaults_are_upstream_2025() {
        let device = cpu_device();
        let gpt: NanoChatGpt = NanoChatGptContractConfig::new()
            .with_vocab_size(64)
            .with_n_layer(2)
            .with_n_head(2)
            .with_n_kv_head(2)
            .with_n_embed(16)
            .with_init_seq_len(8)
            .init(&device);
        assert!(gpt.lm_head.bias.is_none());
        assert_eq!(gpt.norm_eps, 1e-6);
        assert!(gpt.resid_lambdas.is_none() && gpt.x0_lambdas.is_none());
        assert!(gpt.value_embeds.iter().all(Option::is_none));
        assert!(!gpt.is_smeared() && gpt.backout_lambda.is_none());
        assert!(gpt.h.iter().all(|b| {
            b.attn.qk_gain.is_none()
                && b.attn.sliding_window.is_none()
                && b.attn.value_gate.is_none()
        }));

        let structure = NanoChatGptContractConfig::new().to_structure();
        assert_eq!(structure.r_emb.base, 10000);
        assert_eq!(structure.r_emb.round_tables_to, Some(FloatDType::BF16));
        assert_eq!(structure.vocab_size, 50304);
    }

    /// Upstream's `has_ve`: every other layer, the last always.
    #[test]
    fn test_has_value_embed() {
        let layers = |n: usize| {
            (0..n)
                .filter(|&i| has_value_embed(i, n))
                .collect::<Vec<_>>()
        };
        assert_eq!(layers(18), vec![1, 3, 5, 7, 9, 11, 13, 15, 17]);
        assert_eq!(layers(5), vec![0, 2, 4]);
        assert_eq!(layers(1), vec![0]);
    }

    /// Upstream's `_compute_window_sizes`: the pattern tiled, the last layer
    /// long, the short window a quarter of the long one rounded up to 128;
    /// `w` positions back is `w + 1` keys.
    #[test]
    fn test_window_pattern_windows() {
        assert_eq!(
            window_pattern_windows("SSSL", 2048, 6).unwrap(),
            vec![513, 513, 513, 2049, 513, 2049]
        );
        assert_eq!(window_pattern_windows("l", 2048, 3).unwrap(), vec![2049; 3]);
        assert_eq!(window_pattern_windows("S", 4, 2).unwrap(), vec![129, 5]);
        assert!(window_pattern_windows("", 2048, 2).is_err());
        assert!(window_pattern_windows("SXL", 2048, 2).is_err());
    }

    /// `with_upstream_2026` turns on every 2026 option, with upstream's
    /// values, and the structure builds the parts each needs.
    #[test]
    fn test_upstream_2026_structure() {
        let policy = NanoChatGptContractConfig::new()
            .with_vocab_size(60)
            .with_n_layer(3)
            .with_n_head(2)
            .with_n_kv_head(1)
            .with_n_embed(32)
            .with_init_seq_len(4)
            .with_upstream_2026("SL");
        assert_eq!(policy.padded_vocab_size(), 64);
        let structure = policy.to_structure();
        assert_eq!(structure.r_emb.base, 100000);
        assert_eq!(structure.r_emb.round_tables_to, None);
        assert_eq!(structure.wte.n_embedding, 64);
        assert_eq!(structure.vocab_size, 60);
        assert_eq!(
            structure
                .h
                .iter()
                .map(|b| b.attn.sliding_window)
                .collect::<Vec<_>>(),
            vec![Some(129), Some(5), Some(5)]
        );
        assert_eq!(
            structure
                .value_embeds
                .iter()
                .map(|ve| ve.as_ref().map(|c| (c.n_embedding, c.d_model)))
                .collect::<Vec<_>>(),
            vec![Some((64, 16)), None, Some((64, 16))]
        );
        assert!(structure.h.iter().all(|b| b.attn.qk_gain == Some(1.2)));
        assert_eq!(structure.smear_channels, Some(24));
        assert!(structure.residual_lambdas && structure.backout);
    }

    /// A tiny 2026 model whose options all bind: its smear, value gates and
    /// backout set away from their (neutral or near-neutral) initial values,
    /// and windows shorter than the sequences it is run on.
    fn tiny_2026(device: &Device) -> NanoChatGpt {
        let mut gpt: NanoChatGpt = NanoChatGptContractConfig::new()
            .with_vocab_size(30)
            .with_n_layer(3)
            .with_n_head(4)
            .with_n_kv_head(2)
            .with_n_embed(32)
            .with_init_seq_len(4)
            .with_max_seq_len_factor(4)
            .with_upstream_2026("SL")
            .init(device);
        gpt.smear_lambda = Some(Param::from_tensor(Tensor::from_floats([0.7], device)));
        gpt.backout_lambda = Some(Param::from_tensor(Tensor::from_floats([0.3], device)));
        for block in gpt.h.iter_mut() {
            if let Some(gate) = block.attn.value_gate.as_mut() {
                gate.weight = Param::from_tensor(seeded_tensor(
                    7,
                    gate.weight.dims(),
                    Distribution::Normal(0.0, 1.0),
                    device,
                ));
            }
        }
        if let Some(gate) = gpt.smear_gate.as_mut() {
            gate.weight = Param::from_tensor(seeded_tensor(
                11,
                gate.weight.dims(),
                Distribution::Normal(0.0, 1.0),
                device,
            ));
        }
        gpt
    }

    /// A cached decode of a 2026 model, one token at a time after a prefill,
    /// each step given the token before it, computes the logits the whole
    /// sequence does: the smear, the value embeddings and the windows carry
    /// across steps.
    #[test]
    #[serial]
    fn test_upstream_2026_cached_decode_matches_full_pass() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let gpt = tiny_2026(&device);
        assert!(gpt.is_smeared());
        let ids: Tensor<2, Int> =
            Tensor::from_data([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]], &device);
        let n = ids.dims()[1];
        let full = gpt.forward(ids.clone(), &mut None);
        assert_eq!(full.dims(), [1, n, 30]);

        let prefill = 3;
        let mut cache = gpt.new_kv_cache(1);
        let mut steps =
            vec![gpt.forward(ids.clone().slice(s![.., ..prefill]), &mut Some(&mut cache))];
        for i in prefill..n {
            steps.push(gpt.forward_after(
                ids.clone().slice(s![.., i..i + 1]),
                Some(ids.clone().slice(s![.., i - 1..i])),
                &mut Some(&mut cache),
            ));
        }
        let cached = Tensor::cat(steps, 1);
        cached
            .into_data()
            .assert_approx_eq::<f32>(&full.into_data(), Tolerance::rel_abs(1e-3, 1e-4));
    }

    /// A chunk of several tokens after a cached prefix, given the token
    /// before it, computes what the whole sequence does too.
    #[test]
    #[serial]
    fn test_upstream_2026_cached_chunk_matches_full_pass() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let gpt = tiny_2026(&device);
        let ids: Tensor<2, Int> = Tensor::from_data([[7, 2, 9, 4, 4, 1, 0, 3]], &device);
        let full = gpt.forward(ids.clone(), &mut None);

        let mut cache = gpt.new_kv_cache(1);
        let head = gpt.forward(ids.clone().slice(s![.., ..2]), &mut Some(&mut cache));
        let tail = gpt.forward_after(
            ids.clone().slice(s![.., 2..]),
            Some(ids.clone().slice(s![.., 1..2])),
            &mut Some(&mut cache),
        );
        Tensor::cat(vec![head, tail], 1)
            .into_data()
            .assert_approx_eq::<f32>(&full.into_data(), Tolerance::rel_abs(1e-3, 1e-4));
    }

    /// A smeared model continuing a decode without the token before the
    /// step would silently drop the smear: it panics instead.
    #[test]
    #[serial]
    #[should_panic(expected = "needs the token before it")]
    fn test_smeared_continuation_needs_the_previous_token() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let gpt = tiny_2026(&device);
        let mut cache = gpt.new_kv_cache(1);
        gpt.forward(Tensor::zeros([1, 2], &device), &mut Some(&mut cache));
        gpt.forward(Tensor::zeros([1, 1], &device), &mut Some(&mut cache));
    }

    /// Each 2026 option changes the logits: none of them is dead code.
    #[test]
    #[serial]
    fn test_upstream_2026_options_bind() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let ids: Tensor<2, Int> = Tensor::from_data([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3]], &device);
        let gpt = tiny_2026(&device);
        let base = gpt.forward(ids.clone(), &mut None);

        let differs = |other: &NanoChatGpt, what: &str| {
            let diff = (other.forward(ids.clone(), &mut None) - base.clone())
                .abs()
                .max()
                .into_scalar::<f32>();
            assert!(diff > 1e-4, "{what} changed nothing ({diff})");
        };
        let mut g = tiny_2026(&device);
        g.smear_lambda = Some(Param::from_tensor(Tensor::from_floats([0.0], &device)));
        differs(&g, "the smear");
        let mut g = tiny_2026(&device);
        g.backout_lambda = Some(Param::from_tensor(Tensor::from_floats([0.0], &device)));
        differs(&g, "the backout");
        let mut g = tiny_2026(&device);
        g.x0_lambdas = Some(Param::from_tensor(Tensor::zeros([3], &device)));
        differs(&g, "the x0 lambdas");
        let mut g = tiny_2026(&device);
        for block in g.h.iter_mut() {
            block.attn.qk_gain = None;
        }
        differs(&g, "the qk gain");
        let mut g = tiny_2026(&device);
        for block in g.h.iter_mut() {
            block.attn.sliding_window = None;
        }
        differs(&g, "the windows");
        let mut g = tiny_2026(&device);
        for block in g.h.iter_mut() {
            if let Some(gate) = block.attn.value_gate.as_mut() {
                gate.weight = Param::from_tensor(gate.weight.val().zeros_like().sub_scalar(30.0));
            }
        }
        differs(&g, "the value embeddings");
    }

    #[test]
    #[serial]
    fn test_gpt_forward() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let batch_size = 1;
        let seq_len = 100;
        let n_layer = 4;
        let n_embed = 36;

        let vocab_size = 1000;

        let cfg = NanoChatGptContractConfig::new()
            .with_vocab_size(vocab_size)
            .with_n_embed(n_embed)
            .with_n_layer(n_layer);
        let gpt: NanoChatGpt = cfg.init(&device);

        let mut kv_cache = gpt.new_kv_cache(batch_size);

        let input_tokens = Tensor::<2>::random(
            [batch_size, seq_len],
            Distribution::Uniform(0.0, vocab_size as f64),
            &device,
        )
        .int();

        let logits = gpt.forward(input_tokens, &mut Some(&mut kv_cache));
        assert_shape_contract!(
            ["B", "T", "V"],
            &logits.dims(),
            &[("B", batch_size), ("T", seq_len), ("V", vocab_size)]
        );
        assert_eq!(gpt.n_embed(), n_embed);
    }

    /// Asserts that `a` and `b` answer every [`NanoChatGptMeta`] method alike.
    fn assert_meta_agrees(
        a: &impl NanoChatGptMeta,
        b: &impl NanoChatGptMeta,
    ) {
        assert_eq!(a.n_embed(), b.n_embed());
        assert_eq!(a.n_head(), b.n_head());
        assert_eq!(a.n_kv_head(), b.n_kv_head());
        assert_eq!(a.head_dim(), b.head_dim());
        assert_eq!(a.init_seq_len(), b.init_seq_len());
        assert_eq!(a.max_seq_len(), b.max_seq_len());
        assert_eq!(a.n_layer(), b.n_layer());
    }

    /// The policy builds the same `NanoChatGpt` through its structure as
    /// through the blanket `init`, and the module reads back the geometry
    /// its configs declare.
    #[test]
    #[serial]
    fn test_policy_pathways_agree() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let policy = NanoChatGptContractConfig::new()
            .with_init_seq_len(16)
            .with_max_seq_len_factor(2)
            .with_vocab_size(64)
            .with_n_layer(2)
            .with_n_head(4)
            .with_n_kv_head(2)
            .with_n_embed(32);

        let structure = policy.to_structure();
        assert_meta_agrees(&policy, &structure);
        assert_eq!(structure.max_seq_len(), 32);
        assert_eq!(structure.head_dim(), 8);

        let lowered: NanoChatGpt = structure.init(&device);
        let direct: NanoChatGpt = policy.init(&device);

        assert_meta_agrees(&direct, &lowered);
        assert_meta_agrees(&structure, &lowered);
        assert_eq!(lowered.n_embed(), 32);
    }

    /// A tiny model whose rotary table, `max_seq_len`, is 8 positions.
    fn tiny_gpt(device: &Device) -> NanoChatGpt {
        let gpt: NanoChatGpt = NanoChatGptContractConfig::new()
            .with_init_seq_len(4)
            .with_max_seq_len_factor(2)
            .with_vocab_size(16)
            .with_n_layer(2)
            .with_n_head(2)
            .with_n_kv_head(2)
            .with_n_embed(16)
            .init(device);
        assert_eq!(gpt.max_seq_len(), 8);
        gpt
    }

    /// A cached decode may fill the rotary table exactly.
    #[test]
    #[serial]
    fn test_cached_decode_fills_rotary_table() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let gpt = tiny_gpt(&device);
        let mut cache = gpt.new_kv_cache(1);
        gpt.forward(Tensor::zeros([1, 7], &device), &mut Some(&mut cache));
        let logits = gpt.forward(Tensor::zeros([1, 1], &device), &mut Some(&mut cache));
        assert_eq!(logits.dims(), [1, 1, 16]);
        assert_eq!(cache.pos(), 8);
    }

    /// A cached decode that runs past the rotary table fails at the guard,
    /// which counts the cache's position plus the step's length.
    #[test]
    #[serial]
    #[should_panic(expected = "beyond the rotary embeddings table: 8 + 1 > 8")]
    fn test_cached_decode_past_rotary_table_panics() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let gpt = tiny_gpt(&device);
        let mut cache = gpt.new_kv_cache(1);
        gpt.forward(Tensor::zeros([1, 8], &device), &mut Some(&mut cache));
        gpt.forward(Tensor::zeros([1, 1], &device), &mut Some(&mut cache));
    }

    /// Upstream normalizes the embedding before the first block
    /// (`x = norm(wte(idx))`), so the first block sees the same input, and the
    /// logits are the same, whatever the embedding table's scale. Fed a known
    /// embedding table and the same table times 8, the model must agree.
    #[test]
    #[serial]
    fn test_forward_normalizes_the_embedding() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let mut gpt = tiny_gpt(&device);
        let table: Tensor<2> = Tensor::random([16, 16], Distribution::Normal(0.0, 1.0), &device);
        let tokens: Tensor<2, Int> = Tensor::from_data([[3, 1, 4, 1, 5, 9]], &device);

        gpt.wte.weight = Param::from_tensor(table.clone());
        let logits = gpt.forward(tokens.clone(), &mut None);

        gpt.wte.weight = Param::from_tensor(table.mul_scalar(8.0));
        let scaled = gpt.forward(tokens, &mut None);

        scaled
            .into_data()
            .assert_approx_eq::<f32>(&logits.into_data(), Tolerance::rel_abs(1e-3, 1e-4));
    }
}
