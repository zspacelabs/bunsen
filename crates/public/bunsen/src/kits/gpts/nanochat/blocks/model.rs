//! # GPT Module

use burn::{
    Tensor,
    module::Module,
    nn::{
        Embedding,
        EmbeddingConfig,
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
    },
    tensor::Device,
};

use crate::{
    blocks::transformers::{
        attention::csa::{
            CausalSelfAttentionConfig,
            CausalSelfAttentionMeta,
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
    errors::BunsenResult,
    kits::gpts::nanochat::blocks::{
        NanoChatGptBlock,
        NanoChatGptBlockConfig,
    },
    ops::{
        norm::{
            RmsNormOptions,
            rms_norm,
        },
        transformers::attention::{
            KVCache,
            KVCacheConfig,
        },
    },
};

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
}

/// High-level GPT Config.
///
/// User-facing configuration for the nanoChat GPT model, exposing the common
/// hyperparameters (sizes, layer/head counts, vocabulary, softcap): the
/// policy of `NanoChatGpt`'s Stacked Config. It implements
/// [`ToStructureConfig`], expanding into a [`NanoChatGptStructureConfig`], and
/// gets [`ModuleInit`] from that trait's blanket impl, so `.init(&device)`
/// builds the [`NanoChatGpt`] module directly.
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
    #[config(default = "NormalizationConfig::Rms(RmsNormConfig::new(0))")]
    pub norm: NormalizationConfig,
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
}

impl NanoChatGptContractConfig {
    /// Builds the [`NanoChatGptBlockConfig`] for this config.
    pub fn block_config(&self) -> NanoChatGptBlockConfig {
        NanoChatGptBlockConfig::new(
            CausalSelfAttentionConfig::new(self.n_head, self.n_kv_head, self.n_embed)
                .with_norm(self.norm.clone()),
            MlpConfig::new(self.n_embed)
                .with_expansion_factor(self.expansion_factor)
                .with_activation(self.activation.clone())
                .with_act_exponent(Some(2.0)),
        )
        .with_norm(self.norm.clone())
    }
}

impl ToStructureConfig for NanoChatGptContractConfig {
    type Structure = NanoChatGptStructureConfig;

    fn try_to_structure(&self) -> BunsenResult<NanoChatGptStructureConfig> {
        let block_config = self.block_config();
        Ok(NanoChatGptStructureConfig {
            wte: EmbeddingConfig::new(self.vocab_size, self.n_embed),
            h: (0..self.n_layer).map(|_| block_config.clone()).collect(),
            lm_head: LinearConfig::new(self.n_embed, self.vocab_size),
            r_emb: RotaryEmbeddingConfig::new(self.max_seq_len(), self.head_dim()),
            norm: self.norm.clone(),
            init_seq_len: self.init_seq_len,
            softcap: self.softcap,
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
    pub h: Vec<NanoChatGptBlockConfig>,

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
    #[config(default = "NormalizationConfig::Rms(RmsNormConfig::new(0))")]
    pub norm: NormalizationConfig,
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
}

impl ModuleInit<NanoChatGpt> for NanoChatGptStructureConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<NanoChatGpt> {
        let n_embed = self.n_embed();
        Ok(NanoChatGpt {
            wte: self.wte.init(device),
            h: self
                .h
                .iter()
                .enumerate()
                .map(|(layer_idx, c)| c.try_init(layer_idx, device))
                .collect::<BunsenResult<Vec<NanoChatGptBlock>>>()?,
            h_norm: self.norm.clone().with_num_features(n_embed).init(device),
            lm_head: self.lm_head.init(device),
            r_emb: self.r_emb.try_init(device)?,
            init_seq_len: self.init_seq_len,
            softcap: self.softcap,
        })
    }
}

/// nanoChat GPT language model.
///
/// A decoder-only transformer: token embedding, normalized by a
/// parameter-free [`rms_norm`] as upstream's is, a stack of
/// [`NanoChatGptBlock`] layers, a final normalization, and a linear head,
/// not tied to the embedding, producing softcapped vocabulary logits.
/// The normalized embedding starts the residual stream, and each block adds
/// its attention and MLP updates to it.
/// Incremental decoding takes a [`KVCache`] the caller holds, from
/// [`new_kv_cache`](Self::new_kv_cache).
///
/// Built by [`NanoChatGptContractConfig`] (high-level) or
/// [`NanoChatGptStructureConfig`].
#[derive(Module, Debug)]
pub struct NanoChatGpt {
    wte: Embedding,
    h: Vec<NanoChatGptBlock>,
    h_norm: Normalization,
    lm_head: Linear,
    r_emb: RotaryEmbedding,

    init_seq_len: usize,
    softcap: f64,
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
}

impl NanoChatGpt {
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
    /// (0 without a cache) plus `T` exceeds `max_seq_len`.
    pub fn forward(
        &self,
        idx: Tensor<2, Int>,
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
        let mut x = rms_norm(self.wte.forward(idx), &RmsNormOptions::default());

        for block in &self.h {
            x = block.forward(x, &r_emb, kv_cache);
        }
        x = self.h_norm.forward(x);

        let logits = self
            .lm_head
            .forward(x)
            .div_scalar(self.softcap)
            .tanh()
            .mul_scalar(self.softcap);

        assert_shape_contract_periodically!(
            ["B", "T", "V"],
            &logits.dims(),
            &[("B", b), ("T", t), ("V", self.wte.weight.dims()[0])]
        );
        logits
    }

    /// Allocates a new, empty [`KVCache`] for one decode.
    ///
    /// The cache is sized for this model: its KV heads, head width and layer
    /// count, with room for `init_seq_len` positions before it grows. It is
    /// injected state: the caller holds it and passes it to each
    /// [`forward`](Self::forward) of the decode, and the model never owns
    /// one.
    ///
    /// # Arguments
    /// - `batch_size`: the batch size.
    pub fn new_kv_cache(
        &self,
        batch_size: usize,
    ) -> KVCache {
        KVCacheConfig {
            batch_size,
            num_heads: self.n_kv_head(),
            seq_len: self.init_seq_len(),
            head_dim: self.head_dim(),
            num_layers: self.n_layer(),
        }
        .init()
    }

    /// Calculates the estimated FLOPs per token for the model.
    ///
    /// Ref: <https://arxiv.org/abs/2204.02311>
    pub fn estimate_flops_per_token(&self) -> usize {
        let nparams = self.num_params();
        let nparams_embedding = self.wte.num_params();
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
            performance_device,
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
