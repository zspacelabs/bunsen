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
        Backend,
        Config,
        Int,
    },
};

use crate::{
    blocks::transformers::{
        attention::{
            CausalSelfAttentionConfig,
            CausalSelfAttentionMeta,
            KVCache,
            KVCacheConfig,
        },
        embedding::{
            RotaryEmbedding,
            RotaryEmbeddingConfig,
            RotaryEmbeddingMeta,
        },
    },
    contracts::{
        assert_shape_contract_periodically,
        unpack_shape_contract,
    },
    kits::gpts::nanochat::{
        MlpConfig,
        NanoChatGptBlock,
        NanoChatGptBlockConfig,
    },
};

/// Common meta for [`NanoChatGpt`] and [`NanoChatGptConfig`].
pub trait NanoChatGptMeta {
    /// Return the size of the input and output.
    fn n_embed(&self) -> usize;

    /// Return the number of heads.
    fn n_head(&self) -> usize;

    /// Return the number of KV heads.
    fn n_kv_head(&self) -> usize;

    /// Return the size of each head.
    fn head_dim(&self) -> usize {
        self.n_embed() / self.n_head()
    }

    /// Return the initial sequence length.
    fn init_seq_len(&self) -> usize;

    /// Return the maximum sequence length.
    fn max_seq_len(&self) -> usize;

    /// Return the number of layers.
    fn n_layer(&self) -> usize;
}

/// High-level GPT Config.
#[derive(Config, Debug)]
pub struct NanoChatGptConfig {
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

impl NanoChatGptMeta for NanoChatGptConfig {
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

impl NanoChatGptConfig {
    /// Initialize a [`NanoChatGpt`].
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> NanoChatGpt<B> {
        self.into_structure().init(device)
    }

    /// Convert this config into a [`NanoChatGptStructureConfig`].
    pub fn into_structure(self) -> NanoChatGptStructureConfig {
        let block_config = self.block_config();
        NanoChatGptStructureConfig {
            wte: EmbeddingConfig::new(self.vocab_size, self.n_embed),
            h: (0..self.n_layer).map(|_| block_config.clone()).collect(),
            lm_head: LinearConfig::new(self.n_embed, self.vocab_size),
            r_emb: RotaryEmbeddingConfig::new(self.max_seq_len(), self.head_dim()),
            norm: self.norm,
            init_seq_len: self.init_seq_len,
            softcap: self.softcap,
        }
    }

    /// Build the [`NanoChatGptBlockConfig`] for this config.
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

/// Low-level GPT Structure Config.
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

impl NanoChatGptStructureConfig {
    /// Initialize a [`NanoChatGpt`].
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> NanoChatGpt<B> {
        let n_embed = self.n_embed();
        NanoChatGpt {
            wte: self.wte.init(device),
            h: self
                .h
                .into_iter()
                .enumerate()
                .map(|(layer_idx, c)| c.init(layer_idx, device))
                .collect(),
            h_norm: self.norm.clone().with_num_features(n_embed).init(device),
            lm_head: self.lm_head.init(device),
            r_emb: self.r_emb.init(device),
            init_seq_len: self.init_seq_len,
            softcap: self.softcap,
        }
    }
}

/// GPT Module
#[derive(Module, Debug)]
pub struct NanoChatGpt<B: Backend> {
    wte: Embedding<B>,
    h: Vec<NanoChatGptBlock<B>>,
    h_norm: Normalization<B>,
    lm_head: Linear<B>,
    r_emb: RotaryEmbedding<B>,

    init_seq_len: usize,
    softcap: f64,
}

impl<B: Backend> NanoChatGptMeta for NanoChatGpt<B> {
    fn n_embed(&self) -> usize {
        self.wte.weight.dims()[0]
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

impl<B: Backend> NanoChatGpt<B> {
    /// Forward Pass.
    ///
    /// # Arguments
    /// - `idx`: a `[B, T]` input.
    /// - `kv_cache`: a `KVCache`.
    pub fn forward(
        &self,
        idx: Tensor<B, 2, Int>,
        kv_cache: &mut Option<&mut KVCache<B>>,
    ) -> Tensor<B, 3> {
        let [b, t] = unpack_shape_contract!(["B", "T"], &idx.dims());
        assert!(
            t <= self.r_emb.seq_len(),
            "Sequence length grew beyond the rotary embeddings cache: {t} > {}",
            self.r_emb.seq_len()
        );

        let t0 = match kv_cache {
            Some(kv_cache) => kv_cache.pos(),
            None => 0,
        };
        let r_emb = self.r_emb.clip_range(t0..t0 + t);

        let mut x = self.wte.forward(idx);

        // Note: The reference chat has a norm here,
        // but the block has the same norm as the first operation.
        // x = rms_norm(x);

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
            ["B", "T", "D"],
            &logits.dims(),
            &[("B", b), ("T", t), ("D", self.n_embed())]
        );
        logits
    }

    /// Allocate a new [`KVCache`]
    ///
    /// # Arguments
    /// - `batch_size`: the batch size.
    pub fn new_kv_cache(
        &self,
        batch_size: usize,
    ) -> KVCache<B> {
        KVCacheConfig {
            batch_size,
            num_heads: self.n_kv_head(),
            seq_len: self.init_seq_len(),
            head_dim: self.head_dim(),
            num_layers: self.n_layer(),
        }
        .init()
    }

    /// Calculate the estimated FLOPs per token for the model.
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
    use burn::tensor::Distribution;
    use serial_test::serial;

    use super::*;
    use crate::{
        contracts::assert_shape_contract,
        support::testing::PerformanceBackend,
    };

    #[test]
    fn test_gpt_config() {
        let cfg = NanoChatGptConfig::new();
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
        type B = PerformanceBackend;
        let device = Default::default();

        let batch_size = 1;
        let seq_len = 100;
        let n_layer = 4;
        let n_embed = 36;

        let vocab_size = 1000;

        let cfg = NanoChatGptConfig::new()
            .with_vocab_size(vocab_size)
            .with_n_embed(n_embed)
            .with_n_layer(n_layer);
        let gpt: NanoChatGpt<B> = cfg.init(&device);

        let mut kv_cache = gpt.new_kv_cache(batch_size);

        let input_tokens = Tensor::<B, 2>::random(
            [batch_size, seq_len],
            Distribution::Uniform(0.0, vocab_size as f64),
            &device,
        )
        .int();

        let logits = gpt.forward(input_tokens, &mut Some(&mut kv_cache));
        assert_shape_contract!(
            ["B", "T", "D"],
            &logits.dims(),
            &[("B", batch_size), ("T", seq_len), ("D", gpt.n_embed())]
        );
    }
}
