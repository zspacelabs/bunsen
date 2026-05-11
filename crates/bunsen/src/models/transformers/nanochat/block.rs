//! # GPT Block

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::norm::{
        Normalization,
        NormalizationConfig,
        RmsNormConfig,
    },
    prelude::Backend,
};

use crate::{
    blocks::transformers::{
        attention::{
            csa::{
                CausalSelfAttention,
                CausalSelfAttentionConfig,
                CausalSelfAttentionMeta,
            },
            kvcache::KVCache,
        },
        embedding::rotary::RotaryEmbedding,
    },
    models::transformers::nanochat::{
        MLP,
        MLPConfig,
        MLPMeta,
    },
};

/// Common meta for [`GPTBlock`] and [`GPTBlockConfig`].
pub trait GPTBlockMeta {
    /// Return the size of the input and output.
    fn n_embed(&self) -> usize;
}

/// Config for [`GPTBlock`].
#[derive(Config, Debug)]
pub struct GPTBlockConfig {
    /// Causal Self-Attention Config.
    pub attn: CausalSelfAttentionConfig,

    /// MLP Config.
    pub mlp: MLPConfig,

    /// Attention Normalization.
    /// This normalization will be adapted to the appropriate feature count.
    #[config(default = "NormalizationConfig::Rms(RmsNormConfig::new(0))")]
    pub norm: NormalizationConfig,
}

impl GPTBlockMeta for GPTBlockConfig {
    fn n_embed(&self) -> usize {
        self.attn.n_embed()
    }
}

impl GPTBlockConfig {
    /// Initialize a [`GPTBlock`].
    pub fn init<B: Backend>(
        self,
        layer_index: usize,
        device: &B::Device,
    ) -> GPTBlock<B> {
        assert_eq!(self.attn.n_embed(), self.mlp.n_embed());
        let n_embed = self.n_embed();
        GPTBlock {
            input_norm: self.norm.clone().with_num_features(n_embed).init(device),
            attn: self.attn.init(layer_index, device),
            attn_norm: self.norm.clone().with_num_features(n_embed).init(device),
            mlp: self.mlp.init(device),
        }
    }
}

/// GPT Block
#[derive(Module, Debug)]
pub struct GPTBlock<B: Backend> {
    /// Input Normalization.
    pub input_norm: Normalization<B>,

    /// Attention.
    pub attn: CausalSelfAttention<B>,

    /// Attention Normalization.
    pub attn_norm: Normalization<B>,

    /// MLP.
    pub mlp: MLP<B>,
}

impl<B: Backend> GPTBlockMeta for GPTBlock<B> {
    fn n_embed(&self) -> usize {
        self.attn.n_embed()
    }
}

impl<B: Backend> GPTBlock<B> {
    /// Forward Pass.
    ///
    /// # Usage Note
    /// - this block norms on input.
    /// - this block does not norm on output.
    ///
    /// # Arguments
    /// - `input`: a ``[B, T, D]`` input.
    /// - `r_emb`: a ``[1, T, 1, D/2]`` embedding.
    /// - `kv_cache`: optional KV cache.
    ///
    /// # Returns
    /// - the ``[B, T, D]`` block output.
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        r_emb: &RotaryEmbedding<B>,
        kv_cache: &mut Option<&mut KVCache<B>>,
    ) -> Tensor<B, 3> {
        let x = self.input_norm.forward(input);
        let x = self.attn.forward(x, r_emb, kv_cache);
        let x = self.attn_norm.forward(x);
        self.mlp.forward(x)
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;
    use crate::{
        blocks::transformers::embedding::rotary::RotaryEmbeddingConfig,
        contracts::assert_shape_contract,
        support::testing::PerfTestBackend,
    };

    #[test]
    fn test_gpt_block_config() {
        type B = PerfTestBackend;
        let device = Default::default();

        let n_embed = 1024;
        let n_head = 128;
        let n_kv_head = 64;

        let config = GPTBlockConfig::new(
            CausalSelfAttentionConfig::new(n_head, n_kv_head, n_embed),
            MLPConfig::new(n_embed),
        );
        assert_eq!(config.n_embed(), n_embed);
        assert_eq!(config.attn.n_embed(), n_embed);
        assert_eq!(config.attn.n_head(), n_head);
        assert_eq!(config.attn.n_kv_head(), n_kv_head);

        assert_eq!(config.mlp.n_embed(), n_embed);

        let layer_index = 12;
        let block: GPTBlock<B> = config.init(layer_index, &device);

        assert_eq!(block.n_embed(), n_embed);
    }

    #[test]
    fn test_gpt_block_forward() {
        type B = PerfTestBackend;
        let device = Default::default();

        let batch = 2;
        let seq_len = 10;

        let n_embed = 1024;
        let n_head = 128;
        let n_kv_head = 64;
        let layer_index = 12;

        let config = GPTBlockConfig::new(
            CausalSelfAttentionConfig::new(n_head, n_kv_head, n_embed),
            MLPConfig::new(n_embed),
        );

        let block: GPTBlock<B> = config.init(layer_index, &device);

        let input = Tensor::random([batch, seq_len, n_embed], Distribution::Default, &device);

        let r_emb = RotaryEmbeddingConfig::new(seq_len, block.attn.head_dim()).init(&device);
        let mut kv_cache: Option<&mut KVCache<B>> = None;

        let output = block.forward(input.clone(), &r_emb, &mut kv_cache);
        assert_shape_contract!(
            ["B", "T", "D"],
            &output.dims(),
            &[("B", batch), ("T", seq_len), ("D", n_embed)]
        );
    }
}
