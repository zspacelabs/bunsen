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
        attention::csa::{
            CausalSelfAttention,
            CausalSelfAttentionConfig,
            CausalSelfAttentionMeta,
        },
        embedding::RotaryEmbedding,
        mlp::{
            Mlp,
            MlpConfig,
            MlpMeta,
        },
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
        WithOkOrPanic,
    },
    ops::transformers::attention::KVCache,
};

/// Common meta for [`NanoChatGptBlock`] and [`NanoChatGptBlockConfig`].
pub trait NanoChatGptBlockMeta {
    /// Returns the size of the input and output.
    fn n_embed(&self) -> usize;
}

/// Config for [`NanoChatGptBlock`].
#[derive(Config, Debug)]
pub struct NanoChatGptBlockConfig {
    /// Causal Self-Attention Config.
    pub attn: CausalSelfAttentionConfig,

    /// MLP Config.
    pub mlp: MlpConfig,

    /// Attention Normalization.
    /// This normalization will be adapted to the appropriate feature count.
    #[config(default = "NormalizationConfig::Rms(RmsNormConfig::new(0))")]
    pub norm: NormalizationConfig,
}

impl NanoChatGptBlockMeta for NanoChatGptBlockConfig {
    fn n_embed(&self) -> usize {
        self.attn.n_embed()
    }
}

impl NanoChatGptBlockConfig {
    /// Initializes a [`NanoChatGptBlock`].
    ///
    /// Panics on errors.
    pub fn init<B: Backend>(
        &self,
        layer_index: usize,
        device: &B::Device,
    ) -> NanoChatGptBlock<B> {
        self.try_init(layer_index, device).ok_or_panic()
    }

    /// Initializes a [`NanoChatGptBlock`].
    pub fn try_init<B: Backend>(
        &self,
        layer_index: usize,
        device: &B::Device,
    ) -> BunsenResult<NanoChatGptBlock<B>> {
        if self.attn.n_embed() != self.mlp.n_embed() {
            return Err(ConstraintError::new(
                "NanoChatGptBlockConfig",
                "",
                Rule::Relation {
                    lhs: ("attn.n_embed".into(), self.attn.n_embed().to_string()),
                    op: "==",
                    rhs: ("mlp.n_embed".into(), self.mlp.n_embed().to_string()),
                },
            )
            .into());
        }

        let n_embed = self.n_embed();
        Ok(NanoChatGptBlock {
            input_norm: self.norm.clone().with_num_features(n_embed).init(device),
            attn: self.attn.try_init(layer_index, device)?,
            attn_norm: self.norm.clone().with_num_features(n_embed).init(device),
            mlp: self.mlp.try_init(device)?,
        })
    }
}

/// A single nanoChat GPT transformer block.
///
/// A pre-norm residual block, as upstream nanochat's `Block`: the causal
/// self-attention and then the MLP each read a normalized copy of the
/// residual stream, and add their output back to it:
/// `x = x + attn(norm(x)); x = x + mlp(norm(x))`. Used as the repeated unit
/// inside [`NanoChatGpt`](super::NanoChatGpt), which chains the blocks.
///
/// Built by [`NanoChatGptBlockConfig`], whose `init` also takes the block's
/// layer index: its slot in a shared [`KVCache`].
#[derive(Module, Debug)]
pub struct NanoChatGptBlock<B: Backend> {
    /// Normalization of the attention's input.
    pub input_norm: Normalization<B>,

    /// Attention.
    pub attn: CausalSelfAttention<B>,

    /// Normalization of the MLP's input.
    pub attn_norm: Normalization<B>,

    /// MLP.
    pub mlp: Mlp<B>,
}

impl<B: Backend> NanoChatGptBlockMeta for NanoChatGptBlock<B> {
    fn n_embed(&self) -> usize {
        self.attn.n_embed()
    }
}

impl<B: Backend> NanoChatGptBlock<B> {
    /// Forward Pass.
    ///
    /// # Usage Note
    /// - this block norms each sub-layer's input, not the residual stream.
    /// - this block does not norm on output.
    ///
    /// # Arguments
    /// - `input`: a `[B, T, D]` input.
    /// - `r_emb`: a `[1, T, 1, D/2]` embedding.
    /// - `kv_cache`: optional KV cache.
    ///
    /// # Returns
    /// - the `[B, T, D]` block output: `input` plus the attention and MLP
    ///   updates.
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        r_emb: &RotaryEmbedding<B>,
        kv_cache: &mut Option<&mut KVCache<B>>,
    ) -> Tensor<B, 3> {
        let x = input.clone()
            + self
                .attn
                .forward(self.input_norm.forward(input), r_emb, kv_cache);
        x.clone() + self.mlp.forward(self.attn_norm.forward(x))
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
        blocks::transformers::embedding::RotaryEmbeddingConfig,
        contracts::assert_shape_contract,
        support::testing::{
            DeviceMemoryGuard,
            PerformanceBackend,
            assert_tensors_close,
            performance_device,
            seeded_tensor,
        },
    };

    #[test]
    #[serial]
    fn test_gpt_block_config() {
        type B = PerformanceBackend;
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let n_embed = 1024;
        let n_head = 128;
        let n_kv_head = 64;

        let config = NanoChatGptBlockConfig::new(
            CausalSelfAttentionConfig::new(n_head, n_kv_head, n_embed),
            MlpConfig::new(n_embed).with_act_exponent(Some(2.0)),
        );
        assert_eq!(config.n_embed(), n_embed);
        assert_eq!(config.attn.n_embed(), n_embed);
        assert_eq!(config.attn.n_head(), n_head);
        assert_eq!(config.attn.n_kv_head(), n_kv_head);

        assert_eq!(config.mlp.n_embed(), n_embed);

        let layer_index = 12;
        let block: NanoChatGptBlock<B> = config.init(layer_index, &device);

        assert_eq!(block.n_embed(), n_embed);
    }

    #[test]
    #[serial]
    fn test_gpt_block_forward() {
        type B = PerformanceBackend;
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let batch = 2;
        let seq_len = 10;

        let n_embed = 1024;
        let n_head = 128;
        let n_kv_head = 64;
        let layer_index = 12;

        let config = NanoChatGptBlockConfig::new(
            CausalSelfAttentionConfig::new(n_head, n_kv_head, n_embed),
            MlpConfig::new(n_embed).with_act_exponent(Some(2.0)),
        );

        let block: NanoChatGptBlock<B> = config.init(layer_index, &device);

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

    /// The block adds each sub-layer's output back to its input, as upstream
    /// nanochat's `x = x + attn(norm(x)); x = x + mlp(norm(x))` does.
    #[test]
    #[serial]
    fn test_gpt_block_adds_sublayers_to_residual() {
        type B = PerformanceBackend;
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let [batch, seq_len, n_embed] = [2, 5, 32];
        let config = NanoChatGptBlockConfig::new(
            CausalSelfAttentionConfig::new(4, 2, n_embed),
            MlpConfig::new(n_embed).with_act_exponent(Some(2.0)),
        );
        let mut block: NanoChatGptBlock<B> = config.init(0, &device);
        let r_emb = RotaryEmbeddingConfig::new(seq_len, block.attn.head_dim()).init(&device);
        let input =
            seeded_tensor::<3>(1, [batch, seq_len, n_embed], Distribution::Default, &device);

        // With the MLP's output projection zeroed, the block is
        // `x + attn(norm(x))`.
        block.mlp.linear2.weight = Param::from_tensor(block.mlp.linear2.weight.val().zeros_like());
        let attn = block
            .attn
            .forward(block.input_norm.forward(input.clone()), &r_emb, &mut None);
        let output = block.forward(input.clone(), &r_emb, &mut None);
        assert_tensors_close(&output, &(input.clone() + attn), Tolerance::default());

        // With the attention's output projection zeroed too, the block is the
        // identity.
        block.attn.c_proj.weight = Param::from_tensor(block.attn.c_proj.weight.val().zeros_like());
        let output = block.forward(input.clone(), &r_emb, &mut None);
        assert_tensors_close(&output, &input, Tolerance::default());
    }
}
