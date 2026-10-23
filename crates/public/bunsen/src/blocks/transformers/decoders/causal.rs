//! The causal decoder layer.

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::norm::{
        Normalization,
        NormalizationConfig,
        RmsNormConfig,
    },
    tensor::Device,
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
            FeedForward,
            FeedForwardConfig,
            FeedForwardMeta,
        },
    },
    burner::module::{
        ModuleInit,
        Rekey,
    },
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
        WithOkOrPanic,
    },
    ops::transformers::attention::KVCache,
};

/// Common meta for [`CausalDecoderBlock`] and [`CausalDecoderBlockConfig`].
pub trait CausalDecoderBlockMeta {
    /// Returns the size of the input and output.
    fn n_embed(&self) -> usize;
}

/// Config for [`CausalDecoderBlock`].
#[derive(Config, Debug)]
pub struct CausalDecoderBlockConfig {
    /// Causal Self-Attention Config.
    pub attn: CausalSelfAttentionConfig,

    /// The feed-forward's config.
    pub mlp: FeedForwardConfig,

    /// The normalization before each sublayer, adapted to the width (its
    /// feature count is set at init).
    #[config(default = "NormalizationConfig::Rms(RmsNormConfig::new(0))")]
    pub norm: NormalizationConfig,
}

impl CausalDecoderBlockMeta for CausalDecoderBlockConfig {
    fn n_embed(&self) -> usize {
        self.attn.n_embed()
    }
}

impl CausalDecoderBlockConfig {
    /// Initializes a [`CausalDecoderBlock`].
    ///
    /// Panics on errors.
    pub fn init(
        &self,
        layer_index: usize,
        device: &Device,
    ) -> CausalDecoderBlock {
        self.try_init(layer_index, device).ok_or_panic()
    }

    /// Initializes a [`CausalDecoderBlock`].
    pub fn try_init(
        &self,
        layer_index: usize,
        device: &Device,
    ) -> BunsenResult<CausalDecoderBlock> {
        if self.attn.n_embed() != self.mlp.n_embed() {
            return Err(ConstraintError::new(
                "CausalDecoderBlockConfig",
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
        Ok(CausalDecoderBlock {
            attn_norm: self.norm.clone().with_num_features(n_embed).init(device),
            attn: self.attn.try_init(layer_index, device)?,
            mlp_norm: self.norm.clone().with_num_features(n_embed).init(device),
            mlp: self.mlp.try_init(device)?,
        })
    }
}

/// The decoder-only transformer layer: pre-normalized causal
/// self-attention, then a pre-normalized feed-forward, each residual:
/// `x = x + attn(attn_norm(x)); x = x + mlp(mlp_norm(x))`.
///
/// What tells its users apart is config: the norm ([`Normalization`]), the
/// feed-forward ([`FeedForward`]: an `Mlp` or a gated `SwiGLU`), and the
/// attention's own options (grouped-query heads, QK-norm and where it sits).
/// RMS norms, a `ReLU`² `Mlp` and QK-norm after the rotary embedding, it is
/// nanochat's `Block`; RMS norms, a `SwiGLU` and QK-norm before it, Qwen3's
/// decoder layer.
///
/// Built by [`CausalDecoderBlockConfig`], whose `init` also takes the block's
/// layer index: its slot in a shared [`KVCache`].
#[derive(Module, Debug)]
pub struct CausalDecoderBlock {
    /// The attention's input normalization.
    pub attn_norm: Normalization,

    /// Attention.
    pub attn: CausalSelfAttention,

    /// The feed-forward's input normalization.
    pub mlp_norm: Normalization,

    /// The feed-forward.
    pub mlp: FeedForward,
}

impl CausalDecoderBlockMeta for CausalDecoderBlock {
    fn n_embed(&self) -> usize {
        self.attn.n_embed()
    }
}

impl CausalDecoderBlock {
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
    /// - the `[B, T, D]` block output: `input` plus the attention and
    ///   feed-forward updates.
    pub fn forward(
        &self,
        input: Tensor<3>,
        r_emb: &RotaryEmbedding,
        kv_cache: &mut Option<&mut KVCache>,
    ) -> Tensor<3> {
        let x = input.clone()
            + self
                .attn
                .forward(self.attn_norm.forward(input), r_emb, kv_cache);
        x.clone() + self.mlp.forward(self.mlp_norm.forward(x))
    }
}

impl CausalDecoderBlock {
    /// Its tensors as a `transformers` decoder layer names them (Qwen3's,
    /// Llama's), relative to it: `input_layernorm` the attention's norm,
    /// `post_attention_layernorm` the feed-forward's, `self_attn`'s
    /// `{q,k,v,o}_proj` the attention's projections and its `{q,k}_norm` its
    /// QK-norms, and the `SwiGLU`'s projections as they are.
    pub fn transformers_rekey() -> Rekey {
        Rekey::keep_rest()
            .rename("input_layernorm.", "attn_norm.")
            .rename("post_attention_layernorm.", "mlp_norm.")
            .rename("self_attn.q_proj.", "attn.c_q.")
            .rename("self_attn.k_proj.", "attn.c_k.")
            .rename("self_attn.v_proj.", "attn.c_v.")
            .rename("self_attn.o_proj.", "attn.c_proj.")
            .rename("self_attn.", "attn.")
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
        blocks::transformers::{
            embedding::RotaryEmbeddingConfig,
            mlp::MlpConfig,
        },
        contracts::assert_shape_contract,
        support::testing::{
            DeviceMemoryGuard,
            assert_tensors_close,
            performance_device,
            seeded_tensor,
        },
    };

    #[test]
    #[serial]
    fn test_gpt_block_config() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let n_embed = 1024;
        let n_head = 128;
        let n_kv_head = 64;

        let config = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(n_head, n_kv_head, n_embed),
            MlpConfig::new(n_embed).with_act_exponent(Some(2.0)).into(),
        );
        assert_eq!(config.n_embed(), n_embed);
        assert_eq!(config.attn.n_embed(), n_embed);
        assert_eq!(config.attn.n_head(), n_head);
        assert_eq!(config.attn.n_kv_head(), n_kv_head);

        assert_eq!(FeedForwardMeta::n_embed(&config.mlp), n_embed);

        let layer_index = 12;
        let block: CausalDecoderBlock = config.init(layer_index, &device);

        assert_eq!(block.n_embed(), n_embed);
    }

    #[test]
    #[serial]
    fn test_gpt_block_forward() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let batch = 2;
        let seq_len = 10;

        let n_embed = 1024;
        let n_head = 128;
        let n_kv_head = 64;
        let layer_index = 12;

        let config = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(n_head, n_kv_head, n_embed),
            MlpConfig::new(n_embed).with_act_exponent(Some(2.0)).into(),
        );

        let block: CausalDecoderBlock = config.init(layer_index, &device);

        let input = Tensor::random([batch, seq_len, n_embed], Distribution::Default, &device);

        let r_emb = RotaryEmbeddingConfig::new(seq_len, block.attn.head_dim()).init(&device);
        let mut kv_cache: Option<&mut KVCache> = None;

        let output = block.forward(input.clone(), &r_emb, &mut kv_cache);
        assert_shape_contract!(
            ["B", "T", "D"],
            &output.dims(),
            &[("B", batch), ("T", seq_len), ("D", n_embed)]
        );
    }

    /// The block adds each sub-layer's output back to its input:
    /// `x = x + attn(attn_norm(x)); x = x + mlp(mlp_norm(x))`.
    #[test]
    #[serial]
    fn test_gpt_block_adds_sublayers_to_residual() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let [batch, seq_len, n_embed] = [2, 5, 32];
        let config = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(4, 2, n_embed),
            MlpConfig::new(n_embed).with_act_exponent(Some(2.0)).into(),
        );
        let mut block: CausalDecoderBlock = config.init(0, &device);
        let r_emb = RotaryEmbeddingConfig::new(seq_len, block.attn.head_dim()).init(&device);
        let input =
            seeded_tensor::<3>(1, [batch, seq_len, n_embed], Distribution::Default, &device);

        // With the MLP's output projection zeroed, the block is
        // `x + attn(norm(x))`.
        let FeedForward::Mlp(mlp) = &mut block.mlp else {
            panic!("an Mlp feed-forward");
        };
        mlp.linear2.weight = Param::from_tensor(mlp.linear2.weight.val().zeros_like());
        let attn = block
            .attn
            .forward(block.attn_norm.forward(input.clone()), &r_emb, &mut None);
        let output = block.forward(input.clone(), &r_emb, &mut None);
        assert_tensors_close(&output, &(input.clone() + attn), Tolerance::default());

        // With the attention's output projection zeroed too, the block is the
        // identity.
        block.attn.c_proj.weight = Param::from_tensor(block.attn.c_proj.weight.val().zeros_like());
        let output = block.forward(input.clone(), &r_emb, &mut None);
        assert_tensors_close(&output, &input, Tolerance::default());
    }

    /// `transformers`' decoder-layer names, onto the block's own.
    #[test]
    fn test_transformers_rekey() {
        let rekey = CausalDecoderBlock::transformers_rekey();
        for (hf, bunsen) in [
            ("input_layernorm.weight", "attn_norm.weight"),
            ("post_attention_layernorm.weight", "mlp_norm.weight"),
            ("self_attn.q_proj.weight", "attn.c_q.weight"),
            ("self_attn.k_proj.weight", "attn.c_k.weight"),
            ("self_attn.v_proj.weight", "attn.c_v.weight"),
            ("self_attn.o_proj.weight", "attn.c_proj.weight"),
            ("self_attn.q_norm.weight", "attn.q_norm.weight"),
            ("self_attn.k_norm.weight", "attn.k_norm.weight"),
            ("mlp.gate_proj.weight", "mlp.gate_proj.weight"),
            ("mlp.down_proj.weight", "mlp.down_proj.weight"),
        ] {
            assert_eq!(rekey.apply(hf).as_deref(), Some(bunsen), "{hf}");
        }
    }
}
