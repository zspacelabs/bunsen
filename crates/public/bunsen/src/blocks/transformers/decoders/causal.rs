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
        conditioning::{
            AdaptiveScale,
            AdaptiveScaleConfig,
            AdaptiveScaleMeta,
        },
        embedding::RotaryEmbedding,
        mlp::{
            FeedForward,
            FeedForwardConfig,
            FeedForwardMeta,
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

    /// A scale on the feed-forward's normalized input, learned from a
    /// conditioning vector the forward pass is handed
    /// ([`forward_conditioned`](CausalDecoderBlock::forward_conditioned)):
    /// Voxtral Realtime's delay conditioning. `None` scales nothing.
    #[config(default = "None")]
    pub mlp_scale: Option<AdaptiveScaleConfig>,
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
        if let Some(scale) = &self.mlp_scale
            && scale.d_model() != n_embed
        {
            return Err(ConstraintError::new(
                "CausalDecoderBlockConfig",
                "",
                Rule::Relation {
                    lhs: ("mlp_scale.d_model".into(), scale.d_model().to_string()),
                    op: "==",
                    rhs: ("n_embed".into(), n_embed.to_string()),
                },
            )
            .into());
        }
        Ok(CausalDecoderBlock {
            attn_norm: self.norm.clone().with_num_features(n_embed).init(device),
            attn: self.attn.try_init(layer_index, device)?,
            mlp_norm: self.norm.clone().with_num_features(n_embed).init(device),
            mlp: self.mlp.try_init(device)?,
            mlp_scale: self
                .mlp_scale
                .as_ref()
                .map(|c| c.try_init(device))
                .transpose()?,
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
/// decoder layer. With an [`AdaptiveScale`] on the feed-forward's input,
/// `x = x + mlp(mlp_norm(x) * (1 + f(cond)))`, it is Voxtral Realtime's
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

    /// The scale on the feed-forward's normalized input, if the block is
    /// conditioned.
    pub mlp_scale: Option<AdaptiveScale>,
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
    ///
    /// # Panics
    /// If the block is conditioned (it has an
    /// [`mlp_scale`](Self::mlp_scale)): that needs
    /// [`forward_conditioned`](Self::forward_conditioned).
    pub fn forward(
        &self,
        input: Tensor<3>,
        r_emb: &RotaryEmbedding,
        kv_cache: &mut Option<&mut KVCache>,
    ) -> Tensor<3> {
        self.forward_conditioned(input, r_emb, kv_cache, None)
    }

    /// Forward Pass, with the condition a conditioned block scales its
    /// feed-forward's input by.
    ///
    /// # Arguments
    /// - `input`: a `[B, T, D]` input.
    /// - `r_emb`: a `[1, T, 1, D/2]` embedding.
    /// - `kv_cache`: optional KV cache.
    /// - `cond`: the `[B, T, D]` condition, or one that broadcasts to it (`[1,
    ///   1, D]`); `None` for a block without an [`mlp_scale`](Self::mlp_scale).
    ///
    /// # Returns
    /// - the `[B, T, D]` block output.
    ///
    /// # Panics
    /// If `cond` is given to a block without a scale, or withheld from one
    /// with a scale.
    pub fn forward_conditioned(
        &self,
        input: Tensor<3>,
        r_emb: &RotaryEmbedding,
        kv_cache: &mut Option<&mut KVCache>,
        cond: Option<Tensor<3>>,
    ) -> Tensor<3> {
        let x = input.clone()
            + self
                .attn
                .forward(self.attn_norm.forward(input), r_emb, kv_cache);
        let h = self.mlp_norm.forward(x.clone());
        let h = match (&self.mlp_scale, cond) {
            (None, None) => h,
            (Some(scale), Some(cond)) => scale.forward(h, cond),
            (None, Some(_)) => panic!("a condition for a block without an mlp_scale"),
            (Some(_), None) => panic!("a block with an mlp_scale needs its condition"),
        };
        x.clone() + self.mlp.forward(h)
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

    fn conditioned(n_embed: usize) -> CausalDecoderBlockConfig {
        CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(4, 2, n_embed).with_norm(None),
            crate::blocks::transformers::mlp::GatedMlpConfig::new(n_embed, 24).into(),
        )
        .with_mlp_scale(Some(AdaptiveScaleConfig::new(n_embed, 4)))
    }

    /// A conditioned block scales its feed-forward's normalized input:
    /// `x = x + attn(attn_norm(x)); x = x + mlp(mlp_norm(x) * (1 + f(cond)))`.
    #[test]
    fn test_conditioned_block_scales_the_mlp_input() {
        let device = crate::support::testing::cpu_device();
        let [batch, seq_len, n_embed] = [2, 5, 16];
        let block: CausalDecoderBlock = conditioned(n_embed).init(0, &device);
        let scale = block.mlp_scale.as_ref().expect("a conditioned block");
        let r_emb = RotaryEmbeddingConfig::new(seq_len, block.attn.head_dim()).init(&device);
        let input =
            seeded_tensor::<3>(1, [batch, seq_len, n_embed], Distribution::Default, &device);
        let cond = seeded_tensor::<3>(2, [1, 1, n_embed], Distribution::Default, &device);

        let x = input.clone()
            + block
                .attn
                .forward(block.attn_norm.forward(input.clone()), &r_emb, &mut None);
        let expected = x.clone()
            + block
                .mlp
                .forward(scale.forward(block.mlp_norm.forward(x), cond.clone()));
        let output = block.forward_conditioned(input, &r_emb, &mut None, Some(cond));
        assert_tensors_close(&output, &expected, Tolerance::default());
    }

    #[test]
    #[should_panic(expected = "needs its condition")]
    fn test_conditioned_block_needs_its_condition() {
        let device = crate::support::testing::cpu_device();
        let block: CausalDecoderBlock = conditioned(16).init(0, &device);
        let r_emb = RotaryEmbeddingConfig::new(3, block.attn.head_dim()).init(&device);
        let _ = block.forward(Tensor::zeros([1, 3, 16], &device), &r_emb, &mut None);
    }

    /// The scale's width must be the block's.
    #[test]
    fn test_conditioned_block_width() {
        let device = crate::support::testing::cpu_device();
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Illegal)
            .message_contains("mlp_scale.d_model")
            .assert_err(
                &conditioned(16)
                    .with_mlp_scale(Some(AdaptiveScaleConfig::new(8, 4)))
                    .try_init(0, &device),
            );
    }
}
