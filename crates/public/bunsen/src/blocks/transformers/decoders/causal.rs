//! The causal decoder layer.

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::norm::RmsNormConfig,
    prelude::Bool,
    tensor::Device,
};

use crate::{
    blocks::{
        norm::{
            BunsenNormalization,
            BunsenNormalizationConfig,
        },
        transformers::{
            attention::csa::{
                AttentionSides,
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
                TransformerFeedForward,
                TransformerFeedForwardConfig,
                TransformerFeedForwardMeta,
            },
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

/// The side inputs of a [`CausalDecoderBlock`]'s
/// [`forward_with`](CausalDecoderBlock::forward_with): each one only a block
/// configured for it takes.
#[derive(Debug, Default)]
pub struct DecoderSideInputs {
    /// The condition of a block with an
    /// [`mlp_scale`](CausalDecoderBlock::mlp_scale): `[B, T, D]`, or one that
    /// broadcasts to it.
    pub cond: Option<Tensor<3>>,

    /// The `[B, T, H_kv * head_dim]` value embedding of a block whose
    /// attention has a value residual.
    pub value_embed: Option<Tensor<3>>,

    /// `[B, T]`, true at the positions that are not padding, which alone
    /// the attention's queries attend to.
    pub key_valid: Option<Tensor<2, Bool>>,
}

/// Common meta for [`CausalDecoderBlock`] and [`CausalDecoderBlockConfig`].
pub trait CausalDecoderBlockMeta {
    /// Returns the size of the input and output.
    fn n_embed(&self) -> usize;
}

/// A decoder layer's names for its parts, as a checkpoint spells them: what
/// [`CausalDecoderBlock::layer_rekey`] maps onto the block's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecoderLayerKeys {
    /// The attention's input norm.
    pub attn_norm: String,

    /// The feed-forward's input norm.
    pub mlp_norm: String,

    /// The attention's and the feed-forward's output norms, for a block with
    /// post-norms.
    pub post_norms: Option<[String; 2]>,

    /// The attention.
    pub attn: String,

    /// The attention's query, key, value and output projections, under
    /// [`attn`](Self::attn).
    pub attn_projs: [String; 4],
}

impl DecoderLayerKeys {
    /// Llama's line (Qwen3, Mistral, Gemma): `self_attn` with
    /// `{q,k,v,o}_proj`, under the layer's names for its two norms, the
    /// attention's and the feed-forward's (`input_layernorm` and
    /// `post_attention_layernorm` in Llama's own; `self_attn_layer_norm` and
    /// `final_layer_norm` in Voxtral's audio tower).
    pub fn llama(
        attn_norm: impl Into<String>,
        mlp_norm: impl Into<String>,
    ) -> Self {
        Self {
            attn_norm: attn_norm.into(),
            mlp_norm: mlp_norm.into(),
            post_norms: None,
            attn: "self_attn".to_string(),
            attn_projs: ["q_proj", "k_proj", "v_proj", "o_proj"].map(String::from),
        }
    }

    /// With post-norms: the attention's and the feed-forward's output norms
    /// (`post_attention_layernorm` and `post_feedforward_layernorm` in
    /// Gemma 2 and 3).
    pub fn with_post_norms(
        self,
        attn_post_norm: impl Into<String>,
        mlp_post_norm: impl Into<String>,
    ) -> Self {
        Self {
            post_norms: Some([attn_post_norm.into(), mlp_post_norm.into()]),
            ..self
        }
    }
}

/// Config for [`CausalDecoderBlock`].
#[derive(Config, Debug)]
pub struct CausalDecoderBlockConfig {
    /// Causal Self-Attention Config.
    pub attn: CausalSelfAttentionConfig,

    /// The feed-forward's config.
    pub mlp: TransformerFeedForwardConfig,

    /// The normalization before each sublayer, adapted to the width (its
    /// feature count is set at init).
    #[config(default = "BunsenNormalizationConfig::Rms(RmsNormConfig::new(0))")]
    pub norm: BunsenNormalizationConfig,

    /// Whether the attention's input is normalized; without, the attention
    /// takes the residual stream as it is (`ModernBERT`'s first layer, whose
    /// input is the embedding's norm already).
    #[config(default = true)]
    pub attn_norm: bool,

    /// Whether each sublayer's output is normalized too, by a `norm` of its
    /// own, before it is added to the residual stream: the sandwich of
    /// Gemma 2 and 3.
    #[config(default = false)]
    pub post_norms: bool,

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
        let norm = || self.norm.clone().with_num_features(n_embed).init(device);
        let post_norm = || self.post_norms.then(norm);
        Ok(CausalDecoderBlock {
            attn_norm: self.attn_norm.then(norm),
            attn: self.attn.try_init(layer_index, device)?,
            attn_post_norm: post_norm(),
            mlp_norm: norm(),
            mlp: self.mlp.try_init(device)?,
            mlp_post_norm: post_norm(),
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
/// What tells its users apart is config: the norm ([`BunsenNormalization`]),
/// the feed-forward ([`TransformerFeedForward`]: an `TransformerMlp` or a gated
/// `SwiGLU`), and the attention's own options (grouped-query heads, QK-norm and
/// where it sits). RMS norms, a `ReLU`² `TransformerMlp` and QK-norm after the
/// rotary embedding, it is nanochat's `Block`; RMS norms, a `SwiGLU` and
/// QK-norm before it, Qwen3's decoder layer. With an [`AdaptiveScale`] on the
/// feed-forward's input, `x = x + mlp(mlp_norm(x) * (1 + f(cond)))`, it is
/// Voxtral Realtime's decoder layer.
///
/// Without an [`attn_norm`](CausalDecoderBlockConfig::attn_norm), the
/// attention takes the residual stream as it is, `x = x + attn(x)`:
/// `ModernBERT`'s first layer.
///
/// With [`post_norms`](CausalDecoderBlockConfig::post_norms), each sublayer's
/// output is normalized before it is added:
/// `x = x + attn_post_norm(attn(attn_norm(x)))`, and the same for the
/// feed-forward: Gemma 2's and 3's sandwich.
///
/// Built by [`CausalDecoderBlockConfig`], whose `init` also takes the block's
/// layer index: its slot in a shared [`KVCache`].
#[derive(Module, Debug)]
pub struct CausalDecoderBlock {
    /// The attention's input normalization, if the block has one.
    pub attn_norm: Option<BunsenNormalization>,

    /// Attention.
    pub attn: CausalSelfAttention,

    /// The attention's output normalization, if the block has post-norms.
    pub attn_post_norm: Option<BunsenNormalization>,

    /// The feed-forward's input normalization.
    pub mlp_norm: BunsenNormalization,

    /// The feed-forward.
    pub mlp: TransformerFeedForward,

    /// The feed-forward's output normalization, if the block has post-norms.
    pub mlp_post_norm: Option<BunsenNormalization>,

    /// The scale on the feed-forward's normalized input, if the block is
    /// conditioned.
    pub mlp_scale: Option<AdaptiveScale>,
}

impl CausalDecoderBlockMeta for CausalDecoderBlock {
    fn n_embed(&self) -> usize {
        self.attn.n_embed()
    }
}

impl crate::burner::module::InputDTypes for CausalDecoderBlock {
    /// Its attention's: the residual stream is in it.
    fn preferred_dtype(&self) -> burn::tensor::FloatDType {
        crate::burner::module::InputDTypes::preferred_dtype(&self.attn)
    }
}

impl crate::burner::module::ResultDType for CausalDecoderBlock {
    /// Its own: the residual stream's.
    fn result_dtype(
        &self,
        _input: burn::tensor::FloatDType,
    ) -> burn::tensor::FloatDType {
        crate::burner::module::InputDTypes::preferred_dtype(self)
    }
}

impl CausalDecoderBlock {
    /// A checkpoint's decoder-layer names onto this block's, relative to it:
    /// `keys`' two norms, the attention's and the feed-forward's, and with
    /// post-norms their output norms; the attention's projections, and the
    /// rest of the attention (QK-norms) under `attn`; the feed-forward's
    /// projections as they are.
    pub fn layer_rekey(keys: &DecoderLayerKeys) -> Rekey {
        let attn = &keys.attn;
        let [q, k, v, o] = &keys.attn_projs;
        let mut rekey = Rekey::keep_rest()
            .rename(format!("{}.", keys.attn_norm), "attn_norm.")
            .rename(format!("{}.", keys.mlp_norm), "mlp_norm.");
        if let Some([attn_post, mlp_post]) = &keys.post_norms {
            rekey = rekey
                .rename(format!("{attn_post}."), "attn_post_norm.")
                .rename(format!("{mlp_post}."), "mlp_post_norm.");
        }
        rekey
            .rename(format!("{attn}.{q}."), "attn.c_q.")
            .rename(format!("{attn}.{k}."), "attn.c_k.")
            .rename(format!("{attn}.{v}."), "attn.c_v.")
            .rename(format!("{attn}.{o}."), "attn.c_proj.")
            .rename(format!("{attn}."), "attn.")
    }

    /// The attention's input: `x` through the attention's norm, or as it is
    /// without one.
    pub fn attn_input(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        Self::maybe_norm(&self.attn_norm, x)
    }

    /// `norm(x)`, or `x` without a norm.
    fn maybe_norm(
        norm: &Option<BunsenNormalization>,
        x: Tensor<3>,
    ) -> Tensor<3> {
        match norm {
            Some(norm) => norm.forward(x),
            None => x,
        }
    }

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
        self.forward_with(
            input,
            r_emb,
            kv_cache,
            DecoderSideInputs {
                cond,
                ..Default::default()
            },
        )
    }

    /// Forward Pass, with every side input a configured block takes.
    ///
    /// # Arguments
    /// - `input`: a `[B, T, D]` input.
    /// - `r_emb`: a `[1, T, 1, D/2]` embedding.
    /// - `kv_cache`: optional KV cache.
    /// - `sides`: the condition of a block with an
    ///   [`mlp_scale`](Self::mlp_scale), and the value embedding of one whose
    ///   attention has a value residual.
    ///
    /// # Returns
    /// - the `[B, T, D]` block output.
    ///
    /// # Panics
    /// If a side input is given to a block that does not take it, or withheld
    /// from one that does.
    pub fn forward_with(
        &self,
        input: Tensor<3>,
        r_emb: &RotaryEmbedding,
        kv_cache: &mut Option<&mut KVCache>,
        sides: DecoderSideInputs,
    ) -> Tensor<3> {
        #[cfg(any(debug_assertions, test))]
        crate::burner::module::assert_input_dtype(self, input.dtype(), "causal decoder block");
        let DecoderSideInputs {
            cond,
            value_embed,
            key_valid,
        } = sides;
        let attn = self.attn.forward_with(
            self.attn_input(input.clone()),
            r_emb,
            kv_cache,
            AttentionSides {
                value_embed,
                key_valid,
            },
        );
        let x = input + Self::maybe_norm(&self.attn_post_norm, attn);
        let h = self.mlp_norm.forward(x.clone());
        let h = match (&self.mlp_scale, cond) {
            (None, None) => h,
            (Some(scale), Some(cond)) => scale.forward(h, cond),
            (None, Some(_)) => panic!("a condition for a block without an mlp_scale"),
            (Some(_), None) => panic!("a block with an mlp_scale needs its condition"),
        };
        x + Self::maybe_norm(&self.mlp_post_norm, self.mlp.forward(h))
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
            mlp::TransformerMlpConfig,
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
            TransformerMlpConfig::new(n_embed).into(),
        );
        assert_eq!(config.n_embed(), n_embed);
        assert_eq!(config.attn.n_embed(), n_embed);
        assert_eq!(config.attn.n_head(), n_head);
        assert_eq!(config.attn.n_kv_head(), n_kv_head);

        assert_eq!(TransformerFeedForwardMeta::n_embed(&config.mlp), n_embed);

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
            TransformerMlpConfig::new(n_embed).into(),
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

    /// Without an attention norm, the block holds none (its record has no
    /// `attn_norm` keys), and its attention takes the residual stream as it
    /// is: `x = x + attn(x); x = x + mlp(mlp_norm(x))`.
    #[test]
    fn test_block_without_an_attn_norm() {
        let device = crate::support::testing::cpu_device();
        let [batch, seq_len, n_embed] = [2, 5, 16];
        let config = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(4, 2, n_embed),
            TransformerMlpConfig::new(n_embed).into(),
        );
        assert!(config.init(0, &device).attn_norm.is_some());
        let block: CausalDecoderBlock = config.with_attn_norm(false).init(0, &device);
        assert!(block.attn_norm.is_none());
        #[cfg(feature = "store")]
        {
            use burn::store::ModuleSnapshot;

            let paths: Vec<String> = block
                .collect(None, None, false)
                .into_iter()
                .map(|t| t.name)
                .collect();
            assert!(
                paths.iter().all(|p| !p.starts_with("attn_norm")),
                "{paths:?}"
            );
            assert!(paths.iter().any(|p| p.starts_with("mlp_norm")), "{paths:?}");
        }

        let r_emb = RotaryEmbeddingConfig::new(seq_len, block.attn.head_dim()).init(&device);
        let input =
            seeded_tensor::<3>(2, [batch, seq_len, n_embed], Distribution::Default, &device);
        let x = input.clone() + block.attn.forward(input.clone(), &r_emb, &mut None);
        let expected = x.clone() + block.mlp.forward(block.mlp_norm.forward(x));
        let output = block.forward(input, &r_emb, &mut None);
        assert_tensors_close(&output, &expected, Tolerance::default());
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
            TransformerMlpConfig::new(n_embed).into(),
        );
        let mut block: CausalDecoderBlock = config.init(0, &device);
        let r_emb = RotaryEmbeddingConfig::new(seq_len, block.attn.head_dim()).init(&device);
        let input =
            seeded_tensor::<3>(1, [batch, seq_len, n_embed], Distribution::Default, &device);

        // With the MLP's output projection zeroed, the block is
        // `x + attn(norm(x))`.
        let TransformerFeedForward::Mlp(mlp) = &mut block.mlp else {
            panic!("an Mlp feed-forward");
        };
        mlp.linear2.weight = Param::from_tensor(mlp.linear2.weight.val().zeros_like());
        let attn = block
            .attn
            .forward(block.attn_input(input.clone()), &r_emb, &mut None);
        let output = block.forward(input.clone(), &r_emb, &mut None);
        assert_tensors_close(&output, &(input.clone() + attn), Tolerance::default());

        // With the attention's output projection zeroed too, the block is the
        // identity.
        block.attn.c_proj.weight = Param::from_tensor(block.attn.c_proj.weight.val().zeros_like());
        let output = block.forward(input.clone(), &r_emb, &mut None);
        assert_tensors_close(&output, &input, Tolerance::default());
    }

    /// A block with post-norms normalizes each sublayer's output before it
    /// adds it: `x = x + attn_post_norm(attn(attn_norm(x)))`, then
    /// `x = x + mlp_post_norm(mlp(mlp_norm(x)))`.
    #[test]
    fn test_post_norms_sandwich_each_sublayer() {
        let device = crate::support::testing::cpu_device();
        let [batch, seq_len, n_embed] = [2, 5, 16];
        let config = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(4, 2, n_embed),
            crate::blocks::transformers::mlp::TransformerGatedMlpConfig::new(n_embed, 24).into(),
        );
        assert!(config.init(0, &device).attn_post_norm.is_none());
        assert!(config.init(0, &device).mlp_post_norm.is_none());

        let mut block: CausalDecoderBlock = config.with_post_norms(true).init(0, &device);
        // Distinct gains, so a norm in the wrong place shows.
        for (seed, norm) in [
            (3, block.attn_norm.as_mut().expect("an attention norm")),
            (4, block.attn_post_norm.as_mut().expect("a post-norm")),
            (5, &mut block.mlp_norm),
            (6, block.mlp_post_norm.as_mut().expect("a post-norm")),
        ] {
            let BunsenNormalization::Rms(rms) = norm else {
                panic!("an RMS norm");
            };
            rms.gamma = Param::from_tensor(seeded_tensor::<1>(
                seed,
                [n_embed],
                Distribution::Uniform(0.5, 1.5),
                &device,
            ));
        }
        let r_emb = RotaryEmbeddingConfig::new(seq_len, block.attn.head_dim()).init(&device);
        let input =
            seeded_tensor::<3>(1, [batch, seq_len, n_embed], Distribution::Default, &device);

        let post =
            |norm: &Option<BunsenNormalization>, x| norm.as_ref().expect("a post-norm").forward(x);
        let x = input.clone()
            + post(
                &block.attn_post_norm,
                block
                    .attn
                    .forward(block.attn_input(input.clone()), &r_emb, &mut None),
            );
        let expected = x.clone()
            + post(
                &block.mlp_post_norm,
                block.mlp.forward(block.mlp_norm.forward(x)),
            );
        let output = block.forward(input, &r_emb, &mut None);
        assert_tensors_close(&output, &expected, Tolerance::default());
    }

    #[test]
    fn test_layer_rekey_with_post_norms() {
        let rekey = CausalDecoderBlock::layer_rekey(
            &DecoderLayerKeys::llama("input_layernorm", "pre_feedforward_layernorm")
                .with_post_norms("post_attention_layernorm", "post_feedforward_layernorm"),
        );
        for (stored, bunsen) in [
            ("input_layernorm.weight", "attn_norm.weight"),
            ("post_attention_layernorm.weight", "attn_post_norm.weight"),
            ("pre_feedforward_layernorm.weight", "mlp_norm.weight"),
            ("post_feedforward_layernorm.weight", "mlp_post_norm.weight"),
            ("self_attn.q_proj.weight", "attn.c_q.weight"),
            ("self_attn.q_norm.weight", "attn.q_norm.weight"),
            ("mlp.gate_proj.weight", "mlp.gate_proj.weight"),
        ] {
            assert_eq!(rekey.apply(stored).as_deref(), Some(bunsen), "{stored}");
        }
    }

    fn conditioned(n_embed: usize) -> CausalDecoderBlockConfig {
        CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(4, 2, n_embed).with_norm(None),
            crate::blocks::transformers::mlp::TransformerGatedMlpConfig::new(n_embed, 24).into(),
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
                .forward(block.attn_input(input.clone()), &r_emb, &mut None);
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
