//! The Voxtral Realtime text model.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Embedding,
        EmbeddingConfig,
        norm::{
            NormalizationConfig,
            RmsNorm,
            RmsNormConfig,
        },
    },
    prelude::Int,
    tensor::Device,
};

use crate::{
    blocks::transformers::{
        attention::csa::CausalSelfAttentionConfig,
        conditioning::AdaptiveScaleConfig,
        decoders::causal::{
            CausalDecoderBlock,
            CausalDecoderBlockConfig,
        },
        embedding::{
            RotaryDirection,
            RotaryEmbedding,
            RotaryEmbeddingConfig,
        },
        mlp::GatedMlpConfig,
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
    ops::{
        embedding::{
            timestep_embedding,
            unembed,
        },
        transformers::attention::{
            KVCache,
            KVCacheConfig,
            KVCacheModel,
        },
    },
};

/// The positions a text cache allocates up front by default: 80 s of decoder
/// positions.
pub const VOXTRAL_TEXT_KV_CACHE_PRE_SIZE: usize = 1024;

/// The bottleneck of each layer's delay conditioning
/// (`transformers`' `ada_rms_norm`, fixed at 32).
pub const VOXTRAL_ADA_HIDDEN: usize = 32;

/// The base of the delay's sinusoidal embedding.
pub const VOXTRAL_TIME_THETA: f32 = 10_000.0;

/// Common meta for [`VoxtralTextModel`] and [`VoxtralTextConfig`].
pub trait VoxtralTextMeta {
    /// The token count.
    fn n_vocab(&self) -> usize;

    /// The model width.
    fn d_model(&self) -> usize;

    /// The layer count.
    fn n_layers(&self) -> usize;

    /// The key/value head count.
    fn n_kv_heads(&self) -> usize;

    /// The head width.
    fn head_dim(&self) -> usize;

    /// The attention window, in positions, if any.
    fn sliding_window(&self) -> Option<usize>;
}

fn kv_cache_config(meta: &impl VoxtralTextMeta) -> KVCacheConfig {
    KVCacheConfig::new(
        1,
        meta.n_kv_heads(),
        VOXTRAL_TEXT_KV_CACHE_PRE_SIZE,
        meta.head_dim(),
        meta.n_layers(),
    )
}

impl KVCacheModel for VoxtralTextConfig {
    fn kv_cache_config(&self) -> KVCacheConfig {
        kv_cache_config(self)
    }
}

impl KVCacheModel for VoxtralTextModel {
    fn kv_cache_config(&self) -> KVCacheConfig {
        kv_cache_config(self)
    }
}

/// Config for [`VoxtralTextModel`]: `transformers`'
/// `VoxtralRealtimeTextConfig`.
///
/// Implements [`VoxtralTextMeta`].
#[derive(Config, Debug)]
pub struct VoxtralTextConfig {
    /// The token count.
    pub n_vocab: usize,

    /// The model width.
    pub d_model: usize,

    /// The layer count.
    pub n_layers: usize,

    /// The query head count.
    pub n_heads: usize,

    /// The key/value head count.
    pub n_kv_heads: usize,

    /// The head width.
    pub head_dim: usize,

    /// The feed-forward hidden width.
    pub d_ff: usize,

    /// The RMS normalization epsilon.
    #[config(default = "1e-5")]
    pub rms_eps: f64,

    /// The rotary embedding's base.
    #[config(default = "1_000_000")]
    pub rope_theta: usize,

    /// The attention window, in positions; `None` attends to the whole past.
    #[config(default = "Some(8192)")]
    pub sliding_window: Option<usize>,
}

impl VoxtralTextConfig {
    /// `Voxtral-Mini-4B-Realtime-2602`'s (`Ministral-3-3B`'s geometry):
    /// 131072 tokens, 26 layers of width 3072, 32 query and 8 key/value heads
    /// of 128, a 9216-wide feed-forward, an 8192-position window.
    pub fn voxtral_mini_4b_realtime() -> Self {
        Self::new(131_072, 3072, 26, 32, 8, 128, 9216)
    }
}

impl VoxtralTextMeta for VoxtralTextConfig {
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }

    fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }
}

impl ModuleInit<VoxtralTextModel> for VoxtralTextConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<VoxtralTextModel> {
        let norm = |n| RmsNormConfig::new(n).with_epsilon(self.rms_eps);
        let layer = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(self.n_heads, self.n_kv_heads, self.d_model)
                .with_head_dim(Some(self.head_dim))
                .with_norm(None)
                .with_sliding_window(self.sliding_window),
            GatedMlpConfig::new(self.d_model, self.d_ff).into(),
        )
        .with_norm(NormalizationConfig::Rms(norm(0)))
        .with_mlp_scale(Some(AdaptiveScaleConfig::new(
            self.d_model,
            VOXTRAL_ADA_HIDDEN,
        )));
        let layers = (0..self.n_layers)
            .map(|i| layer.try_init(i, device))
            .collect::<BunsenResult<_>>()?;
        Ok(VoxtralTextModel {
            embed_tokens: EmbeddingConfig::new(self.n_vocab, self.d_model).init(device),
            layers,
            norm: norm(self.d_model).init(device),
            rope_theta: self.rope_theta,
        })
    }
}

/// Voxtral Realtime's text model: Mistral's decoder (`Ministral-3`), each
/// layer told the transcription delay, with its head tied to its embedding.
///
/// Its layers are the generic [`CausalDecoderBlock`]: pre-norm RMS,
/// grouped-query causal self-attention with rotary embeddings and a sliding
/// window, no biases and no QK-norm, a `SwiGLU`; and an
/// [`AdaptiveScale`](crate::blocks::transformers::conditioning::AdaptiveScale)
/// on each feed-forward's input, conditioned on the delay's sinusoidal
/// embedding ([`delay_condition`](Self::delay_condition)).
///
/// Its own fields carry the names `transformers`' text model gives them;
/// [`weights::hf_rekey`](crate::kits::speech::voxtral::weights::hf_rekey)
/// maps its layers'.
///
/// Built by [`VoxtralTextConfig`].
/// Implements [`VoxtralTextMeta`].
#[derive(Module, Debug)]
pub struct VoxtralTextModel {
    /// The token embedding, and the tied output head.
    pub embed_tokens: Embedding,

    /// The decoder layers.
    pub layers: Vec<CausalDecoderBlock>,

    /// The final normalization.
    pub norm: RmsNorm,

    /// The rotary embedding's base.
    pub rope_theta: usize,
}

impl VoxtralTextMeta for VoxtralTextModel {
    fn n_vocab(&self) -> usize {
        self.embed_tokens.weight.dims()[0]
    }

    fn d_model(&self) -> usize {
        self.embed_tokens.weight.dims()[1]
    }

    fn n_layers(&self) -> usize {
        self.layers.len()
    }

    fn n_kv_heads(&self) -> usize {
        use crate::blocks::transformers::attention::csa::CausalSelfAttentionMeta;
        self.layers[0].attn.n_kv_head()
    }

    fn head_dim(&self) -> usize {
        self.layers[0].attn.head_dim
    }

    fn sliding_window(&self) -> Option<usize> {
        self.layers[0].attn.sliding_window
    }
}

impl VoxtralTextModel {
    /// The rotary embedding for positions `0..seq_len`.
    pub fn rotary(
        &self,
        seq_len: usize,
        device: &Device,
    ) -> RotaryEmbedding {
        RotaryEmbeddingConfig::new(seq_len, self.head_dim())
            .with_base(self.rope_theta)
            .with_direction(RotaryDirection::Forward)
            .init(device)
    }

    /// The condition every layer is told a delay of `delay_tokens` decoder
    /// positions by: its sinusoidal embedding
    /// ([`timestep_embedding`](crate::ops::embedding::timestep_embedding)).
    ///
    /// # Returns
    /// `[1, 1, d_model]`.
    pub fn delay_condition(
        &self,
        delay_tokens: usize,
        device: &Device,
    ) -> Tensor<3> {
        timestep_embedding(
            delay_tokens as f32,
            self.d_model(),
            VOXTRAL_TIME_THETA,
            device,
        )
        .reshape([1, 1, self.d_model()])
    }

    /// Embeds tokens.
    ///
    /// # Arguments
    /// * `tokens` - `[batch, seq]`.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`.
    pub fn embed(
        &self,
        tokens: Tensor<2, Int>,
    ) -> Tensor<3> {
        #[cfg(any(debug_assertions, test))]
        let [batch, seq] =
            crate::contracts::unpack_shape_contract!(["batch", "seq"], &tokens.dims());

        let y = self.embed_tokens.forward(tokens);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "d_model"],
            &y.dims(),
            &[("batch", batch), ("seq", seq), ("d_model", self.d_model())],
        );

        y
    }

    /// Runs the decoder over input embeddings at positions
    /// `start..start + seq`, through the cache when there is one.
    ///
    /// # Arguments
    /// * `embeds` - `[batch, seq, d_model]`: each position's token embedding
    ///   plus its audio embedding.
    /// * `r_emb` - a rotary embedding covering `start + seq` positions.
    /// * `cond` - the `[1, 1, d_model]`
    ///   [`delay_condition`](Self::delay_condition).
    /// * `kv_cache` - the cache, at position `start`; `None` means `start` is 0
    ///   and nothing is kept.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`, normalized: the head's input.
    pub fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotaryEmbedding,
        cond: &Tensor<3>,
        mut kv_cache: Option<&mut KVCache>,
    ) -> Tensor<3> {
        let [batch, seq] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "d_model"],
            &embeds.dims(),
            &["batch", "seq"],
            &[("d_model", self.d_model())]
        );
        let start = kv_cache.as_ref().map(|c| c.pos()).unwrap_or(0);
        let r_emb = r_emb.clip_range(start..start + seq);
        let mut x = embeds;
        for layer in &self.layers {
            x = layer.forward_conditioned(x, &r_emb, &mut kv_cache, Some(cond.clone()));
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

    /// The tied head: logits over the vocabulary.
    ///
    /// # Arguments
    /// * `hidden` - `[batch, seq, d_model]`, from
    ///   [`forward_hidden`](Self::forward_hidden).
    ///
    /// # Returns
    /// `[batch, seq, n_vocab]`.
    pub fn logits(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        #[cfg(any(debug_assertions, test))]
        let [batch, seq] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "d_model"],
            &hidden.dims(),
            &["batch", "seq"],
            &[("d_model", self.d_model())],
        );

        let y = unembed(&self.embed_tokens, hidden);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "n_vocab"],
            &y.dims(),
            &[("batch", batch), ("seq", seq), ("n_vocab", self.n_vocab())],
        );

        y
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
    };

    /// A config non-default in every field, and the model built from it,
    /// answer the meta alike; a cached decode's hidden states are the
    /// uncached forward's, past the window too.
    #[test]
    fn test_meta_and_cache() {
        let config = VoxtralTextConfig::new(40, 12, 2, 3, 1, 6, 20)
            .with_rms_eps(1e-6)
            .with_rope_theta(500)
            .with_sliding_window(Some(3));
        let device = cpu_device();
        let model: VoxtralTextModel = config.init(&device);
        assert_eq!(model.n_vocab(), config.n_vocab());
        assert_eq!(model.d_model(), config.d_model());
        assert_eq!(model.n_layers(), config.n_layers());
        assert_eq!(model.n_kv_heads(), config.n_kv_heads());
        assert_eq!(model.head_dim(), config.head_dim());
        assert_eq!(model.sliding_window(), config.sliding_window());
        assert_eq!(model.rope_theta, config.rope_theta);
        assert_eq!(model.norm.epsilon, 1e-6);
        assert!(model.layers[0].mlp_scale.is_some());

        let tokens = Tensor::<2, Int>::from_data([[1, 5, 7, 2, 9, 3]], &device);
        let embeds = model.embed(tokens);
        let r_emb = model.rotary(8, &device);
        let cond = model.delay_condition(6, &device);
        let whole = model.forward_hidden(embeds.clone(), &r_emb, &cond, None);
        let mut cache = model.kv_cache_config().with_seq_len(8).init();
        let head = model.forward_hidden(
            embeds.clone().slice_dim(1, 0..4),
            &r_emb,
            &cond,
            Some(&mut cache),
        );
        let tail = model.forward_hidden(embeds.slice_dim(1, 4..6), &r_emb, &cond, Some(&mut cache));
        assert_tensors_close(
            &Tensor::cat(vec![head, tail], 1),
            &whole,
            Tolerance::rel_abs(1e-4, 1e-5),
        );
        assert_eq!(model.logits(whole).dims(), [1, 6, 40]);
        assert_eq!(cond.dims(), [1, 1, 12]);
    }
}
