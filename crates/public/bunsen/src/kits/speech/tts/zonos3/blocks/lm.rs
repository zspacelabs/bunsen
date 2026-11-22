//! [`Zonos3Lm`]: the language model, and its config.

use alloc::vec::Vec;

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Embedding,
        EmbeddingConfig,
        Linear,
        LinearConfig,
        norm::RmsNormConfig,
    },
    prelude::{
        Int,
        s,
    },
    tensor::Device,
};

use crate::{
    blocks::{
        mlp::BunsenGatedMlpConfig,
        norm::{
            BunsenNormalizationConfig,
            GatedNormConfig,
        },
        transformers::{
            attention::csa::{
                AttentionOutputGate,
                CausalSelfAttentionConfig,
                QkNormPlacement,
            },
            decoders::{
                attn_res::{
                    AttnResStack,
                    AttnResStackConfig,
                },
                causal::CausalDecoderBlockConfig,
                stack::{
                    DecoderStackCache,
                    DecoderStackMeta,
                    DecoderStackStructureConfig,
                },
            },
            embedding::{
                RotaryDirection,
                RotaryEmbeddingConfig,
                RotaryPairLayout,
            },
            mixers::{
                GatedDeltaNetConfig,
                TokenMixerConfig,
            },
            mlp::TransformerFeedForwardConfig,
            moe::{
                experts::ExpertStackConfig,
                router::{
                    ChoiceBias,
                    ChoiceBiasMode,
                    ChoiceBiasSite,
                    TopKRouterConfig,
                },
                sparse::SparseMoeConfig,
            },
        },
    },
    burner::{
        module::{
            CanonicalKeys,
            ModuleInit,
            Rekey,
        },
        tensor::Widening,
    },
    contracts::unpack_shape_contract,
    errors::BunsenResult,
    ops::transformers::attention::KVCacheStorage,
};

/// Config for [`Zonos3Lm`](super::Zonos3Lm): ZONOS3-Mini's language model,
/// rows of `n_codebooks + 1` ids in, each row's logits over every codebook's
/// codes out.
#[derive(Config, Debug)]
pub struct Zonos3LmConfig {
    /// The model width, `D`.
    pub d_model: usize,

    /// The layers.
    pub n_layers: usize,

    /// Every `global_period`-th layer (`(l + 1) % global_period == 0`)
    /// attends; the others are gated `DeltaNet`s.
    pub global_period: usize,

    /// The audio codebooks.
    pub n_codebooks: usize,

    /// Codes per codebook; each codebook's vocabulary is two more: the end
    /// of audio, then the pad.
    pub codebook_size: usize,

    /// The text table's rows.
    pub text_rows: usize,

    /// The attention's query heads.
    pub n_heads: usize,

    /// The attention's key and value heads.
    pub n_kv_heads: usize,

    /// Each attention head's width.
    pub head_dim: usize,

    /// The rotary base.
    pub rope_base: f64,

    /// The gated `DeltaNet` heads.
    pub gdn_heads: usize,

    /// Each gated `DeltaNet` head's width.
    pub gdn_head_dim: usize,

    /// The gated `DeltaNet`'s convolution kernel.
    pub gdn_kernel: usize,

    /// The first layer's dense `SwiGLU` hidden width.
    pub ffn_hidden: usize,

    /// The experts of every later layer's mixture.
    pub n_experts: usize,

    /// The experts each row is routed to.
    pub top_k: usize,

    /// Each expert's hidden width.
    pub moe_hidden: usize,

    /// The latent the experts run in.
    pub moe_latent: usize,

    /// The speaker embedding's width.
    pub speaker_dim: usize,

    /// Each condition table's buckets, without its unknown bucket (the table
    /// has one more row).
    pub condition_buckets: Vec<usize>,

    /// The longest sequence of rows.
    pub max_rows: usize,

    /// The RMS norms' epsilon: the embedding's, the sublayers', the
    /// router's, the routes' keys and the final norm.
    #[config(default = 1e-5)]
    pub norm_eps: f64,

    /// The attention's weightless QK-norm epsilon.
    #[config(default = 1e-6)]
    pub qk_norm_eps: f64,

    /// The gated `DeltaNet`'s output norm epsilon.
    #[config(default = 1e-5)]
    pub gdn_norm_eps: f64,

    /// The logits' tanh soft-cap.
    #[config(default = 15.0)]
    pub softcap: f64,
}

impl Zonos3LmConfig {
    /// Whether layer `l` attends (else it is a gated `DeltaNet`).
    pub fn layer_attends(
        &self,
        l: usize,
    ) -> bool {
        (l + 1).is_multiple_of(self.global_period)
    }

    /// An RMS norm at the model's epsilon.
    fn rms(&self) -> BunsenNormalizationConfig {
        BunsenNormalizationConfig::Rms(RmsNormConfig::new(0).with_epsilon(self.norm_eps))
    }

    /// Layer `l`'s block: a global attention or a gated `DeltaNet`, then the
    /// first layer's dense `SwiGLU` or a later layer's latent mixture of
    /// experts; each sublayer behind an RMS norm.
    pub fn layer(
        &self,
        l: usize,
    ) -> CausalDecoderBlockConfig {
        let d = self.d_model;
        let mixer: TokenMixerConfig = if self.layer_attends(l) {
            CausalSelfAttentionConfig::new(self.n_heads, self.n_kv_heads, d)
                .with_head_dim(Some(self.head_dim))
                .with_norm(Some(BunsenNormalizationConfig::Rms(
                    RmsNormConfig::new(0).with_epsilon(self.qk_norm_eps),
                )))
                .with_qk_norm(QkNormPlacement::BeforeRope)
                .with_query_gain(true)
                .with_output_gate(Some(AttentionOutputGate::Headwise))
                .into()
        } else {
            GatedDeltaNetConfig::new(
                d,
                self.gdn_heads,
                self.gdn_heads,
                self.gdn_head_dim,
                self.gdn_head_dim,
            )
            .with_conv_kernel(self.gdn_kernel)
            .with_norm(GatedNormConfig::rms(0, self.gdn_norm_eps))
            .into()
        };
        let ffn: TransformerFeedForwardConfig = if l == 0 {
            BunsenGatedMlpConfig::new(d, self.ffn_hidden).into()
        } else {
            SparseMoeConfig::new(
                TopKRouterConfig::new(d, self.n_experts, self.top_k)
                    .with_choice_bias(Some(ChoiceBias::new(
                        ChoiceBiasSite::Logits,
                        ChoiceBiasMode::Subtract,
                    )))
                    .with_input_norm(Some(self.rms())),
                ExpertStackConfig::new(self.n_experts, self.moe_latent, self.moe_hidden),
            )
            .with_latent(Some(self.moe_latent))
            .into()
        };
        CausalDecoderBlockConfig::new(mixer, ffn).with_norm(self.rms())
    }

    /// The stack: the layers without a residual stream (Full Attention
    /// Residuals), a weightless input norm (the embedding's), the final
    /// norm, and the interleaved rotary table the attention layers share,
    /// which rotates half inputs in f32, as the reference does.
    pub fn stack(&self) -> AttnResStackConfig {
        let layers = (0..self.n_layers).map(|l| self.layer(l)).collect();
        let rotary = RotaryEmbeddingConfig::new(0, self.head_dim)
            .with_base(self.rope_base)
            .with_direction(RotaryDirection::Forward)
            .with_pair_layout(RotaryPairLayout::Interleaved)
            .with_widening(Widening::F32);
        AttnResStackConfig::new(
            DecoderStackStructureConfig::new(layers, self.rms(), rotary)
                .with_input_norm(Some(self.rms()))
                .into(),
        )
        .with_key_eps(self.norm_eps)
    }
}

/// A [`Zonos3Lm`]'s geometry, which its config and the model answer alike.
pub trait Zonos3LmMeta {
    /// The model width, `D`.
    fn d_model(&self) -> usize;

    /// The layers.
    fn n_layers(&self) -> usize;

    /// The audio codebooks, `K`.
    fn n_codebooks(&self) -> usize;

    /// Codes per codebook.
    fn codebook_size(&self) -> usize;

    /// The text table's rows.
    fn text_rows(&self) -> usize;

    /// The speaker embedding's width, `S`.
    fn speaker_dim(&self) -> usize;

    /// Each codebook's vocabulary, `V_a`: the codes, the end of audio and
    /// the pad.
    fn audio_vocab(&self) -> usize {
        self.codebook_size() + 2
    }

    /// The end-of-audio id.
    fn eoa_id(&self) -> usize {
        self.codebook_size()
    }

    /// The audio pad id.
    fn audio_pad_id(&self) -> usize {
        self.codebook_size() + 1
    }
}

impl Zonos3LmMeta for Zonos3LmConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }

    fn n_codebooks(&self) -> usize {
        self.n_codebooks
    }

    fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    fn text_rows(&self) -> usize {
        self.text_rows
    }

    fn speaker_dim(&self) -> usize {
        self.speaker_dim
    }
}

impl Zonos3LmMeta for Zonos3Lm {
    fn d_model(&self) -> usize {
        self.head.weight.dims()[0]
    }

    fn n_layers(&self) -> usize {
        self.stack.stack.n_layers()
    }

    fn n_codebooks(&self) -> usize {
        self.n_codebooks
    }

    fn codebook_size(&self) -> usize {
        self.audio_vocab - 2
    }

    fn text_rows(&self) -> usize {
        self.embedders[self.n_codebooks].weight.dims()[0]
    }

    fn speaker_dim(&self) -> usize {
        self.speaker_projection.weight.dims()[0]
    }
}

impl ModuleInit<Zonos3Lm> for Zonos3LmConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Zonos3Lm> {
        let d = self.d_model;
        let audio_vocab = self.audio_vocab();
        let embedders = (0..self.n_codebooks)
            .map(|_| audio_vocab)
            .chain([self.text_rows])
            .map(|rows| EmbeddingConfig::new(rows, d).init(device))
            .collect();
        Ok(Zonos3Lm {
            embedders,
            speaker_projection: LinearConfig::new(self.speaker_dim, d).init(device),
            condition_embeddings: self
                .condition_buckets
                .iter()
                .map(|&n| EmbeddingConfig::new(n + 1, d).init(device))
                .collect(),
            stack: self.stack().try_init(device)?,
            head: LinearConfig::new(d, self.n_codebooks * audio_vocab)
                .with_bias(false)
                .init(device),
            n_codebooks: self.n_codebooks,
            audio_vocab,
            softcap: self.softcap,
        })
    }
}

/// The rows of a prompt whose embedding is replaced rather than looked up:
/// a speaker row, and the condition rows.
#[derive(Clone, Debug, Default)]
pub struct Zonos3RowConditioning {
    /// A speaker row: its absolute position, and the `[B, S]` speaker
    /// embedding projected in its place, in any float dtype (cast to the
    /// model's before the projection).
    pub speaker: Option<(usize, Tensor<2>)>,

    /// The condition rows: the first one's absolute position, and `[B, C]`
    /// bucket ids; condition `i`'s row, `first + i`, is its table's row for
    /// its bucket.
    pub conditions: Option<(usize, Tensor<2, Int>)>,
}

/// What [`Zonos3Lm::forward_traced`] computed on the way to the logits.
#[derive(Clone, Debug)]
pub struct Zonos3LmTrace {
    /// `[B, T, D]`: the summed row embeddings.
    pub emb_sum: Tensor<3>,

    /// `[B, T, D]`: with the speaker and condition rows replaced.
    pub emb_cond: Tensor<3>,

    /// The stack's bank: the normalized embedding, then each layer's
    /// attention and feed-forward outputs, each `[B, T, D]`.
    pub bank: Vec<Tensor<3>>,

    /// `[B, T, D]`: the stack's output, after its final norm.
    pub hidden: Tensor<3>,
}

/// ZONOS3-Mini's language model: rows of ids in, each row's logits over
/// every codebook's vocabulary out.
///
/// A row is `n_codebooks` audio ids and a text id; its embedding is the sum
/// of each id's row in its own table (the text table last). A speaker row's
/// and the condition rows' embeddings are replaced instead
/// ([`Zonos3RowConditioning`]). The rows go through an [`AttnResStack`] of
/// gated `DeltaNet` and global attention layers, the first with a dense
/// `SwiGLU`, the rest with a latent mixture of experts; one linear then
/// gives each codebook's logits, soft-capped by `c · tanh(x / c)`.
///
/// Built by [`Zonos3LmConfig`].
#[derive(Module, Debug)]
pub struct Zonos3Lm {
    /// The row embeddings: each codebook's, `[V_a, D]`, then the text's.
    pub embedders: Vec<Embedding>,

    /// The speaker embedding's projection, `[S, D]`, with a bias.
    pub speaker_projection: Linear,

    /// Each condition's table, its buckets and an unknown bucket last.
    pub condition_embeddings: Vec<Embedding>,

    /// The layers.
    pub stack: AttnResStack,

    /// Every codebook's logits, `[D, K V_a]`.
    pub head: Linear,

    /// The audio codebooks, `K`.
    #[module(skip)]
    pub n_codebooks: usize,

    /// Each codebook's vocabulary, `V_a`.
    #[module(skip)]
    pub audio_vocab: usize,

    /// The logits' soft-cap.
    #[module(skip)]
    pub softcap: f64,
}

impl Zonos3Lm {
    /// A decode cache for `rows` batch rows of at most `positions` positions
    /// ([`AttnResStack::decode_cache`]).
    pub fn decode_cache(
        &self,
        rows: usize,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> DecoderStackCache {
        self.stack.decode_cache(rows, positions, storage, device)
    }

    /// The rows' embeddings at absolute positions `start..start + T`: the
    /// summed lookups, and the same with the conditioned rows replaced.
    ///
    /// # Arguments
    /// - `ids`: `[B, T, K + 1]`.
    /// - `rows`: the conditioned rows; those outside the span are ignored.
    /// - `start`: the first row's position.
    ///
    /// # Returns
    /// - `([B, T, D], [B, T, D])`.
    pub fn embed(
        &self,
        ids: Tensor<3, Int>,
        rows: &Zonos3RowConditioning,
        start: usize,
    ) -> (Tensor<3>, Tensor<3>) {
        let [b, t] = unpack_shape_contract!(
            ["B", "T", "C"],
            &ids.dims(),
            &["B", "T"],
            &[("C", self.n_codebooks + 1)]
        );
        let sum = self
            .embedders
            .iter()
            .enumerate()
            .map(|(c, table)| {
                table.forward(ids.clone().slice(s![.., .., c..c + 1]).reshape([b, t]))
            })
            .reduce(|a, b| a + b)
            .expect("at least the text table");
        let in_span = |row: usize| (start..start + t).contains(&row).then(|| row - start);

        let mut x = sum.clone();
        if let Some((row, speaker)) = &rows.speaker
            && let Some(at) = in_span(*row)
        {
            let projected = self
                .speaker_projection
                .forward(speaker.clone().cast(sum.dtype()));
            x = x.slice_assign(s![.., at..at + 1, ..], projected.unsqueeze_dim(1));
        }
        if let Some((first, buckets)) = &rows.conditions {
            for (i, table) in self.condition_embeddings.iter().enumerate() {
                if let Some(at) = in_span(first + i) {
                    let looked_up = table.forward(buckets.clone().slice(s![.., i..i + 1]));
                    x = x.slice_assign(s![.., at..at + 1, ..], looked_up);
                }
            }
        }
        (sum, x)
    }

    /// Each row's logits over every codebook's vocabulary.
    ///
    /// # Arguments
    /// - `ids`: `[B, T, K + 1]`, at the cache's position (0 without one).
    /// - `rows`: the conditioned rows, by absolute position.
    /// - `cache`: a decode cache ([`decode_cache`](Self::decode_cache)),
    ///   advanced by `T`; `None` runs the rows alone.
    ///
    /// # Returns
    /// - `[B, T, K, V_a]`, soft-capped.
    pub fn forward(
        &self,
        ids: Tensor<3, Int>,
        rows: &Zonos3RowConditioning,
        cache: Option<&mut DecoderStackCache>,
    ) -> Tensor<4> {
        self.forward_traced(ids, rows, cache).0
    }

    /// [`forward`](Self::forward), with what it computed on the way.
    pub fn forward_traced(
        &self,
        ids: Tensor<3, Int>,
        rows: &Zonos3RowConditioning,
        cache: Option<&mut DecoderStackCache>,
    ) -> (Tensor<4>, Zonos3LmTrace) {
        let [b, t, _] = ids.dims();
        let start = cache.as_ref().map_or(0, |c| c.pos());
        let (emb_sum, emb_cond) = self.embed(ids, rows, start);
        let (hidden, bank) = match cache {
            Some(cache) => self.stack.forward_bank(
                emb_cond.clone(),
                &cache.rotary,
                Some(&mut cache.cache),
                Some(&mut cache.linear),
            ),
            None => {
                let rotary = self.stack.stack.rotary(t, &emb_cond.device());
                self.stack
                    .forward_bank(emb_cond.clone(), &rotary, None, None)
            }
        };
        let c = self.softcap;
        let logits = self
            .head
            .forward(hidden.clone())
            .div_scalar(c)
            .tanh()
            .mul_scalar(c)
            .reshape([b, t, self.n_codebooks, self.audio_vocab]);
        (
            logits,
            Zonos3LmTrace {
                emb_sum,
                emb_cond,
                bank,
                hidden,
            },
        )
    }
}

impl CanonicalKeys for Zonos3Lm {
    fn canonical_rekey(&self) -> Rekey {
        Rekey::keep_rest().mount_canonical("stack.", &self.stack)
    }
}
