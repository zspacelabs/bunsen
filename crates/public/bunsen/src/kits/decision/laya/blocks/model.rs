//! The Laya decision model.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        LayerNormConfig,
        Linear,
        LinearConfig,
    },
    prelude::{
        Bool,
        Int,
    },
    tensor::{
        DType,
        Device,
        activation::{
            gelu,
            softmax,
        },
    },
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        norm::LayerNorm,
        transformers::{
            embedding::{
                TokenEmbedding,
                TokenEmbeddingConfig,
            },
            encoders::residual::{
                ResidualEncoderAttentionBlock,
                ResidualEncoderAttentionBlockConfig,
            },
        },
    },
    burner::{
        module::ModuleInit,
        tensor::param_float_dtype,
    },
    contracts::unpack_shape_contract,
    errors::BunsenResult,
    kits::text::encoders::modernbert::{
        ModernBertConfig,
        ModernBertMeta,
        ModernBertModel,
    },
    ops::transformers::attention::length_mask,
};

/// A Laya question's type: what its options are, and which head
/// temperature calibrates it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LayaQuestionKind {
    /// A pick among named options.
    Choice,

    /// A level on an ordered scale, `level 0` up.
    Score,

    /// A yes/no probability: the options are `false` and `true`, in that
    /// order.
    Noul,
}

impl LayaQuestionKind {
    /// Every kind, by index.
    pub const ALL: [LayaQuestionKind; 3] = [Self::Choice, Self::Score, Self::Noul];

    /// Its index: its row of the type embedding, its temperature's slot.
    pub fn index(self) -> usize {
        self as usize
    }

    /// Its name, as upstream spells it (`choice`, `score`, `noul`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }
}

/// Config for [`LayaModel`]: a [`ModernBertConfig`] encoder and the decision
/// head over it, as upstream's `DecisionModel` builds it.
#[derive(Config, Debug)]
pub struct LayaConfig {
    /// The encoder.
    pub encoder: ModernBertConfig,

    /// The head's encoder layers (`rl_agent_config.json`'s `head_layers`).
    #[config(default = "2")]
    pub head_layers: usize,

    /// The act head's outputs: one per costed act, and one for none.
    #[config(default = "2")]
    pub n_act: usize,

    /// The act head's hidden width.
    #[config(default = "256")]
    pub act_hidden: usize,
}

impl LayaConfig {
    /// The width: the encoder's.
    pub fn d_model(&self) -> usize {
        self.encoder.d_model
    }

    /// A head layer: `PyTorch`'s `nn.TransformerEncoderLayer` as upstream
    /// builds it, pre-norm, `max(1, d / 64)` heads, a `4 d` `ReLU` MLP, and
    /// every projection and norm biased.
    pub fn head_layer(&self) -> ResidualEncoderAttentionBlockConfig {
        let d = self.d_model();
        let heads = (d / 64).max(1);
        ResidualEncoderAttentionBlockConfig::new(d)
            .with_d_head(d / heads)
            .with_key_bias(true)
            .with_activation(BunsenActivationConfig::Relu)
    }
}

impl ModuleInit<LayaModel> for LayaConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<LayaModel> {
        let d = self.d_model();
        let head = (0..self.head_layers)
            .map(|_| self.head_layer().try_init(device))
            .collect::<BunsenResult<_>>()?;
        Ok(LayaModel {
            encoder: self.encoder.try_init(device)?,
            type_emb: TokenEmbeddingConfig::new(LayaQuestionKind::ALL.len(), d).try_init(device)?,
            head,
            scorer: LayaScorer {
                norm: LayerNorm::init(&LayerNormConfig::new(d), device),
                dense: LinearConfig::new(d, d).init(device),
                out: LinearConfig::new(d, 1).init(device),
            },
            act_head: LayaActHead {
                dense: LinearConfig::new(d + 4, self.act_hidden).init(device),
                out: LinearConfig::new(self.act_hidden, self.n_act).init(device),
            },
        })
    }
}

/// The scorer: an option's marker state to its logit, `LayerNorm`, a
/// `Linear`, an exact `GELU` and a `Linear` to one.
#[derive(Module, Debug)]
pub struct LayaScorer {
    /// The input's norm.
    pub norm: LayerNorm,

    /// `d -> d`.
    pub dense: Linear,

    /// `d -> 1`.
    pub out: Linear,
}

impl LayaScorer {
    /// `[.., d]` to `[.., 1]`.
    pub fn forward<const D: usize>(
        &self,
        x: Tensor<D>,
    ) -> Tensor<D> {
        self.out
            .forward(gelu(self.dense.forward(self.norm.forward(x))))
    }
}

/// The act head: the first token's state and four statistics of the answer
/// distribution to the act logits, a `Linear`, an exact `GELU` and a
/// `Linear`.
#[derive(Module, Debug)]
pub struct LayaActHead {
    /// `d + 4 -> act_hidden`.
    pub dense: Linear,

    /// `act_hidden -> n_act`.
    pub out: Linear,
}

/// What [`LayaModel::forward`] gives: per question, its options' logits
/// and its act logits, both in f32.
#[derive(Debug, Clone)]
pub struct LayaOutput {
    /// `[B, K]`: each option's logit, uncalibrated; `-1e4` past a question's
    /// options.
    pub logits: Tensor<2>,

    /// `[B, n_act]`.
    pub act_logits: Tensor<2>,
}

/// Laya, an open decision model (`convaiinnovations/laya`): typed questions
/// about a state answered with probabilities over each question's options,
/// in one forward pass, as upstream's `DecisionModel` computes them.
///
/// Each prompt holds one question, its options each marked by a `[MASK]`
/// token, and the state. The [`ModernBertModel`] encodes it; the question
/// kind's type embedding is added at every position; two encoder layers
/// mix it, over the prompt's tokens alone; each option's marker state is
/// scored to a logit, and the logits past the question's options are
/// `-1e4`. The act head reads the first token's state and the answer
/// distribution's top probability, its margin, its normalized entropy and
/// the option count.
///
/// Built by [`LayaConfig`].
#[derive(Module, Debug)]
pub struct LayaModel {
    /// The encoder.
    pub encoder: ModernBertModel,

    /// `[3, d]`: each question kind's embedding.
    pub type_emb: TokenEmbedding,

    /// The head's encoder layers.
    pub head: Vec<ResidualEncoderAttentionBlock>,

    /// Each option's marker state to its logit.
    pub scorer: LayaScorer,

    /// The act head.
    pub act_head: LayaActHead,
}

impl LayaModel {
    /// The width.
    pub fn d_model(&self) -> usize {
        self.encoder.d_model()
    }

    /// The head's states: the prompts encoded, their question kinds'
    /// embeddings added, and mixed by the head's layers over each prompt's
    /// tokens.
    ///
    /// # Arguments
    /// * `ids` - `[B, T]`, the prompts.
    /// * `lengths` - each prompt's tokens, the rest padding.
    /// * `kinds` - `[B]`, each prompt's [`LayaQuestionKind::index`].
    ///
    /// # Returns
    /// `[B, T, d]`; a padding position's state means nothing.
    pub fn head_states(
        &self,
        ids: Tensor<2, Int>,
        lengths: &[usize],
        kinds: Tensor<1, Int>,
    ) -> Tensor<3> {
        let [b, t] = unpack_shape_contract!(["B", "T"], &ids.dims());
        let device = ids.device();
        let h = self.encoder.forward(ids, Some(lengths));
        let h = h + self.type_emb.forward(kinds.unsqueeze_dim::<2>(1));
        let padding = length_mask(lengths, t, &device)
            .bool_not()
            .unsqueeze_dim::<3>(1)
            .expand([b, t, t]);
        self.head
            .iter()
            .fold(h, |h, layer| layer.forward_masked(h, Some(padding.clone())))
    }

    /// Each question's option logits and act logits.
    ///
    /// # Arguments
    /// * `ids` - `[B, T]`, the prompts, one question each.
    /// * `lengths` - each prompt's tokens, the rest padding.
    /// * `kinds` - `[B]`, each prompt's [`LayaQuestionKind::index`].
    /// * `markers` - `[B, K]`, each option's marker position; any position past
    ///   a question's options.
    /// * `options` - `[B, K]`, true at a question's options.
    pub fn forward(
        &self,
        ids: Tensor<2, Int>,
        lengths: &[usize],
        kinds: Tensor<1, Int>,
        markers: Tensor<2, Int>,
        options: Tensor<2, Bool>,
    ) -> LayaOutput {
        let [b, k] = unpack_shape_contract!(["B", "K"], &markers.dims());
        let d = self.d_model();
        let h = self.head_states(ids, lengths, kinds);

        let at = markers.clamp_min(0).unsqueeze_dim::<3>(2).expand([b, k, d]);
        let marked = h.clone().gather(1, at);
        let logits = self
            .scorer
            .forward(marked)
            .squeeze_dim::<2>(2)
            .cast(DType::F32)
            .mask_fill(options.clone().bool_not(), -1e4);

        // The answer distribution's statistics, as the act head reads them.
        let p = softmax(logits.clone(), 1);
        let count = options
            .int()
            .sum_dim(1)
            .float()
            .cast(DType::F32)
            .clamp_min(2.0);
        let entropy = (p.clone() * p.clone().clamp_min(1e-9).log())
            .sum_dim(1)
            .neg()
            / count.clone().log();
        let top = p.topk(2, 1);
        let first = top.clone().slice_dim(1, 0..1);
        let margin = first.clone() - top.slice_dim(1, 1..2);
        let feats = Tensor::cat(vec![first, margin, entropy, count / 255.0], 1);
        let pooled = h.slice_dim(1, 0..1).squeeze_dim::<2>(1).cast(DType::F32);
        let act_in = Tensor::cat(vec![pooled, feats], 1)
            .cast(param_float_dtype(&self.act_head.dense.weight));
        let act_logits = self
            .act_head
            .out
            .forward(gelu(self.act_head.dense.forward(act_in)))
            .cast(DType::F32);
        LayaOutput { logits, act_logits }
    }
}
