//! The language model's `config.json` section and `-hf` names.

use alloc::{
    collections::BTreeMap,
    format,
    string::String,
    vec::Vec,
};

use serde::Deserialize;

use crate::{
    burner::module::{
        Rekey,
        Select,
        TensorLoadRowSplit,
        TensorLoadTransform,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::tts::zonos3::{
        blocks::Zonos3LmConfig,
        prompt::Zonos3Condition,
    },
};

/// The `lm` section of `Zyphra/ZONOS3-Mini-hf`'s `config.json`, under
/// upstream's names.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Zonos3LmMirror {
    /// The audio pad id, `codebook_size + 1`.
    pub audio_pad_id: usize,
    /// Codes per codebook.
    pub codebook_size: usize,
    /// Each condition table's buckets, without its unknown bucket.
    pub condition_bucket_counts: Vec<usize>,
    /// Each condition's bucket edges.
    pub condition_edges: BTreeMap<String, Vec<f64>>,
    /// The conditions, in table order.
    pub condition_names: Vec<String>,
    /// The model width.
    pub dim: usize,
    /// The end-of-audio id, `codebook_size`.
    pub eoa_id: usize,
    /// The dense feed-forward's hidden width.
    pub ffn_hidden: usize,
    /// Each gated `DeltaNet` head's width.
    pub gdn_head_dim: usize,
    /// The gated `DeltaNet` heads (query, key and value alike).
    pub gdn_heads: usize,
    /// The gated `DeltaNet`'s convolution kernel.
    pub gdn_kernel: usize,
    /// Every `gdn_period`-th layer attends; the others are gated `DeltaNet`s.
    pub gdn_period: usize,
    /// Each attention head's width.
    pub head_dim: usize,
    /// The logits' tanh soft-cap.
    pub loss_softcap: f64,
    /// The longest sequence of rows.
    pub max_seqlen: usize,
    /// Each expert's hidden width.
    pub moe_hidden: usize,
    /// The latent the experts run in.
    pub moe_routed_dim: usize,
    /// The experts each row is routed to.
    pub moe_topk: usize,
    /// The audio codebooks.
    pub n_codebooks: usize,
    /// The experts.
    pub n_experts: usize,
    /// The attention's query heads.
    pub n_heads: usize,
    /// The attention's key and value heads.
    pub n_kv_heads: usize,
    /// The layers.
    pub n_layers: usize,
    /// The RMS norms' epsilon.
    pub norm_eps: f64,
    /// The most frames of reference codes a prompt holds.
    pub reference_max_frames: usize,
    /// The rotary base.
    pub rope_theta: f64,
    /// The speaker embedding's width.
    pub speaker_embedding_dim: usize,
    /// The text table's rows: the vocabulary and its padding row.
    pub text_rows: usize,
    /// The text vocabulary.
    pub text_vocab: usize,
}

impl Zonos3LmMirror {
    /// The [`Zonos3LmConfig`] of the section.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for
    /// special ids other than the end of audio at `codebook_size` and the pad
    /// after it.
    pub fn lm_config(&self) -> BunsenResult<Zonos3LmConfig> {
        let json = self;
        if json.eoa_id != json.codebook_size || json.audio_pad_id != json.codebook_size + 1 {
            return Err(BunsenError::unsupported(format!(
                "end of audio {} and pad {} for {} codes",
                json.eoa_id, json.audio_pad_id, json.codebook_size
            )));
        }
        Ok(Zonos3LmConfig::new(
            json.dim,
            json.n_layers,
            json.gdn_period,
            json.n_codebooks,
            json.codebook_size,
            json.text_rows,
            json.n_heads,
            json.n_kv_heads,
            json.head_dim,
            json.rope_theta,
            json.gdn_heads,
            json.gdn_head_dim,
            json.gdn_kernel,
            json.ffn_hidden,
            json.n_experts,
            json.moe_topk,
            json.moe_hidden,
            json.moe_routed_dim,
            json.speaker_embedding_dim,
            json.condition_bucket_counts.clone(),
            json.max_seqlen,
        )
        .with_norm_eps(json.norm_eps)
        .with_softcap(json.loss_softcap))
    }

    /// The conditions a prompt carries, in table order: each of
    /// `condition_names` with its `condition_edges`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// for a condition without edges.
    pub fn conditions(&self) -> BunsenResult<Vec<Zonos3Condition>> {
        self.condition_names
            .iter()
            .map(|name| {
                let edges = self.condition_edges.get(name).ok_or_else(|| {
                    BunsenError::invalid_resource(format!("`lm.condition_edges` has no `{name}`"))
                })?;
                Ok(Zonos3Condition {
                    name: name.clone(),
                    edges: edges.clone(),
                })
            })
            .collect()
    }
}

/// The prefix the language model's tensors sit under in the `-hf` file.
pub const ZONOS3_LM_PREFIX: &str = "lm.";

/// The packed projections, split by rows into the parts the blocks hold
/// apart, named in the checkpoint's terms:
///
/// - a gated `DeltaNet`'s `attention.in_proj`, `[3 I + I + 2 H, D]`, into
///   `in_proj_qkv` (`q`, `k`, `v`), `in_proj_z`, `in_proj_a` (the decay's
///   input) and `in_proj_b` (the update's);
/// - a global attention's `attention.wkv`, `[2, H_kv d, D]`, into `wk` and
///   `wv`, each `[1, H_kv d, D]`;
/// - the dense feed-forward's `feed_forward.w_in`, `[2, F, D]`, into `w_up`
///   (`w_in[0]`) and `w_gate` (`w_in[1]`), each `[1, F, D]`.
pub fn zonos3_lm_row_splits(config: &Zonos3LmConfig) -> [TensorLoadRowSplit; 3] {
    let (heads, inner) = (config.gdn_heads, config.gdn_heads * config.gdn_head_dim);
    [
        TensorLoadRowSplit::sized(
            ".attention.in_proj.weight",
            [
                (".attention.in_proj_qkv.weight", 3 * inner),
                (".attention.in_proj_z.weight", inner),
                (".attention.in_proj_a.weight", heads),
                (".attention.in_proj_b.weight", heads),
            ],
        ),
        TensorLoadRowSplit::equal(
            ".attention.wkv.weight",
            [".attention.wk.weight", ".attention.wv.weight"],
        ),
        TensorLoadRowSplit::equal(
            ".feed_forward.w_in.weight",
            [".feed_forward.w_up.weight", ".feed_forward.w_gate.weight"],
        ),
    ]
}

/// Layer `l`'s names, after [`zonos3_lm_row_splits`], onto a decoder
/// block's canonical ones: its two norms; its mixer's, as a gated `DeltaNet`
/// or a global attention spells them; its feed-forward's, as the dense
/// `SwiGLU` (layer 0) or the latent mixture spells them. The split parts of
/// a stacked tensor lose their unit axis on the way.
pub fn zonos3_lm_layer_rekey(
    config: &Zonos3LmConfig,
    l: usize,
) -> Rekey {
    let d = config.d_model;
    let rekey = Rekey::drop_rest()
        .rename("attention_norm.", "attn_norm.")
        .rename("ffn_norm.", "mlp_norm.");
    let rekey = if config.layer_attends(l) {
        let kv = config.n_kv_heads * config.head_dim;
        rekey
            .rename("attention.wq.", "attn.c_q.")
            .rename("attention.wk.", "attn.c_k.")
            .rename("attention.wv.", "attn.c_v.")
            .rename("attention.wo.", "attn.c_proj.")
            .rename("attention.gater.", "attn.c_gate.")
            .exact("attention.temp", "attn.query_gain")
            .transform(
                Select::suffix("attn.c_k.weight"),
                TensorLoadTransform::reshape([kv, d]),
            )
            .transform(
                Select::suffix("attn.c_v.weight"),
                TensorLoadTransform::reshape([kv, d]),
            )
            .transform(
                Select::suffix("attn.query_gain"),
                TensorLoadTransform::reshape([config.n_heads]),
            )
    } else {
        rekey
            .rename("attention.in_proj_qkv.", "attn.in_proj_qkv.")
            .rename("attention.in_proj_z.", "attn.in_proj_z.")
            .rename("attention.in_proj_a.", "attn.in_proj_a.")
            .rename("attention.in_proj_b.", "attn.in_proj_b.")
            .rename("attention.conv.", "attn.conv1d.")
            .exact("attention.A_log", "attn.a_log")
            .exact("attention.dt_bias", "attn.dt_bias")
            .exact("attention.out_weight", "attn.norm.norm.weight")
            .rename("attention.wo.", "attn.out_proj.")
    };
    if l == 0 {
        rekey
            .rename("feed_forward.w_up.", "mlp.up_proj.")
            .rename("feed_forward.w_gate.", "mlp.gate_proj.")
            .rename("feed_forward.w_out.", "mlp.down_proj.")
            .transform(
                Select::suffix("mlp.up_proj.weight"),
                TensorLoadTransform::reshape([config.ffn_hidden, d]),
            )
            .transform(
                Select::suffix("mlp.gate_proj.weight"),
                TensorLoadTransform::reshape([config.ffn_hidden, d]),
            )
    } else {
        rekey
            .rename("feed_forward.router.router_linear.", "mlp.router.gate.")
            .rename("feed_forward.router.rmsnorm_eda.", "mlp.router.input_norm.")
            .exact(
                "feed_forward.router.balancing_biases",
                "mlp.router.choice_bias",
            )
            .rename("feed_forward.experts.", "mlp.experts.")
            .rename("feed_forward.latent_down.", "mlp.latent_down.")
            .rename("feed_forward.latent_up.", "mlp.latent_up.")
    }
}

/// The language model's names, after [`zonos3_lm_row_splits`] and without
/// [`ZONOS3_LM_PREFIX`], onto a
/// [`Zonos3Lm`](crate::kits::speech::tts::zonos3::blocks::Zonos3Lm)'s canonical
/// ones: the row tables, the speaker projection and the condition tables as
/// they are; each layer by [`zonos3_lm_layer_rekey`]; the `AttnRes` queries,
/// the final norm (`out_norm`) and the head (`multi_output`). Anything else is
/// dropped.
pub fn zonos3_lm_rekey(config: &Zonos3LmConfig) -> Rekey {
    (0..config.n_layers).fold(
        Rekey::drop_rest()
            .rename("multi_embedder.embedders.", "embedders.")
            .rename("speaker_projection.", "speaker_projection.")
            .rename("condition_embeddings.", "condition_embeddings.")
            .exact("block_attnres_queries", "stack.queries")
            .rename("out_norm.", "stack.stack.norm.")
            .rename("multi_output.", "head."),
        |rekey, l| {
            rekey.mount(
                format!("layers.{l}."),
                format!("stack.stack.layers.{l}."),
                zonos3_lm_layer_rekey(config, l),
            )
        },
    )
}
