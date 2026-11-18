//! Loading `Zyphra/ZONOS3-Mini-hf`'s `lm.*` tensors into a [`Zonos3Lm`].
//!
//! The file names the language model as upstream's reference does
//! (`zonos3.reference.lm`, whose state dict it is, under `lm.`). Its
//! projections are packed where bunsen's blocks hold them apart, so three
//! tensors are split by rows first ([`zonos3_lm_row_splits`]); the names
//! then map onto the blocks' canonical names ([`zonos3_lm_rekey`]), layer by
//! layer, as each layer's kind spells them.

use alloc::{
    format,
    string::String,
    vec::Vec,
};

use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::{
    Zonos3Lm,
    Zonos3LmConfig,
};
use crate::{
    burner::module::{
        Rekey,
        Select,
        TensorLoadRowSplit,
        TensorLoadTransform,
    },
    data::formats::{
        NamedTensorsApplied,
        split_rows,
        unit_scale_tensors,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

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
/// [`ZONOS3_LM_PREFIX`], onto a [`Zonos3Lm`]'s canonical ones: the row
/// tables, the speaker projection and the condition tables as they are;
/// each layer by [`zonos3_lm_layer_rekey`]; the `AttnRes` queries, the final
/// norm (`out_norm`) and the head (`multi_output`). Anything else is
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

/// Whether the parameter at `path` is a norm scale the checkpoint has none
/// for: upstream's weightless RMS norms, on the embedding (the stack's
/// input norm) and on the attention's queries and keys.
fn is_weightless_scale(
    path: &str,
    _containers: &str,
) -> bool {
    path == "stack.stack.input_norm.gamma"
        || path.ends_with("attn.branch.q_norm.gamma")
        || path.ends_with("attn.branch.k_norm.gamma")
}

/// Loads the `-hf` file's tensors (`lm.*`; any others are dropped) into
/// `model`, built from `config`: the packed projections split
/// ([`zonos3_lm_row_splits`]), the names mapped ([`zonos3_lm_rekey`]), the
/// weightless norms' scales set to ones, every floating tensor in `dtype`
/// (a load policy's resolution), or, when `None`, in the dtype `model` was
/// built in. (The file keeps the routers' choice biases in f32; the
/// router reads them in f32, whatever dtype they load in.)
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) as
/// [`split_rows`] and [`Rekey::load`]: a tensor that does not split, does
/// not fit, or a parameter the file lacks;
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
/// naming any `lm.*` tensor the model has no parameter for.
pub fn load_zonos3_lm_tensors(
    model: &mut Zonos3Lm,
    config: &Zonos3LmConfig,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let tensors: Vec<PackTensor> = tensors
        .into_iter()
        .filter(|t| t.name.starts_with(ZONOS3_LM_PREFIX))
        .collect();
    let mut tensors = split_rows(tensors, &zonos3_lm_row_splits(config))?;
    let scales = unit_scale_tensors(model, is_weightless_scale);
    let names: Vec<String> = scales.iter().map(|t| t.name.clone()).collect();
    tensors.extend(scales);

    let first = names
        .iter()
        .fold(Rekey::drop_rest(), |rekey, n| {
            rekey.exact(n.clone(), n.clone())
        })
        .mount(ZONOS3_LM_PREFIX, "", zonos3_lm_rekey(config));
    let last = names
        .iter()
        .fold(Rekey::keep_rest(), |rekey, n| {
            rekey.exact(n.clone(), n.clone())
        })
        .mount_canonical("", model);
    let applied = first.then(last).load(model, tensors, dtype)?;
    if !applied.unused.is_empty() {
        return Err(BunsenError::invalid_resource(format!(
            "ZONOS3 language model tensors the model has no parameter for: {}",
            applied.unused.join(", ")
        )));
    }
    Ok(applied)
}
