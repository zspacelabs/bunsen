//! `NeMo`'s state-dict names as bunsen's module paths.
use crate::{
    blocks::signals::{
        DwStridingSubsamplingMeta,
        FlattenOrder,
    },
    burner::module::Rekey,
};

/// `NeMo`'s `ConvSubsampling` (`pre_encode`) names onto a
/// [`DwStridingSubsampling`](crate::blocks::signals::DwStridingSubsampling)'s,
/// relative to it: the flat `conv.{k}` by
/// [`sequential_rekey`](DwStridingSubsamplingMeta::sequential_rekey), and
/// `out` the projection, its weight permuted when `striding` flattens
/// otherwise than `NeMo` does (channels outer). Anything else is dropped.
///
/// `striding` is the subsampling the tensors load into, a config or a built
/// module: its stage count and flatten order fix its map.
pub fn conv_subsampling_rekey(striding: &impl DwStridingSubsamplingMeta) -> Rekey {
    let conv = striding.sequential_rekey("conv.");
    striding.reflatten_from(
        FlattenOrder::ChannelsFreqs,
        Rekey::drop_rest()
            .custom(move |name| name.starts_with("conv.").then(|| conv.apply(name)))
            .rename("out.", "linear."),
    )
}

/// `NeMo`'s `ConformerLayer` names onto a
/// [`ConformerBlock`](crate::blocks::transformers::encoders::conformer::ConformerBlock)'s,
/// relative to it: the attention's projections, the convolution module's
/// depthwise convolution and batch norm, and the sublayers' norms, renamed;
/// everything else as it is.
pub fn conformer_layer_rekey() -> Rekey {
    Rekey::keep_rest()
        .rename("self_attn.linear_q.", "attn.query.")
        .rename("self_attn.linear_k.", "attn.key.")
        .rename("self_attn.linear_v.", "attn.value.")
        .rename("self_attn.linear_out.", "attn.output.")
        .rename("self_attn.linear_pos.", "attn.pos.")
        .rename("conv.depthwise_conv.", "conv.depthwise.conv.")
        .rename("conv.batch_norm.", "conv.depthwise.norm.")
        .rename("norm_feed_forward1.", "feed_forward1_norm.")
        .rename("norm_self_att.", "attn_norm.")
        .rename("norm_conv.", "conv_norm.")
        .rename("norm_feed_forward2.", "feed_forward2_norm.")
        .rename("norm_out.", "out_norm.")
        .rename("self_attn.", "attn.")
}

/// A `FastConformer`'s tensors as `NeMo`'s `ConformerEncoder` names them,
/// onto the
/// [`SubsampledEncoder`](crate::blocks::transformers::encoders::SubsampledEncoder)
/// a [`ConformerEncoderConfig`](crate::blocks::transformers::encoders::conformer::ConformerEncoderConfig)
/// builds: [`conv_subsampling_rekey`] and [`conformer_layer_rekey`] mounted
/// where they sit. The batch-norm counters, and anything else, are dropped.
///
/// `striding` is the subsampling the tensors load into, a config or a built
/// module: its stage count and flatten order fix its map.
pub fn conformer_encoder_rekey(striding: &impl DwStridingSubsamplingMeta) -> Rekey {
    Rekey::drop_rest()
        .drop_suffix(".num_batches_tracked")
        .mount(
            "pre_encode.",
            "subsampling.",
            conv_subsampling_rekey(striding),
        )
        .mount_indexed("layers.", "transformer.layers.", conformer_layer_rekey())
}

/// A `NeMo` `transformer_encoders.TransformerEncoder` layer's names onto a
/// [`ResidualEncoderAttentionBlock`](crate::blocks::transformers::encoders::residual::ResidualEncoderAttentionBlock)'s,
/// relative to it: `first_sub_layer` the attention, `second_sub_layer` the
/// MLP, `layer_norm_{1,2}` their norms. Anything else is dropped.
pub fn transformer_encoder_layer_rekey() -> Rekey {
    Rekey::drop_rest()
        .rename("first_sub_layer.query_net.", "attn.query.")
        .rename("first_sub_layer.key_net.", "attn.key.")
        .rename("first_sub_layer.value_net.", "attn.value.")
        .rename("first_sub_layer.out_projection.", "attn.output.")
        .rename("second_sub_layer.dense_in.", "mlp.linear1.")
        .rename("second_sub_layer.dense_out.", "mlp.linear2.")
        .rename("layer_norm_1.", "attn_norm.")
        .rename("layer_norm_2.", "mlp_norm.")
}

/// `NeMo`'s `transformer_encoders.TransformerEncoder` names onto a
/// [`ResidualEncoderTransformer`](crate::blocks::transformers::encoders::residual::ResidualEncoderTransformer)'s,
/// relative to it: each `layers.{n}` by
/// [`transformer_encoder_layer_rekey`]. Anything else is dropped.
pub fn transformer_encoder_rekey() -> Rekey {
    Rekey::drop_rest().mount_indexed("layers.", "layers.", transformer_encoder_layer_rekey())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::signals::DwStridingSubsamplingConfig;

    /// `NeMo`'s `ConformerEncoder` names, the subsampling's flat `conv`
    /// list and every block's renames among them.
    #[test]
    fn test_conformer_encoder_rekey() {
        let rekey = conformer_encoder_rekey(&DwStridingSubsamplingConfig::new(80, 4, 16));
        for (nemo, bunsen) in [
            (
                "pre_encode.conv.0.weight",
                Some("subsampling.stem.conv.weight"),
            ),
            (
                "pre_encode.conv.2.bias",
                Some("subsampling.stages.0.depthwise.conv.bias"),
            ),
            (
                "pre_encode.conv.3.weight",
                Some("subsampling.stages.0.pointwise.conv.weight"),
            ),
            (
                "pre_encode.conv.5.weight",
                Some("subsampling.stages.1.depthwise.conv.weight"),
            ),
            (
                "pre_encode.conv.6.bias",
                Some("subsampling.stages.1.pointwise.conv.bias"),
            ),
            ("pre_encode.conv.4.weight", None),
            ("pre_encode.conv.8.weight", None),
            ("pre_encode.out.weight", Some("subsampling.linear.weight")),
            (
                "layers.3.self_attn.linear_pos.weight",
                Some("transformer.layers.3.attn.pos.weight"),
            ),
            (
                "layers.0.self_attn.pos_bias_u",
                Some("transformer.layers.0.attn.pos_bias_u"),
            ),
            (
                "layers.1.conv.batch_norm.running_var",
                Some("transformer.layers.1.conv.depthwise.norm.running_var"),
            ),
            ("layers.1.conv.batch_norm.num_batches_tracked", None),
            (
                "layers.0.norm_feed_forward1.weight",
                Some("transformer.layers.0.feed_forward1_norm.weight"),
            ),
            ("pos_enc.pe", None),
        ] {
            assert_eq!(rekey.apply(nemo).as_deref(), bunsen, "{nemo}");
        }
    }

    /// `NeMo`'s `TransformerEncoder` names, layer by layer.
    #[test]
    fn test_transformer_encoder_rekey() {
        let rekey = transformer_encoder_rekey();
        for (nemo, bunsen) in [
            (
                "layers.17.first_sub_layer.key_net.bias",
                Some("layers.17.attn.key.bias"),
            ),
            (
                "layers.0.second_sub_layer.dense_out.weight",
                Some("layers.0.mlp.linear2.weight"),
            ),
            ("layers.2.layer_norm_2.bias", Some("layers.2.mlp_norm.bias")),
            ("final_layer_norm.weight", None),
        ] {
            assert_eq!(rekey.apply(nemo).as_deref(), bunsen, "{nemo}");
        }
    }
}
