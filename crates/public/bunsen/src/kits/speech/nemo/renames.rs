//! `NeMo`'s state-dict names, as bunsen's module paths.

/// The [`ConformerEncoder`](crate::blocks::conformer::ConformerEncoder) path
/// of one `NeMo` `ConformerEncoder` state-dict name, relative to the encoder
/// (a model's `encoder.` prefix taken off), or `None` for an entry that is not
/// a parameter (`num_batches_tracked`) or not the encoder's.
///
/// `n_stages` is the subsampling's stage count, which fixes how
/// `pre_encode.conv.{k}` (`NeMo`'s flat `Sequential` of convolutions and
/// activations) maps onto the stem and the stages.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`apply_named_tensors`](crate::data::pretrained::apply_named_tensors)
/// resolves to `gamma` / `beta`.
pub fn nemo_conformer_to_bunsen_path(
    name: &str,
    n_stages: usize,
) -> Option<String> {
    if name.ends_with(".num_batches_tracked") {
        return None;
    }
    if let Some(rest) = name.strip_prefix("pre_encode.conv.") {
        let (k, param) = rest.split_once('.')?;
        let k: usize = k.parse().ok()?;
        let module = match k {
            0 => "stem.conv".to_string(),
            k if k >= 2 && (k - 2) / 3 + 1 < n_stages => match (k - 2) % 3 {
                0 => format!("stages.{}.depthwise.conv", (k - 2) / 3),
                1 => format!("stages.{}.pointwise.conv", (k - 2) / 3),
                _ => return None,
            },
            _ => return None,
        };
        return Some(format!("subsampling.{module}.{param}"));
    }
    if let Some(param) = name.strip_prefix("pre_encode.out.") {
        return Some(format!("subsampling.linear.{param}"));
    }
    if !name.starts_with("layers.") {
        return None;
    }
    const RENAMES: &[(&str, &str)] = &[
        (".self_attn.linear_q.", ".self_attn.query."),
        (".self_attn.linear_k.", ".self_attn.key."),
        (".self_attn.linear_v.", ".self_attn.value."),
        (".self_attn.linear_out.", ".self_attn.output."),
        (".self_attn.linear_pos.", ".self_attn.pos."),
        (".conv.depthwise_conv.", ".conv.depthwise.conv."),
        (".conv.batch_norm.", ".conv.depthwise.norm."),
    ];
    let mut path = name.to_string();
    for (from, to) in RENAMES {
        if path.contains(from) {
            path = path.replacen(from, to, 1);
            break;
        }
    }
    Some(path)
}

/// The path, under a stack of
/// [`ResidualEncoderAttentionBlock`](crate::blocks::transformers::encoders::residual::ResidualEncoderAttentionBlock)s
/// named `layers`, of one `NeMo` `transformer_encoders.TransformerEncoder`
/// state-dict name, relative to the encoder (a model's `transformer_encoder.`
/// prefix taken off), or `None` for an entry that is not a layer's.
pub fn nemo_transformer_to_bunsen_path(name: &str) -> Option<String> {
    let rest = name.strip_prefix("layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    const RENAMES: &[(&str, &str)] = &[
        ("first_sub_layer.query_net.", "attn.query."),
        ("first_sub_layer.key_net.", "attn.key."),
        ("first_sub_layer.value_net.", "attn.value."),
        ("first_sub_layer.out_projection.", "attn.output."),
        ("second_sub_layer.dense_in.", "mlp.linear1."),
        ("second_sub_layer.dense_out.", "mlp.linear2."),
        ("layer_norm_1.", "attn_ln."),
        ("layer_norm_2.", "mlp_ln."),
    ];
    let (from, to) = RENAMES.iter().find(|(from, _)| rest.starts_with(from))?;
    Some(format!("layers.{layer}.{to}{}", &rest[from.len()..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformer_paths() {
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
                Some("layers.3.self_attn.pos.weight"),
            ),
            (
                "layers.0.self_attn.pos_bias_u",
                Some("layers.0.self_attn.pos_bias_u"),
            ),
            (
                "layers.1.conv.batch_norm.running_var",
                Some("layers.1.conv.depthwise.norm.running_var"),
            ),
            ("layers.1.conv.batch_norm.num_batches_tracked", None),
            (
                "layers.0.norm_feed_forward1.weight",
                Some("layers.0.norm_feed_forward1.weight"),
            ),
            ("pos_enc.pe", None),
        ] {
            assert_eq!(
                nemo_conformer_to_bunsen_path(nemo, 3).as_deref(),
                bunsen,
                "{nemo}"
            );
        }
    }

    #[test]
    fn test_transformer_paths() {
        assert_eq!(
            nemo_transformer_to_bunsen_path("layers.17.first_sub_layer.key_net.bias").as_deref(),
            Some("layers.17.attn.key.bias")
        );
        assert_eq!(
            nemo_transformer_to_bunsen_path("layers.0.second_sub_layer.dense_out.weight")
                .as_deref(),
            Some("layers.0.mlp.linear2.weight")
        );
        assert_eq!(
            nemo_transformer_to_bunsen_path("layers.2.layer_norm_2.bias").as_deref(),
            Some("layers.2.mlp_ln.bias")
        );
        assert_eq!(
            nemo_transformer_to_bunsen_path("final_layer_norm.weight"),
            None
        );
    }
}
