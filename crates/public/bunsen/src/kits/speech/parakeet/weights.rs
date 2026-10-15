//! Loading `transformers`-named tensors into a [`ParakeetTdt`].
use burn::store::{
    ModuleSnapshot,
    PyTorchToBurnAdapter,
    burn_pack::Tensor as PackTensor,
};

use super::blocks::ParakeetTdt;
use crate::{
    blocks::conformer::DwStridingSubsamplingMeta,
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// The bunsen module path of one `transformers` `ParakeetForTDT` state-dict
/// name, or `None` for an entry that is not a parameter
/// (`num_batches_tracked`).
///
/// `n_stages` is the subsampling's stage count, which fixes how
/// `encoder.subsampling.layers.{k}` (`transformers`' flat `ModuleList` of
/// convolutions and activations) maps onto the stem and the stages.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`PyTorchToBurnAdapter`] resolves to `gamma` / `beta`.
pub fn hf_to_bunsen_path(
    name: &str,
    n_stages: usize,
) -> Option<String> {
    if name.ends_with(".num_batches_tracked") {
        return None;
    }
    if let Some(rest) = name.strip_prefix("encoder.subsampling.layers.") {
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
        return Some(format!("encoder.subsampling.{module}.{param}"));
    }
    if let Some(rest) = name.strip_prefix("decoder.lstm.") {
        let (param, layer) = rest.rsplit_once("_l")?;
        let (proj, kind) = match param {
            "weight_ih" => ("features", "weight"),
            "weight_hh" => ("hidden", "weight"),
            "bias_ih" => ("features", "bias"),
            "bias_hh" => ("hidden", "bias"),
            _ => return None,
        };
        return Some(format!("decoder.lstm.{layer}.{proj}.{kind}"));
    }

    const RENAMES: &[(&str, &str)] = &[
        (".self_attn.q_proj.", ".self_attn.query."),
        (".self_attn.k_proj.", ".self_attn.key."),
        (".self_attn.v_proj.", ".self_attn.value."),
        (".self_attn.o_proj.", ".self_attn.output."),
        (".self_attn.relative_k_proj.", ".self_attn.pos."),
        (".self_attn.bias_u", ".self_attn.pos_bias_u"),
        (".self_attn.bias_v", ".self_attn.pos_bias_v"),
        (".conv.depthwise_conv.", ".conv.depthwise.conv."),
        (".conv.norm.", ".conv.depthwise.norm."),
        ("decoder.decoder_projector.", "decoder.projector."),
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

/// What [`load_hf_tensors`] did.
#[derive(Clone, Debug, Default)]
pub struct HfTensorsApplied {
    /// The module paths loaded.
    pub applied: Vec<String>,

    /// The source names that matched no parameter.
    pub unused: Vec<String>,
}

/// Loads tensors named as `transformers`' `ParakeetForTDT` state dict into
/// `model`.
///
/// The names are rewritten by [`hf_to_bunsen_path`], and the tensors adapted
/// by [`PyTorchToBurnAdapter`]: linear weights transposed, normalization
/// parameters renamed. Any float dtype is accepted; each is cast to its
/// parameter's.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) when
/// a tensor does not fit its parameter, or when a parameter of the model was
/// not among the tensors; the details list them.
pub fn load_hf_tensors(
    model: &mut ParakeetTdt,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<HfTensorsApplied> {
    let n_stages = model.encoder.subsampling.n_stages();
    let mut renamed = Vec::new();
    let mut dropped = Vec::new();
    for mut tensor in tensors {
        match hf_to_bunsen_path(&tensor.name, n_stages) {
            Some(path) => {
                tensor.name = path;
                renamed.push(tensor);
            }
            None => dropped.push(tensor.name),
        }
    }

    let result = model.apply(renamed, None, Some(Box::new(PyTorchToBurnAdapter)), true);
    if !result.errors.is_empty() {
        let details: Vec<String> = result.errors.iter().map(|e| format!("{e:?}")).collect();
        return Err(BunsenError::invalid_resource(format!(
            "{} tensors do not fit the model",
            details.len()
        ))
        .with_details(details.join("\n")));
    }
    if !result.missing.is_empty() {
        let missing: Vec<&str> = result.missing.iter().map(|(p, _)| p.as_str()).collect();
        return Err(BunsenError::invalid_resource(format!(
            "the tensors lack {} of the model's parameters: {}",
            missing.len(),
            missing
                .iter()
                .take(8)
                .copied()
                .collect::<Vec<_>>()
                .join(", ")
        ))
        .with_details(missing.join("\n")));
    }
    let mut unused = result.unused;
    unused.extend(dropped);
    Ok(HfTensorsApplied {
        applied: result.applied,
        unused,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_to_bunsen_path() {
        for (hf, bunsen) in [
            (
                "encoder.subsampling.layers.0.weight",
                Some("encoder.subsampling.stem.conv.weight"),
            ),
            (
                "encoder.subsampling.layers.2.bias",
                Some("encoder.subsampling.stages.0.depthwise.conv.bias"),
            ),
            (
                "encoder.subsampling.layers.3.weight",
                Some("encoder.subsampling.stages.0.pointwise.conv.weight"),
            ),
            (
                "encoder.subsampling.layers.5.weight",
                Some("encoder.subsampling.stages.1.depthwise.conv.weight"),
            ),
            (
                "encoder.subsampling.layers.6.bias",
                Some("encoder.subsampling.stages.1.pointwise.conv.bias"),
            ),
            ("encoder.subsampling.layers.8.weight", None),
            (
                "encoder.subsampling.linear.weight",
                Some("encoder.subsampling.linear.weight"),
            ),
            (
                "encoder.layers.3.self_attn.q_proj.weight",
                Some("encoder.layers.3.self_attn.query.weight"),
            ),
            (
                "encoder.layers.3.self_attn.relative_k_proj.weight",
                Some("encoder.layers.3.self_attn.pos.weight"),
            ),
            (
                "encoder.layers.3.self_attn.bias_u",
                Some("encoder.layers.3.self_attn.pos_bias_u"),
            ),
            (
                "encoder.layers.3.conv.depthwise_conv.weight",
                Some("encoder.layers.3.conv.depthwise.conv.weight"),
            ),
            (
                "encoder.layers.3.conv.norm.running_var",
                Some("encoder.layers.3.conv.depthwise.norm.running_var"),
            ),
            ("encoder.layers.3.conv.norm.num_batches_tracked", None),
            (
                "encoder.layers.3.conv.pointwise_conv1.weight",
                Some("encoder.layers.3.conv.pointwise_conv1.weight"),
            ),
            (
                "encoder.layers.3.norm_out.weight",
                Some("encoder.layers.3.norm_out.weight"),
            ),
            (
                "decoder.lstm.weight_ih_l1",
                Some("decoder.lstm.1.features.weight"),
            ),
            (
                "decoder.lstm.bias_hh_l0",
                Some("decoder.lstm.0.hidden.bias"),
            ),
            (
                "decoder.decoder_projector.bias",
                Some("decoder.projector.bias"),
            ),
            ("joint.head.weight", Some("joint.head.weight")),
            ("encoder_projector.weight", Some("encoder_projector.weight")),
        ] {
            assert_eq!(hf_to_bunsen_path(hf, 3).as_deref(), bunsen, "{hf}");
        }
    }
}
