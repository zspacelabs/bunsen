//! Reading `FermionResearch`'s Phonon-1 archives.
use std::{
    collections::BTreeMap,
    path::Path,
};

use burn::{
    prelude::TensorData,
    store::{
        ModuleStore,
        SafetensorsStore,
        bridge,
        burn_pack::Tensor as PackTensor,
    },
};
use serde::Deserialize;

use crate::{
    data::pretrained::{
        fermion::{
            PhononArchive,
            decode_quint5,
        },
        mlx::dequantize_mlx_affine,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::tokens::ByteLevelVocabulary,
};

/// The member of a Phonon-1 archive that describes its packing.
pub const PHONON_1_MANIFEST: &str = "packed_manifest.json";

/// A decoder linear's shape, from the manifest's `modules`.
#[derive(Debug, Deserialize)]
struct ModuleShape {
    name: String,
    in_features: usize,
}

/// An MLX affine-quantized tensor's packing.
#[derive(Debug, Deserialize)]
struct AffinePacking {
    name: String,
    bits: u32,
    group_size: usize,
    mode: String,
    /// A linear's input width.
    #[serde(default)]
    in_features: Option<usize>,
    /// An embedding's width.
    #[serde(default)]
    dims: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct HybridQuantization {
    #[serde(default)]
    audio_linears: Vec<AffinePacking>,
    #[serde(default)]
    embedding: Option<AffinePacking>,
}

#[derive(Debug, Deserialize)]
struct Shard {
    name: String,
}

/// `packed_manifest.json`: how a Phonon-1 checkpoint is packed.
#[derive(Debug, Deserialize)]
struct Phonon1Manifest {
    modules: Vec<ModuleShape>,
    #[serde(default)]
    hybrid_quantization: Option<HybridQuantization>,
    shards: Vec<Shard>,
}

impl Phonon1Manifest {
    fn parse(json: &[u8]) -> BunsenResult<Self> {
        serde_json::from_slice(json).map_err(|e| {
            BunsenError::invalid_resource(format!("a Phonon-1 `{PHONON_1_MANIFEST}`")).with_cause(e)
        })
    }

    /// The affine packing of every MLX-quantized tensor, by module name.
    fn affine(&self) -> BTreeMap<&str, &AffinePacking> {
        let Some(hq) = &self.hybrid_quantization else {
            return BTreeMap::new();
        };
        hq.audio_linears
            .iter()
            .chain(hq.embedding.as_ref())
            .map(|p| (p.name.as_str(), p))
            .collect()
    }

    /// Every decoder linear's input width, by module name.
    fn in_features(&self) -> BTreeMap<&str, usize> {
        self.modules
            .iter()
            .map(|m| (m.name.as_str(), m.in_features))
            .collect()
    }
}

/// Reads a Phonon-1 archive (`phonon-*.bps.tar.zst`): its weights, expanded
/// to dense tensors named as `transformers`' `Qwen3ASRForConditionalGeneration`
/// state dict, and its vocabulary.
///
/// `FermionResearch`'s Phonon-1 builds of `Qwen3-ASR-0.6B` pack their
/// checkpoint three ways, as `packed_manifest.json` describes:
///
/// - the text decoder's linears in the five-symbol quint5 code
///   ([`decode_quint5`]), their input widths from the manifest's `modules`;
/// - the audio tower's linears and the token embedding (tied to the output
///   head) either MLX affine-quantized ([`dequantize_mlx_affine`]; the bits and
///   group size per tensor from the manifest's `hybrid_quantization`) or
///   `bf16`;
/// - every other tensor `bf16`, the stem's convolution kernels in MLX's `[out,
///   h, w, in]` layout, which this transposes to `[out, in, h, w]`.
///
/// Every value is the stored model's, exactly: quint5 levels and affine codes
/// expand without approximation.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) when
/// the archive cannot be read, a member is missing, or a tensor does not fit
/// its packing; the error carries `path`.
pub fn read_phonon1_archive(path: &Path) -> BunsenResult<(Vec<PackTensor>, ByteLevelVocabulary)> {
    let read = || -> BunsenResult<(Vec<PackTensor>, ByteLevelVocabulary)> {
        let mut archive = PhononArchive::read(path)?;
        let manifest = Phonon1Manifest::parse(archive.expect(PHONON_1_MANIFEST)?)?;
        let text = |archive: &PhononArchive, name: &str| -> BunsenResult<String> {
            String::from_utf8(archive.expect(name)?.to_vec()).map_err(|e| {
                BunsenError::invalid_resource(format!("`{name}` is not UTF-8")).with_cause(e)
            })
        };
        let vocab = ByteLevelVocabulary::from_vocab_json(
            &text(&archive, "vocab.json")?,
            Some(&text(&archive, "tokenizer_config.json")?),
        )?;

        let mut stored = Vec::new();
        for shard in &manifest.shards {
            let bytes = archive.take(&shard.name).ok_or_else(|| {
                BunsenError::invalid_resource(format!(
                    "a Phonon-1 archive without `{}`",
                    shard.name
                ))
            })?;
            let tensors = SafetensorsStore::from_bytes(Some(bytes))
                .get_all_tensors()
                .map_err(|e| {
                    BunsenError::invalid_resource("the store refused a shard")
                        .with_cause(e)
                        .context(shard.name.clone())
                })?
                .values()
                .cloned()
                .collect::<Vec<_>>();
            stored.extend(tensors);
        }
        Ok((expand_phonon1(&manifest, stored)?, vocab))
    };
    read().map_err(|e| e.context(path.display()))
}

/// The dense, `transformers`-named tensors of a Phonon-1 checkpoint's stored
/// tensors.
fn expand_phonon1(
    manifest: &Phonon1Manifest,
    stored: Vec<PackTensor>,
) -> BunsenResult<Vec<PackTensor>> {
    let affine = manifest.affine();
    let in_features = manifest.in_features();
    let mut by_name: BTreeMap<String, PackTensor> =
        stored.into_iter().map(|t| (t.name.clone(), t)).collect();
    let mut out = Vec::new();
    for (module, packing) in &affine {
        if packing.mode != "affine" {
            return Err(BunsenError::invalid_resource(format!(
                "`{module}` is packed `{}`, not affine",
                packing.mode
            )));
        }
        let words = take(&mut by_name, format!("{module}.weight"))?;
        let scales = f32_values(take(&mut by_name, format!("{module}.scales"))?)?;
        let biases = f32_values(take(&mut by_name, format!("{module}.biases"))?)?;
        let [rows, packed] = dims2(&words)?;
        let cols = packed * 32 / packing.bits as usize;
        if let Some(width) = packing.in_features.or(packing.dims)
            && width != cols
        {
            return Err(BunsenError::invalid_resource(format!(
                "`{module}` packs {cols} columns, not the manifest's {width}"
            )));
        }
        let words = bridge::into_data(words)
            .map_err(pack_error)?
            .try_into_vec::<u32>()
            .map_err(|e| BunsenError::invalid_resource(format!("`{module}`'s codes: {e:?}")))?;
        let values = dequantize_mlx_affine(
            &words,
            &scales,
            &biases,
            rows,
            cols,
            packing.bits,
            packing.group_size,
        )
        .map_err(|e| e.context(module.to_string()))?;
        out.push(dense(
            hf_name(&format!("{module}.weight")),
            values,
            vec![rows, cols],
        ));
    }

    let quint5: Vec<String> = by_name
        .keys()
        .filter_map(|n| n.strip_suffix(".quint5_q"))
        .map(String::from)
        .collect();
    for module in quint5 {
        let cols = *in_features.get(module.as_str()).ok_or_else(|| {
            BunsenError::invalid_resource(format!("no input width for `{module}` in the manifest"))
        })?;
        let packed = take(&mut by_name, format!("{module}.quint5_q"))?;
        let alpha = f32_values(take(&mut by_name, format!("{module}.base_alpha"))?)?;
        let residual = f32_values(take(&mut by_name, format!("{module}.residual_scale"))?)?;
        let [rows, _] = dims2(&packed)?;
        let packed = bridge::into_data(packed)
            .map_err(pack_error)?
            .try_into_vec::<u8>()
            .map_err(|e| BunsenError::invalid_resource(format!("`{module}`'s codes: {e:?}")))?;
        let values = decode_quint5(&packed, rows, cols, &alpha, &residual)
            .map_err(|e| e.context(module.clone()))?;
        out.push(dense(
            hf_name(&format!("{module}.weight")),
            values,
            vec![rows, cols],
        ));
    }

    for (name, tensor) in by_name {
        let is_conv = name.starts_with("audio_tower.conv2d") && name.ends_with(".weight");
        let data = bridge::into_data(tensor).map_err(pack_error)?;
        let data = if is_conv {
            // MLX's `[out, h, w, in]` to PyTorch's `[out, in, h, w]`.
            let shape = data.shape().to_vec();
            let [o, h, w, c] = shape[..] else {
                return Err(BunsenError::invalid_resource(format!(
                    "`{name}` has shape {shape:?}, not [out, h, w, in]"
                )));
            };
            let src = data
                .try_into_vec_as::<f32>()
                .map_err(|e| BunsenError::invalid_resource(format!("`{name}`: {e:?}")))?;
            let mut dst = vec![0.0f32; src.len()];
            for oi in 0..o {
                for hi in 0..h {
                    for wi in 0..w {
                        for ci in 0..c {
                            dst[((oi * c + ci) * h + hi) * w + wi] =
                                src[((oi * h + hi) * w + wi) * c + ci];
                        }
                    }
                }
            }
            TensorData::new(dst, [o, c, h, w])
        } else {
            data
        };
        out.push(bridge::from_data(data, hf_name(&name), None));
    }
    Ok(out)
}

/// Removes a stored tensor by name.
fn take(
    by_name: &mut BTreeMap<String, PackTensor>,
    name: String,
) -> BunsenResult<PackTensor> {
    by_name.remove(&name).ok_or_else(|| {
        BunsenError::invalid_resource(format!("a Phonon-1 checkpoint without `{name}`"))
    })
}

/// A Phonon-1 tensor's `transformers` name: the audio tower and text model
/// under `model.`, the tower's `proj1` / `proj2` as the projector.
fn hf_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("audio_tower.proj1.") {
        format!("model.multi_modal_projector.linear_1.{rest}")
    } else if let Some(rest) = name.strip_prefix("audio_tower.proj2.") {
        format!("model.multi_modal_projector.linear_2.{rest}")
    } else if let Some(rest) = name.strip_prefix("model.") {
        format!("model.language_model.{rest}")
    } else {
        format!("model.{name}")
    }
}

fn dense(
    name: String,
    values: Vec<f32>,
    shape: Vec<usize>,
) -> PackTensor {
    bridge::from_data(TensorData::new(values, shape), name, None)
}

fn dims2(t: &PackTensor) -> BunsenResult<[usize; 2]> {
    match t.shape[..] {
        [a, b] => Ok([a, b]),
        _ => Err(BunsenError::invalid_resource(format!(
            "`{}` has shape {:?}, not a matrix",
            t.name, t.shape
        ))),
    }
}

fn f32_values(t: PackTensor) -> BunsenResult<Vec<f32>> {
    let name = t.name.clone();
    bridge::into_data(t)
        .map_err(pack_error)?
        .try_into_vec_as::<f32>()
        .map_err(|e| BunsenError::invalid_resource(format!("`{name}`: {e:?}")))
}

fn pack_error(e: impl std::error::Error + Send + Sync + 'static) -> BunsenError {
    BunsenError::invalid_resource("a tensor the store cannot read").with_cause(e)
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        DType,
        bf16,
    };

    use super::*;

    fn bf16_tensor(
        name: &str,
        values: &[f32],
        shape: Vec<usize>,
    ) -> PackTensor {
        bridge::from_data(
            TensorData::new(values.to_vec(), shape).convert::<bf16>(),
            name.to_string(),
            None,
        )
    }

    fn manifest() -> Phonon1Manifest {
        Phonon1Manifest::parse(
            br#"{
                "modules": [{"name": "model.layers.0.mlp.up_proj", "in_features": 3,
                             "out_features": 2}],
                "hybrid_quantization": {"audio_linears": [
                    {"name": "audio_tower.proj1", "bits": 4, "group_size": 4,
                     "mode": "affine"}]},
                "shards": [{"name": "model-00001.safetensors"}]
            }"#,
        )
        .unwrap()
    }

    /// Every packing expands to the dense tensor its decoder gives, under its
    /// `transformers` name, and a stem kernel is transposed to `[out, in, h,
    /// w]`.
    #[test]
    fn test_expand() {
        // One row of eight 4-bit codes, 0..8, in one little-endian word.
        let word: u32 = (0..8u32).map(|q| q << (4 * q)).sum();
        let quint5_row = [0u8, 0, 0];
        let stored = vec![
            bridge::from_data(
                TensorData::new(vec![word], [1, 1]),
                "audio_tower.proj1.weight".into(),
                None,
            ),
            bf16_tensor("audio_tower.proj1.scales", &[0.5, 2.0], vec![1, 2]),
            bf16_tensor("audio_tower.proj1.biases", &[-1.0, 1.0], vec![1, 2]),
            bf16_tensor("audio_tower.proj1.bias", &[3.0], vec![1]),
            bridge::from_data(
                TensorData::new([quint5_row, quint5_row].concat(), [2, 3]),
                "model.layers.0.mlp.up_proj.quint5_q".into(),
                None,
            ),
            bf16_tensor(
                "model.layers.0.mlp.up_proj.base_alpha",
                &[1.0, 2.0],
                vec![2],
            ),
            bf16_tensor(
                "model.layers.0.mlp.up_proj.residual_scale",
                &[0.25],
                vec![1],
            ),
            bf16_tensor(
                "audio_tower.conv2d1.weight",
                &(0..12).map(|v| v as f32).collect::<Vec<_>>(),
                vec![1, 2, 3, 2],
            ),
            bf16_tensor("model.norm.weight", &[1.0, 1.0], vec![2]),
        ];
        let out: BTreeMap<String, PackTensor> = expand_phonon1(&manifest(), stored)
            .unwrap()
            .into_iter()
            .map(|t| (t.name.clone(), t))
            .collect();
        let values = |name: &str| -> (Vec<usize>, Vec<f32>) {
            let data = bridge::to_data(&out[name]).unwrap();
            (
                data.shape().to_vec(),
                data.try_into_vec_as::<f32>().unwrap(),
            )
        };

        assert_eq!(
            values("model.multi_modal_projector.linear_1.weight"),
            (
                vec![1, 8],
                vec![-1.0, -0.5, 0.0, 0.5, 9.0, 11.0, 13.0, 15.0]
            )
        );
        assert_eq!(
            values("model.multi_modal_projector.linear_1.bias"),
            (vec![1], vec![3.0])
        );
        // Symbol 0 everywhere: `-alpha - residual`.
        assert_eq!(
            values("model.language_model.layers.0.mlp.up_proj.weight"),
            (vec![2, 3], vec![-1.25, -1.25, -1.25, -2.25, -2.25, -2.25])
        );
        // `[1, h=2, w=3, in=2]` to `[1, in, h, w]`.
        assert_eq!(
            values("model.audio_tower.conv2d1.weight"),
            (
                vec![1, 2, 2, 3],
                vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
            )
        );
        assert_eq!(out["model.language_model.norm.weight"].dtype, DType::BF16);
        assert_eq!(out.len(), 5);
    }

    /// A packed tensor without its companions is refused, by name.
    #[test]
    fn test_missing_companion() {
        let stored = vec![bridge::from_data(
            TensorData::new(vec![0u32], [1, 1]),
            "audio_tower.proj1.weight".into(),
            None,
        )];
        let err = expand_phonon1(&manifest(), stored).unwrap_err();
        assert!(
            err.to_string().contains("audio_tower.proj1.scales"),
            "{err}"
        );
    }
}
