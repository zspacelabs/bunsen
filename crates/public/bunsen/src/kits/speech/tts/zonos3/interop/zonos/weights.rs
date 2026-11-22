//! Loading the release's tensors.

use alloc::{
    collections::BTreeMap,
    format,
    string::String,
    vec::Vec,
};

use burn::{
    store::{
        bridge,
        burn_pack::Tensor as PackTensor,
    },
    tensor::FloatDType,
};

use super::{
    ZONOS3_CODEC_PREFIX,
    ZONOS3_LM_PREFIX,
    ZONOS3_SPEAKER_PREFIX,
    zonos3_codec_rekey,
    zonos3_codec_row_splits,
    zonos3_lm_rekey,
    zonos3_lm_row_splits,
    zonos3_speaker_rekey,
};
use crate::{
    burner::module::Rekey,
    data::formats::{
        NamedTensorsApplied,
        split_rows,
        unit_scale_tensors,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::tts::zonos3::{
        codec::{
            Zonos3Codec,
            Zonos3CodecConfig,
        },
        lm::{
            Zonos3Lm,
            Zonos3LmConfig,
        },
        speaker::{
            Zonos3SpeakerEncoder,
            Zonos3SpeakerEncoderConfig,
        },
        voices::Zonos3Voice,
    },
};

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

/// Whether the parameter at `path` is a router's choice bias: read in f32,
/// and stored in f32 by the `-hf` file, so it keeps that width under a half
/// load.
fn is_choice_bias(path: &str) -> bool {
    path.ends_with(".router.choice_bias")
}

/// Loads the `-hf` file's tensors (`lm.*`; any others are dropped) into
/// `model`, built from `config`: the packed projections split
/// ([`zonos3_lm_row_splits`]), the names mapped ([`zonos3_lm_rekey`]), the
/// weightless norms' scales set to ones, every floating tensor in `dtype`
/// (a load policy's resolution), or, when `None`, in the dtype `model` was
/// built in; the routers' choice biases, which the file keeps in f32, stay
/// f32 whatever the dtype.
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
    let applied = first
        .then(last)
        .load_keeping(model, tensors, dtype, is_choice_bias)?;
    if !applied.unused.is_empty() {
        return Err(BunsenError::invalid_resource(format!(
            "ZONOS3 language model tensors the model has no parameter for: {}",
            applied.unused.join(", ")
        )));
    }
    Ok(applied)
}

/// Loads the `-hf` file's tensors (`codec.*`; any others are dropped) into
/// `model`, built from `config`: the packed projections split
/// ([`zonos3_codec_row_splits`]), the names mapped
/// ([`zonos3_codec_rekey`]), every floating tensor in `dtype`, or, when
/// `None`, in the dtype `model` was built in.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) as
/// [`split_rows`] and [`Rekey::load`]; and naming any `codec.*` tensor the
/// model has no parameter for.
pub fn load_zonos3_codec_tensors(
    model: &mut Zonos3Codec,
    config: &Zonos3CodecConfig,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let tensors: Vec<PackTensor> = tensors
        .into_iter()
        .filter(|t| t.name.starts_with(ZONOS3_CODEC_PREFIX))
        .collect();
    let tensors = split_rows(tensors, &zonos3_codec_row_splits())?;
    let rekey = Rekey::drop_rest()
        .mount(ZONOS3_CODEC_PREFIX, "", zonos3_codec_rekey(config))
        .then(Rekey::keep_rest().mount_canonical("", model));
    let applied = rekey.load(model, tensors, dtype)?;
    if !applied.unused.is_empty() {
        return Err(BunsenError::invalid_resource(format!(
            "ZONOS3 codec tensors the model has no parameter for: {}",
            applied.unused.join(", ")
        )));
    }
    Ok(applied)
}

/// Loads the `-hf` file's tensors (`speaker_encoder.*`; any others are
/// dropped) into `model`, built from `config`: the names mapped
/// ([`zonos3_speaker_rekey`]), every floating tensor in `dtype`, or, when
/// `None`, in the dtype `model` was built in. Upstream runs the encoder in
/// f32 whatever the language model's dtype, and its file stores it so.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) as
/// [`Rekey::load`]: a tensor that does not fit, or a parameter the file
/// lacks; and naming any `speaker_encoder.*` tensor the model has no
/// parameter for.
pub fn load_zonos3_speaker_tensors(
    model: &mut Zonos3SpeakerEncoder,
    config: &Zonos3SpeakerEncoderConfig,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let tensors: Vec<PackTensor> = tensors
        .into_iter()
        .filter(|t| t.name.starts_with(ZONOS3_SPEAKER_PREFIX))
        .collect();
    let rekey = Rekey::drop_rest().mount(ZONOS3_SPEAKER_PREFIX, "", zonos3_speaker_rekey(config));
    let applied = rekey.load(model, tensors, dtype)?;
    if !applied.unused.is_empty() {
        return Err(BunsenError::invalid_resource(format!(
            "ZONOS3 speaker-encoder tensors the model has no parameter for: {}",
            applied.unused.join(", ")
        )));
    }
    Ok(applied)
}

/// The language model and the codec of `loaded`'s checkpoint (its
/// `checkpoint`, one `model.safetensors`), built from `lm_config` and
/// `codec_config`: each in the dtype `policy` resolves its stored dtype to
/// on `device` (the language model is bf16 in the file, the codec f32).
///
/// # Errors
/// When the file can't be read, and as [`load_zonos3_lm_tensors`] and
/// [`load_zonos3_codec_tensors`].
#[cfg(feature = "cache")]
pub fn load_zonos3_lm_and_codec(
    loaded: &crate::data::resources::LoadedResourceSet,
    lm_config: &Zonos3LmConfig,
    codec_config: &Zonos3CodecConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<(Zonos3Lm, Zonos3Codec)> {
    use crate::{
        burner::module::ModuleInit,
        data::formats::{
            SafetensorsCheckpoint,
            stored_float_dtype,
        },
        kits::speech::tts::zonos3::pretrained::CHECKPOINT,
    };
    let tensors = SafetensorsCheckpoint::single(loaded.expect(CHECKPOINT)?).tensors()?;
    let stored = |prefix: &str| {
        let part: Vec<_> = tensors
            .iter()
            .filter(|t| t.name.starts_with(prefix))
            .cloned()
            .collect();
        stored_float_dtype(&part).unwrap_or(FloatDType::F32)
    };
    let lm_dtype = policy.resolve(stored(ZONOS3_LM_PREFIX), device, |_| true)?;
    let codec_dtype = policy.resolve(stored(ZONOS3_CODEC_PREFIX), device, |_| true)?;

    let mut lm: Zonos3Lm = lm_config.try_init(device)?;
    load_zonos3_lm_tensors(&mut lm, lm_config, tensors.clone(), Some(lm_dtype))?;
    let mut codec: Zonos3Codec = codec_config.try_init(device)?;
    load_zonos3_codec_tensors(&mut codec, codec_config, tensors, Some(codec_dtype))?;
    Ok((lm, codec))
}

/// The speaker encoder of `loaded`'s checkpoint (its `speaker_encoder.*`
/// tensors), built from `config` on `device`, in f32 whatever the load
/// policy: upstream runs it in f32 (autocast off) beside a bf16 language
/// model, and its file stores it so.
///
/// # Errors
/// When the file can't be read, and as [`load_zonos3_speaker_tensors`].
#[cfg(feature = "cache")]
pub fn load_zonos3_speaker_encoder(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &Zonos3SpeakerEncoderConfig,
    device: &burn::tensor::Device,
) -> BunsenResult<Zonos3SpeakerEncoder> {
    use crate::{
        burner::module::ModuleInit,
        data::formats::SafetensorsCheckpoint,
        kits::speech::tts::zonos3::pretrained::CHECKPOINT,
    };
    let tensors = SafetensorsCheckpoint::single(loaded.expect(CHECKPOINT)?).tensors()?;
    let mut encoder: Zonos3SpeakerEncoder = config.try_init(device)?;
    load_zonos3_speaker_tensors(&mut encoder, config, tensors, Some(FloatDType::F32))?;
    Ok(encoder)
}

/// The voices in `tensors` (a `voices.safetensors`'), by name.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) for
/// a tensor that is neither a voice's codes nor its embedding, codes that are
/// not a non-empty matrix of non-negative integers, an embedding that is not
/// a vector, or a voice missing either.
pub fn zonos3_voices(
    tensors: impl IntoIterator<Item = PackTensor>
) -> BunsenResult<BTreeMap<String, Zonos3Voice>> {
    let mut codes = BTreeMap::new();
    let mut embeddings = BTreeMap::new();
    for tensor in tensors {
        let name = tensor.name.clone();
        let data = bridge::to_data(&tensor)
            .map_err(|e| BunsenError::invalid_resource(format!("cannot read {name}: {e:?}")))?;
        let shape = data.shape().as_slice().to_vec();
        if let Some(voice) = name.strip_suffix(".codes") {
            let [frames, width] = shape[..] else {
                return Err(BunsenError::invalid_resource(format!(
                    "{name}: codes of shape {shape:?}, not [frames, codebooks]"
                )));
            };
            let values = data.convert::<i64>().try_into_vec::<i64>().map_err(|e| {
                BunsenError::invalid_resource(format!("{name}: codes as integers: {e:?}"))
            })?;
            let values = values
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<Vec<u32>, _>>()
                .map_err(|_| BunsenError::invalid_resource(format!("{name}: a negative code")))?;
            if frames == 0 {
                return Err(BunsenError::invalid_resource(format!("{name}: no frames")));
            }
            codes.insert(
                String::from(voice),
                values
                    .chunks(width)
                    .map(<[u32]>::to_vec)
                    .collect::<Vec<_>>(),
            );
        } else if let Some(voice) = name.strip_suffix(".embedding") {
            if shape.len() != 1 {
                return Err(BunsenError::invalid_resource(format!(
                    "{name}: an embedding of shape {shape:?}, not a vector"
                )));
            }
            let values = data.convert::<f32>().try_into_vec::<f32>().map_err(|e| {
                BunsenError::invalid_resource(format!("{name}: an embedding as floats: {e:?}"))
            })?;
            embeddings.insert(String::from(voice), values);
        } else {
            return Err(BunsenError::invalid_resource(format!(
                "{name}: neither a voice's codes nor its embedding"
            )));
        }
    }
    let mut voices = BTreeMap::new();
    for (name, codes) in codes {
        let embedding = embeddings.remove(&name).ok_or_else(|| {
            BunsenError::invalid_resource(format!("voice {name}: codes without an embedding"))
        })?;
        voices.insert(
            name,
            Zonos3Voice {
                codes,
                embedding,
                info: None,
            },
        );
    }
    if let Some(name) = embeddings.keys().next() {
        return Err(BunsenError::invalid_resource(format!(
            "voice {name}: an embedding without codes"
        )));
    }
    Ok(voices)
}

/// The stock voices of `loaded`'s `voices` (a `voices.safetensors`), with
/// what its `voice info` (a `voices.json`) says of them, when the set has
/// one.
///
/// # Errors
/// When a file can't be read, and as [`zonos3_voices`] and
/// [`with_zonos3_voice_info`](super::with_zonos3_voice_info).
#[cfg(feature = "cache")]
pub fn load_zonos3_voices(
    loaded: &crate::data::resources::LoadedResourceSet
) -> BunsenResult<BTreeMap<String, Zonos3Voice>> {
    use crate::{
        data::formats::SafetensorsCheckpoint,
        errors::sys_at,
        kits::speech::tts::zonos3::pretrained::{
            VOICE_INFO,
            VOICES,
        },
    };
    let mut voices =
        zonos3_voices(SafetensorsCheckpoint::single(loaded.expect(VOICES)?).tensors()?)?;
    if let Some(info) = loaded.get(VOICE_INFO) {
        let path = &info.path;
        let json = std::fs::read_to_string(path).map_err(sys_at("read", path))?;
        super::with_zonos3_voice_info(
            &mut voices,
            super::zonos3_voice_info_from_str(&json).map_err(|e| e.context(path.display()))?,
        )?;
    }
    Ok(voices)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use burn::prelude::TensorData;

    use super::*;

    fn tensor(
        name: &str,
        data: TensorData,
    ) -> PackTensor {
        bridge::from_data(data, name.into(), None)
    }

    #[test]
    fn test_voices() {
        let voices = zonos3_voices([
            tensor(
                "Ada.codes",
                TensorData::new(vec![1i32, 2, 3, 4, 5, 6], [2, 3]),
            ),
            tensor("Ada.embedding", TensorData::new(vec![0.5f32, -1.0], [2])),
        ])
        .unwrap();
        let ada = &voices["Ada"];
        assert_eq!(ada.codes, [vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(ada.embedding, [0.5, -1.0]);
    }

    #[test]
    fn test_refusals() {
        let codes = || tensor("Ada.codes", TensorData::new(vec![1i32, 2, 3], [1, 3]));
        let embedding = || tensor("Ada.embedding", TensorData::new(vec![0.5f32], [1]));
        for bad in [
            vec![codes()],
            vec![embedding()],
            vec![
                codes(),
                embedding(),
                tensor("Ada.other", TensorData::new(vec![1f32], [1])),
            ],
            vec![
                tensor("Ada.codes", TensorData::new(vec![-1i32, 2, 3], [1, 3])),
                embedding(),
            ],
            vec![
                tensor("Ada.codes", TensorData::new(vec![1i32, 2, 3], [3])),
                embedding(),
            ],
        ] {
            assert!(zonos3_voices(bad).is_err());
        }
    }
}
