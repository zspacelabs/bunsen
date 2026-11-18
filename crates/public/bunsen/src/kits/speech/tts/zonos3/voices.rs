//! ZONOS3-Mini's stock voices: `Zyphra/ZONOS3-Mini-hf`'s
//! `voices.safetensors` ([`zonos3_voices`]), and what its `voices.json`
//! says of each ([`zonos3_voice_info`]).
//!
//! Each voice is two tensors: `<Name>.codes`, `[F, n]` integers, its
//! reference codes (a reference prefix's), and `<Name>.embedding`, `[S]`
//! floats, its speaker embedding. `voices.json` maps each name to its
//! language, style and gender, and its files' stem.

use alloc::{
    collections::BTreeMap,
    string::String,
    vec::Vec,
};

use burn::store::{
    bridge,
    burn_pack::Tensor as PackTensor,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// One stock voice.
#[derive(Clone, Debug, PartialEq)]
pub struct Zonos3Voice {
    /// Its reference codes, `[F][n]`.
    pub codes: Vec<Vec<u32>>,

    /// Its speaker embedding, `[S]`.
    pub embedding: Vec<f32>,

    /// What the release says of it, when it says
    /// ([`with_zonos3_voice_info`]).
    pub info: Option<Zonos3VoiceInfo>,
}

/// What a release's `voices.json` says of a stock voice, under its names.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Zonos3VoiceInfo {
    /// The language it speaks: `English`.
    pub language: String,

    /// Its accent or delivery, where the release names one: `Irish`,
    /// `British`, `energetic`.
    pub style: Option<String>,

    /// Its gender: `female`, `male`.
    pub gender: String,

    /// The stem of the voice's files upstream: `Cillian_en-Irish_M_1`.
    pub file: String,
}

/// The voices a `voices.json` describes, by name.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) for
/// a document that is not a map of names to voice descriptions.
pub fn zonos3_voice_info(json: &str) -> BunsenResult<BTreeMap<String, Zonos3VoiceInfo>> {
    serde_json::from_str(json)
        .map_err(|e| BunsenError::invalid_resource(format!("a voices.json: {e}")))
}

/// Attaches `info` to the `voices` it names; a voice it does not name keeps
/// none.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) for
/// a description of a voice there is not: the two files disagree.
pub fn with_zonos3_voice_info(
    voices: &mut BTreeMap<String, Zonos3Voice>,
    info: BTreeMap<String, Zonos3VoiceInfo>,
) -> BunsenResult<()> {
    for (name, info) in info {
        let voice = voices.get_mut(&name).ok_or_else(|| {
            BunsenError::invalid_resource(format!("voices.json describes {name}, not a voice"))
        })?;
        voice.info = Some(info);
    }
    Ok(())
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

#[cfg(test)]
mod tests {
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
    fn test_voice_info() {
        let mut voices = zonos3_voices([
            tensor("Ada.codes", TensorData::new(vec![1i32, 2, 3], [1, 3])),
            tensor("Ada.embedding", TensorData::new(vec![0.5f32], [1])),
            tensor("Bo.codes", TensorData::new(vec![1i32, 2, 3], [1, 3])),
            tensor("Bo.embedding", TensorData::new(vec![0.5f32], [1])),
        ])
        .unwrap();
        let info = zonos3_voice_info(
            r#"{"Ada": {"language": "English", "style": "Irish", "gender": "female",
                "file": "Ada_en-Irish_F_1"},
                "Bo": {"language": "French", "style": null, "gender": "male",
                "file": "Bo_fr_M_1"}}"#,
        )
        .unwrap();
        let mut ada_only = info.clone();
        ada_only.remove("Bo");
        with_zonos3_voice_info(&mut voices, ada_only).unwrap();
        assert_eq!(
            voices["Ada"].info.as_ref().unwrap().style.as_deref(),
            Some("Irish")
        );
        assert_eq!(voices["Bo"].info, None, "undescribed");
        with_zonos3_voice_info(&mut voices, info.clone()).unwrap();
        assert_eq!(
            voices["Bo"].info.as_ref().unwrap().style,
            None,
            "no style named"
        );

        let mut stray = info;
        stray.insert("Cy".into(), stray["Ada"].clone());
        assert!(with_zonos3_voice_info(&mut voices, stray).is_err());
        assert!(zonos3_voice_info("[]").is_err());
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
