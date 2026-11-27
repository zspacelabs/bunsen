//! The release's description of its stock voices.

use alloc::{
    collections::BTreeMap,
    format,
    string::String,
};

use serde::Deserialize;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::tts::{
        driver::TtsVoiceInfo,
        zonos3::voices::Zonos3Voice,
    },
};

/// What a release's `voices.json` says of a stock voice, under its names.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Zonos3VoiceInfoMirror {
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

impl Zonos3VoiceInfoMirror {
    /// What the driver says of the voice `name`: its gender and style, and
    /// its language as the release names it (`English`), as its
    /// description.
    pub fn voice_info(
        &self,
        name: &str,
    ) -> TtsVoiceInfo {
        let mut info = TtsVoiceInfo::stock(name);
        info.gender = Some(self.gender.clone());
        info.style.clone_from(&self.style);
        info.description = Some(self.language.clone());
        info
    }
}

/// The voices a `voices.json` describes, by name.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) for
/// a document that is not a map of names to voice descriptions.
pub fn zonos3_voice_info_from_str(
    json: &str
) -> BunsenResult<BTreeMap<String, Zonos3VoiceInfoMirror>> {
    serde_json::from_str(json)
        .map_err(|e| BunsenError::invalid_resource(format!("a voices.json: {e}")))
}

/// Attaches what `info` says to the `voices` it names; a voice it does not
/// name keeps none.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) for
/// a description of a voice there is not: the two files disagree.
pub fn with_zonos3_voice_info(
    voices: &mut BTreeMap<String, Zonos3Voice>,
    info: BTreeMap<String, Zonos3VoiceInfoMirror>,
) -> BunsenResult<()> {
    for (name, info) in info {
        let voice = voices.get_mut(&name).ok_or_else(|| {
            BunsenError::invalid_resource(format!("voices.json describes {name}, not a voice"))
        })?;
        voice.info = Some(info.voice_info(&name));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::errors::testing::ErrorMatcher;

    fn voice() -> Zonos3Voice {
        Zonos3Voice {
            codes: vec![vec![1, 2, 3]],
            embedding: vec![0.5],
            info: None,
        }
    }

    #[test]
    fn test_voice_info() {
        let mut voices =
            BTreeMap::from([("Ada".to_string(), voice()), ("Bo".to_string(), voice())]);
        let info = zonos3_voice_info_from_str(
            r#"{"Ada": {"language": "English", "style": "Irish", "gender": "female",
                "file": "Ada_en-Irish_F_1"},
                "Bo": {"language": "French", "style": null, "gender": "male",
                "file": "Bo_fr_M_1"}}"#,
        )
        .unwrap();
        let mut ada_only = info.clone();
        ada_only.remove("Bo");
        with_zonos3_voice_info(&mut voices, ada_only).unwrap();
        let ada = voices["Ada"].info.as_ref().unwrap();
        assert_eq!(ada.name, "Ada");
        assert_eq!(ada.style.as_deref(), Some("Irish"));
        assert_eq!(ada.gender.as_deref(), Some("female"));
        assert_eq!(ada.description.as_deref(), Some("English"));
        assert_eq!(voices["Bo"].info, None, "undescribed");
        with_zonos3_voice_info(&mut voices, info.clone()).unwrap();
        assert_eq!(
            voices["Bo"].info.as_ref().unwrap().style,
            None,
            "no style named"
        );

        let mut stray = info;
        stray.insert("Cy".into(), stray["Ada"].clone());
        ErrorMatcher::invalid_resource().assert_err(&with_zonos3_voice_info(&mut voices, stray));
        ErrorMatcher::invalid_resource().assert_err(&zonos3_voice_info_from_str("[]"));
    }
}
