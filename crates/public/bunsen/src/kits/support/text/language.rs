//! [`BunsenLanguageTag`]: a well-known language code, a BCP 47 tag.

use alloc::{
    format,
    string::{
        String,
        ToString,
    },
};
use core::{
    fmt,
    str::FromStr,
};

use icu_locale_core::Locale;
use serde::{
    Deserialize,
    Deserializer,
    Serialize,
    Serializer,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// A well-known language code: a BCP 47 tag with Unicode extensions, held as
/// an ICU4X [`Locale`].
///
/// `en`, `en-US`, `zh-Hans-CN`, and CLDR's subdivision form `zh-u-sd-cn51`
/// (Sichuan) all parse. [`parse`](Self::parse) also reads the POSIX form a
/// system locale or a model's own table may use (`en_US`, `en_us`,
/// `en_US.UTF-8`, `de_DE@euro`). A tag prints in BCP 47's canonical case
/// (`en-US`), and is serialized as that string.
///
/// A model's own language keys (Qwen3-TTS's `sichuan_dialect`, ZONOS3's
/// `en_us`) are not tags, even where they parse as one: a model maps its
/// keys to tags, so search and normalizers read one form.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BunsenLanguageTag(Locale);

impl BunsenLanguageTag {
    /// `text`, a BCP 47 tag or a POSIX locale name, as a tag.
    ///
    /// A POSIX name's codeset (`.UTF-8`) and modifier (`@euro`) are dropped,
    /// and its `_` separators read as `-`.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for text that is
    /// neither.
    pub fn parse(text: &str) -> BunsenResult<Self> {
        let trimmed = text.trim();
        let stem = trimmed
            .split(['.', '@'])
            .next()
            .unwrap_or_default()
            .replace('_', "-");
        Locale::try_from_str(&stem).map(Self).map_err(|error| {
            BunsenError::illegal(format!(
                "{text:?} is not a language tag (BCP 47, or a POSIX name like `en_US`): {error}"
            ))
        })
    }

    /// The tag's locale.
    pub fn locale(&self) -> &Locale {
        &self.0
    }

    /// The primary language subtag: `en` of `en-US`, `zh` of `zh-u-sd-cn51`.
    pub fn language(&self) -> &str {
        self.0.id.language.as_str()
    }
}

impl From<Locale> for BunsenLanguageTag {
    fn from(locale: Locale) -> Self {
        Self(locale)
    }
}

impl FromStr for BunsenLanguageTag {
    type Err = BunsenError;

    fn from_str(text: &str) -> BunsenResult<Self> {
        Self::parse(text)
    }
}

impl fmt::Display for BunsenLanguageTag {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Serialize for BunsenLanguageTag {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for BunsenLanguageTag {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::BunsenErrorKind;

    #[test]
    fn test_parse_bcp47() {
        let tag = BunsenLanguageTag::parse("zh-Hans-CN").unwrap();
        assert_eq!(tag.to_string(), "zh-Hans-CN");
        assert_eq!(tag.language(), "zh");

        let sichuan = BunsenLanguageTag::parse("zh-u-sd-cn51").unwrap();
        assert_eq!(sichuan.to_string(), "zh-u-sd-cn51");
        assert_eq!(sichuan.language(), "zh");

        assert_eq!(
            BunsenLanguageTag::parse("EN-us").unwrap().to_string(),
            "en-US"
        );
    }

    #[test]
    fn test_parse_posix() {
        for text in ["en_US", "en_us", "en_US.UTF-8", " en_US@euro "] {
            assert_eq!(
                BunsenLanguageTag::parse(text).unwrap().to_string(),
                "en-US",
                "{text}"
            );
        }
    }

    #[test]
    fn test_parse_rejects() {
        for text in ["", "e", "en us", "123", "english", "sichuan_dialect"] {
            let error = BunsenLanguageTag::parse(text).unwrap_err();
            assert_eq!(error.kind(), BunsenErrorKind::Illegal, "{text}");
        }
    }

    #[test]
    fn test_serde_round_trip() {
        let tag: BunsenLanguageTag = "fr_FR".parse().unwrap();
        let json = serde_json::to_string(&tag).unwrap();
        assert_eq!(json, "\"fr-FR\"");
        assert_eq!(
            serde_json::from_str::<BunsenLanguageTag>(&json).unwrap(),
            tag
        );
        assert!(serde_json::from_str::<BunsenLanguageTag>("\"not a tag\"").is_err());
    }
}
