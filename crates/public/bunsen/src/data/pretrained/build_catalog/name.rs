//! Names: [`CatalogName`], a `/`-separated path of segments, each of
//! which YAML reads as a plain, unquoted key.

use core::{
    fmt,
    str::FromStr,
};

use serde::{
    Deserialize,
    Deserializer,
    Serialize,
    Serializer,
};

use crate::data::pretrained::build_catalog::error::{
    CatalogError,
    CatalogErrorKind,
    CatalogResult,
};

/// The words YAML reads as something other than a string, in any case.
const YAML_WORDS: [&str; 12] = [
    "true", "false", "null", "yes", "no", "on", "off", "y", "n", "nan", "inf", "infinity",
];

/// Whether `segment` is a legal name segment.
///
/// A segment starts with an ASCII letter or `_`, continues with ASCII
/// letters, digits, `_`, `-` and `.`, and is none of the words YAML reads
/// as a bool, a null or a number, in any case. So every segment is a
/// plain, unquoted YAML key, and none is `.` or `..`.
pub fn is_segment(segment: &str) -> bool {
    let mut chars = segment.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
        && !YAML_WORDS.iter().any(|w| w.eq_ignore_ascii_case(segment))
}

/// Checks `segment` with [`is_segment`].
///
/// # Errors
/// [`CatalogErrorKind::Name`] when it is not a legal segment.
pub fn check_segment(segment: &str) -> CatalogResult<()> {
    if is_segment(segment) {
        Ok(())
    } else {
        Err(CatalogError::new(
            CatalogErrorKind::Name,
            format!(
                "`{segment}` is not a name segment: it must start with an ASCII letter or `_`, \
                 continue with ASCII letters, digits, `_`, `-` or `.`, and not be a YAML word"
            ),
        ))
    }
}

/// The `$` word that opens the contracts' section of the base catalog.
pub const CONTRACTS: &str = "$contracts";

/// A full name: one or more segments, written joined by `/`
/// (`qwen/qwen3/qwen3-4b`).
///
/// Every entry in a catalog has one. The segments before the last are its
/// naming scope; the last is its own name, the leaf.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CatalogName(Vec<String>);

impl CatalogName {
    /// Parses a full name.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Name`] when it is empty or a segment is not
    /// legal.
    pub fn parse(name: &str) -> CatalogResult<Self> {
        if name.is_empty() {
            return Err(CatalogError::new(CatalogErrorKind::Name, "a name is empty"));
        }
        let segments: Vec<String> = name.split('/').map(str::to_string).collect();
        // `$contracts` opens the contracts' own namespace; it is no
        // entry's segment.
        let named = match segments.split_first() {
            Some((first, rest)) if first == CONTRACTS => rest,
            _ => &segments[..],
        };
        for segment in named {
            check_segment(segment).map_err(|e| {
                CatalogError::new(
                    CatalogErrorKind::Name,
                    format!("in `{name}`: {}", e.message),
                )
            })?;
        }
        Ok(Self(segments))
    }

    /// The root of the `$contracts` section: contracts and their types
    /// are named under it (`$contracts/whisper`).
    pub fn contracts_root() -> Self {
        Self(vec![CONTRACTS.to_string()])
    }

    /// Whether the name is in the `$contracts` section.
    pub fn in_contracts(&self) -> bool {
        self.0.first().is_some_and(|s| s == CONTRACTS)
    }

    /// The name within `$contracts` (`whisper` for `$contracts/whisper`),
    /// for a name in that section.
    pub fn contract_path(&self) -> Option<String> {
        self.in_contracts()
            .then(|| self.0[1..].join("/"))
            .filter(|path| !path.is_empty())
    }

    /// A name of one segment.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Name`] when `segment` is not legal.
    pub fn root(segment: &str) -> CatalogResult<Self> {
        check_segment(segment)?;
        Ok(Self(vec![segment.to_string()]))
    }

    /// This name with `segment` appended.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Name`] when `segment` is not legal.
    pub fn child(
        &self,
        segment: &str,
    ) -> CatalogResult<Self> {
        check_segment(segment)?;
        let mut segments = self.0.clone();
        segments.push(segment.to_string());
        Ok(Self(segments))
    }

    /// `child`, under `scope` when there is one.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Name`] when `segment` is not legal.
    pub fn under(
        scope: Option<&CatalogName>,
        segment: &str,
    ) -> CatalogResult<Self> {
        match scope {
            Some(scope) => scope.child(segment),
            None => Self::root(segment),
        }
    }

    /// The segments, root first.
    pub fn segments(&self) -> &[String] {
        &self.0
    }

    /// The last segment: the entry's own name.
    pub fn leaf(&self) -> &str {
        self.0.last().expect("a name has a segment")
    }

    /// The naming scope: every segment but the last, if there are any.
    pub fn scope(&self) -> Option<CatalogName> {
        (self.0.len() > 1).then(|| Self(self.0[..self.0.len() - 1].to_vec()))
    }

    /// Whether `self` is `ancestor` or lies under it.
    pub fn is_within(
        &self,
        ancestor: &CatalogName,
    ) -> bool {
        self.0.starts_with(&ancestor.0)
    }

    /// The longest name both `self` and `other` lie within, if any.
    pub fn common_scope(
        &self,
        other: &CatalogName,
    ) -> Option<CatalogName> {
        let n = self
            .0
            .iter()
            .zip(&other.0)
            .take_while(|(a, b)| a == b)
            .count();
        (n > 0).then(|| Self(self.0[..n].to_vec()))
    }
}

impl fmt::Display for CatalogName {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(&self.0.join("/"))
    }
}

impl fmt::Debug for CatalogName {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "`{self}`")
    }
}

impl FromStr for CatalogName {
    type Err = CatalogError;

    fn from_str(s: &str) -> CatalogResult<Self> {
        Self::parse(s)
    }
}

impl Serialize for CatalogName {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for CatalogName {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Self::parse(&s).map_err(serde::de::Error::custom)
    }
}

impl CatalogName {
    /// `self` followed by `other`'s segments.
    pub fn join(
        &self,
        other: &CatalogName,
    ) -> CatalogName {
        let mut segments = self.0.clone();
        segments.extend(other.0.iter().cloned());
        Self(segments)
    }
}

/// A reference to an entry, as a document writes it: a name, resolved
/// relative to where it is written, or, with a leading `/`, from the root.
///
/// A relative reference written in the entry `N` names the first of
/// `N/ref`, `scope(N)/ref`, ..., `ref` that is declared: resolution climbs
/// the namespace from the entry itself to the root, and the nearest name
/// wins.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CatalogRef {
    absolute: bool,
    path: CatalogName,
}

impl CatalogRef {
    /// Parses a reference: a name, with an optional leading `/`.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Name`] when the name isn't legal.
    pub fn parse(reference: &str) -> CatalogResult<Self> {
        let (absolute, rest) = match reference.strip_prefix('/') {
            Some(rest) => (true, rest),
            None => (false, reference),
        };
        Ok(Self {
            absolute,
            path: CatalogName::parse(rest)?,
        })
    }

    /// A reference from the root to `name`.
    pub fn absolute(name: CatalogName) -> Self {
        Self {
            absolute: true,
            path: name,
        }
    }

    /// Whether it resolves from the root only.
    pub fn is_absolute(&self) -> bool {
        self.absolute
    }

    /// The name as written, without the leading `/`.
    pub fn path(&self) -> &CatalogName {
        &self.path
    }

    /// The full names it may resolve to, nearest first, when written in
    /// the entry `from` (or at the root).
    pub fn candidates(
        &self,
        from: Option<&CatalogName>,
    ) -> Vec<CatalogName> {
        let mut out = Vec::new();
        if !self.absolute {
            let mut scope = from.cloned();
            while let Some(s) = scope {
                out.push(s.join(&self.path));
                scope = s.scope();
            }
        }
        out.push(self.path.clone());
        out
    }
}

impl fmt::Display for CatalogRef {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        if self.absolute {
            f.write_str("/")?;
        }
        write!(f, "{}", self.path)
    }
}

impl fmt::Debug for CatalogRef {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "`{self}`")
    }
}

impl FromStr for CatalogRef {
    type Err = CatalogError;

    fn from_str(s: &str) -> CatalogResult<Self> {
        Self::parse(s)
    }
}

impl Serialize for CatalogRef {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for CatalogRef {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Self::parse(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segments_are_plain_yaml_keys() {
        for ok in ["qwen3", "qwen3-0.6b", "tiny.en", "_x", "qwen3.5", "a-b_c.d"] {
            assert!(is_segment(ok), "{ok}");
        }
        for bad in [
            "", "0x", "-a", ".", "..", "a/b", "a b", "yes", "No", "NULL", "inf", "ü", "a:b",
        ] {
            assert!(!is_segment(bad), "{bad}");
        }
    }

    #[test]
    fn test_parse_and_display_round_trip() {
        let name = CatalogName::parse("qwen/qwen3/qwen3-4b").unwrap();
        assert_eq!(name.segments(), ["qwen", "qwen3", "qwen3-4b"]);
        assert_eq!(name.leaf(), "qwen3-4b");
        assert_eq!(name.to_string(), "qwen/qwen3/qwen3-4b");
        assert_eq!(name.scope().unwrap().to_string(), "qwen/qwen3");
        assert_eq!(CatalogName::root("qwen").unwrap().scope(), None);
    }

    #[test]
    fn test_parse_rejects_bad_names() {
        for bad in ["", "/a", "a/", "a//b", "a/yes", "a/0b"] {
            let err = CatalogName::parse(bad).unwrap_err();
            assert_eq!(err.kind, CatalogErrorKind::Name, "{bad}");
        }
    }

    #[test]
    fn test_within_and_common_scope() {
        let a = CatalogName::parse("qwen/qwen3/a").unwrap();
        let b = CatalogName::parse("qwen/qwen3/b").unwrap();
        let c = CatalogName::parse("openai/whisper").unwrap();
        let qwen3 = CatalogName::parse("qwen/qwen3").unwrap();
        assert!(a.is_within(&qwen3));
        assert!(qwen3.is_within(&qwen3));
        assert!(!c.is_within(&qwen3));
        assert_eq!(a.common_scope(&b), Some(qwen3));
        assert_eq!(a.common_scope(&c), None);
    }

    #[test]
    fn test_serde_as_a_string() {
        let name: CatalogName = serde_json::from_str("\"qwen/qwen3\"").unwrap();
        assert_eq!(serde_json::to_string(&name).unwrap(), "\"qwen/qwen3\"");
        assert!(serde_json::from_str::<CatalogName>("\"qwen/yes\"").is_err());
    }

    #[test]
    fn test_references_climb_to_the_root() {
        let from = CatalogName::parse("openai/whisper/large/large-v1").unwrap();
        let relative = CatalogRef::parse("release").unwrap();
        let shown: Vec<String> = relative
            .candidates(Some(&from))
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            shown,
            [
                "openai/whisper/large/large-v1/release",
                "openai/whisper/large/release",
                "openai/whisper/release",
                "openai/release",
                "release",
            ]
        );
        let absolute = CatalogRef::parse("/release").unwrap();
        assert!(absolute.is_absolute());
        assert_eq!(absolute.to_string(), "/release");
        assert_eq!(
            absolute.candidates(Some(&from)),
            [CatalogName::parse("release").unwrap()]
        );
        assert!(CatalogRef::parse("//x").is_err());
        assert!(CatalogRef::parse("/").is_err());
    }
}
