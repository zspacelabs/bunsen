//! Lookup expressions: `name[preset, …]{patch}`.

use alloc::{
    string::{
        String,
        ToString,
    },
    vec::Vec,
};
use core::fmt;

use serde_json::Value;

use crate::{
    data::catalog::presets::PresetSelection,
    errors::{
        BunsenError,
        BunsenErrorKind,
        BunsenResult,
        ParseError,
    },
};

/// A lookup expression, split into the name it looks up and what it asks of
/// the named pretrained's presets.
///
/// The form is `[provider:]name[preset, …]{patch}`, where the box and the
/// patch are each optional:
///
/// - `name` runs to the first `[` or `{`: `qwen/qwen3-8b`, `hf:Qwen/Qwen3-4B`.
///   Names hold neither.
/// - The box names presets, comma-separated, in the preset alphabet (ASCII
///   letters, digits, `-`, `_`, `.`): `[thinking, yarn-128k]`.
/// - The patch is a YAML mapping (JSON is YAML), a final merge patch over the
///   resolved config: `{beam_size: 1}` or `{"vad": {"check_ms": 120}}`. In
///   YAML's flow style a key needs a space after its colon.
///
/// On a shell command line, quote the whole expression: `[` and `]` glob, and
/// braces with commas brace-expand.
///
/// `Display` writes it back with an unspaced box and the patch as a YAML flow
/// mapping, which parses back to the same expression.
///
/// ```
/// use bunsen::kits::support::construct::LookupExpr;
/// use serde_json::json;
///
/// let expr = LookupExpr::parse("openai/small[translate]{beam_size: 1}")?;
/// assert_eq!(expr.name, "openai/small");
/// assert_eq!(expr.selection.presets, ["translate"]);
/// assert_eq!(expr.selection.patch, Some(json!({"beam_size": 1})));
/// assert_eq!(expr.to_string(), "openai/small[translate]{beam_size: 1}");
///
/// let bare = LookupExpr::parse("hf:Qwen/Qwen3-0.6B")?;
/// assert_eq!(bare.name, "hf:Qwen/Qwen3-0.6B");
/// assert!(bare.selection.is_empty());
/// # Ok::<(), bunsen::errors::BunsenError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupExpr {
    /// The name looked up: `[provider:]name`.
    pub name: String,

    /// The presets the box names, and the patch.
    pub selection: PresetSelection,
}

impl LookupExpr {
    /// Splits `spec` into its name, box and patch.
    ///
    /// # Errors
    /// [`Policy`](BunsenErrorKind::Policy), with a [`ParseError`] cause, for
    /// an expression that does not split: no name, a name holding whitespace
    /// or a closing bracket, an unclosed or empty box, a box entry outside the
    /// preset alphabet, a patch that is not a YAML mapping, or anything after
    /// the box that is not a patch.
    pub fn parse(spec: &str) -> BunsenResult<Self> {
        let fail = |why: String| {
            BunsenError::from_cause(
                BunsenErrorKind::Policy,
                ParseError::new("lookup expression")
                    .input(spec)
                    .because(why),
            )
        };

        let text = spec.trim();
        let cut = text.find(['[', '{']).unwrap_or(text.len());
        let name = &text[..cut];
        if name.is_empty() {
            return Err(fail("it names nothing before its `[` or `{`".to_string()));
        }
        if name.contains(|c: char| c.is_whitespace() || c == ']' || c == '}') {
            return Err(fail(format!(
                "the name {name:?} holds whitespace or a closing bracket"
            )));
        }

        let mut rest = &text[cut..];
        let mut presets = Vec::new();
        if let Some(boxed) = rest.strip_prefix('[') {
            let Some(close) = boxed.find(']') else {
                return Err(fail("its `[` is never closed".to_string()));
            };
            let inner = &boxed[..close];
            if inner.trim().is_empty() {
                return Err(fail("its box names no preset".to_string()));
            }
            for entry in inner.split(',') {
                let entry = entry.trim();
                if !is_preset_name(entry) {
                    return Err(fail(format!(
                        "{entry:?} is not a preset name (ASCII letters, digits, `-`, `_`, `.`)"
                    )));
                }
                presets.push(entry.to_string());
            }
            rest = boxed[close + 1..].trim_start();
        }

        let patch = if rest.starts_with('{') {
            let value: Value = serde_saphyr::from_str(rest)
                .map_err(|e| fail(format!("its patch is not YAML: {e}")))?;
            if !value.is_object() {
                return Err(fail("its patch is not a mapping".to_string()));
            }
            Some(value)
        } else if rest.is_empty() {
            None
        } else {
            return Err(fail(format!("{rest:?} follows the box")));
        };

        Ok(Self {
            name: name.to_string(),
            selection: PresetSelection { presets, patch },
        })
    }
}

impl fmt::Display for LookupExpr {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}{}", self.name, self.selection)
    }
}

/// Whether `name` is in the preset alphabet: non-empty ASCII letters,
/// digits, `-`, `_` and `.`.
pub fn is_preset_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::errors::testing::ErrorMatcher;

    fn parse(spec: &str) -> LookupExpr {
        LookupExpr::parse(spec).unwrap()
    }

    #[test]
    fn test_preset_names() {
        assert!(is_preset_name("yarn-128k"));
        assert!(is_preset_name("faster_whisper.v2"));
        assert!(!is_preset_name(""));
        assert!(!is_preset_name("a,b"));
        assert!(!is_preset_name("a b"));
        assert!(!is_preset_name("x[y]"));
    }

    #[test]
    fn test_a_bare_name() {
        let e = parse("qwen/qwen3-8b");
        assert_eq!(e.name, "qwen/qwen3-8b");
        assert!(e.selection.is_empty());
        assert_eq!(e.to_string(), "qwen/qwen3-8b");

        assert_eq!(parse("  tiny  ").name, "tiny");
        assert_eq!(parse("hf:Qwen/Qwen3-4B").name, "hf:Qwen/Qwen3-4B");
    }

    #[test]
    fn test_a_box() {
        let e = parse("qwen/qwen3-8b[yarn-128k, thinking]");
        assert_eq!(e.name, "qwen/qwen3-8b");
        assert_eq!(e.selection.presets, ["yarn-128k", "thinking"]);
        assert_eq!(e.selection.patch, None);
        assert_eq!(e.to_string(), "qwen/qwen3-8b[yarn-128k,thinking]");
        assert_eq!(
            parse("hf:Qwen/Qwen3.5-4B[thinking]").name,
            "hf:Qwen/Qwen3.5-4B"
        );
    }

    #[test]
    fn test_a_patch() {
        let e = parse("silero/vad-16k{vad: {check_ms: 120, keep: true}}");
        assert!(e.selection.presets.is_empty());
        assert_eq!(
            e.selection.patch,
            Some(json!({"vad": {"check_ms": 120, "keep": true}}))
        );

        let json = parse(r#"silero/vad-16k{"vad": {"check_ms": 0.5}}"#);
        assert_eq!(
            json.selection.patch,
            Some(json!({"vad": {"check_ms": 0.5}}))
        );
    }

    #[test]
    fn test_a_box_and_a_patch() {
        let e = parse("foo/bar[translate] {vad: {check_ms: 120}}");
        assert_eq!(e.name, "foo/bar");
        assert_eq!(e.selection.presets, ["translate"]);
        assert_eq!(e.selection.patch, Some(json!({"vad": {"check_ms": 120}})));
        assert_eq!(e.to_string(), "foo/bar[translate]{vad: {check_ms: 120}}");
        assert_eq!(LookupExpr::parse(&e.to_string()).unwrap(), e);
    }

    /// The patch is written as a YAML flow mapping, a string quoted only
    /// where YAML would read it as another thing, and parses back to the
    /// same expression.
    #[test]
    fn test_the_patch_is_written_as_yaml() {
        let e = LookupExpr {
            name: "openai/large-v3".to_string(),
            selection: PresetSelection {
                presets: vec!["translate".to_string()],
                patch: Some(json!({
                    "beam_size": 5,
                    "language": "de",
                    "temperature": 0.5,
                    "timestamps": true,
                    "prompt": null,
                    "stops": [1, 2],
                    "nested": {"a/b": "x.y-z", "empty": {}},
                    "quoted": ["yes", "No", "null", "5", "a b", "a: b", "", "-x", "\"q\"", "ü"],
                })),
            },
        };
        assert_eq!(
            e.to_string(),
            concat!(
                "openai/large-v3[translate]{beam_size: 5, language: de, ",
                "nested: {a/b: x.y-z, empty: {}}, prompt: null, ",
                r#"quoted: ["yes", "No", "null", "5", "a b", "a: b", "", "-x", "\"q\"", "ü"], "#,
                "stops: [1, 2], temperature: 0.5, timestamps: true}",
            )
        );
        assert_eq!(LookupExpr::parse(&e.to_string()).unwrap(), e);
    }

    fn assert_refused(
        spec: &str,
        reason: &str,
    ) {
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .has_cause::<ParseError>()
            .message_contains(reason)
            .assert_err(&LookupExpr::parse(spec));
    }

    #[test]
    fn test_malformed_expressions() {
        assert_refused("", "names nothing");
        assert_refused("[thinking]", "names nothing");
        assert_refused("a b", "holds whitespace");
        assert_refused("a]b", "closing bracket");
        assert_refused("tiny[thinking", "never closed");
        assert_refused("tiny[]", "names no preset");
        assert_refused("tiny[ , ]", "is not a preset name");
        assert_refused("tiny[a b]", "is not a preset name");
        assert_refused("tiny[a]x", "follows the box");
        assert_refused("tiny{beam: [1}", "not YAML");
    }

    #[test]
    fn test_flow_keys_need_a_space() {
        // In YAML's flow style, `beam:1` without a space is one key.
        let e = parse("tiny{beam:1}");
        assert_eq!(e.selection.patch, Some(json!({"beam:1": null})));
        assert_eq!(
            parse("tiny{beam: 1}").selection.patch,
            Some(json!({"beam": 1}))
        );
        // Only one box.
        assert_refused("tiny[a][b]", "follows the box");
    }
}
