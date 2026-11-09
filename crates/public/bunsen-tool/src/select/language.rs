//! The graded match of a requested language against a row's `language` tag.
use serde_json::Value;

/// How a requested language matches a row's languages, best first (RFC 4647).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LanguageMatch {
    /// A row tag the request filters to: the request is the tag, or a prefix
    /// of it on a subtag boundary (`en` matches `en`, `en-US`).
    Filtered,

    /// A row tag matched only once the request is truncated, as RFC 4647's
    /// lookup truncates it (`en-GB` falls back to `en`).
    Truncated,

    /// Only the row's `"*"`: unspecified other languages, so unknown.
    Wildcard,

    /// None of the row's languages.
    Missed,
}

/// `request` (a BCP 47 tag, or `"*"` for any) against a row's `language`
/// tag value, case-insensitively.
pub fn match_language(
    request: &str,
    languages: &Value,
) -> LanguageMatch {
    let tags: Vec<String> = match languages {
        Value::String(s) => vec![s.to_ascii_lowercase()],
        Value::Array(items) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_ascii_lowercase)
            .collect(),
        _ => Vec::new(),
    };
    let request = request.to_ascii_lowercase();
    let wildcard = tags.iter().any(|t| t == "*");
    let named: Vec<&str> = tags
        .iter()
        .map(String::as_str)
        .filter(|t| *t != "*")
        .collect();
    if request == "*" {
        return if tags.is_empty() {
            LanguageMatch::Missed
        } else {
            LanguageMatch::Filtered
        };
    }
    let filters = |range: &str, tag: &str| {
        tag == range || (tag.starts_with(range) && tag.as_bytes().get(range.len()) == Some(&b'-'))
    };
    if named.iter().any(|t| filters(&request, t)) {
        return LanguageMatch::Filtered;
    }
    let mut range = request.as_str();
    while let Some(cut) = range.rfind('-') {
        range = &range[..cut];
        // A singleton is never left at the end of a truncated range.
        if let Some(cut) = range.rfind('-')
            && range.len() - cut == 2
        {
            range = &range[..cut];
        }
        if named.contains(&range) {
            return LanguageMatch::Truncated;
        }
    }
    if wildcard {
        LanguageMatch::Wildcard
    } else {
        LanguageMatch::Missed
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_match_language() {
        use LanguageMatch::*;
        assert_eq!(match_language("en", &json!(["en"])), Filtered);
        assert_eq!(match_language("EN", &json!(["en-US"])), Filtered);
        assert_eq!(
            match_language("en", &json!(["eng"])),
            Missed,
            "a subtag boundary"
        );
        assert_eq!(match_language("en-GB", &json!(["en"])), Truncated);
        assert_eq!(match_language("zh-Hant-TW", &json!(["zh"])), Truncated);
        assert_eq!(match_language("en-GB", &json!(["en-US"])), Missed);
        assert_eq!(match_language("de", &json!(["en", "*"])), Wildcard);
        assert_eq!(match_language("de", &json!(["*"])), Wildcard);
        assert_eq!(match_language("de", &json!(["en"])), Missed);
        assert_eq!(match_language("*", &json!(["en"])), Filtered);
        assert_eq!(match_language("*", &json!([])), Missed);
        assert_eq!(match_language("en", &json!("en")), Filtered);
        assert!(Filtered < Truncated && Truncated < Wildcard && Wildcard < Missed);
    }

    /// A truncated range never ends in a singleton.
    #[test]
    fn test_singletons_are_dropped() {
        assert_eq!(
            match_language("en-a-xyz", &json!(["en"])),
            LanguageMatch::Truncated
        );
        assert_eq!(
            match_language("en-a-xyz", &json!(["en-a"])),
            LanguageMatch::Missed
        );
    }
}
