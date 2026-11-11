//! The typed tests the run shares: how a requested language matches a
//! `languages` tag (RFC 4647).

use serde_json::Value;

/// How a requested language matches a build's languages, best first
/// (RFC 4647).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LanguageMatch {
    /// A tag the request filters to (`en` matches `en`, `en-US`).
    Filtered,
    /// A tag matched only once the request is truncated (`en-GB` to `en`).
    Truncated,
    /// Only the build's `"*"`: unspecified other languages.
    Wildcard,
    /// None of the build's languages.
    Missed,
}

/// `request` (a BCP 47 tag, or `"*"`) against a `languages` value,
/// case-insensitively.
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
    fn test_language_matching() {
        let langs = json!(["en", "zh-Hans", "*"]);
        assert_eq!(match_language("en", &langs), LanguageMatch::Filtered);
        assert_eq!(match_language("zh", &langs), LanguageMatch::Filtered);
        assert_eq!(match_language("en-GB", &langs), LanguageMatch::Truncated);
        assert_eq!(match_language("fr", &langs), LanguageMatch::Wildcard);
        assert_eq!(match_language("fr", &json!(["en"])), LanguageMatch::Missed);
        assert_eq!(
            match_language("de-x-y", &json!(["de"])),
            LanguageMatch::Truncated
        );
    }
}
