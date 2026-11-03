//! "Did you mean?": a failed lookup's nearest names, by edit distance.
//!
//! The library returns a [`LookupError`] with every key its table has; the
//! ranking is the command line's, here.

use bunsen::errors::{
    BunsenError,
    LookupError,
    LookupProblem,
    Namespace,
};

/// The most names a hint offers.
const MOST: usize = 3;

/// How many candidates the error's one-line message names (the library's
/// `Display` counts the rest): a table with more is listed in full.
const SHOWN: usize = 16;

/// The candidates of `lookup` nearest its key, nearest first, at most
/// [`MOST`]: within a Damerau-Levenshtein distance of a third of the key's
/// length (at least 1), case aside. A key without a provider (`well-known:`)
/// is matched against, and offered, the names without theirs; a name offered
/// twice so is offered once.
pub fn nearest(lookup: &LookupError) -> Vec<String> {
    let key = lookup.key.to_lowercase();
    let bare = !key.contains(':');
    let most = (key.chars().count() / 3).max(1);
    let mut ranked: Vec<(usize, String)> = Vec::new();
    for candidate in &lookup.candidates {
        let name = match candidate.split_once(':') {
            Some((_, name)) if bare => name,
            _ => candidate.as_str(),
        };
        let distance = strsim::damerau_levenshtein(&key, &name.to_lowercase());
        if distance <= most && !ranked.iter().any(|(_, n)| n == name) {
            ranked.push((distance, name.to_string()));
        }
    }
    ranked.sort();
    ranked.into_iter().take(MOST).map(|(_, n)| n).collect()
}

/// What to print after `error`, when its cause is a key missing from a
/// table: the names nearest it ("did you mean?"), and, when the message
/// counted some it did not name, every name the table has.
pub fn hint(error: &BunsenError) -> Option<String> {
    let lookup = error.find::<LookupError>()?;
    if lookup.problem != LookupProblem::Missing
        || !matches!(lookup.namespace, Namespace::Table(_))
        || lookup.candidates.is_empty()
    {
        return None;
    }
    let mut lines = Vec::new();
    let nearest = nearest(lookup);
    if let Some((last, rest)) = nearest.split_last() {
        let quoted = |n: &String| format!("`{n}`");
        lines.push(match rest {
            [] => format!("did you mean {}?", quoted(last)),
            rest => format!(
                "did you mean {}, or {}?",
                rest.iter().map(quoted).collect::<Vec<_>>().join(", "),
                quoted(last)
            ),
        });
    }
    if lookup.candidates.len() > SHOWN {
        lines.push(format!("the {} names there are:", lookup.candidates.len()));
        lines.extend(lookup.candidates.iter().map(|c| format!("  {c}")));
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn missing(
        key: &str,
        candidates: &[&str],
    ) -> LookupError {
        LookupError::missing("pretrained", key).with_candidates(candidates.iter().copied())
    }

    /// The nearest names first, case and an unasked provider aside; a far
    /// one is not offered.
    #[test]
    fn test_nearest() {
        let names = [
            "well-known:openai/tiny.en",
            "well-known:openai/tiny",
            "well-known:openai/small",
            "bundled:openai/tiny",
        ];
        assert_eq!(
            nearest(&missing("openai/tinny", &names)),
            ["openai/tiny", "openai/tiny.en"]
        );
        assert_eq!(nearest(&missing("OpenAI/Small", &names)), ["openai/small"]);
        // A key with a provider is matched as written.
        assert_eq!(
            nearest(&missing("bundled:openai/tini", &names)),
            ["bundled:openai/tiny"]
        );
        assert!(nearest(&missing("qwen/qwen3-8b", &names)).is_empty());
        // A short key: one edit.
        assert_eq!(
            nearest(&missing("mult", &["en", "multi", "translate-en"])),
            ["multi"]
        );
    }

    /// A hint for a key missing from a table, through the error's causes;
    /// none for another failure.
    #[test]
    fn test_hint() {
        let error = BunsenError::lookup(missing("translte", &["transcribe", "translate"]))
            .context("the presets of \"openai/small\"");
        assert_eq!(hint(&error).as_deref(), Some("did you mean `translate`?"));

        let three = BunsenError::lookup(missing(
            "qwen3-0.6",
            &["qwen3-0.6b", "qwen3-1.7b", "qwen3-8b", "qwen3-32b-awq"],
        ));
        assert_eq!(
            hint(&three).as_deref(),
            Some("did you mean `qwen3-0.6b`, `qwen3-1.7b`, or `qwen3-8b`?")
        );

        // Past what the message names, every name is listed.
        let many: Vec<String> = (0..20).map(|i| format!("model-{i}")).collect();
        let error = BunsenError::lookup(
            LookupError::missing("pretrained", "zzz").with_candidates(many.iter().cloned()),
        );
        let text = hint(&error).unwrap();
        assert!(
            text.starts_with("the 20 names there are:\n  model-0\n"),
            "{text}"
        );

        assert_eq!(hint(&BunsenError::policy("no")), None);
        let file = BunsenError::lookup(LookupError::path("/nope", LookupProblem::Missing));
        assert_eq!(hint(&file), None);
    }
}
