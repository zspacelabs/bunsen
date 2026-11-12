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
/// `MOST` (3), case aside, in the form the key is written in:
/// - a key without a provider (`catalog:`) is matched against, and offered, the
///   names without theirs; and a key without a group either (`qwen3-1.7b`, not
///   `qwen/qwen3-1.7b`) the names without their group, which a table answers as
///   well, unless two groups share the name, when it is offered with its group;
/// - a name the key extends by a word (`qwen3-1.7b` of `qwen3-1.7b-turbo`)
///   comes first, whatever the distance;
/// - the others within a Damerau-Levenshtein distance of a third of the key's
///   length (at least 1), nearest first.
///
/// A name offered twice so is offered once.
pub fn nearest(lookup: &LookupError) -> Vec<String> {
    let key = lookup.key.to_lowercase();
    let bare = !key.contains(':');
    let grouped = key.contains('/');
    let most = (key.chars().count() / 3).max(1);
    let name_of = |candidate: &str| -> String {
        let name = match candidate.split_once(':') {
            Some((_, name)) if bare => name,
            _ => candidate,
        };
        name.to_string()
    };
    let short_of = |name: &str| -> Option<String> {
        (bare && !grouped)
            .then(|| name.split_once('/').map(|(_, row)| row.to_string()))
            .flatten()
    };
    let names: Vec<String> = lookup.candidates.iter().map(|c| name_of(c)).collect();
    let shorts: Vec<Option<String>> = names.iter().map(|n| short_of(n)).collect();
    let mut ranked: Vec<(bool, usize, String)> = Vec::new();
    for (name, short) in names.iter().zip(&shorts) {
        let offered = match short {
            Some(row) if shorts.iter().filter(|s| s.as_ref() == Some(row)).count() == 1 => row,
            _ => name,
        };
        let compared = short.as_ref().unwrap_or(name).to_lowercase();
        let extends = key
            .strip_prefix(compared.as_str())
            .and_then(|rest| rest.chars().next())
            .is_some_and(|c| matches!(c, '-' | '_' | '.' | '/'));
        let distance = strsim::damerau_levenshtein(&key, &compared);
        if (extends || distance <= most) && !ranked.iter().any(|(_, _, n)| n == offered) {
            ranked.push((!extends, distance, offered.clone()));
        }
    }
    ranked.sort();
    ranked.into_iter().take(MOST).map(|(_, _, n)| n).collect()
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
            "catalog:openai/tiny.en",
            "catalog:openai/tiny",
            "catalog:openai/small",
            "other:openai/tiny",
        ];
        assert_eq!(
            nearest(&missing("openai/tinny", &names)),
            ["openai/tiny", "openai/tiny.en"]
        );
        assert_eq!(nearest(&missing("OpenAI/Small", &names)), ["openai/small"]);
        // A key with a provider is matched as written.
        assert_eq!(
            nearest(&missing("other:openai/tini", &names)),
            ["other:openai/tiny"]
        );
        assert!(nearest(&missing("qwen/qwen3-8b", &names)).is_empty());
        // A short key: one edit.
        assert_eq!(
            nearest(&missing("mult", &["en", "multi", "translate-en"])),
            ["multi"]
        );
    }

    /// A key without a group is matched against, and offered, the names
    /// without theirs, as a table answers them; a name the key extends by a
    /// word comes first, though past the distance. A name two groups share
    /// keeps its group.
    #[test]
    fn test_nearest_ungrouped() {
        let qwen3 = [
            "catalog:qwen/qwen3-0.6b",
            "catalog:qwen/qwen3-0.6b-base",
            "catalog:qwen/qwen3-1.7b",
            "catalog:qwen/qwen3-1.7b-base",
            "catalog:qwen/qwen3-4b",
            "catalog:qwen/qwen3-4b-base",
            "catalog:qwen/qwen3-4b-instruct-2507",
            "catalog:qwen/qwen3-4b-thinking-2507",
            "catalog:qwen/qwen3-8b",
            "catalog:qwen/qwen3-8b-base",
            "catalog:qwen/qwen3-4b-saferl",
            "catalog:qwen/qwen3guard-gen-0.6b",
            "catalog:qwen/qwen3guard-gen-4b",
            "catalog:qwen/qwen3guard-gen-8b",
        ];
        assert_eq!(
            nearest(&missing("qwen3-1.7b-turbo", &qwen3)),
            ["qwen3-1.7b", "qwen3-1.7b-base"]
        );
        // Within the distance, nearest first.
        assert_eq!(
            nearest(&missing("qwen3-8bb", &qwen3)),
            ["qwen3-8b", "qwen3-4b", "qwen3-0.6b"]
        );
        // With a group, as written.
        assert_eq!(
            nearest(&missing("qwen/qwen3-8bb", &qwen3)),
            ["qwen/qwen3-8b", "qwen/qwen3-4b", "qwen/qwen3-0.6b"]
        );

        let shared = ["catalog:openai/tiny", "catalog:other/tiny"];
        assert_eq!(
            nearest(&missing("tinny", &shared)),
            ["openai/tiny", "other/tiny"]
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
