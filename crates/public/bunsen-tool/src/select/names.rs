//! What a name matches among candidates: one row exactly, or a set.
use bunsen::data::pretrained::{
    selection::SelectionCandidate,
    tags::{
        GROUPS,
        KIT,
    },
};

/// What a name matches among candidates, by their indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NameMatch {
    /// A row, exactly: its full id, its bare name, or an alias. The first,
    /// in order, of the rows that answer.
    Exact(usize),

    /// No row exactly, but these, by the first tier that matches any: a
    /// prefix of a full id, bare name or alias; else a `groups` tag; else the
    /// `kit` tag.
    Fuzzy(Vec<usize>),

    /// Nothing.
    Missing,
}

/// A row's bare name: its full id without the group (`openai/large-v3` is
/// `large-v3`).
fn bare(id: &str) -> &str {
    id.rsplit_once('/').map_or(id, |(_, name)| name)
}

/// Whether `text` starts with `prefix`, ignoring ASCII case.
fn starts_with_ignoring_case(
    text: &str,
    prefix: &str,
) -> bool {
    text.len() >= prefix.len()
        && text.is_char_boundary(prefix.len())
        && text[..prefix.len()].eq_ignore_ascii_case(prefix)
}

/// What `name` matches among `candidates`, ignoring ASCII case.
pub fn match_name(
    candidates: &[SelectionCandidate],
    name: &str,
) -> NameMatch {
    let names = |c: &SelectionCandidate| -> Vec<String> {
        let row = &c.pretrained;
        let mut names = vec![row.name.clone(), bare(&row.name).to_string()];
        names.extend(row.aliases.iter().cloned());
        names
    };
    if let Some(i) = candidates
        .iter()
        .position(|c| names(c).iter().any(|n| n.eq_ignore_ascii_case(name)))
    {
        return NameMatch::Exact(i);
    }
    let tier = |matches: &dyn Fn(&SelectionCandidate) -> bool| -> Vec<usize> {
        candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| matches(c))
            .map(|(i, _)| i)
            .collect()
    };
    let lower = name.to_ascii_lowercase();
    for found in [
        tier(&|c| names(c).iter().any(|n| starts_with_ignoring_case(n, name))),
        tier(&|c| {
            c.get(GROUPS)
                .and_then(|g| g.as_array())
                .is_some_and(|groups| {
                    groups
                        .iter()
                        .filter_map(|g| g.as_str())
                        .any(|g| g.to_ascii_lowercase() == lower)
                })
        }),
        tier(&|c| c.text(KIT).is_some_and(|k| k.eq_ignore_ascii_case(name))),
    ] {
        if !found.is_empty() {
            return NameMatch::Fuzzy(found);
        }
    }
    NameMatch::Missing
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::select::testing::candidate;

    fn rows() -> Vec<SelectionCandidate> {
        vec![
            candidate("openai/large-v2", json!({"kit": "whisper"})),
            candidate("openai/large-v3", json!({"kit": "whisper"})),
            candidate("openai/large-v3-turbo", json!({"kit": "whisper"})),
            candidate("openai/tiny", json!({"kit": "whisper"})),
            candidate(
                "qwen/qwen3-0.6b",
                json!({"kit": "qwen3", "groups": ["qwen3"]}),
            ),
            candidate(
                "qwen/qwen3-4b-instruct-2507",
                json!({"kit": "qwen3", "groups": ["qwen3-2507"]}),
            ),
        ]
    }

    #[test]
    fn test_exact() {
        let rows = rows();
        assert_eq!(match_name(&rows, "openai/large-v3"), NameMatch::Exact(1));
        assert_eq!(match_name(&rows, "large-v3"), NameMatch::Exact(1));
        assert_eq!(match_name(&rows, "LARGE-V3"), NameMatch::Exact(1));
        let mut aliased = rows.clone();
        aliased[3].pretrained.aliases.push("t".into());
        assert_eq!(match_name(&aliased, "t"), NameMatch::Exact(3));
    }

    /// The first tier that matches any row is the set: a prefix, then a
    /// group, then a kit.
    #[test]
    fn test_fuzzy_tiers() {
        let rows = rows();
        assert_eq!(match_name(&rows, "large"), NameMatch::Fuzzy(vec![0, 1, 2]));
        assert_eq!(
            match_name(&rows, "openai/large-v"),
            NameMatch::Fuzzy(vec![0, 1, 2])
        );
        assert_eq!(
            match_name(&rows, "qwen3"),
            NameMatch::Fuzzy(vec![4, 5]),
            "a prefix"
        );
        assert_eq!(
            match_name(&rows, "qwen3-2507"),
            NameMatch::Fuzzy(vec![5]),
            "a group"
        );
        assert_eq!(
            match_name(&rows, "whisper"),
            NameMatch::Fuzzy(vec![0, 1, 2, 3]),
            "a kit"
        );
        assert_eq!(match_name(&rows, "gemma"), NameMatch::Missing);
    }
}
