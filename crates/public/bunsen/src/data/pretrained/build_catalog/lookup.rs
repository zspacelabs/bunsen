//! Looking entries up by shorter names, and naming them for people.
//!
//! A query of segments matches an entry when the query's last segment is
//! the entry's leaf and the query's segments appear in the entry's full
//! name in order: `large-v3`, `whisper/large-v3` and `openai/large-v3` all
//! match `openai/whisper/large-v3`. A query resolves when exactly one entry
//! matches, or when it is some entry's full name.

use crate::data::pretrained::build_catalog::{
    document::{
        CatalogEntry,
        EntryKind,
    },
    error::{
        CatalogError,
        CatalogErrorKind,
        CatalogResult,
    },
    name::CatalogName,
    stack::{
        EntryRef,
        PretrainedCatalogStack,
    },
};

/// The kinds a lookup searches when it isn't told which: every kind but
/// `alias`. An alias is found as its target.
pub const SEARCHABLE: [EntryKind; 9] = [
    EntryKind::Build,
    EntryKind::TemplateClass,
    EntryKind::Namespace,
    EntryKind::Contract,
    EntryKind::Release,
    EntryKind::Tag,
    EntryKind::Axis,
    EntryKind::Facet,
    EntryKind::Resource,
];

/// How a query's last segment matches a leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafMatch {
    /// The leaf is the segment.
    Exact,
    /// The leaf starts with the segment.
    Prefix,
}

/// A name for people: the least specified spelling that resolves to the
/// entry, and its full name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisplayName {
    /// The shortest query that resolves to it.
    pub short: String,
    /// Its full name.
    pub full: CatalogName,
}

fn subsequence(
    query: &[&str],
    name: &CatalogName,
    leaf: LeafMatch,
) -> bool {
    let segments = name.segments();
    let (Some((last, init)), Some((name_last, name_init))) =
        (query.split_last(), segments.split_last())
    else {
        return false;
    };
    let leaf_ok = match leaf {
        LeafMatch::Exact => name_last == last,
        LeafMatch::Prefix => name_last.starts_with(last),
    };
    if !leaf_ok {
        return false;
    }
    let mut rest = name_init.iter();
    init.iter().all(|q| rest.any(|s| s == q))
}

impl PretrainedCatalogStack {
    /// The entries of `kinds` that `query` matches, each once. An
    /// alias that matches stands for its target when its target's kind is
    /// among `kinds`; it is matched itself only when `kinds` asks for
    /// aliases and not for its target's kind.
    pub fn matches(
        &self,
        query: &str,
        kinds: &[EntryKind],
        leaf: LeafMatch,
    ) -> Vec<CatalogName> {
        let parts: Vec<&str> = query.split('/').collect();
        let mut found: Vec<CatalogName> = Vec::new();
        for name in self
            .declarations()
            .filter(|e| subsequence(&parts, e.name(), leaf))
            .filter_map(|e| self.as_found(e, kinds))
        {
            if !found.contains(&name) {
                found.push(name);
            }
        }
        found
    }

    /// The name a lookup for `kinds` finds in `entry`: its own, its alias
    /// target's, or none.
    fn as_found(
        &self,
        entry: EntryRef<'_>,
        kinds: &[EntryKind],
    ) -> Option<CatalogName> {
        match entry.body() {
            CatalogEntry::Alias(alias) if kinds.contains(&alias.target.kind) => self
                .resolve_ref(entry.name(), &alias.target.path)
                .map(|target| target.name().clone()),
            _ if kinds.contains(&entry.kind()) => Some(entry.name().clone()),
            _ => None,
        }
    }

    /// The entry the full name `name` stands for in a lookup for `kinds`:
    /// the entry itself, or the target of the alias it names.
    pub fn named_as(
        &self,
        name: &CatalogName,
        kinds: &[EntryKind],
    ) -> Option<CatalogName> {
        self.declaration(name)
            .and_then(|entry| self.as_found(entry, kinds))
    }

    /// The aliases that name `name`, in declaration order.
    pub fn aliases_of(
        &self,
        name: &CatalogName,
    ) -> Vec<CatalogName> {
        self.of_kind(EntryKind::Alias)
            .into_iter()
            .filter(|alias| {
                let CatalogEntry::Alias(a) = alias.body() else {
                    return false;
                };
                self.resolve_ref(alias.name(), &a.target.path)
                    .is_some_and(|t| t.name() == name)
            })
            .map(|alias| alias.name().clone())
            .collect()
    }

    /// Resolves `query` to one entry of `kinds`.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Lookup`] when nothing matches, or more than one
    /// entry does and none is named in full.
    pub fn find(
        &self,
        query: &str,
        kinds: &[EntryKind],
        leaf: LeafMatch,
    ) -> CatalogResult<CatalogName> {
        if let Ok(full) = CatalogName::parse(query)
            && let Some(found) = self.named_as(&full, kinds)
        {
            return Ok(found);
        }
        let mut found = self.matches(query, kinds, leaf);
        match found.len() {
            1 => Ok(found.pop().expect("one")),
            0 => Err(CatalogError::new(
                CatalogErrorKind::Lookup,
                format!("nothing matches `{query}`"),
            )),
            _ => Err(CatalogError::new(
                CatalogErrorKind::Lookup,
                format!(
                    "`{query}` is ambiguous: {}",
                    found
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )),
        }
    }

    /// The build `query` names.
    ///
    /// # Errors
    /// As [`find`](Self::find).
    pub fn find_build(
        &self,
        query: &str,
    ) -> CatalogResult<CatalogName> {
        self.find(query, &[EntryKind::Build], LeafMatch::Exact)
    }

    /// The least specified name that resolves to `name` among every
    /// searchable entry, with its full name. It lengthens when a later
    /// layer adds an entry it would collide with.
    pub fn display_name(
        &self,
        name: &CatalogName,
    ) -> DisplayName {
        let segments = name.segments();
        let (leaf, scope) = segments.split_last().expect("a name has a segment");
        let mut short = name.to_string();
        // Fewest extra segments first; among equals, the ones nearer the
        // root first.
        'search: for size in 0..=scope.len() {
            for picked in choose(scope.len(), size) {
                let mut parts: Vec<&str> = picked.iter().map(|&i| scope[i].as_str()).collect();
                parts.push(leaf);
                let query = parts.join("/");
                if self.matches(&query, &SEARCHABLE, LeafMatch::Exact) == [name.clone()] {
                    short = query;
                    break 'search;
                }
            }
        }
        DisplayName {
            short,
            full: name.clone(),
        }
    }
}

/// The `size`-element subsets of `0..n`, in lexicographic order.
fn choose(
    n: usize,
    size: usize,
) -> Vec<Vec<usize>> {
    fn go(
        start: usize,
        n: usize,
        size: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == size {
            out.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            go(i + 1, n, size, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    go(0, n, size, &mut Vec::new(), &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::pretrained::build_catalog::fixtures;

    #[test]
    fn test_subsequences_match_in_order() {
        let name = CatalogName::parse("openai/whisper/large/large-v3").unwrap();
        let m = |q: &str| {
            let parts: Vec<&str> = q.split('/').collect();
            subsequence(&parts, &name, LeafMatch::Exact)
        };
        assert!(m("large-v3"));
        assert!(m("whisper/large-v3"));
        assert!(m("openai/large-v3"));
        assert!(m("openai/whisper/large/large-v3"));
        assert!(!m("whisper/openai/large-v3"));
        assert!(!m("large"));
        let parts = ["large"];
        assert!(subsequence(&parts, &name, LeafMatch::Prefix));
    }

    #[test]
    fn test_find_by_shorter_names() {
        let stack = fixtures::example_stack().unwrap();
        let v3 = CatalogName::parse("openai/whisper/large/large-v3").unwrap();
        for query in [
            "large-v3",
            "whisper/large-v3",
            "openai/large-v3",
            "openai/whisper/large/large-v3",
        ] {
            assert_eq!(stack.find_build(query).unwrap(), v3, "{query}");
        }
        let err = stack.find_build("large-v4").unwrap_err();
        assert!(err.message.contains("nothing matches"));
        let err = stack
            .find("checkpoint", &SEARCHABLE, LeafMatch::Exact)
            .unwrap_err();
        assert!(err.message.contains("ambiguous"), "{err}");
        assert_eq!(
            stack
                .find("qwen3-0.6b/checkpoint", &SEARCHABLE, LeafMatch::Exact)
                .unwrap()
                .to_string(),
            "qwen/qwen3/qwen3-0.6b/checkpoint"
        );
    }

    #[test]
    fn test_prefix_mode() {
        let stack = fixtures::example_stack().unwrap();
        let turbo = stack
            .find("large-v3-t", &[EntryKind::Build], LeafMatch::Prefix)
            .unwrap();
        assert_eq!(turbo.leaf(), "large-v3-turbo");
        let err = stack
            .find("large", &[EntryKind::Build], LeafMatch::Prefix)
            .unwrap_err();
        assert!(err.message.contains("ambiguous"));
    }

    #[test]
    fn test_template_classes_are_found_like_any_entry() {
        let stack = fixtures::example_stack().unwrap();
        let hybrid = stack
            .find("hybrid", &[EntryKind::TemplateClass], LeafMatch::Exact)
            .unwrap();
        assert_eq!(hybrid.to_string(), "qwen/qwen3/hybrid");
        assert_eq!(stack.display_name(&hybrid).short, "hybrid");
        assert_eq!(
            stack
                .find("template_class_root", &SEARCHABLE, LeafMatch::Exact)
                .unwrap()
                .to_string(),
            "template_class_root"
        );
    }

    #[test]
    fn test_display_names_lengthen_when_a_later_layer_collides() {
        let mut stack = fixtures::example_stack().unwrap();
        let vocabulary = CatalogName::parse("openai/whisper/vocabulary").unwrap();
        assert_eq!(stack.display_name(&vocabulary).short, "vocabulary");
        let tokenizer = CatalogName::parse("qwen/qwen3/qwen3-0.6b/tokenizer").unwrap();
        assert_eq!(stack.display_name(&tokenizer).short, "qwen3-0.6b/tokenizer");
        stack
            .push_yaml(
                "more",
                r#"
local:
  $kind: namespace
  $description: "Local things."
  vocabulary:
    $kind: resource
    $file: v.txt
    $format: text
    $sources: [{ $url: "https://x/v.txt" }]
"#,
            )
            .unwrap();
        let shown = stack.display_name(&vocabulary);
        assert_eq!(shown.short, "openai/vocabulary");
        assert_eq!(shown.full, vocabulary);
    }

    #[test]
    fn test_choose() {
        assert_eq!(choose(3, 0), [Vec::<usize>::new()]);
        assert_eq!(choose(3, 2), [vec![0, 1], vec![0, 2], vec![1, 2]]);
    }
}
