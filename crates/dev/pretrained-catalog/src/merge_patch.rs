//! JSON merge patches (RFC 7396), and the paths a patch touches.
//!
//! The semantics match `bunsen::support::merge_patch`: a `null` member
//! removes a field, an object merges member by member, and anything else
//! replaces. Two patches conflict when one touches a path at or under a
//! path the other touches. A nested empty object is a leaf, so it touches
//! its own path; a root `{}` touches nothing.

use std::fmt;

use serde_json::{
    Map,
    Value,
};

/// A path into a JSON document: object member names, root first.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PatchPath(pub Vec<String>);

impl PatchPath {
    /// Whether `self` is `other` or one of its ancestors.
    pub fn is_prefix_of(
        &self,
        other: &PatchPath,
    ) -> bool {
        other.0.starts_with(&self.0)
    }

    fn child(
        &self,
        name: &str,
    ) -> PatchPath {
        let mut names = self.0.clone();
        names.push(name.to_string());
        PatchPath(names)
    }
}

impl fmt::Display for PatchPath {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        if self.0.is_empty() {
            f.write_str("(root)")
        } else {
            f.write_str(&self.0.join("."))
        }
    }
}

/// Applies `patch` to `target` in place.
pub fn apply(
    target: &mut Value,
    patch: &Value,
) {
    let Value::Object(members) = patch else {
        *target = patch.clone();
        return;
    };
    if !target.is_object() {
        *target = Value::Object(Map::new());
    }
    let Value::Object(fields) = target else {
        unreachable!("made an object above");
    };
    for (name, value) in members {
        if value.is_null() {
            fields.remove(name);
        } else {
            apply(fields.entry(name.clone()).or_insert(Value::Null), value);
        }
    }
}

/// The leaf paths `patch` sets or removes, sorted.
pub fn touched_paths(patch: &Value) -> Vec<PatchPath> {
    fn collect(
        at: &PatchPath,
        members: &Map<String, Value>,
        paths: &mut Vec<PatchPath>,
    ) {
        for (name, value) in members {
            let path = at.child(name);
            match value {
                Value::Object(inner) if !inner.is_empty() => collect(&path, inner, paths),
                _ => paths.push(path),
            }
        }
    }
    let mut paths = Vec::new();
    match patch {
        Value::Object(members) => collect(&PatchPath::default(), members, &mut paths),
        _ => paths.push(PatchPath::default()),
    }
    paths.sort();
    paths
}

/// Two or more patches touching overlapping paths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchConflict {
    /// The shallowest path they overlap at.
    pub path: PatchPath,
    /// Indexes into the patches given, sorted.
    pub patches: Vec<usize>,
}

/// The conflicts among `patches`, by path, reporting each at its
/// shallowest path.
pub fn conflicts(patches: &[&Value]) -> Vec<PatchConflict> {
    let touched: Vec<(PatchPath, usize)> = patches
        .iter()
        .enumerate()
        .flat_map(|(index, patch)| {
            touched_paths(patch)
                .into_iter()
                .map(move |path| (path, index))
        })
        .collect();
    let mut found: Vec<PatchConflict> = Vec::new();
    for (path, index) in &touched {
        for (other, other_index) in &touched {
            if index == other_index || !path.is_prefix_of(other) {
                continue;
            }
            match found.iter_mut().find(|c| &c.path == path) {
                Some(conflict) => conflict.patches.extend([*index, *other_index]),
                None => found.push(PatchConflict {
                    path: path.clone(),
                    patches: vec![*index, *other_index],
                }),
            }
        }
    }
    let shallow: Vec<PatchPath> = found.iter().map(|c| c.path.clone()).collect();
    found.retain(|c| {
        !shallow
            .iter()
            .any(|s| s != &c.path && s.is_prefix_of(&c.path))
    });
    for conflict in &mut found {
        conflict.patches.sort_unstable();
        conflict.patches.dedup();
    }
    found.sort_by(|a, b| a.path.cmp(&b.path));
    found
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_apply_merges_replaces_and_removes() {
        let mut doc = json!({"a": {"b": 1, "c": 2}, "d": 3});
        apply(&mut doc, &json!({"a": {"b": 10, "c": null}, "e": [1]}));
        assert_eq!(doc, json!({"a": {"b": 10}, "d": 3, "e": [1]}));
        apply(&mut doc, &json!(5));
        assert_eq!(doc, json!(5));
    }

    #[test]
    fn test_touched_paths_count_leaves() {
        let paths = touched_paths(&json!({"g": {"t": 1, "p": 2}, "x": {}}));
        let shown: Vec<String> = paths.iter().map(ToString::to_string).collect();
        assert_eq!(shown, ["g.p", "g.t", "x"]);
        assert!(touched_paths(&json!({})).is_empty());
    }

    #[test]
    fn test_conflicts_report_the_shallowest_path() {
        let a = json!({"g": {"t": 1}});
        let b = json!({"g": {"p": 1}});
        let c = json!({"g": {}});
        assert!(conflicts(&[&a, &b]).is_empty());
        let found = conflicts(&[&a, &b, &c]);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].path.to_string(), "g");
        assert_eq!(found[0].patches, [0, 1, 2]);
        assert!(conflicts(&[&json!({}), &a]).is_empty());
    }
}
