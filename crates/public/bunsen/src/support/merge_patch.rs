//! # JSON merge patch
//!
//! [RFC 7386] merge patches over [`serde_json::Value`]: [`apply`] one, list
//! the paths a patch sets ([`touched_paths`]), check that a patch names only
//! fields its base has ([`validate_paths`]), and find where several patches
//! set the same thing ([`conflicts`]).
//!
//! A merge patch is a JSON document shaped like its target. An object member
//! merges into the target's member of the same name, recursively; `null`
//! removes the member; any other value, an array included, replaces it:
//!
//! ```
//! use bunsen::support::merge_patch::apply;
//! use serde_json::json;
//!
//! let mut config = json!({"beam": 5, "sampling": {"temperature": 0.7, "top_p": 0.8}});
//! apply(&mut config, &json!({"beam": null, "sampling": {"top_p": 0.95}}));
//! assert_eq!(config, json!({"sampling": {"temperature": 0.7, "top_p": 0.95}}));
//! ```
//!
//! The operations know nothing of what the documents mean. A typed config
//! is patched by serializing it, applying, and deserializing the result;
//! [`validate_paths`] first catches a misspelled field, which a deserializer
//! that ignores unknown fields would drop silently.
//!
//! [RFC 7386]: https://www.rfc-editor.org/rfc/rfc7386

use core::fmt;

use serde_json::{
    Map,
    Value,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
    LookupError,
};

/// A path into a JSON document: the object member names from the root down.
///
/// `Display` joins the names with `.`; the root, the empty path, prints as
/// `(root)`.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PatchPath(pub Vec<String>);

impl PatchPath {
    /// The member names, from the root down.
    pub fn names(&self) -> &[String] {
        &self.0
    }

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

impl<S: Into<String>> FromIterator<S> for PatchPath {
    fn from_iter<I: IntoIterator<Item = S>>(names: I) -> Self {
        PatchPath(names.into_iter().map(Into::into).collect())
    }
}

/// Applies the merge patch `patch` to `target`, in place: RFC 7386's
/// `MergePatch`.
///
/// When `patch` is an object, `target` becomes an object if it is not one,
/// and each of `patch`'s members is merged into it: a `null` member removes
/// the target's member of that name, and any other member is merged into it
/// recursively. When `patch` is anything else, it replaces `target`.
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

/// The paths `patch` sets, sorted: its leaves.
///
/// A leaf is a member whose value is not a non-empty object: a scalar, an
/// array, `null` (a removal), or `{}` (which makes the member an object).
/// A patch that is not an object sets the whole document, the root path.
pub fn touched_paths(patch: &Value) -> Vec<PatchPath> {
    let mut paths = Vec::new();
    match patch {
        Value::Object(members) => collect_leaves(&PatchPath::default(), members, &mut paths),
        _ => paths.push(PatchPath::default()),
    }
    paths.sort();
    paths
}

fn collect_leaves(
    at: &PatchPath,
    members: &Map<String, Value>,
    paths: &mut Vec<PatchPath>,
) {
    for (name, value) in members {
        let path = at.child(name);
        match value {
            Value::Object(inner) if !inner.is_empty() => collect_leaves(&path, inner, paths),
            _ => paths.push(path),
        }
    }
}

/// Checks that every member `patch` names exists in `base`, at the same
/// path.
///
/// The check descends while both documents hold objects. Where `base` holds
/// something else at a path (`null`, as an unset optional field serializes,
/// or a scalar), the patch's value there replaces it whole, and is not
/// checked further: whether it has the right shape is for whatever reads the
/// result. A patch that is not an object replaces the whole document, and
/// passes.
///
/// # Errors
///
/// [`Lookup`](crate::errors::BunsenErrorKind::Lookup), with a
/// [`LookupError`] cause in the `"field"` table, for the first member
/// `patch` names that `base` lacks: its key is the member's path, and its
/// candidates are the names `base` has beside it.
pub fn validate_paths(
    base: &Value,
    patch: &Value,
) -> BunsenResult<()> {
    validate_paths_open(base, patch, &[])
}

/// [`validate_paths`], with `open` paths whose members are not checked: the
/// paths of maps, which take any key. [`empty_objects`] finds a typed
/// config's maps in its serialization, where they are the empty objects.
///
/// # Errors
/// As [`validate_paths`], for a member outside `open`.
pub fn validate_paths_open(
    base: &Value,
    patch: &Value,
    open: &[PatchPath],
) -> BunsenResult<()> {
    validate_at(&PatchPath::default(), base, patch, open)
}

/// The paths of the empty objects in `value`, sorted: in a serialized typed
/// config, its empty maps (a struct serializes with its fields).
pub fn empty_objects(value: &Value) -> Vec<PatchPath> {
    fn walk(
        at: &PatchPath,
        value: &Value,
        out: &mut Vec<PatchPath>,
    ) {
        if let Value::Object(members) = value {
            if members.is_empty() && !at.0.is_empty() {
                out.push(at.clone());
            }
            for (name, v) in members {
                walk(&at.child(name), v, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(&PatchPath::default(), value, &mut out);
    out.sort();
    out
}

fn validate_at(
    at: &PatchPath,
    base: &Value,
    patch: &Value,
    open: &[PatchPath],
) -> BunsenResult<()> {
    if open.contains(at) {
        return Ok(());
    }
    let (Value::Object(fields), Value::Object(members)) = (base, patch) else {
        return Ok(());
    };
    for (name, value) in members {
        let path = at.child(name);
        match fields.get(name) {
            Some(field) => validate_at(&path, field, value, open)?,
            None if open.contains(&path) => {}
            None => {
                return Err(BunsenError::lookup(
                    LookupError::missing("field", path.to_string())
                        .with_candidates(fields.keys().cloned()),
                ));
            }
        }
    }
    Ok(())
}

/// A path that more than one patch sets, from [`conflicts`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchConflict {
    /// The path: the shallowest of the overlapping paths the patches set.
    pub path: PatchPath,

    /// The indices of the patches that set it or a path under it, ascending.
    pub patches: Vec<usize>,
}

/// The paths that two or more of `patches` set, sorted by path.
///
/// Two patches conflict where one sets a path that the other sets too, or
/// sets a path under it: applied in either order, they could leave different
/// results there. Each conflict is reported once, at the shallowest path
/// involved, with every patch that reaches it. Patches that touch disjoint
/// paths have no conflicts, and apply in any order to the same result.
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
                Some(conflict) => {
                    conflict.patches.extend([*index, *other_index]);
                }
                None => found.push(PatchConflict {
                    path: path.clone(),
                    patches: vec![*index, *other_index],
                }),
            }
        }
    }

    // A conflict under another conflict's path is reported by that one: its
    // patches all reach under the shallow path, so they are listed there.
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
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    fn applied(
        target: Value,
        patch: Value,
    ) -> Value {
        let mut target = target;
        apply(&mut target, &patch);
        target
    }

    fn path(names: &[&str]) -> PatchPath {
        names.iter().copied().collect()
    }

    /// RFC 7386, Appendix A: every example, as target, patch and result.
    #[test]
    fn test_apply_rfc_7386_appendix_a() {
        let cases = [
            (json!({"a": "b"}), json!({"a": "c"}), json!({"a": "c"})),
            (
                json!({"a": "b"}),
                json!({"b": "c"}),
                json!({"a": "b", "b": "c"}),
            ),
            (json!({"a": "b"}), json!({"a": null}), json!({})),
            (
                json!({"a": "b", "b": "c"}),
                json!({"a": null}),
                json!({"b": "c"}),
            ),
            (json!({"a": ["b"]}), json!({"a": "c"}), json!({"a": "c"})),
            (json!({"a": "c"}), json!({"a": ["b"]}), json!({"a": ["b"]})),
            (
                json!({"a": {"b": "c"}}),
                json!({"a": {"b": "d", "c": null}}),
                json!({"a": {"b": "d"}}),
            ),
            (
                json!({"a": [{"b": "c"}]}),
                json!({"a": [1]}),
                json!({"a": [1]}),
            ),
            (json!(["a", "b"]), json!(["c", "d"]), json!(["c", "d"])),
            (json!({"a": "b"}), json!(["c"]), json!(["c"])),
            (json!({"a": "foo"}), json!(null), json!(null)),
            (json!({"a": "foo"}), json!("bar"), json!("bar")),
            (
                json!({"e": null}),
                json!({"a": 1}),
                json!({"e": null, "a": 1}),
            ),
            (
                json!([1, 2]),
                json!({"a": "b", "c": null}),
                json!({"a": "b"}),
            ),
            (
                json!({}),
                json!({"a": {"bb": {"ccc": null}}}),
                json!({"a": {"bb": {}}}),
            ),
        ];
        for (target, patch, expected) in cases {
            assert_eq!(
                applied(target.clone(), patch.clone()),
                expected,
                "{target} + {patch}"
            );
        }
    }

    /// RFC 7386, section 3: the worked example.
    #[test]
    fn test_apply_rfc_7386_section_3() {
        let target = json!({
            "title": "Goodbye!",
            "author": {"givenName": "John", "familyName": "Doe"},
            "tags": ["example", "sample"],
            "content": "This will be unchanged"
        });
        let patch = json!({
            "title": "Hello!",
            "phoneNumber": "+01-123-456-7890",
            "author": {"familyName": null},
            "tags": ["example"]
        });
        assert_eq!(
            applied(target, patch),
            json!({
                "title": "Hello!",
                "author": {"givenName": "John"},
                "tags": ["example"],
                "content": "This will be unchanged",
                "phoneNumber": "+01-123-456-7890"
            })
        );
    }

    #[test]
    fn test_patch_path_display() {
        assert_eq!(PatchPath::default().to_string(), "(root)");
        assert_eq!(path(&["sampling", "top_p"]).to_string(), "sampling.top_p");
        assert!(path(&["a"]).is_prefix_of(&path(&["a", "b"])));
        assert!(path(&["a"]).is_prefix_of(&path(&["a"])));
        assert!(!path(&["a", "b"]).is_prefix_of(&path(&["a"])));
        assert!(PatchPath::default().is_prefix_of(&path(&["a"])));
    }

    #[test]
    fn test_touched_paths() {
        assert_eq!(
            touched_paths(&json!({
                "beam": 1,
                "sampling": {"top_p": 0.9, "top_k": null},
                "stop": ["a"],
                "vad": {}
            })),
            vec![
                path(&["beam"]),
                path(&["sampling", "top_k"]),
                path(&["sampling", "top_p"]),
                path(&["stop"]),
                path(&["vad"]),
            ]
        );
        assert_eq!(touched_paths(&json!({})), Vec::<PatchPath>::new());
        assert_eq!(touched_paths(&json!(3)), vec![PatchPath::default()]);
    }

    #[test]
    fn test_validate_paths_accepts_known_fields() {
        let base = json!({"beam": 5, "sampling": {"temperature": 0.7}, "yarn": null});
        validate_paths(&base, &json!({})).unwrap();
        validate_paths(&base, &json!({"beam": null})).unwrap();
        validate_paths(&base, &json!({"sampling": {"temperature": 1.0}})).unwrap();
        // Under a non-object base value, the patch replaces it whole.
        validate_paths(&base, &json!({"yarn": {"factor": 4.0}})).unwrap();
        validate_paths(&base, &json!({"beam": {"width": 2}})).unwrap();
        validate_paths(&base, &json!("replace everything")).unwrap();
    }

    #[test]
    fn test_validate_paths_rejects_unknown_fields() {
        let base = json!({"sampling": {"temperature": 0.7, "top_p": 0.8}});
        let result = validate_paths(&base, &json!({"sampling": {"tempurature": 1.0}}));
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_eq("no field \"sampling.tempurature\"; there are: temperature, top_p")
            .has_cause::<LookupError>()
            .assert_err(&result);

        let result = validate_paths(&base, &json!({"beam": 1}));
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_eq("no field \"beam\"; there are: sampling")
            .assert_err(&result);
    }

    #[test]
    fn test_conflicts_disjoint_patches_have_none() {
        let a = json!({"sampling": {"temperature": 0.6}});
        let b = json!({"sampling": {"top_p": 0.95}, "beam": 1});
        assert_eq!(conflicts(&[&a, &b]), vec![]);
        assert_eq!(conflicts(&[&a]), vec![]);
        assert_eq!(conflicts(&[]), vec![]);
    }

    #[test]
    fn test_conflicts_same_path() {
        let a = json!({"sampling": {"temperature": 0.6}});
        let b = json!({"sampling": {"temperature": 0.7}});
        let c = json!({"beam": 2});
        assert_eq!(
            conflicts(&[&a, &c, &b]),
            vec![PatchConflict {
                path: path(&["sampling", "temperature"]),
                patches: vec![0, 2],
            }]
        );
    }

    #[test]
    fn test_conflicts_report_the_shallowest_path() {
        // One patch replaces `sampling` whole; two others set fields under
        // it, and also collide with each other there.
        let whole = json!({"sampling": null});
        let a = json!({"sampling": {"temperature": 0.6}});
        let b = json!({"sampling": {"temperature": 0.7, "top_p": 0.9}});
        assert_eq!(
            conflicts(&[&a, &whole, &b]),
            vec![PatchConflict {
                path: path(&["sampling"]),
                patches: vec![0, 1, 2],
            }]
        );
    }

    #[test]
    fn test_conflicts_sorted_by_path() {
        let a = json!({"z": 1, "a": 1});
        let b = json!({"z": 2, "a": 2});
        let found = conflicts(&[&a, &b]);
        assert_eq!(
            found.iter().map(|c| c.path.to_string()).collect::<Vec<_>>(),
            vec!["a", "z"]
        );
    }

    /// Patches with no conflicts commute: either order, the same result.
    #[test]
    fn test_disjoint_patches_commute() {
        let base = json!({"beam": 5, "sampling": {"temperature": 0.7, "top_p": 0.8}});
        let a = json!({"sampling": {"temperature": 0.6}});
        let b = json!({"sampling": {"top_p": null}, "beam": 1});
        assert!(conflicts(&[&a, &b]).is_empty());

        let mut ab = base.clone();
        apply(&mut ab, &a);
        apply(&mut ab, &b);
        let mut ba = base;
        apply(&mut ba, &b);
        apply(&mut ba, &a);
        assert_eq!(ab, ba);
        assert_eq!(ab, json!({"beam": 1, "sampling": {"temperature": 0.6}}));
    }

    /// Under an open path, any member passes; elsewhere the check holds.
    #[test]
    fn test_validate_paths_open() {
        let base = json!({"generation": {"kwargs": {}, "stop": [1]}, "beam": 1});
        let open = empty_objects(&base);
        assert_eq!(open, vec![path(&["generation", "kwargs"])]);
        validate_paths_open(
            &base,
            &json!({"generation": {"kwargs": {"any": true}}}),
            &open,
        )
        .unwrap();
        // Once the map holds a key, another may still join it.
        let held = json!({"generation": {"kwargs": {"any": true}, "stop": [1]}, "beam": 1});
        validate_paths_open(
            &held,
            &json!({"generation": {"kwargs": {"other": 1}}}),
            &open,
        )
        .unwrap();
        assert!(validate_paths_open(&base, &json!({"generation": {"stopp": [2]}}), &open).is_err());
        assert!(validate_paths(&base, &json!({"generation": {"kwargs": {"any": true}}})).is_err());
        assert_eq!(empty_objects(&json!({})), Vec::<PatchPath>::new());
    }
}
