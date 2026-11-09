//! # Tag rules: what `cargo validate catalog` holds every catalog's tags to
//!
//! bunsen holds the tag vocabulary as data (`data::pretrained::tags`); the
//! rules are enforced here, when a catalog is built, and nowhere at run time.
//! A build fails on a break:
//!
//! - a key that is neither a shared key nor one of the kit's own
//!   (`{kit}.name`);
//! - a value not of its key's shape: a `language` tag's tags not canonical BCP
//!   47, a `version`'s version not semver;
//! - a row writing a key the catalog writes (`kit`);
//! - a row missing a key its kit requires.
//!
//! The build writes each row's `dtype` tag from its resources' dtypes, which
//! must agree, as must a `dtype` the row wrote by hand.

use std::collections::BTreeSet;

use bunsen::data::pretrained::{
    PretrainedCatalog,
    Tags,
    tags::{
        DTYPE,
        KitTags,
        TagAuthor,
        TagDef,
        TagShape,
        shared_tag,
    },
};
use icu_locale::{
    Locale,
    LocaleCanonicalizer,
};
use serde_json::Value;

/// Each kit's tag rules, by the kit's name: its own keys, and the keys every
/// row carries. A kit not listed has none.
const KIT_TAGS: &[(&str, KitTags)] = &[];

/// The tag rules of `kit`.
pub fn kit_tags(kit: &str) -> KitTags {
    KIT_TAGS
        .iter()
        .find(|(name, _)| *name == kit)
        .map_or(KitTags::NONE, |(_, rules)| *rules)
}

/// A resource's dtype as its publisher names it (`bfloat16`), in the
/// vocabulary's spelling (`bf16`).
pub fn short_dtype(publisher: &str) -> Option<&'static str> {
    match publisher {
        "float32" | "f32" => Some("f32"),
        "float16" | "f16" => Some("f16"),
        "bfloat16" | "bf16" => Some("bf16"),
        _ => None,
    }
}

/// The definition of `key` in `kit`'s vocabulary: a shared key, or one of the
/// kit's own.
fn definition(
    kit: &str,
    rules: &KitTags,
    key: &str,
) -> Result<&'static TagDef, String> {
    match key.split_once('.') {
        Some((prefix, _)) if prefix != kit => Err(format!(
            "the tag {key:?} is prefixed with another kit's name than {kit:?}"
        )),
        Some(_) => rules
            .keys
            .iter()
            .find(|def| def.key == key)
            .ok_or_else(|| format!("the tag {key:?} is not one of the {kit} kit's own")),
        None => shared_tag(key).ok_or_else(|| {
            let known: Vec<&str> = bunsen::data::pretrained::tags::VOCABULARY
                .iter()
                .map(|d| d.key)
                .collect();
            format!(
                "the tag {key:?} is not in the shared vocabulary ({})",
                known.join(", ")
            )
        }),
    }
}

/// `tag`, canonicalized as BCP 47 (`EN-us` -> `en-US`, `eng` -> `en`).
fn canonical_language(tag: &str) -> Result<String, String> {
    let mut locale =
        Locale::try_from_str(tag).map_err(|e| format!("{tag:?} is not a BCP 47 tag: {e:?}"))?;
    LocaleCanonicalizer::new_extended().canonicalize(&mut locale);
    Ok(locale.to_string())
}

/// Checks `value` has `def`'s shape.
fn check_value(
    def: &TagDef,
    value: &Value,
) -> Result<(), String> {
    let key = def.key;
    let fail = |what: &str| Err(format!("the tag {key:?}: {value} is not {what}"));
    match def.shape {
        TagShape::Bool => match value {
            Value::Bool(_) => Ok(()),
            _ => fail("true or false"),
        },
        TagShape::Count => match value.as_u64() {
            Some(_) => Ok(()),
            None => fail("a whole number"),
        },
        TagShape::Text => match value {
            Value::String(s) if !s.is_empty() => Ok(()),
            _ => fail("text"),
        },
        TagShape::OneOf(members) => match value.as_str() {
            Some(s) if members.contains(&s) => Ok(()),
            _ => fail(&format!("one of {}", members.join(", "))),
        },
        TagShape::TextList => match value {
            Value::Array(items)
                if items
                    .iter()
                    .all(|v| v.as_str().is_some_and(|s| !s.is_empty())) =>
            {
                Ok(())
            }
            _ => fail("a list of names"),
        },
        TagShape::Languages => {
            let Some(items) = value.as_array().filter(|items| !items.is_empty()) else {
                return fail("a list of BCP 47 tags");
            };
            let mut seen = BTreeSet::new();
            for item in items {
                let Some(tag) = item.as_str() else {
                    return fail("a list of BCP 47 tags");
                };
                if !seen.insert(tag) {
                    return Err(format!("the tag {key:?}: {tag:?} twice"));
                }
                if tag == "*" {
                    continue;
                }
                let canonical =
                    canonical_language(tag).map_err(|e| format!("the tag {key:?}: {e}"))?;
                if canonical != tag {
                    return Err(format!(
                        "the tag {key:?}: {tag:?} is not canonical; write {canonical:?}"
                    ));
                }
            }
            Ok(())
        }
        TagShape::Version => {
            let Some(fields) = value.as_object() else {
                return fail("{group, version}");
            };
            if fields.len() != 2 {
                return fail("{group, version}, and nothing else");
            }
            match (
                fields.get("group").and_then(Value::as_str),
                fields.get("version").and_then(Value::as_str),
            ) {
                (Some(group), Some(version)) if !group.is_empty() => {
                    semver::Version::parse(version).map(|_| ()).map_err(|e| {
                        format!("the tag {key:?}: {version:?} is not a semver version: {e}")
                    })
                }
                _ => fail("{group, version}"),
            }
        }
        TagShape::Scores => match value.as_object() {
            Some(scores) if scores.values().all(Value::is_number) => Ok(()),
            _ => fail("an object of numbers"),
        },
    }
}

/// Checks tags written by hand: every key known, no key the catalog writes,
/// every value of its key's shape.
fn check_authored(
    kit: &str,
    rules: &KitTags,
    what: &str,
    tags: &Tags,
) -> Result<(), String> {
    for (key, value) in tags.iter() {
        let def = definition(kit, rules, key).map_err(|e| format!("{what}: {e}"))?;
        if def.author == TagAuthor::Catalog {
            return Err(format!(
                "{what}: the tag {key:?} is written by the catalog, never by hand"
            ));
        }
        check_value(def, value).map_err(|e| format!("{what}: {e}"))?;
    }
    Ok(())
}

/// Checks every hand-written tag of `catalog`: its families', groups' and
/// rows'.
pub fn check_sources(catalog: &PretrainedCatalog) -> Result<(), String> {
    let kit = catalog.kit.as_str();
    let rules = kit_tags(kit);
    for family in &catalog.families {
        check_authored(
            kit,
            &rules,
            &format!("{kit}: the family {:?}", family.name),
            &family.tags,
        )?;
    }
    for group in &catalog.groups {
        check_authored(
            kit,
            &rules,
            &format!("{kit}: the group {:?}", group.name),
            &group.tags,
        )?;
        for row in &group.rows {
            check_authored(
                kit,
                &rules,
                &format!("{kit}: the row {}/{}", group.name, row.name),
                &row.tags,
            )?;
        }
    }
    Ok(())
}

/// Writes each row's `dtype` tag from its resources' dtypes: they must agree,
/// and a `dtype` the row wrote by hand must agree with them. A row whose
/// resources carry no dtype keeps what it wrote.
pub fn write_dtypes(catalog: &mut PretrainedCatalog) -> Result<(), String> {
    let kit = catalog.kit.clone();
    let mut written = Vec::new();
    for (g, group) in catalog.groups.iter().enumerate() {
        for (r, row) in group.rows.iter().enumerate() {
            let what = format!("{kit}: the row {}/{}", group.name, row.name);
            let mut dtypes = BTreeSet::new();
            for name in &row.bundles {
                let Some(bundle) = catalog.bundle(name) else {
                    continue;
                };
                for resource in &bundle.resources {
                    let Some(publisher) = &resource.dtype else {
                        continue;
                    };
                    let short = short_dtype(publisher).ok_or_else(|| {
                        format!(
                            "{what}: {}'s dtype {publisher:?} has no tag spelling",
                            resource.file
                        )
                    })?;
                    dtypes.insert(short);
                }
            }
            let derived = match dtypes.len() {
                0 => continue,
                1 => *dtypes.iter().next().unwrap(),
                _ => {
                    return Err(format!(
                        "{what}: its resources disagree on the dtype: {dtypes:?}"
                    ));
                }
            };
            if let Some(authored) = row.tags.text(DTYPE)
                && authored != derived
            {
                return Err(format!(
                    "{what}: its dtype tag {authored:?} contradicts its resources' {derived:?}"
                ));
            }
            written.push((g, r, derived));
        }
    }
    for (g, r, dtype) in written {
        catalog.groups[g].rows[r].tags.insert(DTYPE, dtype);
    }
    Ok(())
}

/// Checks every row carries the keys its kit requires, its tags laid as the
/// table lays them.
pub fn check_required(catalog: &PretrainedCatalog) -> Result<(), String> {
    let rules = kit_tags(&catalog.kit);
    for group in &catalog.groups {
        for row in &group.rows {
            let tags = catalog
                .row_tags(group, row)
                .map_err(|e| format!("{}: {e}", catalog.kit))?;
            for key in rules.required {
                if !tags.contains_key(*key) {
                    return Err(format!(
                        "{}: the row {}/{} lacks the tag {key:?}, which every {} row carries",
                        catalog.kit, group.name, row.name, catalog.kit
                    ));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use bunsen::data::pretrained::tags::{
        LANGUAGE,
        THINKING,
        VERSION,
        VOCABULARY,
    };
    use serde_json::json;

    use super::*;

    fn tags(value: Value) -> Tags {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn test_shapes() {
        let ok = |def: &TagDef, v: Value| check_value(def, &v).unwrap();
        let bad = |def: &TagDef, v: Value| assert!(check_value(def, &v).is_err(), "{v}");
        ok(&THINKING, json!("switch"));
        bad(&THINKING, json!("sometimes"));
        bad(&THINKING, json!(true));
        ok(&LANGUAGE, json!(["en"]));
        ok(&LANGUAGE, json!(["en", "zh", "*"]));
        ok(&LANGUAGE, json!(["*"]));
        ok(&LANGUAGE, json!(["zh-TW", "yue"]));
        bad(&LANGUAGE, json!([]));
        bad(&LANGUAGE, json!("en"));
        bad(&LANGUAGE, json!(["en", "en"]));
        bad(&LANGUAGE, json!(["not a tag"]));
        ok(
            &VERSION,
            json!({"group": "whisper-large", "version": "3.0.0"}),
        );
        ok(&VERSION, json!({"group": "qwen3", "version": "3.0.0+2507"}));
        bad(&VERSION, json!({"group": "qwen3", "version": "3"}));
        bad(&VERSION, json!({"group": "qwen3"}));
        bad(&VERSION, json!({"group": "", "version": "1.0.0"}));
        bad(
            &VERSION,
            json!({"group": "g", "version": "1.0.0", "extra": 1}),
        );
        let parameters = shared_tag("parameters").unwrap();
        ok(parameters, json!(600000000u64));
        bad(parameters, json!(-1));
        bad(parameters, json!(1.5));
        let scores = shared_tag("scores").unwrap();
        ok(scores, json!({"wer": 1.9, "rtf": 0}));
        bad(scores, json!({"wer": "low"}));
        assert!(VOCABULARY.len() >= 12);
    }

    /// A language tag must be written canonically: the error names the
    /// canonical spelling.
    #[test]
    fn test_canonical_languages() {
        assert_eq!(canonical_language("en").unwrap(), "en");
        assert_eq!(canonical_language("EN-us").unwrap(), "en-US");
        assert_eq!(canonical_language("eng").unwrap(), "en");
        let e = check_value(&LANGUAGE, &json!(["eng"])).unwrap_err();
        assert!(e.contains("write \"en\""), "{e}");
    }

    fn catalog(row_tags: Value) -> PretrainedCatalog {
        PretrainedCatalog::from_json(
            &json!({
                "kit": "toy", "name": "well-known", "description": "toys",
                "groups": [{"name": "a", "description": "a", "rows": [
                    {"name": "m", "description": "m", "bundles": ["b"], "tags": row_tags}
                ]}],
                "bundles": [{"name": "b", "description": "b", "namespace": "a", "bases": [],
                             "resources": [{"key": "checkpoint", "file": "m.safetensors",
                                            "sha256": "00", "dtype": "bfloat16"}]}]
            })
            .to_string(),
        )
        .unwrap()
    }

    #[test]
    fn test_sources() {
        check_sources(&catalog(json!({"thinking": "never", "toy.anything": 1}))).unwrap_err();
        check_sources(&catalog(json!({"thinking": "never"}))).unwrap();
        let e = check_sources(&catalog(json!({"kit": "toy"}))).unwrap_err();
        assert!(e.contains("written by the catalog"), "{e}");
        let e = check_sources(&catalog(json!({"size": 7}))).unwrap_err();
        assert!(e.contains("not in the shared vocabulary"), "{e}");
        let e = check_sources(&catalog(json!({"other.key": 7}))).unwrap_err();
        assert!(e.contains("another kit"), "{e}");
    }

    /// The catalog writes a row's dtype from its resources, in the
    /// vocabulary's spelling; a hand-written one must agree.
    #[test]
    fn test_write_dtypes() {
        let mut built = catalog(json!({}));
        write_dtypes(&mut built).unwrap();
        assert_eq!(built.groups[0].rows[0].tags, tags(json!({"dtype": "bf16"})));

        let mut agreeing = catalog(json!({"dtype": "bf16"}));
        write_dtypes(&mut agreeing).unwrap();
        let mut contradicting = catalog(json!({"dtype": "f32"}));
        let e = write_dtypes(&mut contradicting).unwrap_err();
        assert!(e.contains("contradicts"), "{e}");
    }

    #[test]
    fn test_required() {
        check_required(&catalog(json!({}))).unwrap();
        assert_eq!(kit_tags("toy"), KitTags::NONE);
    }
}
