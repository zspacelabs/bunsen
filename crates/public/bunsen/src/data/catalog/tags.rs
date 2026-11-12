//! Values and their types: a type expression resolved against the stack,
//! a value checked against it, and the values that axis uses and facets
//! imply.

use alloc::collections::{
    BTreeMap,
    BTreeSet,
};

use serde_json::Value;

use crate::{
    data::catalog::{
        document::{
            AxisKind,
            CatalogEntry,
            EntryKind,
            Primitive,
            TypeExpr,
        },
        error::{
            CatalogErrorExt,
            CatalogRule,
            catalog_error,
        },
        name::CatalogName,
        stack::BunsenCatalogStack,
    },
    errors::BunsenResult,
};

/// A type expression with its references to `type` entries resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedType {
    /// A primitive.
    Primitive(Primitive),
    /// A list of the inner type.
    List(Box<ResolvedType>),
    /// An object of the inner type, by any key.
    Map(Box<ResolvedType>),
    /// One of these strings.
    OneOf(Vec<String>),
    /// Named fields: each field's type, and whether it may be left out.
    Record(BTreeMap<String, (ResolvedType, bool)>),
}

impl BunsenCatalogStack {
    /// Resolves `expr`, written in the entry `from`: each reference to a
    /// `type` entry, resolved from where it is written, is replaced by the
    /// type's own expression.
    ///
    /// # Errors
    /// [`CatalogRule::Reference`] for a reference that names no type,
    /// and a type that names itself.
    pub fn resolve_type(
        &self,
        from: &CatalogName,
        expr: &TypeExpr,
    ) -> BunsenResult<ResolvedType> {
        self.resolve_type_in(from, expr, &mut BTreeSet::new())
    }

    fn resolve_type_in(
        &self,
        from: &CatalogName,
        expr: &TypeExpr,
        open: &mut BTreeSet<CatalogName>,
    ) -> BunsenResult<ResolvedType> {
        Ok(match expr {
            TypeExpr::Primitive(p) => ResolvedType::Primitive(*p),
            TypeExpr::List(inner) => {
                ResolvedType::List(Box::new(self.resolve_type_in(from, inner, open)?))
            }
            TypeExpr::Map(inner) => {
                ResolvedType::Map(Box::new(self.resolve_type_in(from, inner, open)?))
            }
            TypeExpr::OneOf(choices) => ResolvedType::OneOf(choices.clone()),
            TypeExpr::Record(fields) => ResolvedType::Record(
                fields
                    .iter()
                    .map(|(field, spec)| {
                        self.resolve_type_in(from, &spec.type_expr, open)
                            .map(|t| (field.clone(), (t, spec.optional)))
                            .map_err(|e| {
                                catalog_error(
                                    e.rule().unwrap_or(CatalogRule::Structure),
                                    format!("the field `{field}`: {}", e.problem_message()),
                                )
                            })
                    })
                    .collect::<BunsenResult<_>>()?,
            ),
            TypeExpr::Named(reference) => {
                let decl = self.expect_ref(from, reference, "type", &[EntryKind::Type])?;
                let CatalogEntry::Type(t) = decl.body() else {
                    unreachable!("expected a type")
                };
                let name = decl.name().clone();
                if !open.insert(name.clone()) {
                    return Err(catalog_error(
                        CatalogRule::Reference,
                        format!("the type `{name}` names itself"),
                    ));
                }
                let resolved = self.resolve_type_in(&name, &t.type_expr, open)?;
                open.remove(&name);
                resolved
            }
        })
    }
}

/// Checks `value` against `ty`, with a reason when it doesn't fit.
///
/// The check is weak: a record's declared fields must be present (unless
/// optional) and fit, and fields it doesn't declare pass. The load checks
/// what the catalog doesn't.
pub fn check_value(
    ty: &ResolvedType,
    value: &Value,
) -> Result<(), String> {
    let fits = match ty {
        ResolvedType::Primitive(p) => match p {
            Primitive::Bool => value.is_boolean(),
            Primitive::Count => value.is_u64(),
            Primitive::Number => value.is_number(),
            Primitive::Text => value.is_string(),
            Primitive::Languages => value.as_array().is_some_and(|items| {
                !items.is_empty()
                    && items
                        .iter()
                        .all(|item| item.as_str().is_some_and(is_language))
            }),
            Primitive::Object => value.is_object(),
        },
        ResolvedType::List(inner) => {
            let Some(items) = value.as_array() else {
                return Err(format!("`{value}` is not {}", describe(ty)));
            };
            for (i, item) in items.iter().enumerate() {
                check_value(inner, item).map_err(|why| format!("item {i}: {why}"))?;
            }
            true
        }
        ResolvedType::Map(inner) => {
            let Some(entries) = value.as_object() else {
                return Err(format!("`{value}` is not {}", describe(ty)));
            };
            for (key, item) in entries {
                check_value(inner, item).map_err(|why| format!("`{key}`: {why}"))?;
            }
            true
        }
        ResolvedType::OneOf(choices) => value
            .as_str()
            .is_some_and(|s| choices.iter().any(|c| c == s)),
        ResolvedType::Record(fields) => {
            let Some(entries) = value.as_object() else {
                return Err(format!("`{value}` is not {}", describe(ty)));
            };
            for (field, (field_type, optional)) in fields {
                match entries.get(field) {
                    Some(item) => {
                        check_value(field_type, item).map_err(|why| format!("`{field}`: {why}"))?
                    }
                    None if *optional => {}
                    None => return Err(format!("it has no `{field}`")),
                }
            }
            true
        }
    };
    if fits {
        Ok(())
    } else {
        Err(format!("`{value}` is not {}", describe(ty)))
    }
}

fn describe(ty: &ResolvedType) -> String {
    match ty {
        ResolvedType::Primitive(p) => match p {
            Primitive::Bool => "`true` or `false`".into(),
            Primitive::Count => "a whole number".into(),
            Primitive::Number => "a number".into(),
            Primitive::Text => "text".into(),
            Primitive::Languages => "a non-empty list of language tags (or `*`)".into(),
            Primitive::Object => "an object".into(),
        },
        ResolvedType::List(_) => "a list".into(),
        ResolvedType::Map(_) | ResolvedType::Record(_) => "an object".into(),
        ResolvedType::OneOf(choices) => format!("one of {choices:?}"),
    }
}

/// Whether `s` is shaped like a BCP 47 language tag, or is `*`.
fn is_language(s: &str) -> bool {
    s == "*"
        || (!s.is_empty()
            && s.split('-').all(|part| {
                (1..=8).contains(&part.len()) && part.chars().all(|c| c.is_ascii_alphanumeric())
            }))
}

/// The tag a used axis implies on a build.
///
/// A bool axis `X` gives `always` when only `X` is offered, `never` when
/// only `no-X` is, and `switch` when both are. A categorical axis gives the
/// members offered.
pub fn axis_tag_value(
    kind: AxisKind,
    positive: &str,
    offers: &[String],
) -> Value {
    match kind {
        AxisKind::Bool => {
            let on = offers.iter().any(|o| o == positive);
            let off = offers.iter().any(|o| o != positive);
            Value::String(
                match (on, off) {
                    (true, true) => "switch",
                    (true, false) => "always",
                    _ => "never",
                }
                .into(),
            )
        }
        AxisKind::Categorical => Value::Array(offers.iter().cloned().map(Value::String).collect()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_values_checked_by_type() {
        let p = ResolvedType::Primitive;
        assert!(check_value(&p(Primitive::Bool), &json!(true)).is_ok());
        assert!(check_value(&p(Primitive::Bool), &json!("true")).is_err());
        assert!(check_value(&p(Primitive::Count), &json!(600000000u64)).is_ok());
        assert!(check_value(&p(Primitive::Count), &json!(-1)).is_err());
        assert!(check_value(&p(Primitive::Count), &json!(1.5)).is_err());
        assert!(check_value(&p(Primitive::Number), &json!(1.5)).is_ok());
        assert!(check_value(&p(Primitive::Languages), &json!(["en", "zh-Hans", "*"])).is_ok());
        assert!(check_value(&p(Primitive::Languages), &json!([])).is_err());
        assert!(check_value(&p(Primitive::Languages), &json!(["en us"])).is_err());
        let scores = ResolvedType::Map(Box::new(p(Primitive::Number)));
        assert!(check_value(&scores, &json!({"wer.en": 4.2})).is_ok());
        assert!(check_value(&scores, &json!({"wer.en": "4.2"})).is_err());
        let one_of = ResolvedType::OneOf(vec!["a".into(), "b".into()]);
        assert!(check_value(&one_of, &json!("a")).is_ok());
        assert!(check_value(&one_of, &json!("c")).is_err());
        let texts = ResolvedType::List(Box::new(p(Primitive::Text)));
        assert!(check_value(&texts, &json!(["a", 1])).is_err());
    }

    #[test]
    fn test_records_are_checked_weakly() {
        let p = ResolvedType::Primitive;
        let geometry = ResolvedType::Record(
            [
                (
                    "layers".to_string(),
                    (ResolvedType::List(Box::new(p(Primitive::Count))), false),
                ),
                ("width".to_string(), (p(Primitive::Count), true)),
            ]
            .into(),
        );
        // Extra fields pass; an optional one may be missing.
        assert!(
            check_value(
                &geometry,
                &json!({"layers": [3, 4, 6, 3], "block": "basic"})
            )
            .is_ok()
        );
        let missing = check_value(&geometry, &json!({"width": 64})).unwrap_err();
        assert!(missing.contains("it has no `layers`"), "{missing}");
        let wrong = check_value(&geometry, &json!({"layers": [3, "4"]})).unwrap_err();
        assert!(wrong.contains("`layers`: item 1"), "{wrong}");
    }

    #[test]
    fn test_axis_tags() {
        let offers = |o: &[&str]| o.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let tag = |o: &[&str]| axis_tag_value(AxisKind::Bool, "thinking", &offers(o));
        assert_eq!(tag(&["thinking", "no-thinking"]), json!("switch"));
        assert_eq!(tag(&["thinking"]), json!("always"));
        assert_eq!(tag(&["no-thinking"]), json!("never"));
        assert_eq!(
            axis_tag_value(AxisKind::Categorical, "task", &offers(&["transcribe"])),
            json!(["transcribe"])
        );
    }
}
