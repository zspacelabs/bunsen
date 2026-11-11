//! Type expressions: what a tag, an argument slot or a record field takes.

use alloc::collections::BTreeMap;
use core::fmt;

use serde::{
    Deserialize,
    Deserializer,
    Serialize,
    Serializer,
};
use serde_json::{
    Map,
    Value,
};

use crate::data::pretrained::build_catalog::name::CatalogRef;

/// A primitive type: a `$` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Primitive {
    /// `$bool`: `true` or `false`.
    Bool,
    /// `$count`: a whole number, zero or more.
    Count,
    /// `$number`: any number.
    Number,
    /// `$text`: any string.
    Text,
    /// `$languages`: a non-empty list of BCP 47 language tags, `"*"` among
    /// them for "others".
    Languages,
    /// `$object`: any JSON object, unchecked.
    Object,
}

impl Primitive {
    /// Every primitive.
    pub const ALL: [Primitive; 6] = [
        Self::Bool,
        Self::Count,
        Self::Number,
        Self::Text,
        Self::Languages,
        Self::Object,
    ];

    /// The `$` word that spells it.
    pub fn word(self) -> &'static str {
        match self {
            Self::Bool => "$bool",
            Self::Count => "$count",
            Self::Number => "$number",
            Self::Text => "$text",
            Self::Languages => "$languages",
            Self::Object => "$object",
        }
    }

    /// The primitive `word` spells.
    pub fn from_word(word: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.word() == word)
    }
}

/// A type expression: what a tag, a contract's argument slot or a record
/// field takes.
///
/// The primitives are `$` words, a closed set: `$bool`, `$count`, `$number`,
/// `$text`, `$languages` and `$object`. The structural forms are `$` maps:
/// `{$list: T}`, `{$map: T}`, `{$one_of: [..]}` and
/// `{$record: {field: {$type, $description, $optional}}}`. Anything else
/// written as a string is a reference to a `type` entry, resolved from
/// where the expression is written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeExpr {
    /// A primitive.
    Primitive(Primitive),
    /// `{$list: T}`: a list of T.
    List(Box<TypeExpr>),
    /// `{$map: T}`: an object of T, by any key.
    Map(Box<TypeExpr>),
    /// `{$one_of: [..]}`: one of these strings.
    OneOf(Vec<String>),
    /// `{$record: {..}}`: named fields.
    Record(BTreeMap<String, FieldSpec>),
    /// A `type` entry, by reference.
    Named(CatalogRef),
}

/// One field of a record type.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldSpec {
    /// What the field takes.
    #[serde(rename = "$type")]
    pub type_expr: TypeExpr,

    /// What the field is, for a person.
    #[serde(
        rename = "$description",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Option<String>,

    /// Whether a value may leave it out.
    #[serde(rename = "$optional", default, skip_serializing_if = "is_false")]
    pub optional: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// The structural `$` words.
const FORMS: [&str; 4] = ["$list", "$map", "$one_of", "$record"];

impl TypeExpr {
    /// The expression `value` writes.
    ///
    /// # Errors
    /// Why it isn't a type expression, for a person.
    pub fn from_value(value: &Value) -> Result<Self, String> {
        match value {
            Value::String(word) if word.starts_with('$') => Primitive::from_word(word)
                .map(Self::Primitive)
                .ok_or_else(|| {
                    let words: Vec<&str> = Primitive::ALL.iter().map(|p| p.word()).collect();
                    format!(
                        "`{word}` is not a type word: the primitives are {}",
                        words.join(", ")
                    )
                }),
            Value::String(name) => CatalogRef::parse(name)
                .map(Self::Named)
                .map_err(|e| format!("`{name}`: {}", e.message)),
            Value::Object(form) => {
                let mut fields = form.iter();
                let (Some((key, inner)), None) = (fields.next(), fields.next()) else {
                    return Err(format!(
                        "a type form is a map of one key, one of {}",
                        FORMS.join(", ")
                    ));
                };
                match key.as_str() {
                    "$list" => Ok(Self::List(Box::new(Self::from_value(inner)?))),
                    "$map" => Ok(Self::Map(Box::new(Self::from_value(inner)?))),
                    "$one_of" => {
                        let choices: Vec<String> = serde_json::from_value(inner.clone())
                            .map_err(|_| "`$one_of` takes a list of strings".to_string())?;
                        if choices.is_empty() {
                            return Err("`$one_of` takes at least one string".into());
                        }
                        Ok(Self::OneOf(choices))
                    }
                    "$record" => {
                        let Value::Object(spec) = inner else {
                            return Err("`$record` takes a map of fields".into());
                        };
                        let mut record = BTreeMap::new();
                        for (field, spec) in spec {
                            let spec: FieldSpec = serde_json::from_value(spec.clone())
                                .map_err(|e| format!("the field `{field}`: {e}"))?;
                            record.insert(field.clone(), spec);
                        }
                        Ok(Self::Record(record))
                    }
                    other => Err(format!(
                        "`{other}` is not a type form: the forms are {}",
                        FORMS.join(", ")
                    )),
                }
            }
            other => Err(format!(
                "{other} is not a type expression: write a `$` word, a type's name, or a form"
            )),
        }
    }

    /// The expression as it is written.
    pub fn to_value(&self) -> Value {
        let form = |key: &str, inner: Value| {
            let mut map = Map::new();
            map.insert(key.to_string(), inner);
            Value::Object(map)
        };
        match self {
            Self::Primitive(p) => Value::String(p.word().to_string()),
            Self::Named(name) => Value::String(name.to_string()),
            Self::List(inner) => form("$list", inner.to_value()),
            Self::Map(inner) => form("$map", inner.to_value()),
            Self::OneOf(choices) => form(
                "$one_of",
                Value::Array(choices.iter().cloned().map(Value::String).collect()),
            ),
            Self::Record(fields) => form(
                "$record",
                Value::Object(
                    fields
                        .iter()
                        .map(|(name, spec)| {
                            (
                                name.clone(),
                                serde_json::to_value(spec).expect("a field spec serializes"),
                            )
                        })
                        .collect(),
                ),
            ),
        }
    }
}

impl fmt::Display for TypeExpr {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}", self.to_value())
    }
}

impl Serialize for TypeExpr {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        self.to_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TypeExpr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        Self::from_value(&value).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_expressions_round_trip() {
        for written in [
            json!("$count"),
            json!("$languages"),
            json!("geometry"),
            json!({"$list": "$count"}),
            json!({"$map": "$number"}),
            json!({"$one_of": ["basic", "bottleneck"]}),
            json!({"$record": {
                "block": {"$type": {"$one_of": ["basic", "bottleneck"]}, "$description": "The block."},
                "layers": {"$type": {"$list": "$count"}},
                "width": {"$type": "$count", "$optional": true}
            }}),
        ] {
            let expr = TypeExpr::from_value(&written).unwrap();
            assert_eq!(expr.to_value(), written);
        }
    }

    #[test]
    fn test_bad_expressions_are_refused() {
        for (written, why) in [
            (json!("$counts"), "not a type word"),
            (json!({"$tuple": []}), "not a type form"),
            (
                json!({"$list": "$count", "$map": "$count"}),
                "a map of one key",
            ),
            (json!({"$one_of": []}), "at least one"),
            (
                json!({"$record": {"a": {"$typo": "$count"}}}),
                "the field `a`",
            ),
            (json!(3), "not a type expression"),
        ] {
            let err = TypeExpr::from_value(&written).unwrap_err();
            assert!(err.contains(why), "{written}: {err}");
        }
    }
}
