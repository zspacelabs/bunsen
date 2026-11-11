//! Tag values: checking one against its [`TagType`], and the values that
//! axis uses and facets imply.

use serde_json::Value;

use crate::data::pretrained::build_catalog::document::{
    AxisKind,
    TagType,
};

/// Checks `value` against `tag_type`, with a reason when it doesn't fit.
pub fn check_value(
    tag_type: &TagType,
    value: &Value,
) -> Result<(), String> {
    let ok = match tag_type {
        TagType::Bool => value.is_boolean(),
        TagType::Count => value.is_u64(),
        TagType::Text => value.is_string(),
        TagType::TextList => value
            .as_array()
            .is_some_and(|items| items.iter().all(Value::is_string)),
        TagType::Languages => value.as_array().is_some_and(|items| {
            !items.is_empty()
                && items
                    .iter()
                    .all(|item| item.as_str().is_some_and(is_language))
        }),
        TagType::Scores => value
            .as_object()
            .is_some_and(|scores| scores.values().all(Value::is_number)),
        TagType::OneOf(choices) => value
            .as_str()
            .is_some_and(|s| choices.iter().any(|c| c == s)),
    };
    if ok {
        Ok(())
    } else {
        Err(format!("`{value}` is not {}", describe(tag_type)))
    }
}

fn describe(tag_type: &TagType) -> String {
    match tag_type {
        TagType::Bool => "`true` or `false`".into(),
        TagType::Count => "a whole number".into(),
        TagType::Text => "text".into(),
        TagType::TextList => "a list of text".into(),
        TagType::Languages => "a non-empty list of language tags (or `*`)".into(),
        TagType::Scores => "an object of numbers".into(),
        TagType::OneOf(choices) => format!("one of {choices:?}"),
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
        assert!(check_value(&TagType::Bool, &json!(true)).is_ok());
        assert!(check_value(&TagType::Bool, &json!("true")).is_err());
        assert!(check_value(&TagType::Count, &json!(600000000u64)).is_ok());
        assert!(check_value(&TagType::Count, &json!(-1)).is_err());
        assert!(check_value(&TagType::Count, &json!(1.5)).is_err());
        assert!(check_value(&TagType::Languages, &json!(["en", "zh-Hans", "*"])).is_ok());
        assert!(check_value(&TagType::Languages, &json!([])).is_err());
        assert!(check_value(&TagType::Languages, &json!(["en us"])).is_err());
        assert!(check_value(&TagType::Scores, &json!({"wer.en": 4.2})).is_ok());
        assert!(check_value(&TagType::Scores, &json!({"wer.en": "4.2"})).is_err());
        let one_of = TagType::OneOf(vec!["a".into(), "b".into()]);
        assert!(check_value(&one_of, &json!("a")).is_ok());
        assert!(check_value(&one_of, &json!("c")).is_err());
        assert!(check_value(&TagType::TextList, &json!(["a", 1])).is_err());
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
