use serde_json::Value;

pub(super) fn is_false(b: &bool) -> bool {
    !*b
}

pub(super) fn empty_object() -> Value {
    Value::Object(serde_json::Map::new())
}

pub(super) fn is_empty_object(value: &Value) -> bool {
    value.as_object().is_some_and(serde_json::Map::is_empty)
}
