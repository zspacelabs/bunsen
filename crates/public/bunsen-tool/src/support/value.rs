use serde::{
    Serialize,
    Serializer,
    ser::{
        SerializeMap,
        SerializeSeq,
    },
};

/// A document: a tree of strings, lists and maps whose keys keep the order
/// they were added in.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A string.
    Str(String),

    /// A list.
    List(Vec<Value>),

    /// A map, in the order its keys were added.
    Map(Vec<(String, Value)>),
}

impl Value {
    /// A map of `entries`, in their order.
    pub fn map<K: Into<String>>(entries: impl IntoIterator<Item = (K, Value)>) -> Value {
        Value::Map(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// A string.
    pub fn str(s: impl Into<String>) -> Value {
        Value::Str(s.into())
    }

    /// The value under `key`, in a map.
    pub fn get(
        &self,
        key: &str,
    ) -> Option<&Value> {
        match self {
            Value::Map(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

impl Serialize for Value {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match self {
            Value::Str(s) => serializer.serialize_str(s),
            Value::List(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Value::Map(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (k, v) in entries {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}
