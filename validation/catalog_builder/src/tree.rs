//! Reading and editing catalog trees ([`OrderedNode`]s) in place: a map's
//! keys keep their order, and a key set again stays where it was.

use bunsen::data::catalog::document::OrderedNode;
use serde_json::Value;

/// The entry's `$kind`, when `node` is an entry.
pub fn kind_of(node: &OrderedNode) -> Option<&str> {
    match get(node, "$kind") {
        Some(OrderedNode::Scalar(Value::String(kind))) => Some(kind),
        _ => None,
    }
}

/// The children of an entry or a document: its bare keys whose values are
/// entries, in order.
pub fn children(node: &OrderedNode) -> impl Iterator<Item = (&String, &OrderedNode)> {
    let entries: &[(String, OrderedNode)] = match node {
        OrderedNode::Map(entries) => entries,
        _ => &[],
    };
    entries
        .iter()
        .filter(|(k, v)| !k.starts_with('$') && v.is_entry())
        .map(|(k, v)| (k, v))
}

/// The node at `path`, key by key from `root`.
pub fn node_at<'n>(
    root: &'n OrderedNode,
    path: &[String],
) -> Option<&'n OrderedNode> {
    path.iter()
        .try_fold(root, |node, segment| get(node, segment))
}

/// The node at `path`, key by key from `root`, to edit.
pub fn node_at_mut<'n>(
    root: &'n mut OrderedNode,
    path: &[String],
) -> Option<&'n mut OrderedNode> {
    path.iter()
        .try_fold(root, |node, segment| get_mut(node, segment))
}

/// The value under `key`, in a map.
pub fn get<'n>(
    node: &'n OrderedNode,
    key: &str,
) -> Option<&'n OrderedNode> {
    node.get(key)
}

/// The value under `key`, in a map, to edit.
pub fn get_mut<'n>(
    node: &'n mut OrderedNode,
    key: &str,
) -> Option<&'n mut OrderedNode> {
    match node {
        OrderedNode::Map(entries) => entries.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

/// Sets `key` in a map: in place when it is there, else at the end.
pub fn set(
    node: &mut OrderedNode,
    key: &str,
    value: OrderedNode,
) {
    if let OrderedNode::Map(entries) = node {
        match entries.iter_mut().find(|(k, _)| k == key) {
            Some((_, slot)) => *slot = value,
            None => entries.push((key.to_string(), value)),
        }
    }
}

/// Removes `key` from a map, and returns its value.
pub fn take(
    node: &mut OrderedNode,
    key: &str,
) -> Option<OrderedNode> {
    let OrderedNode::Map(entries) = node else {
        return None;
    };
    let at = entries.iter().position(|(k, _)| k == key)?;
    Some(entries.remove(at).1)
}

/// The map under `key`, set to an empty one when there is none.
pub fn get_or_insert_map<'n>(
    node: &'n mut OrderedNode,
    key: &str,
) -> &'n mut OrderedNode {
    if get(node, key).is_none() {
        set(node, key, OrderedNode::Map(Vec::new()));
    }
    get_mut(node, key).expect("just set")
}

/// The string under `key`.
///
/// # Errors
/// When there is none, or it isn't a string.
pub fn string_field<'n>(
    node: &'n OrderedNode,
    key: &str,
) -> Result<&'n str, String> {
    match get(node, key) {
        Some(OrderedNode::Scalar(Value::String(s))) => Ok(s),
        _ => Err(format!("`{key}` is a string")),
    }
}

/// The string under `key`, when there is one.
///
/// # Errors
/// When it isn't a string.
pub fn string_opt(
    node: &OrderedNode,
    key: &str,
) -> Result<Option<String>, String> {
    match get(node, key) {
        None => Ok(None),
        Some(OrderedNode::Scalar(Value::String(s))) => Ok(Some(s.clone())),
        Some(_) => Err(format!("`{key}` is a string")),
    }
}

/// The whole number under `key`, when there is one.
///
/// # Errors
/// When it isn't a whole number.
pub fn u64_opt(
    node: &OrderedNode,
    key: &str,
) -> Result<Option<u64>, String> {
    match get(node, key) {
        None => Ok(None),
        Some(OrderedNode::Scalar(v)) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("`{key}` is a whole number")),
        Some(_) => Err(format!("`{key}` is a whole number")),
    }
}
