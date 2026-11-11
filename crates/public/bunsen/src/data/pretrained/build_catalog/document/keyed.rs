//! The keyed form's reader and writer, through an order-keeping tree.

use alloc::{
    collections::{
        BTreeMap,
        BTreeSet,
    },
    fmt,
};

use serde::{
    Deserialize,
    Deserializer,
    Serialize,
    Serializer,
    de::{
        self,
        DeserializeOwned,
        MapAccess,
        SeqAccess,
        Visitor,
    },
    ser::{
        SerializeMap,
        SerializeSeq,
    },
};
use serde_json::Value;

use super::{
    AxisEntry,
    AxisUse,
    BuildEntry,
    CatalogEntry,
    EntryKind,
    FacetEntry,
    GroupEntry,
    PresetEntry,
    PresetSetEntry,
    PretrainedCatalog,
    ReleaseEntry,
    ResourceEntry,
    TagEntry,
    TemplateClassEntry,
    empty_object,
    is_empty_object,
};
use crate::data::pretrained::build_catalog::{
    error::{
        CatalogError,
        CatalogErrorKind,
        CatalogResult,
    },
    name::CatalogRef,
};

/// A YAML or JSON document as a tree whose maps keep their order and hold
/// each key once: the form a catalog is read and written through.
///
/// A catalog editor reads a document as a tree, changes it, and writes it
/// back with every map's order kept, as written;
/// [`PretrainedCatalog::from_node`] reads the catalog the tree writes, and
/// [`PretrainedCatalog::to_node`] writes one back. A repeated key is refused
/// when the tree is read.
#[derive(Clone, Debug, PartialEq)]
pub enum OrderedNode {
    /// A map, in the order written; its keys are strings, each once.
    Map(Vec<(String, OrderedNode)>),
    /// A list.
    Seq(Vec<OrderedNode>),
    /// A scalar: null, a bool, a number or a string.
    Scalar(Value),
}

impl OrderedNode {
    /// Reads a YAML document.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Parse`] when it isn't YAML, or repeats a key.
    pub fn from_yaml(yaml: &str) -> CatalogResult<Self> {
        serde_saphyr::from_str(yaml).map_err(|e| parse_error("", e))
    }

    /// Reads a JSON document.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Parse`] when it isn't JSON, or repeats a key.
    pub fn from_json(json: &str) -> CatalogResult<Self> {
        serde_json::from_str(json).map_err(|e| parse_error("", e))
    }

    /// Writes the tree as YAML, in its order.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Parse`] when it doesn't serialize.
    pub fn to_yaml(&self) -> CatalogResult<String> {
        serde_saphyr::to_string(self).map_err(|e| parse_error("", e))
    }

    /// Writes the tree as pretty JSON, in its order.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Parse`] when it doesn't serialize.
    pub fn to_json(&self) -> CatalogResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| parse_error("", e))
    }

    /// `value` as a tree, its maps in serialization order.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Parse`] when it doesn't serialize.
    pub fn from_serialize<T: Serialize>(value: &T) -> CatalogResult<Self> {
        let text = serde_json::to_string(value).map_err(|e| parse_error("", e))?;
        Self::from_json(&text)
    }

    /// The value under `key`, in a map.
    pub fn get(
        &self,
        key: &str,
    ) -> Option<&OrderedNode> {
        match self {
            Self::Map(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Whether this is an entry: a map with a `$kind` key.
    pub fn is_entry(&self) -> bool {
        matches!(self, Self::Map(entries) if entries.iter().any(|(k, _)| k == "$kind"))
    }

    /// The tree as a JSON value; a map's order is lost.
    pub fn into_value(self) -> Value {
        match self {
            Self::Map(entries) => Value::Object(
                entries
                    .into_iter()
                    .map(|(k, v)| (k, v.into_value()))
                    .collect(),
            ),
            Self::Seq(items) => Value::Array(items.into_iter().map(Self::into_value).collect()),
            Self::Scalar(value) => value,
        }
    }

    /// `value` as a tree, for the catalog's own types, which always
    /// serialize.
    fn of<T: Serialize>(value: &T) -> Self {
        Self::from_serialize(value).expect("the catalog's types serialize")
    }
}

impl<'de> Deserialize<'de> for OrderedNode {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(OrderedNodeVisitor)
    }
}

struct OrderedNodeVisitor;

impl<'de> Visitor<'de> for OrderedNodeVisitor {
    type Value = OrderedNode;

    fn expecting(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str("a YAML or JSON value")
    }

    fn visit_bool<E>(
        self,
        v: bool,
    ) -> Result<OrderedNode, E> {
        Ok(OrderedNode::Scalar(Value::Bool(v)))
    }

    fn visit_i64<E>(
        self,
        v: i64,
    ) -> Result<OrderedNode, E> {
        Ok(OrderedNode::Scalar(Value::from(v)))
    }

    fn visit_u64<E>(
        self,
        v: u64,
    ) -> Result<OrderedNode, E> {
        Ok(OrderedNode::Scalar(Value::from(v)))
    }

    fn visit_f64<E: de::Error>(
        self,
        v: f64,
    ) -> Result<OrderedNode, E> {
        serde_json::Number::from_f64(v)
            .map(|n| OrderedNode::Scalar(Value::Number(n)))
            .ok_or_else(|| E::custom(format!("{v} is not a finite number")))
    }

    fn visit_str<E>(
        self,
        v: &str,
    ) -> Result<OrderedNode, E> {
        Ok(OrderedNode::Scalar(Value::String(v.to_string())))
    }

    fn visit_string<E>(
        self,
        v: String,
    ) -> Result<OrderedNode, E> {
        Ok(OrderedNode::Scalar(Value::String(v)))
    }

    fn visit_unit<E>(self) -> Result<OrderedNode, E> {
        Ok(OrderedNode::Scalar(Value::Null))
    }

    fn visit_none<E>(self) -> Result<OrderedNode, E> {
        Ok(OrderedNode::Scalar(Value::Null))
    }

    fn visit_some<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<OrderedNode, D::Error> {
        OrderedNode::deserialize(deserializer)
    }

    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> Result<OrderedNode, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element()? {
            items.push(item);
        }
        Ok(OrderedNode::Seq(items))
    }

    fn visit_map<A: MapAccess<'de>>(
        self,
        mut map: A,
    ) -> Result<OrderedNode, A::Error> {
        let mut entries = Vec::new();
        let mut seen = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("the key `{key}` is repeated")));
            }
            entries.push((key, map.next_value()?));
        }
        Ok(OrderedNode::Map(entries))
    }
}

impl Serialize for OrderedNode {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match self {
            Self::Map(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (k, v) in entries {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
            Self::Seq(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Self::Scalar(value) => value.serialize(serializer),
        }
    }
}

impl Serialize for PretrainedCatalog {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        write_catalog(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PretrainedCatalog {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        read_catalog(OrderedNode::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

impl Serialize for CatalogEntry {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        write_entry(self).serialize(serializer)
    }
}

fn parse_error(
    path: &str,
    message: impl fmt::Display,
) -> CatalogError {
    let error = CatalogError::new(CatalogErrorKind::Parse, message.to_string());
    if path.is_empty() {
        error
    } else {
        error.at(path)
    }
}

fn child_path(
    path: &str,
    key: &str,
) -> String {
    if path.is_empty() {
        key.to_string()
    } else {
        format!("{path}/{key}")
    }
}

fn expect_map(
    node: OrderedNode,
    path: &str,
    what: &str,
) -> CatalogResult<Vec<(String, OrderedNode)>> {
    match node {
        OrderedNode::Map(entries) => Ok(entries),
        _ => Err(parse_error(path, format!("{what} is not a map"))),
    }
}

fn typed<T: DeserializeOwned>(
    fields: serde_json::Map<String, Value>,
    path: &str,
    what: &str,
) -> CatalogResult<T> {
    serde_json::from_value(Value::Object(fields))
        .map_err(|e| parse_error(path, format!("{what}: {e}")))
}

/// Reads a catalog document from its tree.
pub(super) fn read_catalog(node: OrderedNode) -> CatalogResult<PretrainedCatalog> {
    let mut catalog = PretrainedCatalog::default();
    for (key, value) in expect_map(node, "", "the catalog")? {
        match key.as_str() {
            "$description" => match value {
                OrderedNode::Scalar(Value::String(text)) => catalog.description = Some(text),
                _ => return Err(parse_error("", "`$description` is not a string")),
            },
            _ if key.starts_with('$') => {
                return Err(parse_error(
                    "",
                    format!("the catalog has no `{key}`: its `$` word is `$description`"),
                ));
            }
            _ => {
                let path = key.clone();
                catalog.entries.push(read_entry(key, value, &path)?);
            }
        }
    }
    Ok(catalog)
}

fn entry_kind(
    text: &str,
    path: &str,
) -> CatalogResult<EntryKind> {
    EntryKind::from_name(text).ok_or_else(|| parse_error(path, format!("`{text}` is not a kind")))
}

fn read_entry(
    name: String,
    node: OrderedNode,
    path: &str,
) -> CatalogResult<CatalogEntry> {
    let entries = expect_map(node, path, "an entry")?;
    let kind = match entries.iter().find(|(k, _)| k == "$kind") {
        Some((_, OrderedNode::Scalar(Value::String(kind)))) => entry_kind(kind, path)?,
        Some(_) => return Err(parse_error(path, "`$kind` is not a string")),
        None => return Err(parse_error(path, "an entry has no `$kind`")),
    };
    let has_presets = matches!(kind, EntryKind::Build | EntryKind::TemplateClass);
    let mut fields = serde_json::Map::new();
    let mut presets = PresetSetEntry::default();
    let mut tags = BTreeMap::new();
    let mut children = Vec::new();
    for (key, value) in entries {
        if key == "$kind" {
            continue;
        }
        if key == "$presets" && has_presets {
            presets = read_presets(value, path)?;
        } else if key.starts_with('$') {
            fields.insert(key, value.into_value());
        } else {
            let child = match kind {
                EntryKind::Namespace | EntryKind::Contract => true,
                EntryKind::Build => value.is_entry(),
                EntryKind::TemplateClass => false,
                _ => {
                    return Err(parse_error(
                        path,
                        format!(
                            "{} has no bare keys: `{key}` is neither a `$` word nor allowed",
                            crate::data::pretrained::build_catalog::stack::with_article(
                                kind.as_str()
                            )
                        ),
                    ));
                }
            };
            if child {
                let child_path = child_path(path, &key);
                children.push(read_entry(key, value, &child_path)?);
            } else {
                let tag = CatalogRef::parse(&key)
                    .map_err(|e| parse_error(path, format!("the tag `{key}`: {}", e.message)))?;
                tags.insert(tag, value.into_value());
            }
        }
    }
    let what = kind.as_str();
    Ok(match kind {
        EntryKind::Namespace | EntryKind::Contract => {
            let group = GroupEntry {
                name,
                children,
                ..typed(fields, path, what)?
            };
            if kind == EntryKind::Namespace {
                CatalogEntry::Namespace(group)
            } else {
                CatalogEntry::Contract(group)
            }
        }
        EntryKind::Release => CatalogEntry::Release(ReleaseEntry {
            name,
            ..typed(fields, path, what)?
        }),
        EntryKind::Build => CatalogEntry::Build(BuildEntry {
            name,
            tags,
            presets,
            children,
            ..typed(fields, path, what)?
        }),
        EntryKind::TemplateClass => CatalogEntry::TemplateClass(TemplateClassEntry {
            name,
            tags,
            presets,
            ..typed(fields, path, what)?
        }),
        EntryKind::Tag => CatalogEntry::Tag(TagEntry {
            name,
            ..typed(fields, path, what)?
        }),
        EntryKind::Axis => CatalogEntry::Axis(AxisEntry {
            name,
            ..typed(fields, path, what)?
        }),
        EntryKind::Facet => CatalogEntry::Facet(FacetEntry {
            name,
            ..typed(fields, path, what)?
        }),
        EntryKind::Resource => CatalogEntry::Resource(ResourceEntry {
            name,
            ..typed(fields, path, what)?
        }),
    })
}

/// An axis use's fields, under its key in `$axes`.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AxisUseFields {
    #[serde(rename = "$offers")]
    offers: Vec<String>,
    #[serde(rename = "$default")]
    default: String,
}

fn read_presets(
    node: OrderedNode,
    path: &str,
) -> CatalogResult<PresetSetEntry> {
    let mut set = PresetSetEntry::default();
    for (key, value) in expect_map(node, path, "`$presets`")? {
        match key.as_str() {
            "$common" => {
                set.common = value.into_value();
                if set.common.is_null() {
                    set.common = empty_object();
                }
            }
            "$default" => {
                set.default = serde_json::from_value(value.into_value())
                    .map_err(|e| parse_error(path, format!("`$presets.$default`: {e}")))?;
            }
            "$axes" => {
                for (axis, value) in expect_map(value, path, "`$presets.$axes`")? {
                    let what = format!("the use of `{axis}`");
                    let OrderedNode::Map(entries) = value else {
                        return Err(parse_error(path, format!("{what} is not a map")));
                    };
                    let fields: AxisUseFields = typed(
                        entries
                            .into_iter()
                            .map(|(k, v)| (k, v.into_value()))
                            .collect(),
                        path,
                        &what,
                    )?;
                    let axis = CatalogRef::parse(&axis)
                        .map_err(|e| parse_error(path, format!("{what}: {}", e.message)))?;
                    set.axes.push(AxisUse {
                        axis,
                        offers: fields.offers,
                        default: fields.default,
                    });
                }
            }
            _ if key.starts_with('$') => {
                return Err(parse_error(
                    path,
                    format!(
                        "`$presets` has no `{key}`: its `$` words are `$common`, `$axes` and \
                         `$default`, and its other keys are presets"
                    ),
                ));
            }
            _ => {
                let what = format!("the preset `{key}`");
                let OrderedNode::Map(entries) = value else {
                    return Err(parse_error(path, format!("{what} is not a map")));
                };
                let preset: PresetEntry = typed(
                    entries
                        .into_iter()
                        .map(|(k, v)| (k, v.into_value()))
                        .collect(),
                    path,
                    &what,
                )?;
                set.presets.push(PresetEntry {
                    name: key,
                    ..preset
                });
            }
        }
    }
    Ok(set)
}

/// Writes a catalog document as its tree.
pub(super) fn write_catalog(catalog: &PretrainedCatalog) -> OrderedNode {
    let mut entries = Vec::new();
    if let Some(description) = &catalog.description {
        entries.push((
            "$description".to_string(),
            OrderedNode::Scalar(Value::String(description.clone())),
        ));
    }
    for entry in &catalog.entries {
        entries.push((entry.name().to_string(), write_entry(entry)));
    }
    OrderedNode::Map(entries)
}

/// An entry's keyed form, without its name: `$kind`, its fields in their
/// declared order, its tags, `$presets`, then its children.
pub(super) fn write_entry(entry: &CatalogEntry) -> OrderedNode {
    let (fields, presets, tags, children) = match entry {
        CatalogEntry::Namespace(e) | CatalogEntry::Contract(e) => {
            (OrderedNode::of(e), None, None, e.children.as_slice())
        }
        CatalogEntry::Release(e) => (OrderedNode::of(e), None, None, &[][..]),
        CatalogEntry::Build(e) => (
            OrderedNode::of(e),
            Some(&e.presets),
            Some(&e.tags),
            e.children.as_slice(),
        ),
        CatalogEntry::TemplateClass(e) => {
            (OrderedNode::of(e), Some(&e.presets), Some(&e.tags), &[][..])
        }
        CatalogEntry::Tag(e) => (OrderedNode::of(e), None, None, &[][..]),
        CatalogEntry::Axis(e) => (OrderedNode::of(e), None, None, &[][..]),
        CatalogEntry::Facet(e) => (OrderedNode::of(e), None, None, &[][..]),
        CatalogEntry::Resource(e) => (OrderedNode::of(e), None, None, &[][..]),
    };
    let mut out = vec![(
        "$kind".to_string(),
        OrderedNode::Scalar(Value::String(entry.kind().as_str().to_string())),
    )];
    if let OrderedNode::Map(fields) = fields {
        out.extend(fields);
    }
    for (tag, value) in tags.into_iter().flatten() {
        out.push((tag.to_string(), OrderedNode::of(value)));
    }
    if let Some(presets) = presets
        && !presets.is_empty()
    {
        out.push(("$presets".to_string(), write_presets(presets)));
    }
    for child in children {
        out.push((child.name().to_string(), write_entry(child)));
    }
    OrderedNode::Map(out)
}

fn write_presets(set: &PresetSetEntry) -> OrderedNode {
    let mut out = Vec::new();
    if !is_empty_object(&set.common) {
        out.push(("$common".to_string(), OrderedNode::of(&set.common)));
    }
    if !set.axes.is_empty() {
        let axes = set
            .axes
            .iter()
            .map(|using| {
                let fields = AxisUseFields {
                    offers: using.offers.clone(),
                    default: using.default.clone(),
                };
                (using.axis.to_string(), OrderedNode::of(&fields))
            })
            .collect();
        out.push(("$axes".to_string(), OrderedNode::Map(axes)));
    }
    if !set.default.is_empty() {
        out.push(("$default".to_string(), OrderedNode::of(&set.default)));
    }
    for preset in &set.presets {
        out.push((preset.name.clone(), OrderedNode::of(preset)));
    }
    OrderedNode::Map(out)
}
