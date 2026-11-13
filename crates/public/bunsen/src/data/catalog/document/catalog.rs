use crate::{
    data::catalog::{
        document::{
            CatalogEntry,
            OrderedNode,
        },
        error::{
            CatalogRule,
            catalog_error,
        },
    },
    errors::BunsenResult,
};

/// One catalog document: `$description`, `$contracts` in the base catalog,
/// and the top-level entries keyed by their segments ([module](super)).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BunsenCatalog {
    /// What the catalog holds, for a person.
    pub description: Option<String>,

    /// The `$contracts` section: contracts and their types, in the order
    /// written. Only the base catalog has one, and it must; `None` when the
    /// document doesn't write it.
    pub contracts: Option<Vec<CatalogEntry>>,

    /// The top-level entries, in the order written.
    pub entries: Vec<CatalogEntry>,
}

impl BunsenCatalog {
    /// Parses a YAML document.
    ///
    /// # Errors
    /// [`CatalogRule::Parse`] when it isn't a catalog.
    pub fn from_yaml(yaml: &str) -> BunsenResult<Self> {
        Self::from_node(OrderedNode::from_yaml(yaml)?)
    }

    /// Writes the catalog as YAML.
    ///
    /// # Errors
    /// [`CatalogRule::Parse`] when it doesn't serialize.
    pub fn to_yaml(&self) -> BunsenResult<String> {
        serde_saphyr::to_string(self).map_err(|e| catalog_error(CatalogRule::Parse, e.to_string()))
    }

    /// Parses a JSON document.
    ///
    /// # Errors
    /// [`CatalogRule::Parse`] when it isn't a catalog.
    pub fn from_json(json: &str) -> BunsenResult<Self> {
        Self::from_node(OrderedNode::from_json(json)?)
    }

    /// Reads the catalog `node` writes.
    ///
    /// # Errors
    /// [`CatalogRule::Parse`] when it isn't a catalog, with the path of
    /// the entry at fault.
    pub fn from_node(node: OrderedNode) -> BunsenResult<Self> {
        BunsenCatalog::read_keyed(node)
    }

    /// The catalog as a tree: its entries, children, presets and axis uses
    /// in their order, and each entry's `$` fields in their declared order.
    pub fn to_node(&self) -> OrderedNode {
        self.write_keyed()
    }

    /// Writes the catalog as pretty JSON.
    ///
    /// # Errors
    /// [`CatalogRule::Parse`] when it doesn't serialize.
    pub fn to_json(&self) -> BunsenResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| catalog_error(CatalogRule::Parse, e.to_string()))
    }
}
