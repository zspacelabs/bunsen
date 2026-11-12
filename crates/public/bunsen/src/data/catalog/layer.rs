//! One loaded catalog: [`BunsenCatalogLayer`], the document plus its
//! entries flattened to full names.
//!
//! Flattening resolves the nesting: a child's full name is its container's
//! full name plus its own segment. The layer keeps the document as it was
//! read; the flat entries are a view of it.

use std::path::{
    Path,
    PathBuf,
};

use crate::{
    data::catalog::{
        document::{
            BunsenCatalog,
            CatalogEntry,
            EntryKind,
        },
        error::{
            CatalogErrorExt,
            CatalogResult,
            CatalogRule,
            catalog_error,
            catalog_errors,
        },
        name::{
            CONTRACTS,
            CatalogName,
        },
    },
    errors::BunsenError,
};

/// One entry of a layer, under its full name.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerEntry {
    /// The full name.
    pub name: CatalogName,

    /// The entry, with its children taken out: the full name places it.
    pub entry: CatalogEntry,

    /// Whether the entry was written nested in a container.
    pub nested: bool,
}

impl LayerEntry {
    /// The entry's kind.
    pub fn kind(&self) -> EntryKind {
        self.entry.kind()
    }

    /// How the occurrence meets an existing declaration of its name.
    pub fn merge(&self) -> crate::data::catalog::document::MergeMode {
        self.entry.merge()
    }
}

/// One catalog in a stack: a label, where it was read from, the document,
/// and its flat entries.
#[derive(Clone, Debug)]
pub struct BunsenCatalogLayer {
    label: String,
    path: Option<PathBuf>,
    document: BunsenCatalog,
    entries: Vec<LayerEntry>,
}

impl BunsenCatalogLayer {
    /// Flattens `document` into a layer. This checks only what flattening
    /// needs: name segments, and which kinds may nest where. Everything else
    /// is checked when the layer joins a stack.
    ///
    /// # Errors
    /// Every [`CatalogRule::Name`] and [`CatalogRule::Structure`]
    /// problem found.
    pub fn new(
        label: impl Into<String>,
        path: Option<PathBuf>,
        document: BunsenCatalog,
    ) -> CatalogResult<Self> {
        let label = label.into();
        let mut entries = Vec::new();
        let mut errors = Vec::new();
        if let Some(contracts) = &document.contracts {
            let root = CatalogName::contracts_root();
            flatten(
                contracts,
                Some((&root, EntryKind::Contract)),
                &mut entries,
                &mut errors,
            );
        }
        flatten(&document.entries, None, &mut entries, &mut errors);
        if errors.is_empty() {
            Ok(Self {
                label,
                path,
                document,
                entries,
            })
        } else {
            Err(catalog_errors(
                errors.into_iter().map(|e| e.in_layer(&label)).collect(),
            ))
        }
    }

    /// The label the layer was loaded under.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The file it was read from, if any.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// The document as read.
    pub fn document(&self) -> &BunsenCatalog {
        &self.document
    }

    /// The flat entries, in document order (a container before its
    /// children).
    pub fn entries(&self) -> &[LayerEntry] {
        &self.entries
    }
}

fn flatten(
    entries: &[CatalogEntry],
    container: Option<(&CatalogName, EntryKind)>,
    out: &mut Vec<LayerEntry>,
    errors: &mut Vec<BunsenError>,
) {
    for entry in entries {
        let kind = entry.kind();
        let name = match name_of(entry, container) {
            Ok(name) => name,
            Err(e) => {
                errors.push(e);
                continue;
            }
        };
        let mut flat = entry.clone();
        let children = flat.take_children();
        out.push(LayerEntry {
            name: name.clone(),
            entry: flat,
            nested: container.is_some(),
        });
        flatten(&children, Some((&name, kind)), out, errors);
    }
}

fn name_of(
    entry: &CatalogEntry,
    container: Option<(&CatalogName, EntryKind)>,
) -> Result<CatalogName, BunsenError> {
    let kind = entry.kind();
    let segment = entry.name();
    let named = |scope: Option<&CatalogName>| {
        CatalogName::under(scope, segment).map_err(|e| match scope {
            Some(scope) => e.at(format!("{scope}/{segment}")),
            None => e.at(segment),
        })
    };
    let Some((scope, container_kind)) = container else {
        let name = named(None)?;
        if kind == EntryKind::Contract {
            return Err(catalog_error(
                CatalogRule::Structure,
                "a contract is declared in the base catalog's `$contracts`",
            )
            .at(&name));
        }
        return Ok(name);
    };
    let name = named(Some(scope))?;
    let allowed = match container_kind {
        EntryKind::Namespace => !matches!(kind, EntryKind::Facet | EntryKind::Contract),
        EntryKind::Contract => matches!(kind, EntryKind::Contract | EntryKind::Type),
        EntryKind::Build => matches!(kind, EntryKind::Resource | EntryKind::Tag | EntryKind::Axis),
        _ => false,
    };
    if !allowed {
        let container = if scope.segments() == [CONTRACTS] {
            "`$contracts`".to_string()
        } else {
            crate::data::catalog::stack::with_article(container_kind.as_str())
        };
        return Err(catalog_error(
            CatalogRule::Structure,
            format!(
                "{container} can't hold {}{}",
                crate::data::catalog::stack::with_article(kind.as_str()),
                if kind == EntryKind::Contract {
                    "; contracts are declared in the base catalog's `$contracts`"
                } else {
                    ""
                }
            ),
        )
        .at(&name));
    }
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(yaml: &str) -> CatalogResult<BunsenCatalogLayer> {
        BunsenCatalogLayer::new("test", None, BunsenCatalog::from_yaml(yaml).unwrap())
    }

    fn names(layer: &BunsenCatalogLayer) -> Vec<String> {
        layer.entries().iter().map(|e| e.name.to_string()).collect()
    }

    #[test]
    fn test_nesting_names_the_entries() {
        let nested = layer(
            r#"
qwen:
  $kind: namespace
  $description: "Qwen."
  qwen3:
    $kind: namespace
    $description: "Qwen3."
    qwen3-4b: { $kind: build, $description: "B." }
"#,
        )
        .unwrap();
        assert_eq!(
            names(&nested),
            ["qwen", "qwen/qwen3", "qwen/qwen3/qwen3-4b"]
        );
        assert!(!nested.entries()[0].nested);
        assert!(nested.entries()[2].nested);
        // The flat entries hold no children: the full name places them.
        assert!(nested.entries()[0].entry.children().is_empty());
    }

    #[test]
    fn test_template_classes_nest_and_name_their_base_apart() {
        let l = layer(
            r#"
child: { $kind: template_class, $base: base_class, $description: "C." }
g:
  $kind: namespace
  $description: "G."
  nested: { $kind: template_class, $base: child, $description: "N." }
  flat: { $kind: template_class, $description: "F." }
"#,
        )
        .unwrap();
        assert_eq!(names(&l), ["child", "g", "g/nested", "g/flat"]);
    }

    #[test]
    fn test_misplaced_entries_are_refused() {
        let errors = layer(
            r#"
g:
  $kind: namespace
  $description: "G."
  f: { $kind: facet, $description: "F." }
  b:
    $kind: build
    $description: "B."
    inner: { $kind: namespace, $description: "I." }
"#,
        )
        .unwrap_err();
        assert_eq!(errors.problems().len(), 2, "{errors}");
        assert!(errors.mentions("can't hold a facet"));
        assert!(errors.mentions("a build can't hold a namespace"));
        assert!(
            errors
                .problems()
                .iter()
                .all(|p| p.layer.as_deref() == Some("test"))
        );
    }

    #[test]
    fn test_bad_segments_are_refused() {
        let errors = layer("\"no\": { $kind: facet, $description: \"d\" }\n").unwrap_err();
        assert!(errors.has(CatalogRule::Name));
    }
}
