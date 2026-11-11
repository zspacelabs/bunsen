//! One loaded catalog: [`PretrainedCatalogLayer`], the document plus its
//! entries flattened to full names.
//!
//! Flattening resolves the nesting shortcut: a child's full name is its
//! container's full name plus its own segment, as if it had named the
//! container as its `parent`. The layer keeps the document as it was read;
//! the flat entries are a view of it.

use std::path::{
    Path,
    PathBuf,
};

use crate::{
    document::{
        CatalogEntry,
        EntryKind,
        PretrainedCatalog,
    },
    error::{
        CatalogError,
        CatalogErrorKind,
        CatalogErrors,
    },
    name::CatalogName,
};

/// One entry of a layer, under its full name.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerEntry {
    /// The full name.
    pub name: CatalogName,

    /// The entry, with its `children` taken out and its naming `parent`
    /// cleared: the full name carries both.
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
    pub fn merge(&self) -> crate::document::MergeMode {
        self.entry.merge()
    }
}

/// One catalog in a stack: a label, where it was read from, the document,
/// and its flat entries.
#[derive(Clone, Debug)]
pub struct PretrainedCatalogLayer {
    label: String,
    path: Option<PathBuf>,
    document: PretrainedCatalog,
    entries: Vec<LayerEntry>,
}

impl PretrainedCatalogLayer {
    /// Flattens `document` into a layer. This checks only what flattening
    /// needs: name segments, which kinds may nest where, and that a nested
    /// entry doesn't also state a `parent`. Everything else is checked when
    /// the layer joins a stack.
    ///
    /// # Errors
    /// Every [`CatalogErrorKind::Name`] and [`CatalogErrorKind::Structure`]
    /// problem found.
    pub fn new(
        label: impl Into<String>,
        path: Option<PathBuf>,
        document: PretrainedCatalog,
    ) -> Result<Self, CatalogErrors> {
        let label = label.into();
        let mut entries = Vec::new();
        let mut errors = Vec::new();
        flatten(&document.entries, None, &mut entries, &mut errors);
        if errors.is_empty() {
            Ok(Self {
                label,
                path,
                document,
                entries,
            })
        } else {
            Err(CatalogErrors(
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
    pub fn document(&self) -> &PretrainedCatalog {
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
    errors: &mut Vec<CatalogError>,
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
        flat.clear_naming_parent();
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
) -> Result<CatalogName, CatalogError> {
    let kind = entry.kind();
    let segment = entry.name();
    let named = |scope: Option<&CatalogName>| {
        CatalogName::under(scope, segment).map_err(|e| match scope {
            Some(scope) => e.at(format!("{scope}/{segment}")),
            None => e.at(segment),
        })
    };
    let Some((scope, container_kind)) = container else {
        // A naming `parent` is read from the root, with or without a `/`.
        return match kind {
            EntryKind::Facet => named(None),
            _ => named(entry.naming_parent().map(|p| p.path())),
        };
    };
    let name = named(Some(scope))?;
    let allowed = match container_kind {
        EntryKind::Namespace | EntryKind::Contract => kind != EntryKind::Facet,
        EntryKind::Build => matches!(kind, EntryKind::Resource | EntryKind::Tag | EntryKind::Axis),
        _ => false,
    };
    if !allowed {
        return Err(CatalogError::new(
            CatalogErrorKind::Structure,
            format!(
                "{} can't hold {}",
                crate::stack::with_article(container_kind.as_str()),
                crate::stack::with_article(kind.as_str())
            ),
        )
        .at(&name));
    }
    if entry.naming_parent().is_some() {
        return Err(CatalogError::new(
            CatalogErrorKind::Structure,
            "a nested entry may not also state a `parent`: its container is its parent",
        )
        .at(&name));
    }
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(yaml: &str) -> Result<PretrainedCatalogLayer, CatalogErrors> {
        PretrainedCatalogLayer::new("test", None, PretrainedCatalog::from_yaml(yaml).unwrap())
    }

    fn names(layer: &PretrainedCatalogLayer) -> Vec<String> {
        layer.entries().iter().map(|e| e.name.to_string()).collect()
    }

    #[test]
    fn test_nesting_and_parent_name_the_same_entries() {
        let nested = layer(
            r#"
entries:
  - kind: namespace
    name: qwen
    description: "Qwen."
    children:
      - kind: namespace
        name: qwen3
        description: "Qwen3."
        children:
          - { kind: build, name: qwen3-4b, description: "B." }
"#,
        )
        .unwrap();
        let flat = layer(
            r#"
entries:
  - { kind: namespace, name: qwen, description: "Qwen." }
  - { kind: namespace, name: qwen3, parent: qwen, description: "Qwen3." }
  - { kind: build, name: qwen3-4b, parent: qwen/qwen3, description: "B." }
"#,
        )
        .unwrap();
        assert_eq!(
            names(&nested),
            ["qwen", "qwen/qwen3", "qwen/qwen3/qwen3-4b"]
        );
        assert_eq!(names(&nested), names(&flat));
        let strip = |l: &PretrainedCatalogLayer| {
            l.entries()
                .iter()
                .map(|e| e.entry.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(strip(&nested), strip(&flat));
        assert!(nested.entries()[2].nested);
        assert!(!flat.entries()[2].nested);
    }

    #[test]
    fn test_template_classes_nest_and_name_their_base_apart() {
        let l = layer(
            r#"
entries:
  - { kind: template_class, name: child, base: base_class, description: "C." }
  - kind: namespace
    name: g
    description: "G."
    children:
      - { kind: template_class, name: nested, base: child, description: "N." }
  - { kind: template_class, name: flat, parent: /g, description: "F." }
"#,
        )
        .unwrap();
        assert_eq!(names(&l), ["child", "g", "g/nested", "g/flat"]);
    }

    #[test]
    fn test_misplaced_entries_are_refused() {
        let errors = layer(
            r#"
entries:
  - kind: namespace
    name: g
    description: "G."
    children:
      - { kind: facet, name: f, description: "F." }
      - { kind: namespace, name: h, parent: g, description: "H." }
      - kind: build
        name: b
        description: "B."
        children:
          - { kind: namespace, name: inner, description: "I." }
"#,
        )
        .unwrap_err();
        assert_eq!(errors.0.len(), 3, "{errors}");
        assert!(errors.mentions("can't hold a facet"));
        assert!(errors.mentions("may not also state a `parent`"));
        assert!(errors.mentions("a build can't hold a namespace"));
        assert!(errors.0.iter().all(|e| e.layer.as_deref() == Some("test")));
    }

    #[test]
    fn test_bad_segments_are_refused() {
        let errors =
            layer("entries:\n  - { kind: facet, name: \"no\", description: d }\n").unwrap_err();
        assert!(errors.has(CatalogErrorKind::Name));
    }
}
