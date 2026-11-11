//! # bunsen's built-in pretrained catalog
//!
//! The layers of the build catalog bunsen imports, as the catalog builder
//! (`validation/catalog_builder`, `cargo catalog update`) writes them from
//! their probe-contract sources: `catalog/MANIFEST.txt`, listing the layers
//! in import order, each with the bunsen feature it needs, and the layer
//! files, compiled in. [`files`] hands them out in that order;
//! `bunsen::data::pretrained::build_catalog::builtin_stack` imports them.
//!
//! The crate knows nothing of the catalog format: it carries the text.

use include_dir::{
    Dir,
    include_dir,
};

/// The catalog directory, compiled in.
static CATALOG: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/catalog");

/// The manifest's file name.
pub const MANIFEST: &str = "MANIFEST.txt";

/// One layer of the catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CatalogFile {
    /// The layer's file name (`qwen3.yml`): YAML, or JSON by its `.json`
    /// extension.
    pub name: &'static str,

    /// The bunsen feature the layer needs, when it needs one: a build
    /// without it skips the layer.
    pub feature: Option<&'static str>,

    /// The layer's text.
    pub text: &'static str,
}

impl CatalogFile {
    /// The layer's name: its file name without the extension.
    pub fn label(&self) -> &'static str {
        self.name
            .rsplit_once('.')
            .map_or(self.name, |(stem, _)| stem)
    }
}

fn text_of(name: &str) -> &'static str {
    CATALOG
        .get_file(name)
        .and_then(|f| f.contents_utf8())
        .unwrap_or_else(|| panic!("the catalog has no UTF-8 file {name}"))
}

/// The layers, in the manifest's order.
///
/// # Panics
/// When the manifest names a file the directory lacks: the directory is
/// built whole, so a test of this crate catches it.
pub fn files() -> Vec<CatalogFile> {
    text_of(MANIFEST)
        .lines()
        .map(|line| line.split('#').next().unwrap_or_default().trim())
        .filter(|line| !line.is_empty())
        .map(|line| {
            let mut words = line.split_whitespace();
            let name = words.next().expect("a line has a file");
            CatalogFile {
                name,
                feature: words.next(),
                text: text_of(name),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest names every layer file, once, and nothing else is in
    /// the directory.
    #[test]
    fn test_the_manifest_lists_the_directory() {
        let files = files();
        assert!(!files.is_empty());
        let mut listed: Vec<&str> = files.iter().map(|f| f.name).collect();
        listed.push(MANIFEST);
        listed.sort_unstable();
        let mut present: Vec<&str> = CATALOG.files().filter_map(|f| f.path().to_str()).collect();
        present.sort_unstable();
        assert_eq!(listed, present);
        assert_eq!(files[0].label(), "prelude");
        assert_eq!(files[0].feature, None);
    }
}
