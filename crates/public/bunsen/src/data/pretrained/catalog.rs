//! Catalogues: a kit's pretrained table as data, loaded at run time.

use std::{
    path::PathBuf,
    sync::OnceLock,
};

use serde::{
    Deserialize,
    Serialize,
};

use super::{
    Fuse,
    Pretrained,
    PretrainedGroup,
    PretrainedTable,
    Resource,
    ResourceBundle,
    Source,
    not_found,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
    ResultContext,
};

/// A kit's pretrained table, as the catalog tool writes it: the kit's
/// `catalog.json`, which the kit embeds and loads with
/// [`to_table`](Self::to_table).
///
/// Groups hold their rows (a row's name is unique within its group, as
/// `group/row` addresses it); rows name their resource bundles, and a bundle
/// is listed once however many rows use it. The tool generates the file from a
/// hand-written source (`cargo validate catalog`); the file is checked in, and
/// `--check` keeps the two in step.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PretrainedCatalog {
    /// The kit, for messages.
    pub kit: String,

    /// The table's name.
    pub name: String,

    /// A line for a listing.
    pub description: String,

    /// The table's groups, in listing order, each with its rows.
    pub groups: Vec<CatalogGroup>,

    /// The resource bundles, by name, in any order: a row lists the ones it
    /// is made of.
    pub bundles: Vec<CatalogBundle>,
}

/// One group of a [`PretrainedCatalog`]: its listing fields, and its rows by
/// name.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogGroup {
    /// Its name: the `group` of `group/row`.
    pub name: String,

    /// A line for a listing.
    pub description: String,

    /// The license its rows are distributed under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// Where its rows are published.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,

    /// Its rows, in listing order; a name is unique within the group.
    pub rows: Vec<CatalogRow>,
}

/// One row of a [`PretrainedCatalog`]: a [`Pretrained`], its resources named.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogRow {
    /// Its name, unique within its group.
    pub name: String,

    /// Other names it answers to.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,

    /// A line for a listing.
    pub description: String,

    /// The license it is distributed under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// Where it is published.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,

    /// The prefab it instantiates, by name in the kit's prefab map.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefab: Option<String>,

    /// The resource bundles it is made of, by name, fused strictly in
    /// order.
    pub bundles: Vec<String>,
}

/// One resource bundle of a [`PretrainedCatalog`]: a [`ResourceBundle`] before
/// its bases are appended to its resources' sources.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogBundle {
    /// Its name, unique within its catalog.
    pub name: String,

    /// A line for a listing.
    pub description: String,

    /// The license it is distributed under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// Where it is published.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,

    /// The cache segment its files live under: who published them.
    pub namespace: String,

    /// The places every file name is appended to, in preference order.
    pub bases: Vec<CatalogBase>,

    /// Its files, in listing order.
    pub resources: Vec<CatalogResource>,
}

/// A place a file can be had from: a URL to append the file name to, or a
/// directory another tool keeps the file in.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum CatalogBase {
    /// A URL; a base has the file name appended, a source is used as is.
    Url(String),

    /// A directory another tool keeps its files in, under their bare names.
    LocalDir(LocalDirBase),
}

/// A directory another tool keeps its files in: `openai-whisper`'s download
/// root, say.
///
/// A caller overrides the directory by [`name`](Self::name) through
/// [`PretrainedCacheOptions::local_dirs`](super::PretrainedCacheOptions::local_dirs);
/// otherwise it is the first of [`default`](Self::default) that resolves
/// ([`resolve_dir_template`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalDirBase {
    /// The source's name, for overrides and messages.
    pub name: String,

    /// Where the directory is when nobody overrides it: path templates, the
    /// first that resolves wins.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub default: Vec<String>,
}

/// One file of a [`CatalogBundle`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogResource {
    /// The bundle's name for it: a kit's key constant.
    pub key: String,

    /// The file name, under every base.
    pub file: String,

    /// Lowercase hex SHA-256 of the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,

    /// A label for a listing, and for the kit's reader.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,

    /// The file's size in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,

    /// The element type a checkpoint stores its weights in, as its
    /// publisher names it (`bfloat16`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,

    /// Sources of this file alone, tried before the bundle's bases; a
    /// source is used as is, with no file name appended.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<CatalogBase>,
}

/// The directory a path template names, or `None` when it does not
/// resolve.
///
/// A template is a path whose leading `~` is the home directory and whose
/// `$NAME` segments are environment variables: `~/.cache/whisper`,
/// `$XDG_CACHE_HOME/whisper`. A template that names an unset variable, or
/// `~` without a home directory, does not resolve.
pub fn resolve_dir_template(template: &str) -> Option<PathBuf> {
    resolve_dir_template_with(template, &|name| std::env::var(name).ok(), &|| {
        std::env::home_dir()
    })
}

fn resolve_dir_template_with(
    template: &str,
    env: &dyn Fn(&str) -> Option<String>,
    home: &dyn Fn() -> Option<PathBuf>,
) -> Option<PathBuf> {
    let mut path = PathBuf::new();
    for (i, segment) in template.split('/').enumerate() {
        if i == 0 && segment == "~" {
            path.push(home()?);
        } else if let Some(name) = segment.strip_prefix('$') {
            path.push(env(name).filter(|v| !v.is_empty())?);
        } else if i == 0 && segment.is_empty() {
            path.push("/");
        } else if !segment.is_empty() {
            path.push(segment);
        }
    }
    Some(path)
}

impl CatalogBase {
    /// The source a file of `file` has under this base: a URL with the file
    /// name appended, or the directory.
    fn source_for(
        &self,
        file: &str,
    ) -> Source {
        match self {
            Self::Url(url) => Source::Url(format!("{}/{file}", url.trim_end_matches('/'))),
            Self::LocalDir(dir) => dir.to_source(),
        }
    }

    /// The source this is, used as is.
    fn to_source(&self) -> Source {
        match self {
            Self::Url(url) => Source::Url(url.clone()),
            Self::LocalDir(dir) => dir.to_source(),
        }
    }
}

impl LocalDirBase {
    fn to_source(&self) -> Source {
        Source::LocalDir {
            name: self.name.clone(),
            dir: self.default.iter().find_map(|t| resolve_dir_template(t)),
        }
    }
}

impl CatalogBundle {
    /// The owned bundle: every resource with this bundle's bases appended to
    /// its own sources.
    pub fn to_resource_bundle(&self) -> ResourceBundle {
        ResourceBundle {
            name: self.name.clone(),
            description: self.description.clone(),
            license: self.license.clone(),
            origin: self.origin.clone(),
            resources: self
                .resources
                .iter()
                .map(|r| {
                    let resource = Resource {
                        key: r.key.clone(),
                        file: r.file.clone(),
                        sha256: r.sha256.clone(),
                        kind: r.kind.clone(),
                        size: r.size,
                        dtype: r.dtype.clone(),
                        namespace: self.namespace.clone(),
                        sources: r
                            .sources
                            .iter()
                            .map(CatalogBase::to_source)
                            .chain(self.bases.iter().map(|b| b.source_for(&r.file)))
                            .collect(),
                    };
                    (r.key.clone(), resource)
                })
                .collect(),
        }
    }
}

impl PretrainedCatalog {
    /// A catalog from the JSON the catalog tool writes.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when it does not parse as a catalog.
    pub fn from_json(json: &str) -> BunsenResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| BunsenError::invalid_resource("a pretrained catalog").with_cause(e))
    }

    /// The bundle called `name`.
    pub fn bundle(
        &self,
        name: &str,
    ) -> Option<&CatalogBundle> {
        self.bundles.iter().find(|b| b.name == name)
    }

    /// The owned row: its bundles fused strictly under its own name and
    /// listing fields.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a bundle the
    /// catalog does not have;
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when two of its
    /// bundles share a key.
    pub fn to_pretrained(
        &self,
        row: &CatalogRow,
    ) -> BunsenResult<Pretrained> {
        let mut resources = ResourceBundle::new(&row.name);
        resources.description = row.description.clone();
        resources.license = row.license.clone();
        resources.origin = row.origin.clone();
        for name in &row.bundles {
            let bundle = self.bundle(name).ok_or_else(|| {
                let names: Vec<&str> = self.bundles.iter().map(|b| b.name.as_str()).collect();
                not_found(Some(&row.name), "resource bundle", name, &names)
            })?;
            resources = resources.fuse(bundle.to_resource_bundle(), Fuse::Strict)?;
        }
        Ok(Pretrained {
            name: row.name.clone(),
            aliases: row.aliases.clone(),
            description: row.description.clone(),
            license: row.license.clone(),
            origin: row.origin.clone(),
            prefab: row.prefab.clone(),
            resources,
        })
    }

    /// The owned table: every group with its rows, in listing order.
    ///
    /// # Errors
    /// As [`to_pretrained`](Self::to_pretrained), under a frame naming the
    /// kit.
    pub fn to_table(&self) -> BunsenResult<PretrainedTable> {
        let groups = self
            .groups
            .iter()
            .map(|group| {
                let items = group
                    .rows
                    .iter()
                    .map(|row| self.to_pretrained(row))
                    .collect::<BunsenResult<_>>()?;
                Ok(PretrainedGroup {
                    name: group.name.clone(),
                    description: group.description.clone(),
                    license: group.license.clone(),
                    origin: group.origin.clone(),
                    items,
                })
            })
            .collect::<BunsenResult<_>>()
            .with_context(|| format!("the {} catalog", self.kit))?;
        Ok(PretrainedTable {
            name: self.name.clone(),
            description: self.description.clone(),
            groups,
        })
    }
}

/// A kit's catalog, embedded at build time and parsed on first use:
/// `static FOO_CATALOG: EmbeddedCatalog =
/// EmbeddedCatalog::new("foo", include_str!("catalog.json"));`.
///
/// The JSON is the catalog tool's output and the kit's tests load it
/// ([`PretrainedCatalog::validate`]), so a malformed catalog is a broken
/// build, and the accessors panic on one rather than return an error a
/// caller could do nothing with.
pub struct EmbeddedCatalog {
    kit: &'static str,
    json: &'static str,
    parsed: OnceLock<PretrainedCatalog>,
}

impl EmbeddedCatalog {
    /// The catalog `json`, of `kit`.
    pub const fn new(
        kit: &'static str,
        json: &'static str,
    ) -> Self {
        Self {
            kit,
            json,
            parsed: OnceLock::new(),
        }
    }

    /// The parsed catalog.
    ///
    /// # Panics
    /// If the embedded JSON is not a catalog: a broken build, which the
    /// kit's tests catch.
    pub fn catalog(&self) -> &PretrainedCatalog {
        self.parsed.get_or_init(|| {
            PretrainedCatalog::from_json(self.json)
                .unwrap_or_else(|e| panic!("the {} kit's embedded catalog: {e}", self.kit))
        })
    }

    /// The owned table, behind a kit's `well-known:` provider.
    ///
    /// # Panics
    /// As [`catalog`](Self::catalog), and if the catalog does not load.
    pub fn table(&self) -> PretrainedTable {
        self.catalog()
            .to_table()
            .unwrap_or_else(|e| panic!("the {} kit's embedded catalog: {e}", self.kit))
    }

    /// The owned resource bundle called `name`.
    ///
    /// # Panics
    /// As [`catalog`](Self::catalog), and if there is no such bundle.
    pub fn resource_bundle(
        &self,
        name: &str,
    ) -> ResourceBundle {
        self.catalog()
            .bundle(name)
            .unwrap_or_else(|| panic!("the {} kit's catalog has no bundle {name:?}", self.kit))
            .to_resource_bundle()
    }
}

impl core::fmt::Debug for EmbeddedCatalog {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("EmbeddedCatalog")
            .field("kit", &self.kit)
            .finish_non_exhaustive()
    }
}

impl PretrainedCatalog {
    /// Every row, group by group.
    pub fn rows(&self) -> impl Iterator<Item = &CatalogRow> {
        self.groups.iter().flat_map(|g| g.rows.iter())
    }

    /// Checks the catalog hangs together, as a kit's test does: it loads,
    /// every digest is 64 lowercase hex digits, every prefab a row names is
    /// one of `prefabs`, and every bundle is some row's.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// naming the first problem found, or the error
    /// [`to_table`](Self::to_table) gives.
    pub fn validate(
        &self,
        prefabs: &[&str],
    ) -> BunsenResult<()> {
        self.to_table()?;
        let problem = |what: String| {
            Err(BunsenError::invalid_resource(format!(
                "the {} catalog: {what}",
                self.kit
            )))
        };
        for bundle in &self.bundles {
            for r in &bundle.resources {
                if let Some(sha) = &r.sha256
                    && !(sha.len() == 64
                        && sha
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
                {
                    return problem(format!(
                        "{}: {}: a malformed sha256 {sha:?}",
                        bundle.name, r.file
                    ));
                }
            }
            if !self.rows().any(|row| row.bundles.contains(&bundle.name)) {
                return problem(format!("bundle {:?} is no row's", bundle.name));
            }
        }
        for row in self.rows() {
            if let Some(prefab) = &row.prefab
                && !prefabs.contains(&prefab.as_str())
            {
                return problem(format!(
                    "row {:?} names prefab {prefab:?}, which the kit lacks",
                    row.name
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    const CATALOG: &str = r#"{
        "kit": "toy",
        "name": "well-known",
        "description": "the toy table",
        "groups": [
            { "name": "a", "description": "group a", "license": "MIT", "rows": [
                { "name": "small", "aliases": ["s"], "description": "a small one",
                  "bundles": ["a/small.pt", "a/vocab"] },
                { "name": "large", "description": "a large one", "prefab": "large",
                  "bundles": ["a/large.pt", "a/vocab"] }
            ] },
            { "name": "b", "description": "group b", "rows": [
                { "name": "small", "description": "b's small one, another row",
                  "bundles": ["a/large.pt"] }
            ] }
        ],
        "bundles": [
            { "name": "a/small.pt", "description": "the small checkpoint", "namespace": "a",
              "bases": [ { "local_dir": { "name": "upstream" } }, { "url": "https://a.example/m/" } ],
              "resources": [ { "key": "checkpoint", "file": "small.pt", "sha256": "aa", "kind": "pt",
                               "size": 7, "dtype": "float32" } ] },
            { "name": "a/large.pt", "description": "the large checkpoint", "namespace": "a",
              "bases": [ { "url": "https://a.example/m" } ],
              "resources": [ { "key": "checkpoint", "file": "large.pt", "sha256": "bb" } ] },
            { "name": "a/vocab", "description": "the vocabulary", "namespace": "a",
              "bases": [ { "url": "https://a.example/v" } ],
              "resources": [ { "key": "vocabulary", "file": "v.txt", "sha256": "cc",
                               "sources": [ { "url": "https://mirror.example/v.txt" } ] } ] }
        ]
    }"#;

    /// A catalog loads into the owned table: groups in order, each row's
    /// bundles fused under its name, a bundle's bases appended to its
    /// resources after their own sources, shared bundles listed once.
    #[test]
    fn test_to_table() {
        let catalog = PretrainedCatalog::from_json(CATALOG).unwrap();
        let table = catalog.to_table().unwrap();
        assert_eq!(table.name, "well-known");
        let group = &table.groups[0];
        assert_eq!(group.license.as_deref(), Some("MIT"));
        let names: Vec<&str> = group.items.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["small", "large"]);

        let small = &group.items[0];
        assert_eq!(small.aliases, ["s"]);
        assert_eq!(small.resources.name, "small");
        assert_eq!(small.resources.keys(), ["checkpoint", "vocabulary"]);
        let checkpoint = small.resources.get("checkpoint").unwrap();
        assert_eq!(checkpoint.size, Some(7));
        assert_eq!(checkpoint.dtype.as_deref(), Some("float32"));
        assert_eq!(checkpoint.namespace, "a");
        assert_eq!(
            checkpoint.sources,
            [
                Source::LocalDir {
                    name: "upstream".into(),
                    dir: None
                },
                Source::Url("https://a.example/m/small.pt".into()),
            ]
        );
        let vocabulary = small.resources.get("vocabulary").unwrap();
        assert_eq!(
            vocabulary.urls(),
            ["https://mirror.example/v.txt", "https://a.example/v/v.txt"]
        );
        assert_eq!(group.items[1].prefab.as_deref(), Some("large"));
        // A name is unique within its group: `b/small` is another row.
        let b_small = &table.groups[1].items[0];
        assert_eq!(b_small.name, "small");
        assert_eq!(b_small.resources.keys(), ["checkpoint"]);
        assert_eq!(
            b_small.resources.get("checkpoint").unwrap().file,
            "large.pt"
        );
    }

    /// A catalog survives a round trip through JSON unchanged.
    #[test]
    fn test_round_trip() {
        let catalog = PretrainedCatalog::from_json(CATALOG).unwrap();
        let json = serde_json::to_string_pretty(&catalog).unwrap();
        assert_eq!(PretrainedCatalog::from_json(&json).unwrap(), catalog);
    }

    /// A row naming a missing bundle, two bundles sharing a key, and an
    /// unknown field are refused.
    #[test]
    fn test_refusals() {
        let mut catalog = PretrainedCatalog::from_json(CATALOG).unwrap();
        catalog.groups[0].rows[0].bundles.push("a/nowhere".into());
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .display_contains("a/nowhere")
            .assert_err(&catalog.to_table());

        let mut catalog = PretrainedCatalog::from_json(CATALOG).unwrap();
        catalog.groups[0].rows[1].bundles.push("a/small.pt".into());
        assert!(catalog.to_table().is_err());

        let unknown = CATALOG.replacen("\"kit\"", "\"colour\": \"red\", \"kit\"", 1);
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .assert_err(&PretrainedCatalog::from_json(&unknown));
    }

    /// `validate` passes a sound catalog, and refuses a malformed digest, an
    /// unknown prefab, and a bundle no row uses.
    #[test]
    fn test_validate() {
        let catalog = PretrainedCatalog::from_json(CATALOG).unwrap();
        assert!(
            catalog.validate(&["large"]).is_err(),
            "the digests are short"
        );

        let full = CATALOG
            .replace("\"aa\"", &format!("\"{}\"", "a".repeat(64)))
            .replace("\"bb\"", &format!("\"{}\"", "b".repeat(64)))
            .replace("\"cc\"", &format!("\"{}\"", "c".repeat(64)));
        let catalog = PretrainedCatalog::from_json(&full).unwrap();
        catalog.validate(&["large"]).unwrap();
        assert!(catalog.validate(&[]).is_err(), "no prefab large");

        let mut unused = catalog.clone();
        unused.groups[0].rows.retain(|r| r.name != "large");
        unused.groups.truncate(1);
        assert!(
            unused.validate(&["large"]).is_err(),
            "a/large.pt is no row's"
        );
    }

    /// An embedded catalog parses once and hands out its table and bundles.
    #[test]
    fn test_embedded() {
        static EMBEDDED: EmbeddedCatalog = EmbeddedCatalog::new("toy", CATALOG);
        assert_eq!(EMBEDDED.table().groups[1].items.len(), 1);
        assert_eq!(EMBEDDED.resource_bundle("a/vocab").keys(), ["vocabulary"]);
        assert!(std::ptr::eq(EMBEDDED.catalog(), EMBEDDED.catalog()));
    }

    /// A template resolves `~` and `$NAME` segments, and does not resolve
    /// when a variable is unset or empty, or there is no home.
    #[test]
    fn test_resolve_dir_template() {
        let env = |name: &str| match name {
            "XDG_CACHE_HOME" => Some("/x/cache".to_string()),
            "EMPTY" => Some(String::new()),
            _ => None,
        };
        let home = || Some(PathBuf::from("/home/u"));
        let resolve = |t: &str| resolve_dir_template_with(t, &env, &home);
        assert_eq!(
            resolve("$XDG_CACHE_HOME/whisper"),
            Some(PathBuf::from("/x/cache/whisper"))
        );
        assert_eq!(
            resolve("~/.cache/whisper"),
            Some(PathBuf::from("/home/u/.cache/whisper"))
        );
        assert_eq!(resolve("/abs/dir"), Some(PathBuf::from("/abs/dir")));
        assert_eq!(resolve("$UNSET/whisper"), None);
        assert_eq!(resolve("$EMPTY/whisper"), None);
        assert_eq!(resolve_dir_template_with("~/x", &env, &|| None), None);
    }
}
