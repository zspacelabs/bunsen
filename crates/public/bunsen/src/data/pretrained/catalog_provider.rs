//! A build catalog's builds of one contract, as a provider's rows.

use alloc::{
    collections::BTreeMap,
    format,
    string::{
        String,
        ToString,
    },
    vec::Vec,
};

use super::{
    Pretrained,
    PretrainedProvider,
    Tags,
    WELL_KNOWN,
    tags::KIT,
};
use crate::{
    data::{
        catalog::{
            CatalogName,
            CatalogRef,
            CatalogRule,
            PretrainedCatalogStack,
            ResolvedBuild,
            builtin_stack,
            catalog_error,
            document::{
                CatalogEntry,
                EntryKind,
                ResourceEntry,
                ResourceSource,
            },
            lookup::{
                LeafMatch,
                name_matches,
            },
            presets::ResolvedPresetSet,
        },
        resources::{
            Resource,
            ResourceBundle,
            Source,
            resolve_dir_template,
        },
    },
    errors::BunsenResult,
};

/// The argument slot a contract states a geometry in
/// ([`PretrainedRef::stated`](super::PretrainedRef::stated)): for weights
/// that carry none (resnet's, and the `stated` variants of qwen3-asr's and
/// voxtral's), the geometry they are built in; for weights that carry their
/// own (whisper's), the one they must have.
pub const GEOMETRY: &str = "geometry";

/// A provider over the builds of one contract in a build catalog: each a
/// row, named in full (`qwen/qwen3/qwen3-0.6b`).
///
/// A contract is what a kit's loader implements
/// ([`catalog_contracts`](crate::kits::support::catalog_contracts)), so the
/// builds of a kit's contract, its variants' among them, are the rows its
/// loader can build. The rows are made once, when the provider is: a stack
/// is not shared between threads, and a provider is.
///
/// A lookup takes a name as the catalog's lookups do: a build's full name,
/// or a shorter spelling whose segments are among its name's in order
/// (`qwen/qwen3-0.6b`, `qwen3-0.6b`), or one of its aliases'; a spelling two
/// rows match is ambiguous.
#[derive(Clone, Debug)]
pub struct BuildCatalogProvider {
    name: String,
    description: String,
    rows: Vec<Pretrained>,
    /// The names a lookup tests, each with its row: the build's full name
    /// and each of its aliases'.
    names: Vec<(CatalogName, usize)>,
}

impl BuildCatalogProvider {
    /// The builds of `contract` (`qwen3-lm`; `qwen3-asr` for both of its
    /// variants) in `stack`, as rows of a provider called `name`, each
    /// tagged with `kit` ([`KIT`]).
    ///
    /// # Errors
    /// A contract `$contracts` doesn't have, or a build that doesn't
    /// resolve or doesn't make a row ([`build_row`]).
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        stack: &PretrainedCatalogStack,
        contract: &str,
        kit: &str,
    ) -> BunsenResult<Self> {
        let contract_name = stack
            .expect_contract(&CatalogRef::parse(contract)?)?
            .name()
            .clone();
        let mut builds = Vec::new();
        for build in stack.builds_of_contract(&contract_name) {
            builds.push((stack.resolve_build(&build)?, stack.aliases_of(&build)));
        }
        Self::of_builds(name, description, builds.iter(), kit)
    }

    /// The built-in catalog's builds of `contract`, as the [`WELL_KNOWN`]
    /// provider's rows, described by the contract.
    ///
    /// The built-in catalog ([`builtin_stack`]) is imported and resolved once
    /// in a process, the first time it is asked for.
    ///
    /// # Errors
    /// As [`builtin_stack`], and
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a contract
    /// `$contracts` doesn't have.
    pub fn builtin(
        contract: &str,
        kit: &str,
    ) -> BunsenResult<Self> {
        let snapshot = builtin_snapshot()?;
        let contract_name = CatalogName::contracts_root().join(CatalogRef::parse(contract)?.path());
        let description = snapshot.contracts.get(&contract_name).ok_or_else(|| {
            catalog_error(
                CatalogRule::Lookup,
                format!("the built-in catalog has no contract `{contract}`"),
            )
        })?;
        let builds = snapshot.builds.iter().filter(|(build, _)| {
            build
                .contract
                .as_ref()
                .is_some_and(|c| c == &contract_name || c.is_within(&contract_name))
        });
        Self::of_builds(WELL_KNOWN, description.clone(), builds, kit)
    }

    /// [`builtin`](Self::builtin), for a kit's default providers: the
    /// built-in catalog is compiled in and tested whole.
    ///
    /// # Panics
    /// When [`builtin`](Self::builtin) fails, a bug in the built-in catalog.
    pub fn expect_builtin(
        contract: &str,
        kit: &str,
    ) -> Self {
        Self::builtin(contract, kit)
            .unwrap_or_else(|e| panic!("the built-in catalog's `{contract}` builds: {e}"))
    }

    fn of_builds<'a>(
        name: impl Into<String>,
        description: impl Into<String>,
        builds: impl Iterator<Item = &'a (ResolvedBuild, Vec<CatalogName>)>,
        kit: &str,
    ) -> BunsenResult<Self> {
        let mut rows = Vec::new();
        let mut names = Vec::new();
        for (build, aliases) in builds {
            let index = rows.len();
            names.push((build.name.clone(), index));
            names.extend(aliases.iter().map(|a| (a.clone(), index)));
            rows.push(build_row(build, aliases, kit)?);
        }
        Ok(Self {
            name: name.into(),
            description: description.into(),
            rows,
            names,
        })
    }

    /// The rows, in the catalog's order.
    pub fn rows(&self) -> &[Pretrained] {
        &self.rows
    }

    /// The row `name` names, if any.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) when two rows
    /// match it and neither is named in full.
    pub fn find(
        &self,
        name: &str,
    ) -> BunsenResult<Option<&Pretrained>> {
        if let Some((_, row)) = self
            .names
            .iter()
            .find(|(n, _)| n.to_string().eq_ignore_ascii_case(name))
        {
            return Ok(Some(&self.rows[*row]));
        }
        let mut found: Vec<usize> = Vec::new();
        for (n, row) in &self.names {
            if name_matches(name, n, LeafMatch::Exact) && !found.contains(row) {
                found.push(*row);
            }
        }
        match found.as_slice() {
            [] => Ok(None),
            [one] => Ok(Some(&self.rows[*one])),
            many => Err(catalog_error(
                CatalogRule::Lookup,
                format!(
                    "`{name}` is ambiguous: {}",
                    many.iter()
                        .map(|&i| self.rows[i].name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )),
        }
    }
}

impl PretrainedProvider for BuildCatalogProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn list(&self) -> Vec<Pretrained> {
        self.rows.clone()
    }

    fn lookup(
        &self,
        name: &str,
    ) -> BunsenResult<Option<Pretrained>> {
        Ok(self.find(name)?.cloned())
    }
}

/// `build` as a row: its full name, the leaves of its `aliases`, its listing
/// fields, its bound resources by slot (a sharded one as `{slot}.index` and
/// `{slot}.{n:05}`, numbered from 1), its presets, its bound arguments, and
/// its tags, each by its own name, with `kit`.
///
/// # Errors
/// A resource source that has no URL form (none yet).
pub fn build_row(
    build: &ResolvedBuild,
    aliases: &[CatalogName],
    kit: &str,
) -> BunsenResult<Pretrained> {
    let name = build.name.to_string();
    let mut resources = ResourceBundle::new(&name);
    resources.description = build.description.clone();
    resources.license = build.license.clone();
    resources.origin = build.origin.clone();
    for (slot, bound) in &build.resources {
        let namespace = bound.resource.to_string();
        for resource in slot_resources(slot, &bound.entry, &namespace) {
            resources = resources.with_resource(resource);
        }
    }
    let mut tags = Tags::new();
    for (tag, value) in &build.tags {
        tags.insert(tag.leaf(), value.value.clone());
    }
    if let Some(release) = &build.release {
        tags.insert(
            "release",
            serde_json::json!({
                "group": release.group.to_string(),
                "released": release.released.canonical(),
            }),
        );
    }
    tags.insert(KIT.key, kit);
    Ok(Pretrained {
        name,
        aliases: aliases.iter().map(|a| a.leaf().to_string()).collect(),
        description: build.description.clone(),
        license: build.license.clone(),
        origin: build.origin.clone(),
        prefab: None,
        resources,
        presets: build.presets.clone(),
        args: build
            .args
            .iter()
            .map(|(slot, arg)| (slot.clone(), arg.value.clone()))
            .collect::<BTreeMap<_, _>>(),
        tags,
    })
}

/// What the built-in catalog resolves to, kept for the process: every build
/// with its aliases, every template class's presets, every contract's
/// description, and every resource. A stack is not shared between threads; this
/// is.
struct BuiltinSnapshot {
    builds: Vec<(ResolvedBuild, Vec<CatalogName>)>,
    classes: BTreeMap<CatalogName, ResolvedPresetSet>,
    contracts: BTreeMap<CatalogName, String>,
    resources: BTreeMap<CatalogName, ResourceEntry>,
}

fn builtin_snapshot() -> BunsenResult<&'static BuiltinSnapshot> {
    static SNAPSHOT: std::sync::OnceLock<Result<BuiltinSnapshot, String>> =
        std::sync::OnceLock::new();
    let snapshot = SNAPSHOT.get_or_init(|| {
        let take = || -> BunsenResult<BuiltinSnapshot> {
            let stack = builtin_stack()?;
            let mut builds = Vec::new();
            for build in stack.of_kind(EntryKind::Build) {
                let name = build.name().clone();
                builds.push((stack.resolve_build(&name)?, stack.aliases_of(&name)));
            }
            let mut classes = BTreeMap::new();
            for class in stack.of_kind(EntryKind::TemplateClass) {
                let name = class.name().clone();
                classes.insert(name.clone(), stack.resolve_template_class(&name)?.presets);
            }
            let mut resources = BTreeMap::new();
            for resource in stack.of_kind(EntryKind::Resource) {
                if let CatalogEntry::Resource(r) = resource.body() {
                    resources.insert(resource.name().clone(), r.clone());
                }
            }
            let mut contracts = BTreeMap::new();
            for contract in stack.of_kind(EntryKind::Contract) {
                if let CatalogEntry::Contract(c) = contract.body() {
                    contracts.insert(contract.name().clone(), c.description.clone());
                }
            }
            Ok(BuiltinSnapshot {
                builds,
                classes,
                contracts,
                resources,
            })
        };
        take().map_err(|e| e.to_string())
    });
    snapshot
        .as_ref()
        .map_err(|e| catalog_error(CatalogRule::Parse, format!("the built-in catalog: {e}")))
}

/// The presets of the built-in catalog's template class `class`, by its
/// full name (`openai/whisper/multilingual`): what a kit gives a model of
/// the class that no build describes, such as an `hf:` repo's.
///
/// # Errors
/// As [`builtin_stack`], and [`Lookup`](crate::errors::BunsenErrorKind::Lookup)
/// for a class the catalog doesn't have.
pub fn builtin_template_presets(class: &str) -> BunsenResult<ResolvedPresetSet> {
    let snapshot = builtin_snapshot()?;
    let name = CatalogName::parse(class)?;
    snapshot.classes.get(&name).cloned().ok_or_else(|| {
        catalog_error(
            CatalogRule::Lookup,
            format!("the built-in catalog has no template class `{class}`"),
        )
    })
}

/// The resource bundle of the built-in catalog's build `name`: its row's files,
/// by slot, under the build's full name. `name` is any spelling the catalog's
/// lookups take (`qwen/qwen3-0.6b`, `qwen3-0.6b`), or an alias.
///
/// # Errors
/// As [`builtin_stack`], and [`Lookup`](crate::errors::BunsenErrorKind::Lookup)
/// for a name no build, or more than one, answers to.
pub fn builtin_build_bundle(name: &str) -> BunsenResult<ResourceBundle> {
    let snapshot = builtin_snapshot()?;
    let all = BuildCatalogProvider::of_builds(WELL_KNOWN, "", snapshot.builds.iter(), "")?;
    let row = all.find(name)?.ok_or_else(|| {
        catalog_error(
            CatalogRule::Lookup,
            format!("the built-in catalog has no build `{name}`"),
        )
    })?;
    Ok(row.resources.clone())
}

/// The built-in catalog's resource `resource`, by its full name
/// (`openai/whisper/multilingual.tiktoken`), as a resource bundle under
/// `key`: what code that names a file of the catalog's loads, such as a
/// checkpoint's vocabulary. A sharded resource is its index and shards.
///
/// # Errors
/// As [`builtin_stack`], and [`Lookup`](crate::errors::BunsenErrorKind::Lookup)
/// for a resource the catalog doesn't have.
pub fn builtin_resource_bundle(
    resource: &str,
    key: &str,
) -> BunsenResult<ResourceBundle> {
    let snapshot = builtin_snapshot()?;
    let name = CatalogName::parse(resource)?;
    let entry = snapshot.resources.get(&name).ok_or_else(|| {
        catalog_error(
            CatalogRule::Lookup,
            format!("the built-in catalog has no resource `{resource}`"),
        )
    })?;
    let mut bundle = ResourceBundle::new(resource);
    if let Some(description) = &entry.description {
        bundle.description = description.clone();
    }
    for r in slot_resources(key, entry, &name.to_string()) {
        bundle = bundle.with_resource(r);
    }
    Ok(bundle)
}

/// The resources a slot bound to `entry` is: one, or a sharded resource's
/// index and shards.
fn slot_resources(
    slot: &str,
    entry: &ResourceEntry,
    namespace: &str,
) -> Vec<Resource> {
    let resource =
        |key: String, file: &str, sha256: Option<&String>, size, kind: &str, sources| Resource {
            key,
            file: file.to_string(),
            sha256: sha256.cloned(),
            kind: Some(kind.to_string()),
            size,
            dtype: entry.dtype.clone(),
            namespace: namespace.to_string(),
            sources,
        };
    let sources_of = |sources: &mut dyn Iterator<Item = &ResourceSource>, file: &str| {
        sources.map(|s| to_source(s, file)).collect::<Vec<_>>()
    };
    if entry.shards.is_empty() {
        return alloc::vec![resource(
            slot.to_string(),
            &entry.file,
            entry.sha256.as_ref(),
            entry.size,
            &entry.format,
            sources_of(&mut entry.sources.iter(), &entry.file),
        )];
    }
    let shard_format = entry.format.strip_suffix("-index").unwrap_or(&entry.format);
    let mut out = alloc::vec![resource(
        format!("{slot}.index"),
        &entry.file,
        entry.sha256.as_ref(),
        entry.size,
        &entry.format,
        sources_of(&mut entry.sources.iter(), &entry.file),
    )];
    for (n, shard) in entry.shards.iter().enumerate() {
        out.push(resource(
            format!("{slot}.{:05}", n + 1),
            &shard.file,
            Some(&shard.sha256),
            Some(shard.size),
            shard_format,
            sources_of(&mut entry.shard_sources(shard), &shard.file),
        ));
    }
    out
}

/// A catalog source as a source of `file`: a URL as it is; a base URL or a
/// Hub repo with the file's path appended; a local dir resolved from its
/// templates.
fn to_source(
    source: &ResourceSource,
    file: &str,
) -> Source {
    match source {
        ResourceSource::Url(url) => Source::Url(url.clone()),
        ResourceSource::BaseUrl(base) => {
            Source::Url(format!("{}/{file}", base.trim_end_matches('/')))
        }
        ResourceSource::Hub { repo, revision } => Source::Url(format!(
            "https://huggingface.co/{repo}/resolve/{revision}/{file}"
        )),
        ResourceSource::LocalDir { name, default } => Source::LocalDir {
            name: name.clone(),
            dir: default.iter().find_map(|t| resolve_dir_template(t)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::catalog::fixtures;

    fn provider(contract: &str) -> BuildCatalogProvider {
        BuildCatalogProvider::new(
            "examples",
            "the example catalogs",
            &fixtures::example_stack().unwrap(),
            contract,
            "toy",
        )
        .unwrap()
    }

    /// A contract's builds are the rows, named in full, tagged with the
    /// kit and the catalog's implied tags.
    #[test]
    fn test_the_rows_are_the_contracts_builds() {
        let whisper = provider("whisper-asr");
        let names: Vec<&str> = whisper.rows().iter().map(|r| r.name.as_str()).collect();
        assert!(
            names.contains(&"openai/whisper/large/large-v3"),
            "{names:?}"
        );
        assert!(names.iter().all(|n| n.starts_with("openai/whisper/")));
        let v3 = whisper.find("large-v3").unwrap().unwrap();
        assert_eq!(v3.tags.get("kit"), Some(&serde_json::json!("toy")));
        assert_eq!(
            v3.tags.get("contract"),
            Some(&serde_json::json!("whisper-asr"))
        );
        assert!(v3.presets.axis_of("translate").is_some());
    }

    /// A resource's sources: a place serves the file by its path, a URL
    /// names it, a local dir resolves from its templates.
    #[test]
    fn test_sources() {
        let whisper = provider("whisper-asr");
        let v1 = whisper.find("large-v1").unwrap().unwrap();
        let checkpoint = v1.resources.get("checkpoint").unwrap();
        // Cached under the resource's full name.
        assert_eq!(
            checkpoint.namespace,
            "openai/whisper/large/large-v1/checkpoint"
        );
        assert_eq!(checkpoint.dtype.as_deref(), Some("float16"));
        assert!(
            matches!(&checkpoint.sources[0], Source::LocalDir { name, .. } if name == "openai-whisper")
        );
        let Source::Url(url) = &checkpoint.sources[1] else {
            panic!("{:?}", checkpoint.sources)
        };
        assert!(url.ends_with("/large-v1.pt"), "{url}");
        assert!(v1.resources.get("vocabulary").is_some());
    }

    /// Lookups take the shorter spellings the catalog's do; a name no row
    /// has is "not here"; one two rows match is ambiguous (the built-in
    /// catalog's two `resnet34`s).
    #[test]
    fn test_lookups() {
        let whisper = provider("whisper-asr");
        for name in [
            "openai/whisper/large/large-v3",
            "openai/large-v3",
            "large-v3",
            "LARGE-V3",
        ] {
            assert_eq!(
                whisper.lookup(name).unwrap().unwrap().name,
                "openai/whisper/large/large-v3",
                "{name}"
            );
        }
        assert!(whisper.lookup("qwen3-0.6b").unwrap().is_none());
        assert!(whisper.lookup("large").unwrap().is_none(), "a namespace");

        // The resnet builds are in the built-in catalog's `kits` layer.
        if !cfg!(feature = "kits") {
            return;
        }

        let resnet = BuildCatalogProvider::builtin("resnet-classifier", "resnet").unwrap();
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Lookup)
            .message_contains("`resnet34` is ambiguous: torchvision/resnet34, timm/resnet34")
            .assert_err(&resnet.lookup("resnet34"));
        assert_eq!(
            resnet.lookup("timm/resnet34").unwrap().unwrap().name,
            "timm/resnet34"
        );
    }
}
