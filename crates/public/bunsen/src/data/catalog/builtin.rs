//! The built-in catalog: the layers `bunsen-catalog` carries, imported.
//!
//! `bunsen-catalog` holds the catalog the catalog builder writes from its
//! probe-contract sources (`validation/catalog_builder`): the prelude, whose
//! `$contracts` are the kits' own (`kits::support::catalog_contracts`), the
//! commons, and a layer per kit, each with the bunsen feature it needs.
//! [`builtin_stack`] imports them in order, skipping a layer whose feature
//! this build of bunsen doesn't have.

use alloc::{
    collections::BTreeMap,
    format,
    string::{
        String,
        ToString,
    },
    vec::Vec,
};

use crate::data::{
    catalog::{
        BunsenCatalog,
        BunsenCatalogStack,
        CatalogName,
        ResolvedBuild,
        document::{
            CatalogEntry,
            EntryKind,
            ResourceEntry,
        },
        error::{
            CatalogErrorExt,
            CatalogResult,
            CatalogRule,
            catalog_error,
        },
        loader_form::entry_resource_set,
        lookup::{
            LeafMatch,
            name_matches,
        },
        presets::ResolvedPresetSet,
    },
    resources::ResourceSet,
};

/// Whether this build of bunsen has `feature`, as the built-in catalog's
/// manifest names it.
///
/// # Errors
/// [`CatalogRule::Parse`] for a feature bunsen doesn't know.
fn compiled(feature: &str) -> CatalogResult<bool> {
    match feature {
        "kits" => Ok(cfg!(feature = "kits")),
        "speech" => Ok(cfg!(feature = "speech")),
        other => Err(catalog_error(
            CatalogRule::Parse,
            format!("the built-in catalog names the feature `{other}`, which bunsen doesn't have"),
        )),
    }
}

/// The built-in catalog's layers, in import order, with whether this build
/// imports each.
///
/// # Errors
/// [`CatalogRule::Parse`] for a layer whose feature bunsen doesn't know.
pub fn builtin_layers() -> CatalogResult<Vec<(bunsen_catalog::CatalogFile, bool)>> {
    bunsen_catalog::files()
        .into_iter()
        .map(|file| {
            let imported = match file.feature {
                None => true,
                Some(feature) => compiled(feature).map_err(|e| e.in_layer(file.label()))?,
            };
            Ok((file, imported))
        })
        .collect()
}

/// The built-in catalog, as a stack: each of its layers this build of
/// bunsen has the feature for, in order.
///
/// # Errors
/// Whatever a push finds, which would be a bug in the built catalog: the
/// builder imports every layer before it writes them.
pub fn builtin_stack() -> CatalogResult<BunsenCatalogStack> {
    let mut stack = BunsenCatalogStack::new();
    for (file, imported) in builtin_layers()? {
        if !imported {
            continue;
        }
        let label = file.label();
        let document = if file.name.ends_with(".json") {
            BunsenCatalog::from_json(file.text)
        } else {
            BunsenCatalog::from_yaml(file.text)
        }
        .map_err(|e| e.in_layer(label))?;
        stack.push(label, None, document)?;
    }
    Ok(stack)
}

/// The built-in catalog, resolved once in a process: every build with its
/// aliases, every template class's presets, every contract's description,
/// and every resource. A stack is not shared between threads; this is.
///
/// ```rust
/// # #[cfg(feature = "kits")] {
/// use bunsen::data::catalog::BuiltinCatalog;
///
/// let catalog = BuiltinCatalog::get()?;
/// let build = catalog.build("qwen3-0.6b")?;
/// assert_eq!(build.name.to_string(), "qwen/qwen3/qwen3-0.6b");
/// assert!(build.resource_set().get("checkpoint").is_some());
/// # }
/// # Ok::<(), bunsen::errors::BunsenError>(())
/// ```
#[derive(Debug)]
pub struct BuiltinCatalog {
    builds: Vec<ResolvedBuild>,
    /// The names a lookup tests, each with its build's index: the build's
    /// full name and each of its aliases'.
    names: Vec<(CatalogName, usize)>,
    classes: BTreeMap<CatalogName, ResolvedPresetSet>,
    contracts: BTreeMap<CatalogName, String>,
    resources: BTreeMap<CatalogName, ResourceEntry>,
}

impl BuiltinCatalog {
    /// The process's built-in catalog, imported ([`builtin_stack`]) and
    /// resolved the first time it is asked for.
    ///
    /// # Errors
    /// As [`builtin_stack`] and [`BunsenCatalogStack::resolve_build`]: a bug
    /// in the built catalog, which its tests import whole.
    pub fn get() -> CatalogResult<&'static BuiltinCatalog> {
        static BUILTIN: std::sync::OnceLock<Result<BuiltinCatalog, String>> =
            std::sync::OnceLock::new();
        BUILTIN
            .get_or_init(|| {
                builtin_stack()
                    .and_then(|stack| Self::resolve(&stack))
                    .map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| catalog_error(CatalogRule::Parse, format!("the built-in catalog: {e}")))
    }

    fn resolve(stack: &BunsenCatalogStack) -> CatalogResult<Self> {
        let mut builds = Vec::new();
        let mut names = Vec::new();
        for build in stack.of_kind(EntryKind::Build) {
            let build = stack.resolve_build(build.name())?;
            names.push((build.name.clone(), builds.len()));
            names.extend(build.aliases.iter().map(|a| (a.clone(), builds.len())));
            builds.push(build);
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
        Ok(Self {
            builds,
            names,
            classes,
            contracts,
            resources,
        })
    }

    /// The built-in catalog as a stack, imported anew: a stack is not shared
    /// between threads.
    ///
    /// # Errors
    /// As [`builtin_stack`].
    pub fn stack(&self) -> CatalogResult<BunsenCatalogStack> {
        builtin_stack()
    }

    /// Every build, resolved, in the catalog's order.
    pub fn builds(&self) -> &[ResolvedBuild] {
        &self.builds
    }

    /// The build `name` names: its full name, or any spelling the catalog's
    /// lookups take (`qwen/qwen3-0.6b`, `qwen3-0.6b`, case aside), or an
    /// alias's.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a name no
    /// build, or more than one, answers to.
    pub fn build(
        &self,
        name: &str,
    ) -> CatalogResult<&ResolvedBuild> {
        let found = find_spelling(&self.names, name, |i| self.builds[i].name.to_string())?;
        found.map(|i| &self.builds[i]).ok_or_else(|| {
            catalog_error(
                CatalogRule::Lookup,
                format!("the built-in catalog has no build `{name}`"),
            )
        })
    }

    /// The resource `name` names, by its full name
    /// (`openai/whisper/multilingual.tiktoken`), as a set that holds it in
    /// `slot`: what code that names a file of the catalog's loads, such as a
    /// checkpoint's vocabulary. A sharded resource is its index and shards.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a resource the
    /// catalog doesn't have.
    pub fn resource(
        &self,
        name: &str,
        slot: &str,
    ) -> CatalogResult<ResourceSet> {
        let full = CatalogName::parse(name)?;
        let entry = self.resources.get(&full).ok_or_else(|| {
            catalog_error(
                CatalogRule::Lookup,
                format!("the built-in catalog has no resource `{name}`"),
            )
        })?;
        Ok(entry_resource_set(&full.to_string(), entry, slot))
    }

    /// The presets of the template class `class`, by its full name
    /// (`openai/whisper/multilingual`): what a kit gives a model of the class
    /// that no build describes, such as an `hf:` repo's.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a class the
    /// catalog doesn't have.
    pub fn template_presets(
        &self,
        class: &str,
    ) -> CatalogResult<ResolvedPresetSet> {
        let name = CatalogName::parse(class)?;
        self.classes.get(&name).cloned().ok_or_else(|| {
            catalog_error(
                CatalogRule::Lookup,
                format!("the built-in catalog has no template class `{class}`"),
            )
        })
    }

    /// The description of the contract `name`, by its full name under
    /// `$contracts`.
    pub fn contract_description(
        &self,
        name: &CatalogName,
    ) -> Option<&str> {
        self.contracts.get(name).map(String::as_str)
    }
}

/// The index `spelling` names among `names`: a full name (case aside), or
/// a shorter spelling whose segments are among one's in order; `label`
/// names an index for the error when two match.
///
/// # Errors
/// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) when two indices
/// match and neither is named in full.
pub(crate) fn find_spelling(
    names: &[(CatalogName, usize)],
    spelling: &str,
    label: impl Fn(usize) -> String,
) -> CatalogResult<Option<usize>> {
    if let Some((_, i)) = names
        .iter()
        .find(|(n, _)| n.to_string().eq_ignore_ascii_case(spelling))
    {
        return Ok(Some(*i));
    }
    let mut found: Vec<usize> = Vec::new();
    for (n, i) in names {
        if name_matches(spelling, n, LeafMatch::Exact) && !found.contains(i) {
            found.push(*i);
        }
    }
    match found.as_slice() {
        [] => Ok(None),
        [one] => Ok(Some(*one)),
        many => Err(catalog_error(
            CatalogRule::Lookup,
            format!(
                "`{spelling}` is ambiguous: {}",
                many.iter()
                    .map(|&i| label(i))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The built-in catalog imports whole, and its base layer is first.
    #[test]
    fn test_the_builtin_stack_imports() {
        let stack = builtin_stack().unwrap();
        let layers = builtin_layers().unwrap();
        assert_eq!(
            stack.layers().len(),
            layers.iter().filter(|(_, imported)| *imported).count()
        );
        assert_eq!(stack.layers()[0].label(), "prelude");
    }

    /// Its contracts are the compiled kits' own, as their fragments write
    /// them: a kit's changed `contract.yml` needs a `cargo catalog update`.
    #[cfg(all(feature = "kits", feature = "store"))]
    #[test]
    fn test_the_builtin_contracts_are_the_kits() {
        use crate::{
            data::catalog::document::OrderedNode,
            kits::support::catalog_contracts::{
                contracts_section,
                kit_catalog_contracts,
            },
        };

        let stack = builtin_stack().unwrap();
        let built = stack.layers()[0].document().contracts.clone().unwrap();
        let fragments = BunsenCatalog::from_node(OrderedNode::Map(vec![(
            "$contracts".to_string(),
            contracts_section(&kit_catalog_contracts()).unwrap(),
        )]))
        .unwrap()
        .contracts
        .unwrap();
        #[cfg(feature = "speech")]
        assert_eq!(built, fragments);
        #[cfg(not(feature = "speech"))]
        for contract in &fragments {
            assert!(built.contains(contract), "{}", contract.name());
        }
    }

    /// A build of each kit resolves, by its short name.
    #[cfg(feature = "speech")]
    #[test]
    fn test_builtin_builds_resolve() {
        let stack = builtin_stack().unwrap();
        for (name, contract) in [
            ("qwen3-0.6b", "qwen3-lm"),
            ("qwen3.5-0.8b", "qwen3.5-lm"),
            ("gemma-3-1b-it", "gemma-lm"),
            ("rl-d20", "nanochat-lm"),
            ("laya", "laya-decision"),
            ("torchvision/resnet50", "resnet-classifier"),
            ("large", "whisper-asr"),
            ("phonon-2", "parakeet-asr"),
            ("phonon-1", "qwen3-asr/stated"),
            ("voxtral-mini-4b-realtime", "voxtral-asr/stated"),
            ("sortformer-v2", "sortformer-diarizer"),
            ("vad-16k", "silero-vad"),
        ] {
            let found = stack.find_build(name).unwrap();
            let build = stack.resolve_build(&found).unwrap();
            assert_eq!(
                build.tag("contract"),
                Some(&serde_json::json!(contract)),
                "{name}"
            );
        }
    }
}
