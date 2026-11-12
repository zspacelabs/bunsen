//! Resolving a build: [`ResolvedBuild`], everything the catalog says about
//! it once its template-class chain, its memberships and its stack's
//! declarations are taken into account.

use alloc::collections::{
    BTreeMap,
    BTreeSet,
};

use serde_json::Value;

use crate::{
    data::catalog::{
        document::{
            ArgSlot,
            CatalogEntry,
            EntryKind,
            Membership,
            ResourceEntry,
            ResourceSlot,
        },
        error::{
            CatalogErrorExt,
            CatalogRule,
            catalog_error,
            catalog_errors,
        },
        name::{
            CatalogName,
            CatalogRef,
            check_segment,
        },
        presets::{
            PresetLink,
            ResolvedPresetSet,
        },
        release_time::ReleaseTime,
        stack::BunsenCatalogStack,
        tags::{
            axis_tag_value,
            check_value,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// Where a resolved tag's value comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TagSource {
    /// Written by this template class or build (the lowest in the chain
    /// that writes it).
    Written(CatalogName),
    /// Implied by a use of the axis of the same name.
    Axis,
    /// Implied by membership in a group of the facet of the same name.
    Facet,
    /// The [`CONTRACT_TAG`], implied by the build's contract.
    Contract,
}

/// The tag every build carries naming its contract, as a select query can
/// read it. No tag, axis or facet may be declared under this name.
pub const CONTRACT_TAG: &str = "contract";

/// One resolved tag.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedTag {
    /// The value.
    pub value: Value,
    /// Where it comes from.
    pub source: TagSource,
}

/// One bound resource slot.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedResource {
    /// The resource's full name.
    pub resource: CatalogName,
    /// The resource's declaration.
    pub entry: ResourceEntry,
    /// Who bound the slot: the lowest template class or build that does.
    pub set_by: CatalogName,
}

/// One bound argument slot.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedArg {
    /// The value.
    pub value: Value,
    /// Who bound the slot: the lowest template class or build that does.
    pub set_by: CatalogName,
}

/// A build's release membership.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedRelease {
    /// The release group.
    pub group: CatalogName,
    /// When the build was released.
    pub released: ReleaseTime,
}

/// Everything the catalog says about one build.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedBuild {
    /// The build's full name.
    pub name: CatalogName,
    /// The aliases that name it, by their full names.
    pub aliases: Vec<CatalogName>,
    /// What it is, for a person.
    pub description: String,
    /// The checkpoint's license.
    pub license: Option<String>,
    /// Where the checkpoint comes from.
    pub origin: Option<String>,
    /// The template-class chain, root first.
    pub template_chain: Vec<CatalogName>,
    /// Every group it is in.
    pub memberships: BTreeSet<CatalogName>,
    /// Its group in each facet it is in, by facet.
    pub facets: BTreeMap<CatalogName, CatalogName>,
    /// Its contract, under `$contracts`: the one its `$contract` names, or
    /// its template class's.
    pub contract: Option<CatalogName>,
    /// Its release membership, if it has one.
    pub release: Option<ResolvedRelease>,
    /// Its tags, written and implied, by tag name.
    pub tags: BTreeMap<CatalogName, ResolvedTag>,
    /// Its presets.
    pub presets: ResolvedPresetSet,
    /// Its contract's resource slots, bound, by slot name.
    pub resources: BTreeMap<String, ResolvedResource>,
    /// Its contract's argument slots, bound, by slot name.
    pub args: BTreeMap<String, ResolvedArg>,
    /// The origins its `$hub` sources are had from, in order: its catalog's
    /// Hub mirror set ([`BunsenCatalogStack::hub_origins`]); the Hub's own
    /// origin when empty.
    pub hub_origins: Vec<String>,
}

impl ResolvedBuild {
    /// A build no catalog declares, named `name`: what a provider that
    /// makes its builds itself answers with (an `hf:` repo's), its
    /// resources and the rest filled in after. It has no aliases, chain,
    /// memberships, contract, release, tags, presets, resources or
    /// arguments.
    pub fn new(
        name: CatalogName,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name,
            aliases: Vec::new(),
            description: description.into(),
            license: None,
            origin: None,
            template_chain: Vec::new(),
            memberships: BTreeSet::new(),
            facets: BTreeMap::new(),
            contract: None,
            release: None,
            tags: BTreeMap::new(),
            presets: ResolvedPresetSet::default(),
            resources: BTreeMap::new(),
            args: BTreeMap::new(),
            hub_origins: Vec::new(),
        }
    }

    /// The build with `entry`, the resource `resource`, bound to its slot
    /// `slot`, set by the build itself: how a provider that makes its builds
    /// itself gives one its files.
    pub fn with_resource(
        mut self,
        slot: impl Into<String>,
        resource: CatalogName,
        entry: ResourceEntry,
    ) -> Self {
        let set_by = self.name.clone();
        self.resources.insert(
            slot.into(),
            ResolvedResource {
                resource,
                entry,
                set_by,
            },
        );
        self
    }

    /// Whether `name` names the build: its full name or an alias's, or the
    /// leaf of an alias, ignoring ASCII case.
    pub fn answers_to(
        &self,
        name: &str,
    ) -> bool {
        let full = |n: &CatalogName| n.to_string().eq_ignore_ascii_case(name);
        full(&self.name)
            || self
                .aliases
                .iter()
                .any(|a| full(a) || a.leaf().eq_ignore_ascii_case(name))
    }

    /// The value of the tag `name`, if it has one.
    pub fn tag(
        &self,
        name: &str,
    ) -> Option<&Value> {
        let name = CatalogName::parse(name).ok()?;
        self.tags.get(&name).map(|t| &t.value)
    }
}

/// What the catalog says about one template class, as its builds inherit
/// it.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedTemplateClass {
    /// The class's full name.
    pub name: CatalogName,
    /// What it shares, for a person.
    pub description: String,
    /// Its chain of base classes, root first, ending at itself.
    pub chain: Vec<CatalogName>,
    /// Every group its builds join through the chain.
    pub memberships: BTreeSet<CatalogName>,
    /// The contract its builds have unless they name another: the nearest
    /// in its chain that names one.
    pub contract: Option<CatalogName>,
    /// Its tags, written along the chain and implied by its axis uses.
    pub tags: BTreeMap<CatalogName, ResolvedTag>,
    /// Its presets, folded along the chain.
    pub presets: ResolvedPresetSet,
}

impl BunsenCatalogStack {
    /// Resolves the build `name`.
    ///
    /// # Errors
    /// Every problem found: a reference that doesn't resolve, a facet
    /// broken, a tag wrong, a preset set that doesn't fold, a dependency
    /// key its contract doesn't declare, and so on.
    pub fn resolve_build(
        &self,
        name: &CatalogName,
    ) -> BunsenResult<ResolvedBuild> {
        let mut errors: Vec<BunsenError> = Vec::new();
        let decl = self
            .expect(name, "build", &[EntryKind::Build])
            .map_err(|e| e.at(name))?;
        let CatalogEntry::Build(build) = decl.body() else {
            unreachable!("expected a build");
        };
        let mut report = |e: BunsenError| errors.push(e.at(name));

        if build.description.trim().is_empty() {
            report(catalog_error(
                CatalogRule::Structure,
                "it has no description",
            ));
        }

        // The template-class chain.
        let chain = match self.template_chain(name, build.template_class.as_ref()) {
            Ok(chain) => chain,
            Err(e) => {
                report(e);
                Vec::new()
            }
        };

        // Memberships.
        let mut release = None;
        for membership in &build.join {
            match membership {
                Membership::Plain(group) => {
                    if let Err(e) = self.check_plain_join(name, group) {
                        report(e);
                    }
                }
                Membership::Release(r) => {
                    let group = match self.expect_ref(
                        name,
                        &r.group,
                        "release group",
                        &[EntryKind::Release],
                    ) {
                        Ok(d) => d.name().clone(),
                        Err(e) => {
                            report(e);
                            continue;
                        }
                    };
                    let released = match ReleaseTime::parse(&r.released) {
                        Ok(t) => t,
                        Err(e) => {
                            report(e);
                            continue;
                        }
                    };
                    if release.is_some() {
                        report(catalog_error(
                            CatalogRule::Release,
                            "it is in two release groups; a build is in at most one",
                        ));
                    }
                    release = Some(ResolvedRelease { group, released });
                }
            }
        }
        let memberships = self.build_memberships(name);
        // The contract: the build's `$contract`, else the nearest class's.
        let named = [(name, build.contract.as_ref())]
            .into_iter()
            .chain(
                chain
                    .iter()
                    .rev()
                    .map(|l| (l.name, l.class.contract.as_ref())),
            )
            .find_map(|(owner, c)| c.map(|c| (owner, c)));
        let contract = match named {
            None => {
                report(catalog_error(
                    CatalogRule::Contract,
                    "it names no contract: a build, or its template class, names one with \
                     `$contract`",
                ));
                None
            }
            Some((_, reference)) => match self.expect_contract(reference) {
                Ok(decl) => {
                    let contract = decl.name().clone();
                    if let CatalogEntry::Contract(c) = decl.body()
                        && c.abstract_
                    {
                        let variants: Vec<String> = self
                            .contract_variants(&contract)
                            .iter()
                            .filter_map(CatalogName::contract_path)
                            .collect();
                        report(catalog_error(
                            CatalogRule::Contract,
                            format!(
                                "its contract `{}` is abstract: a build names one of its \
                                 variants: {}",
                                reference.path(),
                                variants.join(", ")
                            ),
                        ));
                    }
                    Some(contract)
                }
                Err(e) => {
                    report(e);
                    None
                }
            },
        };

        // Facets: at most one group of each, and exactly one of each
        // required facet.
        let mut by_facet: BTreeMap<CatalogName, Vec<CatalogName>> = BTreeMap::new();
        for group in &memberships {
            if let Some(facet) = self.declaration(group).and_then(|d| self.group_facet(d)) {
                by_facet.entry(facet).or_default().push(group.clone());
            }
        }
        let mut facets = BTreeMap::new();
        for (facet, groups) in &by_facet {
            if groups.len() > 1 {
                report(catalog_error(
                    CatalogRule::Facet,
                    format!("it is in more than one `{facet}` group: {groups:?}"),
                ));
            }
            facets.insert(facet.clone(), groups[0].clone());
        }
        let mut required: BTreeSet<CatalogName> = BTreeSet::new();
        let wanted = chain
            .iter()
            .map(|l| (l.name, &l.class.required_facets))
            .chain([(name, &build.required_facets)]);
        for (owner, refs) in wanted {
            for facet in refs {
                match self.expect_ref(owner, facet, "facet", &[EntryKind::Facet]) {
                    Ok(d) => {
                        required.insert(d.name().clone());
                    }
                    Err(e) => report(e),
                }
            }
        }
        for facet in &required {
            if !facets.contains_key(facet) {
                report(catalog_error(
                    CatalogRule::Facet,
                    format!("it is in no `{facet}` group, and its chain requires one"),
                ));
            }
        }

        // Presets.
        let mut links: Vec<PresetLink<'_>> = chain
            .iter()
            .map(|l| PresetLink {
                owner: l.name,
                set: &l.class.presets,
            })
            .collect();
        links.push(PresetLink {
            owner: name,
            set: &build.presets,
        });
        let presets = match ResolvedPresetSet::fold(&links, &|l, a| self.axis_info(l.owner, a)) {
            Ok(set) => set,
            Err(found) => {
                for e in found {
                    report(e);
                }
                ResolvedPresetSet {
                    common: Value::Object(Default::default()),
                    presets: Vec::new(),
                    axes: Vec::new(),
                    default: Vec::new(),
                }
            }
        };
        for using in &presets.axes {
            if let Err(e) = self.check_scoped_use(name, &memberships, &using.axis.name, "use") {
                report(e);
            }
        }

        // Tags: the chain's, root first, then the build's; then implied.
        let mut tags: BTreeMap<CatalogName, ResolvedTag> = BTreeMap::new();
        let written = chain
            .iter()
            .map(|l| (l.name, &l.class.tags))
            .chain([(name, &build.tags)]);
        for (owner, values) in written {
            for (tag, value) in values {
                let tag = match self.check_tag_value(owner, tag, value) {
                    Ok(tag) => tag,
                    Err(e) => {
                        report(e);
                        continue;
                    }
                };
                if let Err(e) = self.check_scoped_use(name, &memberships, &tag, "set") {
                    report(e);
                }
                tags.insert(
                    tag,
                    ResolvedTag {
                        value: value.clone(),
                        source: TagSource::Written(owner.clone()),
                    },
                );
            }
        }
        for using in &presets.axes {
            tags.insert(
                using.axis.name.clone(),
                ResolvedTag {
                    value: axis_tag_value(using.axis.kind, using.axis.positive(), &using.offers),
                    source: TagSource::Axis,
                },
            );
        }
        for (facet, group) in &facets {
            tags.insert(
                facet.clone(),
                ResolvedTag {
                    value: Value::String(group.to_string()),
                    source: TagSource::Facet,
                },
            );
        }
        if let Some(path) = contract.as_ref().and_then(CatalogName::contract_path) {
            tags.insert(
                CatalogName::root(CONTRACT_TAG).expect("a legal segment"),
                ResolvedTag {
                    value: Value::String(path),
                    source: TagSource::Contract,
                },
            );
        }
        for group in &memberships {
            let Some(g_decl) = self.declaration(group) else {
                continue;
            };
            let Some(g) = g_decl.group() else {
                continue;
            };
            for tag in &g.requires {
                let Some(tag) = self.resolve_ref(group, tag).map(|d| d.name().clone()) else {
                    continue;
                };
                if !tags.contains_key(&tag) {
                    report(catalog_error(
                        CatalogRule::Tag,
                        format!("it has no value for `{tag}`, which its group `{group}` requires"),
                    ));
                }
            }
        }

        // The contract's slots, from its outermost ancestor in.
        let lineage = contract
            .as_ref()
            .map(|c| self.contract_lineage(c))
            .unwrap_or_default();
        let contract_path = contract
            .as_ref()
            .and_then(CatalogName::contract_path)
            .unwrap_or_default();
        let resource_slots: BTreeMap<&String, &ResourceSlot> = lineage
            .iter()
            .flat_map(|(_, c)| c.resources.iter())
            .collect();
        let arg_slots: BTreeMap<&String, (&CatalogName, &ArgSlot)> = lineage
            .iter()
            .flat_map(|(n, c)| c.args.iter().map(move |(k, slot)| (k, (n, slot))))
            .collect();

        // Bindings: the chain's, root first, then the build's.
        let mut bound: BTreeMap<String, (&CatalogName, &CatalogRef)> = BTreeMap::new();
        let mut given: BTreeMap<String, (&CatalogName, &Value)> = BTreeMap::new();
        let binders = chain
            .iter()
            .map(|l| (l.name, &l.class.resources, &l.class.args))
            .chain([(name, &build.resources, &build.args)]);
        for (owner, resources, args) in binders {
            for (slot, resource) in resources {
                bound.insert(slot.clone(), (owner, resource));
            }
            for (slot, value) in args {
                given.insert(slot.clone(), (owner, value));
            }
        }
        let mut resources = BTreeMap::new();
        for (slot, (owner, reference)) in &bound {
            if let Err(e) = check_segment(slot) {
                report(e);
                continue;
            }
            let spec = resource_slots.get(slot);
            if contract.is_some() && spec.is_none() {
                report(catalog_error(
                    CatalogRule::Dependency,
                    format!(
                        "the resource slot `{slot}` is not one of its contract `{contract_path}`'s"
                    ),
                ));
            }
            let resource = match self.check_dependency_target(owner, reference) {
                Ok(resource) => resource,
                Err(e) => {
                    report(e);
                    continue;
                }
            };
            let Some(entry) = self.resource(&resource).cloned() else {
                continue;
            };
            if let Some(spec) = spec
                && !spec.formats.is_empty()
                && !spec.formats.contains(&entry.format)
            {
                report(catalog_error(
                    CatalogRule::Dependency,
                    format!(
                        "the resource slot `{slot}` takes {}, and `{resource}` is `{}`",
                        spec.formats
                            .iter()
                            .map(|f| format!("`{f}`"))
                            .collect::<Vec<_>>()
                            .join(" or "),
                        entry.format
                    ),
                ));
            }
            resources.insert(
                slot.clone(),
                ResolvedResource {
                    resource,
                    entry,
                    set_by: (*owner).clone(),
                },
            );
        }
        let mut args = BTreeMap::new();
        for (slot, (owner, value)) in &given {
            let Some((declared_in, spec)) = arg_slots.get(slot) else {
                if contract.is_some() {
                    report(catalog_error(
                        CatalogRule::Dependency,
                        format!(
                            "the argument slot `{slot}` is not one of its contract \
                             `{contract_path}`'s"
                        ),
                    ));
                }
                continue;
            };
            match self
                .resolve_type(declared_in, &spec.type_expr)
                .map_err(|e| e.problem_message())
                .and_then(|ty| check_value(&ty, value))
            {
                Ok(()) => {
                    args.insert(
                        slot.clone(),
                        ResolvedArg {
                            value: (*value).clone(),
                            set_by: (*owner).clone(),
                        },
                    );
                }
                Err(why) => report(catalog_error(
                    CatalogRule::Dependency,
                    format!("the argument `{slot}`: {why}"),
                )),
            }
        }
        for (slot, spec) in &resource_slots {
            if !spec.optional && !bound.contains_key(*slot) {
                report(catalog_error(
                    CatalogRule::Dependency,
                    format!("it binds no `{slot}`, which its contract `{contract_path}` requires"),
                ));
            }
        }
        for (slot, (_, spec)) in &arg_slots {
            if !spec.optional && !given.contains_key(*slot) {
                report(catalog_error(
                    CatalogRule::Dependency,
                    format!(
                        "it gives no `{slot}` argument, which its contract `{contract_path}` \
                         requires"
                    ),
                ));
            }
        }

        if !errors.is_empty() {
            return Err(catalog_errors(errors));
        }
        Ok(ResolvedBuild {
            name: name.clone(),
            aliases: self.aliases_of(name),
            description: build.description.clone(),
            license: build.license.clone(),
            origin: build.origin.clone(),
            template_chain: chain.into_iter().map(|l| l.name.clone()).collect(),
            memberships,
            facets,
            contract,
            release,
            tags,
            presets,
            resources,
            args,
            hub_origins: self.hub_origins(),
        })
    }

    /// Resolves the template class `name`: its chain of base classes, the
    /// groups its builds join through the chain, its tags and its presets,
    /// folded root first. The catalog's checks ran when the stack imported
    /// it; this only reads what the state holds.
    ///
    /// # Errors
    /// [`CatalogRule::Reference`] when `name` isn't a template class,
    /// and every problem folding its chain finds.
    pub fn resolve_template_class(
        &self,
        name: &CatalogName,
    ) -> BunsenResult<ResolvedTemplateClass> {
        let decl = self
            .expect(name, "template class", &[EntryKind::TemplateClass])
            .map_err(|e| e.at(name))?;
        let CatalogEntry::TemplateClass(class) = decl.body() else {
            unreachable!("expected a template class");
        };
        let chain = self
            .template_chain(name, Some(&CatalogRef::absolute(name.clone())))
            .map_err(|e| e.at(name))?;
        let mut joined = Vec::new();
        for link in &chain {
            for group in &link.class.join {
                if let Some(d) = self.resolve_ref(link.name, group)
                    && d.kind().is_group()
                {
                    joined.push(d.name().clone());
                }
            }
        }
        let memberships = self.group_closure(joined);
        let links: Vec<PresetLink<'_>> = chain
            .iter()
            .map(|l| PresetLink {
                owner: l.name,
                set: &l.class.presets,
            })
            .collect();
        let presets = ResolvedPresetSet::fold(&links, &|l, a| self.axis_info(l.owner, a))
            .map_err(|found| catalog_errors(found.into_iter().map(|e| e.at(name)).collect()))?;
        let mut tags: BTreeMap<CatalogName, ResolvedTag> = BTreeMap::new();
        for link in &chain {
            for (tag, value) in &link.class.tags {
                if let Some(d) = self.resolve_ref(link.name, tag) {
                    tags.insert(
                        d.name().clone(),
                        ResolvedTag {
                            value: value.clone(),
                            source: TagSource::Written(link.name.clone()),
                        },
                    );
                }
            }
        }
        for using in &presets.axes {
            tags.insert(
                using.axis.name.clone(),
                ResolvedTag {
                    value: axis_tag_value(using.axis.kind, using.axis.positive(), &using.offers),
                    source: TagSource::Axis,
                },
            );
        }
        let contract = chain
            .iter()
            .rev()
            .find_map(|l| l.class.contract.as_ref())
            .and_then(|r| self.expect_contract(r).ok())
            .map(|d| d.name().clone());
        Ok(ResolvedTemplateClass {
            name: name.clone(),
            description: class.description.clone(),
            chain: chain.into_iter().map(|l| l.name.clone()).collect(),
            memberships,
            contract,
            tags,
            presets,
        })
    }

    /// A tag or axis declared in a scope may be set or used only by that
    /// scope's members (a group's) or by the scope itself (a build's).
    fn check_scoped_use(
        &self,
        build: &CatalogName,
        memberships: &BTreeSet<CatalogName>,
        declared: &CatalogName,
        verb: &str,
    ) -> Result<(), BunsenError> {
        let Some(scope) = declared.scope() else {
            return Ok(());
        };
        let allowed = memberships.contains(&scope) || &scope == build;
        if allowed {
            Ok(())
        } else {
            Err(catalog_error(
                CatalogRule::Tag,
                format!(
                    "`{declared}` is scoped to `{scope}`, which it is not in: only members may \
                     {verb} it"
                ),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::data::catalog::{
        presets::PresetSelection,
        testing::fixtures,
    };

    fn n(s: &str) -> CatalogName {
        CatalogName::parse(s).unwrap()
    }

    fn resolve(
        stack: &BunsenCatalogStack,
        query: &str,
    ) -> ResolvedBuild {
        let name = stack.find_build(query).unwrap();
        stack.resolve_build(&name).unwrap()
    }

    fn refused(yaml: &str) -> BunsenError {
        let mut stack = fixtures::example_stack().unwrap();
        stack.push_yaml("bad", yaml).unwrap_err()
    }

    #[test]
    fn test_a_hybrid_build() {
        let stack = fixtures::example_stack().unwrap();
        let b = resolve(&stack, "qwen3-0.6b");
        assert_eq!(
            b.template_chain,
            [n("template_class_root"), n("qwen/qwen3/hybrid")]
        );
        assert_eq!(b.facets[&n("family")], n("qwen/qwen3"));
        assert_eq!(b.facets[&n("publisher")], n("qwen"));
        assert_eq!(b.contract, Some(n("$contracts/qwen3-lm")));
        assert_eq!(b.tag("contract"), Some(&json!("qwen3-lm")));
        assert_eq!(b.tags[&n("contract")].source, TagSource::Contract);
        assert_eq!(b.tag("thinking"), Some(&json!("switch")));
        assert_eq!(b.tag("family"), Some(&json!("qwen/qwen3")));
        assert_eq!(b.tag("publisher"), Some(&json!("qwen")));
        assert_eq!(b.tag("instruct"), Some(&json!(true)));
        assert_eq!(b.tag("parameters"), Some(&json!(600000000u64)));
        assert_eq!(b.tags[&n("thinking")].source, TagSource::Axis);
        assert_eq!(b.tags[&n("family")].source, TagSource::Facet);
        assert_eq!(
            b.tags[&n("instruct")].source,
            TagSource::Written(n("qwen/qwen3/hybrid"))
        );

        let think = n("thinking");
        assert_eq!(b.presets.bool_member(&think, false).unwrap(), "no-thinking");
        let off = b
            .presets
            .select(json!({}), &PresetSelection::of(["no-thinking"]))
            .unwrap();
        assert_eq!(off.applied, ["no-thinking"]);
        assert_eq!(
            off.document,
            json!({
                "generation": {"max_new_tokens": 2048, "temperature": 0.7, "top_p": 0.8},
                "chat": {"enable_thinking": false},
            })
        );
    }

    #[test]
    fn test_a_build_narrows_its_class_to_a_fixed_mode() {
        let stack = fixtures::example_stack().unwrap();
        let base = resolve(&stack, "qwen3-0.6b-base");
        assert_eq!(base.tag("thinking"), Some(&json!("never")));
        assert_eq!(base.tag("instruct"), Some(&json!(false)));
        assert_eq!(
            base.tags[&n("instruct")].source,
            TagSource::Written(n("qwen/qwen3/qwen3-0.6b-base"))
        );
        // The class's `thinking` preset is pruned: not offered, not selectable.
        assert!(base.presets.get("thinking").is_none());
        let err = base.presets.bool_member(&n("thinking"), true).unwrap_err();
        assert!(
            err.problem_message().contains("fixed at `no-thinking`"),
            "{err}"
        );
        // Free presets survive the narrowing.
        assert!(base.presets.get("greedy").is_some());
    }

    #[test]
    fn test_a_child_class_fixes_the_mode() {
        let stack = fixtures::example_stack().unwrap();
        let b = resolve(&stack, "qwen3-4b-thinking-2507");
        assert_eq!(
            b.template_chain,
            [
                n("template_class_root"),
                n("qwen/qwen3/hybrid"),
                n("qwen/qwen3/thinking-only")
            ]
        );
        assert_eq!(b.tag("thinking"), Some(&json!("always")));
        assert!(b.memberships.contains(&n("qwen/qwen3/qwen3-2507")));
        let applied = b
            .presets
            .select(json!({}), &PresetSelection::default())
            .unwrap();
        assert_eq!(applied.applied, ["thinking"]);
    }

    #[test]
    fn test_a_categorical_axis_narrowed() {
        let stack = fixtures::example_stack().unwrap();
        let turbo = resolve(&stack, "large-v3-turbo");
        assert_eq!(turbo.tag("task"), Some(&json!(["transcribe"])));
        let err = turbo
            .presets
            .check_selection(&PresetSelection::of(["translate"]))
            .unwrap_err();
        assert!(
            err.problem_message().contains("fixed at `transcribe`"),
            "{err}"
        );
        assert!(turbo.release.is_none());

        let v3 = resolve(&stack, "large-v3");
        assert_eq!(v3.tag("task"), Some(&json!(["transcribe", "translate"])));
        assert_eq!(
            v3.release.as_ref().unwrap().released.to_string(),
            "2023-11-06"
        );
        assert_eq!(v3.tag("openai/whisper/timestamps"), Some(&json!(true)));
        // A categorical axis has no negation.
        assert_eq!(v3.presets.negate("transcribe"), None);
    }

    #[test]
    fn test_dependencies_resolve_through_the_namespace() {
        let stack = fixtures::example_stack().unwrap();
        let base = resolve(&stack, "qwen3-0.6b-base");
        let config = &base.resources["config"];
        assert_eq!(config.resource, n("qwen/qwen3/qwen3-0.6b/config"));
        assert_eq!(config.set_by, n("qwen/qwen3/qwen3-0.6b-base"));
        let v1 = resolve(&stack, "large-v1");
        assert_eq!(
            v1.resources["vocabulary"].resource,
            n("openai/whisper/vocabulary")
        );
    }

    #[test]
    fn test_tag_rules() {
        let errors = refused(
            r#"
qwen:
  $kind: namespace
  $merge: update
  qwen3:
    $kind: namespace
    $merge: update
    tagged:
      $kind: build
      $description: "Bad tags."
      $template_class: hybrid
      parameters: many
      thinking: always
      family: qwen3
      mood: happy
      openai/whisper/timestamps: true
"#,
        );
        for text in [
            "`\"many\"` is not a whole number",
            "implied by the axis use",
            "implied by facet membership",
            "no tag `mood`",
            "scoped to `openai/whisper`",
        ] {
            assert!(errors.mentions(text), "{text}: {errors}");
        }
    }

    #[test]
    fn test_group_requirements() {
        let errors = refused(
            r#"
qwen:
  $kind: namespace
  $merge: update
  qwen3:
    $kind: namespace
    $merge: update
    bare: { $kind: build, $description: "No parameters." }
"#,
        );
        assert!(
            errors.mentions("no value for `parameters`, which its group `qwen/qwen3` requires"),
            "{errors}"
        );
    }

    #[test]
    fn test_dependency_rules() {
        let errors = refused(
            r#"
qwen:
  $kind: namespace
  $merge: update
  qwen3:
    $kind: namespace
    $merge: update
    deps:
      $kind: build
      $description: "Bad dependencies."
      $template_class: hybrid
      $resources: { checkpoint: qwen/qwen3/qwen3-0.6b/checkpoint, weights: qwen/qwen3/qwen3-0.6b/checkpoint, tokenizer: qwen/qwen3/nope }
      parameters: 1
"#,
        );
        for text in [
            "`weights` is not one of its contract `qwen3-lm`'s",
            "no resource `qwen/qwen3/nope`",
            "no `config`, which its contract `qwen3-lm` requires",
        ] {
            assert!(errors.mentions(text), "{text}: {errors}");
        }

        // A later catalog may use resources an earlier one declares.
        let mut stack = fixtures::example_stack().unwrap();
        stack
            .push_yaml(
                "borrows",
                r#"
qwen:
  $kind: namespace
  $merge: update
  qwen3:
    $kind: namespace
    $merge: update
    borrower:
      $kind: build
      $description: "Uses resources declared below it."
      $template_class: hybrid
      $resources: { checkpoint: qwen/qwen3/qwen3-0.6b/checkpoint, config: qwen/qwen3/qwen3-0.6b/config, tokenizer: qwen/qwen3/qwen3-0.6b/tokenizer }
      parameters: 1
"#,
            )
            .unwrap();
        let b = stack.resolve_build(&n("qwen/qwen3/borrower")).unwrap();
        assert_eq!(
            b.resources["tokenizer"].resource,
            n("qwen/qwen3/qwen3-0.6b/tokenizer")
        );
    }

    #[test]
    fn test_widening_a_class_is_refused() {
        let errors = refused(
            r#"
qwen:
  $kind: namespace
  $merge: update
  qwen3:
    $kind: namespace
    $merge: update
    wide:
      $kind: build
      $description: "Tries to switch a thinking-only class."
      $template_class: thinking-only
      parameters: 1
      $presets:
        $axes:
          thinking: { $offers: [thinking, no-thinking], $default: thinking }
"#,
        );
        assert!(errors.mentions("may only narrow"), "{errors}");
    }

    #[test]
    fn test_a_build_names_one_contract() {
        // A build's own `$contract` overrides its class's; the slots it
        // binds must be the new contract's.
        let other = refused(
            r#"
qwen:
  $kind: namespace
  $merge: update
  qwen3:
    $kind: namespace
    $merge: update
    elsewhere:
      $kind: build
      $description: "Names Whisper's contract over its class's."
      $template_class: hybrid
      $contract: whisper-asr
      $resources: { config: qwen3-0.6b/config }
      parameters: 1
"#,
        );
        assert!(
            other.mentions("the resource slot `config` is not one of its contract `whisper-asr`'s"),
            "{other}"
        );
        assert!(
            other.mentions("it binds no `checkpoint`, which its contract `whisper-asr` requires"),
            "{other}"
        );
        let none = refused(
            "qwen:\n  $kind: namespace\n  $merge: update\n  qwen3:\n    $kind: namespace\n    $merge: update\n    free: { $kind: build, $description: \"F.\", parameters: 1 }\n",
        );
        assert!(none.mentions("it names no contract"), "{none}");
        assert!(none.has(CatalogRule::Contract));
        let missing = refused(
            "qwen:\n  $kind: namespace\n  $merge: update\n  qwen3:\n    $kind: namespace\n    $merge: update\n    lost: { $kind: build, $description: \"L.\", $template_class: hybrid, $contract: qwen9-lm, parameters: 1 }\n",
        );
        assert!(
            missing.mentions("`$contracts` has no contract `qwen9-lm`"),
            "{missing}"
        );
    }

    /// A base catalog of variant contracts, typed arguments and a family.
    const VARIANTS: &str = r#"
$contracts:
  asr:
    $kind: contract
    $abstract: true
    $description: "Shared by both variants."
    $resources:
      checkpoint: { $format: [safetensors, archive] }
      vocabulary: { $format: [tokenizer.json] }
    geometry:
      $kind: type
      $description: "The dims a loader can't read from an archive."
      $type:
        $record:
          layers: { $type: $count }
          width: { $type: $count, $optional: true }
    probed:
      $kind: contract
      $description: "Its geometry read from config.json."
      $resources: { config: { $format: [json] } }
    stated:
      $kind: contract
      $description: "Its geometry stated."
      $args: { geometry: { $type: geometry, $description: "The dims." } }
family: { $kind: facet, $description: "Families." }
template_class_root: { $kind: template_class, $description: "The root." }
lab:
  $kind: namespace
  $facet: family
  $description: "A lab."
  w: { $kind: resource, $file: w.bin, $format: archive, $sources: [{ $url: "https://x/w" }] }
  t: { $kind: resource, $file: t.json, $format: tokenizer.json, $sources: [{ $url: "https://x/t" }] }
  c: { $kind: resource, $file: config.json, $format: json, $sources: [{ $url: "https://x/c" }] }
"#;

    fn variants() -> BunsenCatalogStack {
        let mut stack = BunsenCatalogStack::new();
        stack.push_yaml("base", VARIANTS).unwrap();
        stack
    }

    fn build_in(
        stack: &BunsenCatalogStack,
        build: &str,
    ) -> BunsenResult<BunsenCatalogStack> {
        let mut stack = stack.clone();
        stack.push_yaml(
            "builds",
            &format!("lab:\n  $kind: namespace\n  $merge: update\n  {build}\n"),
        )?;
        Ok(stack)
    }

    #[test]
    fn test_variants_args_and_types() {
        let base = variants();
        // A stated build binds the shared slots and the variant's argument.
        let stack = build_in(
            &base,
            "x: { $kind: build, $description: \"X.\", $contract: asr/stated, $resources: { checkpoint: w, vocabulary: t }, $args: { geometry: { layers: 18, extra: true } } }",
        )
        .unwrap();
        let x = stack.resolve_build(&n("lab/x")).unwrap();
        assert_eq!(x.contract, Some(n("$contracts/asr/stated")));
        assert_eq!(x.tag("contract"), Some(&json!("asr/stated")));
        assert_eq!(
            x.args["geometry"].value,
            json!({"layers": 18, "extra": true})
        );
        assert_eq!(x.resources["checkpoint"].resource, n("lab/w"));
        assert_eq!(stack.builds_of_contract(&n("$contracts/asr")), [n("lab/x")]);
        // A probed build binds `config` instead.
        assert!(
            build_in(
                &base,
                "probed-build: { $kind: build, $description: \"P.\", $contract: asr/probed, $resources: { checkpoint: w, vocabulary: t, config: c } }",
            )
            .is_ok()
        );
        let refused = |build: &str| build_in(&base, build).unwrap_err();
        // The abstract parent can't be named.
        let abstract_ = refused(
            "z: { $kind: build, $description: \"Z.\", $contract: asr, $resources: { checkpoint: w, vocabulary: t } }",
        );
        assert!(
            abstract_.mentions(
                "its contract `asr` is abstract: a build names one of its variants: asr/probed, \
                 asr/stated"
            ),
            "{abstract_}"
        );
        // An argument is checked against its type; one the variant lacks is
        // refused, and so is one it requires and the build leaves out.
        let typed = refused(
            "z: { $kind: build, $description: \"Z.\", $contract: asr/stated, $resources: { checkpoint: w, vocabulary: t }, $args: { geometry: { width: 3 } } }",
        );
        assert!(
            typed.mentions("the argument `geometry`: it has no `layers`"),
            "{typed}"
        );
        let unknown = refused(
            "z: { $kind: build, $description: \"Z.\", $contract: asr/probed, $resources: { checkpoint: w, vocabulary: t, config: c }, $args: { geometry: { layers: 1 } } }",
        );
        assert!(
            unknown
                .mentions("the argument slot `geometry` is not one of its contract `asr/probed`'s"),
            "{unknown}"
        );
        let absent = refused(
            "z: { $kind: build, $description: \"Z.\", $contract: asr/stated, $resources: { checkpoint: w, vocabulary: t } }",
        );
        assert!(
            absent.mentions(
                "it gives no `geometry` argument, which its contract `asr/stated` requires"
            ),
            "{absent}"
        );
        // A slot's format is the resource's.
        let format = refused(
            "z: { $kind: build, $description: \"Z.\", $contract: asr/probed, $resources: { checkpoint: c, vocabulary: t, config: c } }",
        );
        assert!(
            format.mentions("the resource slot `checkpoint` takes `safetensors` or `archive`, and `lab/c` is `json`"),
            "{format}"
        );
    }

    #[test]
    fn test_contract_declarations_are_checked() {
        let base = |contracts: &str| {
            let mut stack = BunsenCatalogStack::new();
            stack
                .push_yaml("base", &format!("$contracts:\n{contracts}"))
                .unwrap_err()
        };
        let restated = base(
            "  a:\n    $kind: contract\n    $description: \"A.\"\n    $resources: { w: {} }\n    b: { $kind: contract, $description: \"B.\", $resources: { w: {} } }\n",
        );
        assert!(
            restated
                .mentions("the slot `w` is already its contract `a`'s: a variant only adds slots"),
            "{restated}"
        );
        let hollow = base("  a: { $kind: contract, $abstract: true, $description: \"A.\" }\n");
        assert!(
            hollow.mentions("it is abstract, and has no variant a build can name"),
            "{hollow}"
        );
        let untyped = base(
            "  a: { $kind: contract, $description: \"A.\", $args: { g: { $type: nothing } } }\n",
        );
        assert!(
            untyped.mentions("the argument slot `g`: no type `nothing`"),
            "{untyped}"
        );
        let cyclic = base(
            "  a:\n    $kind: contract\n    $description: \"A.\"\n    t: { $kind: type, $description: \"T.\", $type: { $list: t } }\n",
        );
        assert!(
            cyclic.mentions("the type `$contracts/a/t` names itself"),
            "{cyclic}"
        );
        // Only the base catalog declares `$contracts`, and it must.
        let mut stack = BunsenCatalogStack::new();
        let missing = stack
            .push_yaml("base", "$description: \"No contracts.\"\n")
            .unwrap_err();
        assert!(
            missing.mentions("the base catalog declares `$contracts`"),
            "{missing}"
        );
        let late = refused("$contracts: {}\n");
        assert!(
            late.mentions("only the base catalog declares `$contracts`"),
            "{late}"
        );
        let misplaced = refused("c: { $kind: contract, $description: \"C.\" }\n");
        assert!(
            misplaced.mentions("in the base catalog's `$contracts`"),
            "{misplaced}"
        );
    }

    #[test]
    fn test_two_release_groups_are_refused() {
        let errors = refused(
            r#"
openai:
  $kind: namespace
  $merge: update
  whisper:
    $kind: namespace
    $merge: update
    other-line: { $kind: release, $description: "Another sequence." }
    twice:
      $kind: build
      $description: "In two release groups."
      $template_class: multilingual
      $join: [{ $group: openai/whisper/large/release, $released: "2030-01" }, { $group: openai/whisper/other-line, $released: "2030-02" }]
      parameters: 1
"#,
        );
        assert!(errors.mentions("two release groups"), "{errors}");
    }
}
