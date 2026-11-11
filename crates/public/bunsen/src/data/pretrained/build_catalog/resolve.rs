//! Resolving a build: [`ResolvedBuild`], everything the catalog says about
//! it once its template-class chain, its memberships and its stack's
//! declarations are taken into account.

use alloc::collections::{
    BTreeMap,
    BTreeSet,
};

use serde_json::Value;

use crate::data::pretrained::build_catalog::{
    document::{
        CatalogEntry,
        EntryKind,
        Membership,
        ResourceEntry,
    },
    error::{
        CatalogError,
        CatalogErrorKind,
        CatalogErrors,
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
    stack::PretrainedCatalogStack,
    tags::axis_tag_value,
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

/// One resolved dependency.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedDependency {
    /// The resource's full name.
    pub resource: CatalogName,
    /// The resource's declaration.
    pub entry: ResourceEntry,
    /// Who set the key: the lowest template class or build that does.
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
    /// Its contract: the one `contract` group it is in.
    pub contract: Option<CatalogName>,
    /// Its release membership, if it has one.
    pub release: Option<ResolvedRelease>,
    /// Its tags, written and implied, by tag name.
    pub tags: BTreeMap<CatalogName, ResolvedTag>,
    /// Its presets.
    pub presets: ResolvedPresetSet,
    /// Its dependencies, by key.
    pub dependencies: BTreeMap<String, ResolvedDependency>,
}

impl ResolvedBuild {
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
    /// Its tags, written along the chain and implied by its axis uses.
    pub tags: BTreeMap<CatalogName, ResolvedTag>,
    /// Its presets, folded along the chain.
    pub presets: ResolvedPresetSet,
}

impl PretrainedCatalogStack {
    /// Resolves the build `name`.
    ///
    /// # Errors
    /// Every problem found: a reference that doesn't resolve, a facet
    /// broken, a tag wrong, a preset set that doesn't fold, a dependency
    /// key its contract doesn't declare, and so on.
    pub fn resolve_build(
        &self,
        name: &CatalogName,
    ) -> Result<ResolvedBuild, CatalogErrors> {
        let mut errors: Vec<CatalogError> = Vec::new();
        let decl = self
            .expect(name, "build", &[EntryKind::Build])
            .map_err(|e| CatalogErrors::from(e.at(name)))?;
        let CatalogEntry::Build(build) = decl.body() else {
            unreachable!("expected a build");
        };
        let mut report = |e: CatalogError| errors.push(e.at(name));

        if build.description.trim().is_empty() {
            report(CatalogError::new(
                CatalogErrorKind::Structure,
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
                        report(CatalogError::new(
                            CatalogErrorKind::Release,
                            "it is in two release groups; a build is in at most one",
                        ));
                    }
                    release = Some(ResolvedRelease { group, released });
                }
            }
        }
        let memberships = self.build_memberships(name);
        // The contract: a build is in exactly one.
        let contracts: Vec<CatalogName> = memberships
            .iter()
            .filter(|g| {
                self.declaration(g)
                    .is_some_and(|d| d.kind() == EntryKind::Contract)
            })
            .cloned()
            .collect();
        match contracts.len() {
            0 => report(CatalogError::new(
                CatalogErrorKind::Contract,
                "it is in no contract; every build is in exactly one",
            )),
            1 => {}
            _ => report(CatalogError::new(
                CatalogErrorKind::Contract,
                format!("it is in more than one contract: {contracts:?}"),
            )),
        }
        let contract = contracts.first().cloned();

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
                report(CatalogError::new(
                    CatalogErrorKind::Facet,
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
                report(CatalogError::new(
                    CatalogErrorKind::Facet,
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
        if let Some(contract) = &contract {
            tags.insert(
                CatalogName::root(CONTRACT_TAG).expect("a legal segment"),
                ResolvedTag {
                    value: Value::String(contract.to_string()),
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
                    report(CatalogError::new(
                        CatalogErrorKind::Tag,
                        format!("it has no value for `{tag}`, which its group `{group}` requires"),
                    ));
                }
            }
        }

        // Dependencies: the chain's, root first, then the build's.
        let mut keys: BTreeMap<String, (&CatalogName, &CatalogRef)> = BTreeMap::new();
        let set = chain
            .iter()
            .map(|l| (l.name, &l.class.dependencies))
            .chain([(name, &build.dependencies)]);
        for (owner, deps) in set {
            for (key, resource) in deps {
                keys.insert(key.clone(), (owner, resource));
            }
        }
        let mut vocabulary: BTreeSet<&String> = BTreeSet::new();
        let mut required_keys: Vec<(&CatalogName, &String)> = Vec::new();
        if let Some(contract) = &contract
            && let Some(keys) = self
                .declaration(contract)
                .and_then(|d| d.group())
                .and_then(|g| g.dependency_keys.as_ref())
        {
            vocabulary.extend(keys.required.iter().chain(&keys.optional));
            required_keys.extend(keys.required.iter().map(|k| (contract, k)));
        }
        let mut dependencies = BTreeMap::new();
        for (key, (owner, resource)) in &keys {
            if let Err(e) = check_segment(key) {
                report(e);
                continue;
            }
            if let Some(contract) = &contract
                && !vocabulary.contains(key)
            {
                report(CatalogError::new(
                    CatalogErrorKind::Dependency,
                    format!("the dependency key `{key}` is not one of its contract `{contract}`'s"),
                ));
            }
            let resource = match self.check_dependency_target(owner, resource) {
                Ok(resource) => resource,
                Err(e) => {
                    report(e);
                    continue;
                }
            };
            let Some(entry) = self.resource(&resource).cloned() else {
                continue;
            };
            dependencies.insert(
                key.clone(),
                ResolvedDependency {
                    resource,
                    entry,
                    set_by: (*owner).clone(),
                },
            );
        }
        for (contract, key) in required_keys {
            if !keys.contains_key(key) {
                report(CatalogError::new(
                    CatalogErrorKind::Dependency,
                    format!("it has no `{key}`, which its contract `{contract}` requires"),
                ));
            }
        }

        if !errors.is_empty() {
            return Err(CatalogErrors(errors));
        }
        Ok(ResolvedBuild {
            name: name.clone(),
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
            dependencies,
        })
    }

    /// Resolves the template class `name`: its chain of base classes, the
    /// groups its builds join through the chain, its tags and its presets,
    /// folded root first. The catalog's checks ran when the stack imported
    /// it; this only reads what the state holds.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Reference`] when `name` isn't a template class,
    /// and every problem folding its chain finds.
    pub fn resolve_template_class(
        &self,
        name: &CatalogName,
    ) -> Result<ResolvedTemplateClass, CatalogErrors> {
        let decl = self
            .expect(name, "template class", &[EntryKind::TemplateClass])
            .map_err(|e| CatalogErrors::from(e.at(name)))?;
        let CatalogEntry::TemplateClass(class) = decl.body() else {
            unreachable!("expected a template class");
        };
        let chain = self
            .template_chain(name, Some(&CatalogRef::absolute(name.clone())))
            .map_err(|e| CatalogErrors::from(e.at(name)))?;
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
            .map_err(|found| CatalogErrors(found.into_iter().map(|e| e.at(name)).collect()))?;
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
        Ok(ResolvedTemplateClass {
            name: name.clone(),
            description: class.description.clone(),
            chain: chain.into_iter().map(|l| l.name.clone()).collect(),
            memberships,
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
    ) -> Result<(), CatalogError> {
        let Some(scope) = declared.scope() else {
            return Ok(());
        };
        let allowed = memberships.contains(&scope) || &scope == build;
        if allowed {
            Ok(())
        } else {
            Err(CatalogError::new(
                CatalogErrorKind::Tag,
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
    use crate::data::pretrained::build_catalog::{
        fixtures,
        presets::PresetSelection,
    };

    fn n(s: &str) -> CatalogName {
        CatalogName::parse(s).unwrap()
    }

    fn resolve(
        stack: &PretrainedCatalogStack,
        query: &str,
    ) -> ResolvedBuild {
        let name = stack.find_build(query).unwrap();
        stack.resolve_build(&name).unwrap()
    }

    fn refused(yaml: &str) -> CatalogErrors {
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
        assert_eq!(b.contract, Some(n("qwen3-lm")));
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
        assert!(err.message.contains("fixed at `no-thinking`"), "{err}");
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
        assert!(err.message.contains("fixed at `transcribe`"), "{err}");
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
        let config = &base.dependencies["config"];
        assert_eq!(config.resource, n("qwen/qwen3/qwen3-0.6b/config"));
        assert_eq!(config.set_by, n("qwen/qwen3/qwen3-0.6b-base"));
        let v1 = resolve(&stack, "large-v1");
        assert_eq!(
            v1.dependencies["vocabulary"].resource,
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
      $dependencies: { checkpoint: qwen/qwen3/qwen3-0.6b/checkpoint, weights: qwen/qwen3/qwen3-0.6b/checkpoint, tokenizer: qwen/qwen3/nope }
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
      $dependencies: { checkpoint: qwen/qwen3/qwen3-0.6b/checkpoint, config: qwen/qwen3/qwen3-0.6b/config, tokenizer: qwen/qwen3/qwen3-0.6b/tokenizer }
      parameters: 1
"#,
            )
            .unwrap();
        let b = stack.resolve_build(&n("qwen/qwen3/borrower")).unwrap();
        assert_eq!(
            b.dependencies["tokenizer"].resource,
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
    fn test_exactly_one_contract() {
        let two = refused(
            r#"
qwen:
  $kind: namespace
  $merge: update
  qwen3:
    $kind: namespace
    $merge: update
    both:
      $kind: build
      $description: "In two contracts."
      $template_class: hybrid
      $join: [whisper-asr]
      parameters: 1
"#,
        );
        assert!(two.mentions("more than one contract"), "{two}");
        let none = refused(
            "qwen:\n  $kind: namespace\n  $merge: update\n  qwen3:\n    $kind: namespace\n    $merge: update\n    free: { $kind: build, $description: \"F.\", parameters: 1 }\n",
        );
        assert!(none.mentions("in no contract"), "{none}");
        assert!(none.has(CatalogErrorKind::Contract));
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
