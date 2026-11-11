//! [`PretrainedCatalogStack`]: catalogs imported one after another into one
//! materialized state.
//!
//! Importing a catalog applies its occurrences, in order, to a copy of the
//! state: `$merge: deny` declares a name (an existing one is a conflict),
//! `$merge: update` updates it as its kind defines, and `$merge: replace`
//! substitutes the occurrence's own fields for the entry's ([`MergeMode`]).
//! The copy is then validated as a whole, and committed only if it is
//! clean. The state before the import was valid, so every problem found is
//! the importing catalog's, and it is refused whole.
//!
//! Queries read the state. References ([`CatalogRef`]) resolve from the
//! entry they are written in, climbing the namespace to the root; the
//! nearest declared name is taken, whatever its kind, and a wrong kind is
//! an error. A name is unique in its scope, and may repeat in other scopes,
//! nested or not. A catalog may not change what an earlier catalog's
//! reference resolves to: a name that would capture one, by being nearer
//! to it, is refused. Only a `replace` changes an entry, and it does so for
//! every reference to it.
//!
//! The imported catalogs are kept, as read, for provenance: each entry
//! records the catalogs that declared, updated and replaced it.

use alloc::collections::{
    BTreeMap,
    BTreeSet,
};
use core::cell::RefCell;
use std::path::{
    Path,
    PathBuf,
};

use crate::data::pretrained::build_catalog::{
    document::{
        CatalogEntry,
        EntryKind,
        GroupEntry,
        Membership,
        MergeMode,
        PretrainedCatalog,
        ResourceEntry,
        TEMPLATE_CLASS_ROOT,
        TemplateClassEntry,
    },
    error::{
        CatalogError,
        CatalogErrorKind,
        CatalogErrors,
        CatalogResult,
    },
    layer::{
        LayerEntry,
        PretrainedCatalogLayer,
    },
    name::{
        CatalogName,
        CatalogRef,
        check_segment,
    },
    release_time::ReleaseTime,
    tags::check_value,
};

/// One change an import made to an entry: the catalog, and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryChange {
    /// The index of the imported catalog.
    pub layer: usize,
    /// How its occurrence met the entry.
    pub merge: MergeMode,
}

/// One entry of the state: its name, its fields as merged, and the changes
/// that made them.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedEntry {
    /// The full name.
    pub name: CatalogName,
    /// The entry's fields after every import: no children (they are entries
    /// of their own), and `merge` at its default.
    pub entry: CatalogEntry,
    /// The changes, in import order; the first declared it.
    pub history: Vec<EntryChange>,
}

/// An entry of the state.
#[derive(Clone, Copy, Debug)]
pub struct EntryRef<'a> {
    /// The entry.
    pub entry: &'a MaterializedEntry,
}

impl<'a> EntryRef<'a> {
    /// The full name.
    pub fn name(&self) -> &'a CatalogName {
        &self.entry.name
    }

    /// The kind.
    pub fn kind(&self) -> EntryKind {
        self.entry.entry.kind()
    }

    /// The entry's merged fields.
    pub fn body(&self) -> &'a CatalogEntry {
        &self.entry.entry
    }

    /// The group fields, for a `namespace` or `contract`.
    pub fn group(&self) -> Option<&'a GroupEntry> {
        self.entry.entry.as_group()
    }

    /// The changes that made the entry, in import order.
    pub fn history(&self) -> &'a [EntryChange] {
        &self.entry.history
    }

    /// The index of the catalog that declared it.
    pub fn declared_in(&self) -> usize {
        self.entry.history[0].layer
    }
}

/// An occurrence of a name in an imported catalog, as written.
#[derive(Clone, Copy, Debug)]
pub struct Occurrence<'a> {
    /// The index of the catalog.
    pub layer: usize,
    /// The flat entry, as written.
    pub entry: &'a LayerEntry,
}

/// One link of a template-class chain.
#[derive(Clone, Copy, Debug)]
pub struct TemplateLink<'a> {
    /// The class's full name.
    pub name: &'a CatalogName,
    /// The class.
    pub class: &'a TemplateClassEntry,
}

/// Catalogs imported in order, and the state they materialize.
#[derive(Clone, Debug, Default)]
pub struct PretrainedCatalogStack {
    layers: Vec<PretrainedCatalogLayer>,
    entries: Vec<MaterializedEntry>,
    index: BTreeMap<CatalogName, usize>,
    /// Every reference the imported catalogs' validation resolved, by the
    /// entry it is written in, and the entry it resolved to.
    resolved: BTreeMap<(CatalogName, CatalogRef), CatalogName>,
    /// While a catalog is validated, the references it resolves.
    recording: RefCell<Option<BTreeMap<(CatalogName, CatalogRef), CatalogName>>>,
}

impl PretrainedCatalogStack {
    /// An empty stack.
    pub fn new() -> Self {
        Self::default()
    }

    /// Imports `document` as the next catalog, labelled `label`.
    ///
    /// # Errors
    /// Every problem found; the stack is left as it was.
    pub fn push(
        &mut self,
        label: impl Into<String>,
        path: Option<PathBuf>,
        document: PretrainedCatalog,
    ) -> Result<(), CatalogErrors> {
        let layer = PretrainedCatalogLayer::new(label, path, document)?;
        let label = layer.label().to_string();
        let declares_contracts = layer.document().contracts.is_some();
        let refusal = match (self.layers.is_empty(), declares_contracts) {
            (true, false) => Some(
                "the base catalog declares `$contracts`, the contracts every build names (it \
                 may be empty)",
            ),
            (false, true) => Some(
                "only the base catalog declares `$contracts`: a contract is an interface \
                 bunsen's loaders implement, and a later catalog can't add one",
            ),
            _ => None,
        };
        if let Some(why) = refusal {
            return Err(CatalogErrors::from(
                CatalogError::new(CatalogErrorKind::Contract, why).in_layer(&label),
            ));
        }
        let mut next = self.clone();
        next.layers.push(layer);
        let mut errors = next.apply_top();
        if errors.is_empty() {
            errors = next.check_captures();
        }
        if errors.is_empty() {
            next.forget_replaced();
            *next.recording.borrow_mut() = Some(BTreeMap::new());
            errors = next.validate_top();
            let recorded = next.recording.borrow_mut().take().unwrap_or_default();
            next.resolved.extend(recorded);
        }
        if errors.is_empty() {
            *self = next;
            Ok(())
        } else {
            Err(CatalogErrors(
                errors.into_iter().map(|e| e.in_layer(&label)).collect(),
            ))
        }
    }

    /// Parses `yaml` and [`push`](Self::push)es it.
    ///
    /// # Errors
    /// A [`CatalogErrorKind::Parse`] problem, or every problem `push` finds.
    pub fn push_yaml(
        &mut self,
        label: impl Into<String>,
        yaml: &str,
    ) -> Result<(), CatalogErrors> {
        let label = label.into();
        let document = PretrainedCatalog::from_yaml(yaml)
            .map_err(|e| CatalogErrors::from(e.in_layer(&label)))?;
        self.push(label, None, document)
    }

    /// Reads a YAML (or, by its `.json` extension, JSON) file and pushes it,
    /// labelled by its file stem.
    ///
    /// # Errors
    /// A problem reading or parsing it, or every problem `push` finds.
    pub fn push_file(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<(), CatalogErrors> {
        let path = path.as_ref();
        let label = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let text = std::fs::read_to_string(path).map_err(|e| {
            CatalogErrors::from(
                CatalogError::new(CatalogErrorKind::Parse, format!("{}: {e}", path.display()))
                    .in_layer(&label),
            )
        })?;
        let document = if path.extension().is_some_and(|x| x == "json") {
            PretrainedCatalog::from_json(&text)
        } else {
            PretrainedCatalog::from_yaml(&text)
        }
        .map_err(|e| CatalogErrors::from(e.in_layer(&label)))?;
        self.push(label, Some(path.to_path_buf()), document)
    }

    /// The imported catalogs, in import order.
    pub fn layers(&self) -> &[PretrainedCatalogLayer] {
        &self.layers
    }

    /// Every occurrence of `name` in the imported catalogs, in import order.
    pub fn occurrences<'a>(
        &'a self,
        name: &'a CatalogName,
    ) -> impl Iterator<Item = Occurrence<'a>> + 'a {
        self.layers.iter().enumerate().flat_map(move |(layer, l)| {
            l.entries()
                .iter()
                .filter(move |e| &e.name == name)
                .map(move |entry| Occurrence { layer, entry })
        })
    }

    /// Every entry of the state, in the order the names were declared.
    pub fn declarations(&self) -> impl Iterator<Item = EntryRef<'_>> {
        self.entries.iter().map(|entry| EntryRef { entry })
    }

    /// The entry `name`.
    pub fn declaration(
        &self,
        name: &CatalogName,
    ) -> Option<EntryRef<'_>> {
        self.index.get(name).map(|&i| EntryRef {
            entry: &self.entries[i],
        })
    }

    /// The resource `name`, as merged.
    pub fn resource(
        &self,
        name: &CatalogName,
    ) -> Option<&ResourceEntry> {
        match self.declaration(name)?.body() {
            CatalogEntry::Resource(r) => Some(r),
            _ => None,
        }
    }

    /// Resolves `reference`, written in the entry `from`: the nearest of
    /// its candidates that is declared.
    pub fn resolve_ref(
        &self,
        from: &CatalogName,
        reference: &CatalogRef,
    ) -> Option<EntryRef<'_>> {
        let found = reference
            .candidates(Some(from))
            .into_iter()
            .find_map(|candidate| self.declaration(&candidate));
        if let (Some(found), Some(recording)) = (found, self.recording.borrow_mut().as_mut()) {
            recording.insert((from.clone(), reference.clone()), found.name().clone());
        }
        found
    }

    /// The references earlier catalogs resolved that now resolve elsewhere:
    /// a name the top catalog adds is nearer to them than what they named.
    fn check_captures(&self) -> Vec<CatalogError> {
        self.resolved
            .iter()
            .filter_map(|((from, reference), target)| {
                let now = self.resolve_ref(from, reference)?;
                (now.name() != target).then(|| {
                    CatalogError::new(
                        CatalogErrorKind::Conflict,
                        format!(
                            "it would capture `{reference}`, written in `{from}`, which resolves \
                             to `{target}`: a catalog can't change an earlier catalog's reference"
                        ),
                    )
                    .at(now.name())
                })
            })
            .collect()
    }

    /// Drops the recorded references of the entries the top catalog
    /// replaces: their validation records the new fields' references.
    fn forget_replaced(&mut self) {
        let top = self.layers.len() - 1;
        let replaced: BTreeSet<CatalogName> = self.layers[top]
            .entries()
            .iter()
            .filter(|o| o.merge() == MergeMode::Replace)
            .map(|o| o.name.clone())
            .collect();
        self.resolved
            .retain(|(from, _), _| !replaced.contains(from));
    }

    /// Resolves `reference` (as [`resolve_ref`](Self::resolve_ref)) to an
    /// entry of one of `kinds`.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Reference`] when it resolves to nothing, or the
    /// nearest entry it names is of another kind.
    pub fn expect_ref(
        &self,
        from: &CatalogName,
        reference: &CatalogRef,
        what: &str,
        kinds: &[EntryKind],
    ) -> CatalogResult<EntryRef<'_>> {
        match self.resolve_ref(from, reference) {
            Some(decl) if kinds.contains(&decl.kind()) => Ok(decl),
            Some(decl) => Err(CatalogError::new(
                CatalogErrorKind::Reference,
                format!(
                    "`{reference}` resolves to `{}`, which is {}, not {}",
                    decl.name(),
                    with_article(decl.kind().as_str()),
                    with_article(what)
                ),
            )),
            None => Err(CatalogError::new(
                CatalogErrorKind::Reference,
                format!("no {what} `{reference}` is visible from `{from}`"),
            )),
        }
    }

    /// Resolves the full name `name` to an entry of one of `kinds`.
    pub(crate) fn expect(
        &self,
        name: &CatalogName,
        what: &str,
        kinds: &[EntryKind],
    ) -> CatalogResult<EntryRef<'_>> {
        match self.declaration(name) {
            Some(decl) if kinds.contains(&decl.kind()) => Ok(decl),
            Some(decl) => Err(CatalogError::new(
                CatalogErrorKind::Reference,
                format!(
                    "`{name}` is {}, not {}",
                    with_article(decl.kind().as_str()),
                    with_article(what)
                ),
            )),
            None => Err(CatalogError::new(
                CatalogErrorKind::Reference,
                format!("no {what} `{name}`"),
            )),
        }
    }

    /// The entries of `kind`.
    pub fn of_kind(
        &self,
        kind: EntryKind,
    ) -> Vec<EntryRef<'_>> {
        self.declarations().filter(|e| e.kind() == kind).collect()
    }

    /// The direct children of `name`: the entries in its scope.
    pub fn children(
        &self,
        name: &CatalogName,
    ) -> Vec<EntryRef<'_>> {
        self.declarations()
            .filter(|e| e.name().scope().as_ref() == Some(name))
            .collect()
    }

    /// The facet a group declares itself in, resolved from the group.
    pub fn group_facet(
        &self,
        group: EntryRef<'_>,
    ) -> Option<CatalogName> {
        let facet = group.group()?.facet.as_ref()?;
        self.resolve_ref(group.name(), facet)
            .filter(|d| d.kind() == EntryKind::Facet)
            .map(|d| d.name().clone())
    }

    /// The groups of `facet` ("all families"): the groups that declare it
    /// as their own facet. Their members are not among them.
    pub fn facet_groups(
        &self,
        facet: &CatalogName,
    ) -> Vec<EntryRef<'_>> {
        self.declarations()
            .filter(|e| self.group_facet(*e).as_ref() == Some(facet))
            .collect()
    }

    /// The builds that are members of `group`, directly or through other
    /// groups or their template-class chains.
    pub fn members_of(
        &self,
        group: &CatalogName,
    ) -> Vec<CatalogName> {
        self.of_kind(EntryKind::Build)
            .into_iter()
            .filter(|b| self.build_memberships(b.name()).contains(group))
            .map(|b| b.name().clone())
            .collect()
    }

    /// The builds in some group of `facet` ("things in families", as
    /// against [`facet_groups`](Self::facet_groups), "all families").
    pub fn builds_in_facet(
        &self,
        facet: &CatalogName,
    ) -> Vec<CatalogName> {
        let groups: BTreeSet<CatalogName> = self
            .facet_groups(facet)
            .iter()
            .map(|g| g.name().clone())
            .collect();
        self.of_kind(EntryKind::Build)
            .into_iter()
            .filter(|b| {
                self.build_memberships(b.name())
                    .iter()
                    .any(|m| groups.contains(m))
            })
            .map(|b| b.name().clone())
            .collect()
    }

    /// The groups an entry is directly in: its naming scope, when that is a
    /// group, and the groups it joins. A build's template-class chain
    /// contributes its classes' joins. References that don't resolve to a
    /// group are skipped (validation reports them).
    pub(crate) fn direct_groups(
        &self,
        entry: EntryRef<'_>,
    ) -> Vec<CatalogName> {
        let mut out = Vec::new();
        if let Some(scope) = entry.name().scope()
            && self
                .declaration(&scope)
                .is_some_and(|d| d.kind().is_group())
        {
            out.push(scope);
        }
        let mut join = |from: &CatalogName, refs: &mut dyn Iterator<Item = &CatalogRef>| {
            for r in refs {
                if let Some(d) = self.resolve_ref(from, r)
                    && d.kind().is_group()
                {
                    out.push(d.name().clone());
                }
            }
        };
        match entry.body() {
            CatalogEntry::Namespace(g) => join(entry.name(), &mut g.join.iter()),
            CatalogEntry::Release(r) => join(entry.name(), &mut r.join.iter()),
            CatalogEntry::Build(b) => {
                join(entry.name(), &mut b.join.iter().map(Membership::group));
                if let Ok(chain) = self.template_chain(entry.name(), b.template_class.as_ref()) {
                    for link in chain {
                        join(link.name, &mut link.class.join.iter());
                    }
                }
            }
            _ => {}
        }
        out
    }

    /// The membership closure of the groups `start`: each, and every group
    /// it is in, transitively.
    pub(crate) fn group_closure(
        &self,
        start: Vec<CatalogName>,
    ) -> BTreeSet<CatalogName> {
        let mut seen = BTreeSet::new();
        let mut todo = start;
        while let Some(name) = todo.pop() {
            let Some(decl) = self.declaration(&name) else {
                continue;
            };
            if !decl.kind().is_group() || !seen.insert(name) {
                continue;
            }
            todo.extend(self.direct_groups(decl));
        }
        seen
    }

    /// Every group the build `name` is in.
    pub fn build_memberships(
        &self,
        name: &CatalogName,
    ) -> BTreeSet<CatalogName> {
        match self.declaration(name) {
            Some(decl) => self.group_closure(self.direct_groups(decl)),
            None => BTreeSet::new(),
        }
    }

    /// The template-class chain that `start` (written in `from`) names, up
    /// to [`TEMPLATE_CLASS_ROOT`], root first. With no `start`, the chain is
    /// the root alone. Each class's `base` resolves from that class.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Reference`] when a link names no template class
    /// or the root isn't declared; [`CatalogErrorKind::Structure`] on a
    /// cycle.
    pub fn template_chain(
        &self,
        from: &CatalogName,
        start: Option<&CatalogRef>,
    ) -> CatalogResult<Vec<TemplateLink<'_>>> {
        let root = CatalogName::root(TEMPLATE_CLASS_ROOT).expect("a legal segment");
        let class_kind = [EntryKind::TemplateClass];
        let the_root = || {
            self.declaration(&root)
                .filter(|d| d.kind() == EntryKind::TemplateClass)
                .ok_or_else(|| {
                    CatalogError::new(
                        CatalogErrorKind::Reference,
                        format!("no template class `{root}` (the prelude declares it)"),
                    )
                })
        };
        let mut next = match start {
            Some(r) => self.expect_ref(from, r, "template class", &class_kind)?,
            None => the_root()?,
        };
        let mut chain: Vec<TemplateLink<'_>> = Vec::new();
        loop {
            if chain.iter().any(|l| l.name == next.name()) {
                return Err(CatalogError::new(
                    CatalogErrorKind::Structure,
                    format!("the template-class chain loops at `{}`", next.name()),
                ));
            }
            let CatalogEntry::TemplateClass(class) = next.body() else {
                unreachable!("expected a template class");
            };
            chain.push(TemplateLink {
                name: next.name(),
                class,
            });
            if next.name() == &root {
                break;
            }
            next = match &class.base {
                Some(base) => self.expect_ref(next.name(), base, "template class", &class_kind)?,
                None => the_root()?,
            };
        }
        chain.reverse();
        Ok(chain)
    }

    /// The members of the `release` group `group`, ordered by release time.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Reference`] when `group` isn't a release group.
    pub fn releases(
        &self,
        group: &CatalogName,
    ) -> CatalogResult<Vec<(CatalogName, ReleaseTime)>> {
        self.expect(group, "release group", &[EntryKind::Release])?;
        let mut out: Vec<(CatalogName, ReleaseTime)> = self
            .of_kind(EntryKind::Build)
            .into_iter()
            .filter_map(|b| {
                let CatalogEntry::Build(build) = b.body() else {
                    return None;
                };
                build.join.iter().find_map(|m| match m {
                    Membership::Release(r)
                        if self
                            .resolve_ref(b.name(), &r.group)
                            .is_some_and(|d| d.name() == group) =>
                    {
                        ReleaseTime::parse(&r.released)
                            .ok()
                            .map(|t| (b.name().clone(), t))
                    }
                    _ => None,
                })
            })
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        Ok(out)
    }

    /// The latest member of the release group `group`.
    ///
    /// # Errors
    /// As [`releases`](Self::releases).
    pub fn latest(
        &self,
        group: &CatalogName,
    ) -> CatalogResult<Option<CatalogName>> {
        Ok(self.releases(group)?.pop().map(|(name, _)| name))
    }

    /// The label of the catalog an entry was declared in.
    fn declared_label(
        &self,
        entry: EntryRef<'_>,
    ) -> &str {
        self.layers[entry.declared_in()].label()
    }

    /// Applies the top catalog's occurrences, in order, to the state.
    fn apply_top(&mut self) -> Vec<CatalogError> {
        let top = self.layers.len() - 1;
        let occurrences: Vec<LayerEntry> = self.layers[top].entries().to_vec();
        let mut errors = Vec::new();
        let mut replaced: BTreeSet<CatalogName> = BTreeSet::new();
        for occurrence in occurrences {
            let name = occurrence.name.clone();
            let merge = occurrence.merge();
            let change = EntryChange { layer: top, merge };
            let mut fields = occurrence.entry.clone();
            reset_merge(&mut fields);
            let at = |e: CatalogError| e.at(&name);
            let existing = self.index.get(&name).copied();
            match (merge, existing) {
                (MergeMode::Deny, None) => {
                    self.index.insert(name.clone(), self.entries.len());
                    self.entries.push(MaterializedEntry {
                        name,
                        entry: fields,
                        history: vec![change],
                    });
                }
                (MergeMode::Deny, Some(i)) => {
                    let first = EntryRef {
                        entry: &self.entries[i],
                    };
                    errors.push(at(CatalogError::new(
                        CatalogErrorKind::Conflict,
                        format!(
                            "`{name}` is already declared in `{}`, as {}",
                            self.declared_label(first),
                            with_article(first.kind().as_str())
                        ),
                    )));
                }
                (MergeMode::Update | MergeMode::Replace, None) => {
                    let verb = if merge == MergeMode::Update {
                        "updates"
                    } else {
                        "replaces"
                    };
                    errors.push(at(CatalogError::new(
                        CatalogErrorKind::Merge,
                        format!("{verb} `{name}`, which no earlier catalog or this one declares"),
                    )));
                }
                (MergeMode::Update | MergeMode::Replace, Some(i))
                    if self.entries[i].entry.kind() != fields.kind() =>
                {
                    let (verb, rule) = if merge == MergeMode::Update {
                        ("updates", "")
                    } else {
                        ("replaces", ": a replacement keeps the kind")
                    };
                    errors.push(at(CatalogError::new(
                        CatalogErrorKind::Merge,
                        format!(
                            "{verb} `{name}` as {}, but it is declared {}{rule}",
                            with_article(fields.kind().as_str()),
                            with_article(self.entries[i].entry.kind().as_str())
                        ),
                    )));
                }
                (MergeMode::Replace, Some(i)) => {
                    if !replaced.insert(name.clone()) {
                        errors.push(at(CatalogError::new(
                            CatalogErrorKind::Merge,
                            format!("`{name}` is replaced twice in one catalog"),
                        )));
                        continue;
                    }
                    self.entries[i].entry = fields;
                    self.entries[i].history.push(change);
                }
                (MergeMode::Update, Some(i)) => match update(&mut self.entries[i].entry, &fields) {
                    Ok(()) => self.entries[i].history.push(change),
                    Err(e) => errors.push(at(e)),
                },
            }
        }
        errors
    }

    /// Validates the state after the top catalog is applied.
    fn validate_top(&self) -> Vec<CatalogError> {
        let top = self.layers.len() - 1;
        let occurrences: Vec<&LayerEntry> = self.layers[top].entries().iter().collect();
        let mut errors = Vec::new();
        let mut builds: BTreeSet<CatalogName> = BTreeSet::new();
        let mut changes_entries = false;
        for occurrence in &occurrences {
            let Some(entry) = self.declaration(&occurrence.name) else {
                continue;
            };
            let at = |e: CatalogError| e.at(entry.name());
            match occurrence.merge() {
                MergeMode::Deny | MergeMode::Replace => {
                    changes_entries |= occurrence.merge() == MergeMode::Replace;
                    errors.extend(self.check_naming_scope(entry).into_iter().map(at));
                    errors.extend(self.check_declaration(entry).into_iter().map(at));
                    if entry.kind() == EntryKind::Build {
                        builds.insert(entry.name().clone());
                    }
                }
                MergeMode::Update => {
                    changes_entries = true;
                    errors.extend(self.check_update(entry, occurrence).into_iter().map(at));
                    if entry.kind() == EntryKind::Alias {
                        errors.extend(self.check_declaration(entry).into_iter().map(at));
                    }
                }
            }
        }
        // A replacement or an update can change any build's resolution.
        if changes_entries {
            builds.extend(
                self.of_kind(EntryKind::Build)
                    .iter()
                    .map(|b| b.name().clone()),
            );
        }
        for build in &builds {
            if let Err(found) = self.resolve_build(build) {
                errors.extend(found.0);
            }
        }
        errors.extend(self.check_release_order(&builds));
        errors
    }

    /// What an update checks against the state: a namespace's expected
    /// facet. (What it may state is checked as it is applied.)
    fn check_update(
        &self,
        entry: EntryRef<'_>,
        occurrence: &LayerEntry,
    ) -> Vec<CatalogError> {
        let CatalogEntry::Namespace(group) = &occurrence.entry else {
            return Vec::new();
        };
        let Some(expected) = &group.facet else {
            return Vec::new();
        };
        match self.expect_ref(entry.name(), expected, "facet", &[EntryKind::Facet]) {
            Err(e) => vec![e],
            Ok(expected) => {
                let declared = self.group_facet(entry);
                if declared.as_ref() == Some(expected.name()) {
                    Vec::new()
                } else {
                    let declared = declared.map_or("no facet".to_string(), |f| format!("`{f}`"));
                    vec![CatalogError::new(
                        CatalogErrorKind::Merge,
                        format!(
                            "expects `{}` in the facet `{}`, but it is declared with {declared}",
                            entry.name(),
                            expected.name()
                        ),
                    )]
                }
            }
        }
    }

    fn check_naming_scope(
        &self,
        entry: EntryRef<'_>,
    ) -> Vec<CatalogError> {
        let Some(scope) = entry.name().scope() else {
            return Vec::new();
        };
        // The layer placed contracts and types in `$contracts` and in
        // contracts; the section's root is no entry.
        if scope == CatalogName::contracts_root() {
            return Vec::new();
        }
        let child_of_build = matches!(
            entry.kind(),
            EntryKind::Resource | EntryKind::Tag | EntryKind::Axis
        );
        match self.declaration(&scope) {
            Some(decl) if decl.kind().is_container_group() => Vec::new(),
            Some(decl) if decl.kind() == EntryKind::Contract => Vec::new(),
            Some(decl) if decl.kind() == EntryKind::Build && child_of_build => Vec::new(),
            Some(decl) => vec![CatalogError::new(
                CatalogErrorKind::Structure,
                format!(
                    "{} can't hold {}",
                    with_article(decl.kind().as_str()),
                    with_article(entry.kind().as_str())
                ),
            )],
            None => vec![CatalogError::new(
                CatalogErrorKind::Reference,
                format!("its scope `{scope}` is not declared"),
            )],
        }
    }

    fn check_declaration(
        &self,
        entry: EntryRef<'_>,
    ) -> Vec<CatalogError> {
        let mut errors = Vec::new();
        fn push<T>(
            errors: &mut Vec<CatalogError>,
            r: CatalogResult<T>,
        ) {
            if let Err(e) = r {
                errors.push(e);
            }
        }
        let described = |d: &str| {
            if d.trim().is_empty() {
                Err(CatalogError::new(
                    CatalogErrorKind::Structure,
                    "it has no description",
                ))
            } else {
                Ok(())
            }
        };
        let from = entry.name();
        match entry.body() {
            CatalogEntry::Namespace(g) => {
                push(
                    &mut errors,
                    described(g.description.as_deref().unwrap_or("")),
                );
                if let Some(facet) = &g.facet {
                    push(
                        &mut errors,
                        self.expect_ref(from, facet, "facet", &[EntryKind::Facet]),
                    );
                }
                for joined in &g.join {
                    push(&mut errors, self.check_plain_join(from, joined));
                }
                for tag in &g.requires {
                    push(
                        &mut errors,
                        self.expect_ref(from, tag, "tag", &[EntryKind::Tag, EntryKind::Axis]),
                    );
                }
                errors.extend(self.check_join_cycle(entry));
                errors.extend(self.check_facet_nesting(entry));
            }
            CatalogEntry::Contract(c) => {
                push(&mut errors, described(&c.description));
                errors.extend(self.check_contract(entry, c));
            }
            CatalogEntry::Type(t) => {
                push(&mut errors, described(&t.description));
                push(&mut errors, self.resolve_type(from, &t.type_expr));
            }
            CatalogEntry::Release(r) => {
                push(&mut errors, described(&r.description));
                for joined in &r.join {
                    push(&mut errors, self.check_plain_join(from, joined));
                }
                errors.extend(self.check_join_cycle(entry));
            }
            CatalogEntry::Facet(f) => {
                push(&mut errors, described(&f.description));
                push(&mut errors, reserved_tag_name(entry.name()));
            }
            CatalogEntry::Tag(t) => {
                push(&mut errors, described(&t.description));
                push(&mut errors, reserved_tag_name(entry.name()));
                if matches!(
                    t.tag_type,
                    crate::data::pretrained::build_catalog::document::TypeExpr::Primitive(
                        crate::data::pretrained::build_catalog::document::Primitive::Bool
                    )
                ) {
                    push(&mut errors, positive_bool_name("tag", entry.name()));
                }
                push(&mut errors, self.resolve_type(from, &t.tag_type));
            }
            CatalogEntry::Axis(a) => {
                push(&mut errors, described(&a.description));
                push(&mut errors, reserved_tag_name(entry.name()));
                if a.axis == crate::data::pretrained::build_catalog::document::AxisKind::Bool {
                    push(&mut errors, positive_bool_name("axis", entry.name()));
                }
                let problem = match a.axis {
                    crate::data::pretrained::build_catalog::document::AxisKind::Bool
                        if !a.members.is_empty() =>
                    {
                        Some("a bool axis's members are generated, never written")
                    }
                    crate::data::pretrained::build_catalog::document::AxisKind::Categorical
                        if a.members.len() < 2 =>
                    {
                        Some("a categorical axis has two or more members")
                    }
                    _ => None,
                };
                if let Some(problem) = problem {
                    push::<()>(
                        &mut errors,
                        Err(CatalogError::new(CatalogErrorKind::Preset, problem)),
                    );
                }
                let mut seen = BTreeSet::new();
                for member in &a.members {
                    push(&mut errors, check_segment(member));
                    if !seen.insert(member) {
                        push::<()>(
                            &mut errors,
                            Err(CatalogError::new(
                                CatalogErrorKind::Preset,
                                format!("the member `{member}` is listed twice"),
                            )),
                        );
                    }
                }
            }
            CatalogEntry::Alias(a) => {
                if a.target.kind == EntryKind::Alias {
                    push::<()>(
                        &mut errors,
                        Err(CatalogError::new(
                            CatalogErrorKind::Reference,
                            "an alias names an entry, not another alias",
                        )),
                    );
                } else {
                    push(
                        &mut errors,
                        self.expect_ref(
                            from,
                            &a.target.path,
                            a.target.kind.as_str(),
                            &[a.target.kind],
                        ),
                    );
                }
            }
            CatalogEntry::Resource(r) => {
                let mut bad = |m: String| {
                    errors.push(CatalogError::new(CatalogErrorKind::Structure, m));
                };
                if let Err(why) = check_file_path(&r.file) {
                    bad(why);
                }
                if r.format.trim().is_empty() {
                    bad("a resource names its format".into());
                }
                if let Some(sha) = &r.sha256
                    && !is_sha256(sha)
                {
                    bad(format!("`{sha}` is not a lowercase hex SHA-256"));
                }
                if r.sources.is_empty() {
                    bad("a resource names at least one source".into());
                }
                for source in &r.sources {
                    if let Err(why) = check_source(source) {
                        bad(why);
                    }
                }
                let mut files = BTreeSet::from([r.file.as_str()]);
                for shard in &r.shards {
                    if let Err(why) = check_file_path(&shard.file) {
                        bad(format!("the shard `{}`: {why}", shard.file));
                    }
                    if !files.insert(shard.file.as_str()) {
                        bad(format!("the file `{}` is listed twice", shard.file));
                    }
                    if !is_sha256(&shard.sha256) {
                        bad(format!(
                            "the shard `{}`: `{}` is not a lowercase hex SHA-256",
                            shard.file, shard.sha256
                        ));
                    }
                    for source in &shard.sources {
                        if let Err(why) = check_source(source) {
                            bad(format!("the shard `{}`: {why}", shard.file));
                        }
                    }
                    if r.shard_sources(shard).next().is_none() {
                        bad(format!(
                            "the shard `{}` has no source: the resource's are `$url`s, which \
                             name one file each; give it its own `$sources`, or the resource a \
                             place (`$hub`, `$base_url`, `$local_dir`)",
                            shard.file
                        ));
                    }
                }
            }
            CatalogEntry::TemplateClass(t) => {
                push(&mut errors, described(&t.description));
                let is_root =
                    entry.name().scope().is_none() && entry.name().leaf() == TEMPLATE_CLASS_ROOT;
                if is_root && t.base.is_some() {
                    push::<()>(
                        &mut errors,
                        Err(CatalogError::new(
                            CatalogErrorKind::Structure,
                            "the template-class root has no base",
                        )),
                    );
                }
                let this = CatalogRef::absolute(entry.name().clone());
                let chain = self.template_chain(from, Some(&this));
                if let Err(e) = &chain {
                    errors.push(e.clone());
                }
                for joined in &t.join {
                    push(&mut errors, self.check_plain_join(from, joined));
                }
                for facet in &t.required_facets {
                    push(
                        &mut errors,
                        self.expect_ref(from, facet, "facet", &[EntryKind::Facet]),
                    );
                }
                for (tag, value) in &t.tags {
                    push(&mut errors, self.check_tag_value(from, tag, value));
                }
                if let Some(contract) = &t.contract {
                    push(&mut errors, self.expect_contract(contract));
                }
                for (slot, resource) in &t.resources {
                    push(&mut errors, check_segment(slot));
                    push(&mut errors, self.check_dependency_target(from, resource));
                }
                for slot in t.args.keys() {
                    push(&mut errors, check_segment(slot));
                }
                if let Ok(chain) = chain {
                    let links: Vec<
                        crate::data::pretrained::build_catalog::presets::PresetLink<'_>,
                    > = chain
                        .iter()
                        .map(
                            |l| crate::data::pretrained::build_catalog::presets::PresetLink {
                                owner: l.name,
                                set: &l.class.presets,
                            },
                        )
                        .collect();
                    if let Err(found) =
                        crate::data::pretrained::build_catalog::presets::ResolvedPresetSet::fold(
                            &links,
                            &|l, a| self.axis_info(l.owner, a),
                        )
                    {
                        errors.extend(found);
                    }
                }
            }
            CatalogEntry::Build(_) => {}
        }
        errors
    }

    /// A plain join: to a `namespace`. Joining a release
    /// needs a release timestamp, which only a build's membership carries.
    pub(crate) fn check_plain_join(
        &self,
        from: &CatalogName,
        joined: &CatalogRef,
    ) -> CatalogResult<CatalogName> {
        let decl = self.expect_ref(
            from,
            joined,
            "group",
            &[EntryKind::Namespace, EntryKind::Release],
        )?;
        if decl.kind() == EntryKind::Release {
            return Err(CatalogError::new(
                CatalogErrorKind::Release,
                format!(
                    "joins the release group `{}` without a `$released` timestamp; only a build \
                     joins a release, as `{{$group, $released}}`",
                    decl.name()
                ),
            ));
        }
        Ok(decl.name().clone())
    }

    /// Checks a written tag value; returns the tag's full name.
    pub(crate) fn check_tag_value(
        &self,
        from: &CatalogName,
        tag: &CatalogRef,
        value: &serde_json::Value,
    ) -> CatalogResult<CatalogName> {
        let Some(decl) = self.resolve_ref(from, tag) else {
            return Err(CatalogError::new(
                CatalogErrorKind::Reference,
                format!("no tag `{tag}` is visible from `{from}`; every tag is declared"),
            ));
        };
        let name = decl.name();
        match decl.body() {
            CatalogEntry::Tag(t) => self
                .resolve_type(name, &t.tag_type)
                .map_err(|e| e.message)
                .and_then(|ty| check_value(&ty, value))
                .map(|()| name.clone())
                .map_err(|why| {
                    CatalogError::new(CatalogErrorKind::Tag, format!("the tag `{name}`: {why}"))
                }),
            CatalogEntry::Axis(_) => Err(CatalogError::new(
                CatalogErrorKind::Tag,
                format!("the tag `{name}` is implied by the axis use; it is never written"),
            )),
            CatalogEntry::Facet(_) => Err(CatalogError::new(
                CatalogErrorKind::Tag,
                format!("the tag `{name}` is implied by facet membership; it is never written"),
            )),
            _ => Err(CatalogError::new(
                CatalogErrorKind::Reference,
                format!(
                    "`{tag}` resolves to `{name}`, which is {}, not a tag",
                    with_article(decl.kind().as_str())
                ),
            )),
        }
    }

    /// Resolves a dependency's resource; returns its full name.
    pub(crate) fn check_dependency_target(
        &self,
        from: &CatalogName,
        resource: &CatalogRef,
    ) -> CatalogResult<CatalogName> {
        self.expect_ref(from, resource, "resource", &[EntryKind::Resource])
            .map(|d| d.name().clone())
    }

    /// The contract `reference` names in `$contracts`
    /// (`qwen3-asr/stated` is `$contracts/qwen3-asr/stated`).
    ///
    /// # Errors
    /// [`CatalogErrorKind::Contract`] when `$contracts` has no such
    /// contract.
    pub fn expect_contract(
        &self,
        reference: &CatalogRef,
    ) -> CatalogResult<EntryRef<'_>> {
        let name = CatalogName::contracts_root().join(reference.path());
        match self.declaration(&name) {
            Some(decl) if decl.kind() == EntryKind::Contract => Ok(decl),
            _ => Err(CatalogError::new(
                CatalogErrorKind::Contract,
                format!("`$contracts` has no contract `{}`", reference.path()),
            )),
        }
    }

    /// The contract `name` and its ancestors, the outermost first: the
    /// contracts whose slots it has.
    pub fn contract_lineage(
        &self,
        name: &CatalogName,
    ) -> Vec<(
        CatalogName,
        &crate::data::pretrained::build_catalog::document::ContractEntry,
    )> {
        let mut lineage = Vec::new();
        let mut at = Some(name.clone());
        while let Some(current) = at {
            let Some(CatalogEntry::Contract(c)) = self.declaration(&current).map(|d| d.body())
            else {
                break;
            };
            at = current.scope();
            lineage.push((current, c));
        }
        lineage.reverse();
        lineage
    }

    /// The contracts nested in `name`, at any depth.
    pub fn contract_variants(
        &self,
        name: &CatalogName,
    ) -> Vec<CatalogName> {
        self.of_kind(EntryKind::Contract)
            .iter()
            .map(|c| c.name().clone())
            .filter(|c| c != name && c.is_within(name))
            .collect()
    }

    /// The builds whose contract is `contract` or one of its variants.
    pub fn builds_of_contract(
        &self,
        contract: &CatalogName,
    ) -> Vec<CatalogName> {
        self.of_kind(EntryKind::Build)
            .iter()
            .filter(|b| {
                self.resolve_build(b.name()).is_ok_and(|r| {
                    r.contract
                        .as_ref()
                        .is_some_and(|c| c == contract || c.is_within(contract))
                })
            })
            .map(|b| b.name().clone())
            .collect()
    }

    /// A contract's own rules: its slot names, its argument slots' types,
    /// a variant adding slots rather than restating its parents', and an
    /// abstract contract having a variant a build can name.
    fn check_contract(
        &self,
        entry: EntryRef<'_>,
        contract: &crate::data::pretrained::build_catalog::document::ContractEntry,
    ) -> Vec<CatalogError> {
        let mut errors = Vec::new();
        let name = entry.name();
        let lineage = self.contract_lineage(name);
        let inherited: BTreeMap<&String, &CatalogName> = lineage
            .iter()
            .filter(|(n, _)| n != name)
            .flat_map(|(n, c)| c.resources.keys().chain(c.args.keys()).map(move |k| (k, n)))
            .collect();
        for slot in contract.resources.keys().chain(contract.args.keys()) {
            if let Err(e) = check_segment(slot) {
                errors.push(e);
            }
            if let Some(parent) = inherited.get(slot) {
                errors.push(CatalogError::new(
                    CatalogErrorKind::Contract,
                    format!(
                        "the slot `{slot}` is already its contract `{}`'s: a variant only adds \
                         slots",
                        parent.contract_path().unwrap_or_default()
                    ),
                ));
            }
        }
        if contract
            .resources
            .keys()
            .any(|k| contract.args.contains_key(k))
        {
            errors.push(CatalogError::new(
                CatalogErrorKind::Contract,
                "a slot is either a resource or an argument, not both",
            ));
        }
        for (slot, spec) in &contract.resources {
            if spec.formats.iter().any(|f| f.trim().is_empty()) {
                errors.push(CatalogError::new(
                    CatalogErrorKind::Contract,
                    format!("the resource slot `{slot}` lists an empty format"),
                ));
            }
        }
        for (slot, spec) in &contract.args {
            if let Err(e) = self.resolve_type(name, &spec.type_expr) {
                errors.push(CatalogError::new(
                    e.kind,
                    format!("the argument slot `{slot}`: {}", e.message),
                ));
            }
        }
        if contract.abstract_ {
            let concrete = self.contract_variants(name).iter().any(|v| {
                matches!(
                    self.declaration(v).map(|d| d.body()),
                    Some(CatalogEntry::Contract(c)) if !c.abstract_
                )
            });
            if !concrete {
                errors.push(CatalogError::new(
                    CatalogErrorKind::Contract,
                    "it is abstract, and has no variant a build can name",
                ));
            }
        }
        errors
    }

    /// Resolves an axis use's axis.
    pub(crate) fn axis_info(
        &self,
        from: &CatalogName,
        axis: &CatalogRef,
    ) -> CatalogResult<crate::data::pretrained::build_catalog::presets::AxisInfo> {
        let decl = self.expect_ref(from, axis, "axis", &[EntryKind::Axis])?;
        let CatalogEntry::Axis(a) = decl.body() else {
            unreachable!("expected an axis")
        };
        Ok(crate::data::pretrained::build_catalog::presets::AxisInfo {
            name: decl.name().clone(),
            kind: a.axis,
            members: a.member_names(),
        })
    }

    fn check_join_cycle(
        &self,
        start: EntryRef<'_>,
    ) -> Vec<CatalogError> {
        let mut seen = BTreeSet::new();
        let mut todo = self.direct_groups(start);
        while let Some(name) = todo.pop() {
            if &name == start.name() {
                return vec![CatalogError::new(
                    CatalogErrorKind::Structure,
                    format!(
                        "`{}` is, through its memberships, a member of itself",
                        start.name()
                    ),
                )];
            }
            if !seen.insert(name.clone()) {
                continue;
            }
            if let Some(d) = self.declaration(&name)
                && d.kind().is_group()
            {
                todo.extend(self.direct_groups(d));
            }
        }
        Vec::new()
    }

    fn check_facet_nesting(
        &self,
        entry: EntryRef<'_>,
    ) -> Vec<CatalogError> {
        let Some(facet) = self.group_facet(entry) else {
            return Vec::new();
        };
        let mut above = self.group_closure(self.direct_groups(entry));
        above.remove(entry.name());
        above
            .iter()
            .filter(|g| {
                self.declaration(g)
                    .and_then(|d| self.group_facet(d))
                    .as_ref()
                    == Some(&facet)
            })
            .map(|g| {
                CatalogError::new(
                    CatalogErrorKind::Facet,
                    format!(
                        "a `{facet}` group can't be within another `{facet}` group, `{g}`: its \
                         members would be in both"
                    ),
                )
            })
            .collect()
    }

    fn check_release_order(
        &self,
        builds: &BTreeSet<CatalogName>,
    ) -> Vec<CatalogError> {
        let mut touched: BTreeSet<CatalogName> = BTreeSet::new();
        for build in builds {
            let Some(entry) = self.declaration(build) else {
                continue;
            };
            if let CatalogEntry::Build(b) = entry.body() {
                for m in &b.join {
                    if let Membership::Release(r) = m
                        && let Some(d) = self.resolve_ref(entry.name(), &r.group)
                    {
                        touched.insert(d.name().clone());
                    }
                }
            }
        }
        let mut errors = Vec::new();
        for group in &touched {
            let Ok(members) = self.releases(group) else {
                continue;
            };
            for (i, (a, ta)) in members.iter().enumerate() {
                for (b, tb) in &members[i + 1..] {
                    if ta.overlaps(tb) {
                        errors.push(
                            CatalogError::new(
                                CatalogErrorKind::Release,
                                format!(
                                    "`{a}` ({ta}) and `{b}` ({tb}) overlap in `{group}`: pin \
                                     one of them more precisely"
                                ),
                            )
                            .at(group),
                        );
                    }
                }
            }
        }
        errors
    }
}

/// Whether `sha` is a lowercase hex SHA-256.
fn is_sha256(sha: &str) -> bool {
    sha.len() == 64
        && sha
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

/// A resource file's path: `/`-separated segments, none empty, `.` or
/// `..`, with no leading `/` and no `\`.
fn check_file_path(path: &str) -> Result<(), String> {
    let fine = !path.is_empty()
        && !path.contains('\\')
        && path
            .split('/')
            .all(|segment| !segment.is_empty() && segment != "." && segment != "..");
    if fine {
        Ok(())
    } else {
        Err(format!(
            "`{path}` is not a file path: `/`-separated names, none empty, `.` or `..`, and no \
             leading `/`"
        ))
    }
}

/// A source's own fields.
fn check_source(
    source: &crate::data::pretrained::build_catalog::document::ResourceSource
) -> Result<(), String> {
    use crate::data::pretrained::build_catalog::document::ResourceSource;
    match source {
        ResourceSource::Url(url) | ResourceSource::BaseUrl(url) if url.trim().is_empty() => {
            Err("a URL source is empty".into())
        }
        ResourceSource::LocalDir { name, .. } if name.trim().is_empty() => {
            Err("a `$local_dir` source names no directory".into())
        }
        _ => Ok(()),
    }
}

/// Clears an entry's `merge`: the state holds merged entries.
fn reset_merge(entry: &mut CatalogEntry) {
    match entry {
        CatalogEntry::Namespace(e) => e.merge = MergeMode::Deny,
        CatalogEntry::Contract(e) => e.merge = MergeMode::Deny,
        CatalogEntry::Type(e) => e.merge = MergeMode::Deny,
        CatalogEntry::Release(e) => e.merge = MergeMode::Deny,
        CatalogEntry::Build(e) => e.merge = MergeMode::Deny,
        CatalogEntry::TemplateClass(e) => e.merge = MergeMode::Deny,
        CatalogEntry::Tag(e) => e.merge = MergeMode::Deny,
        CatalogEntry::Axis(e) => e.merge = MergeMode::Deny,
        CatalogEntry::Facet(e) => e.merge = MergeMode::Deny,
        CatalogEntry::Resource(e) => e.merge = MergeMode::Deny,
        CatalogEntry::Alias(e) => e.merge = MergeMode::Deny,
    }
}

/// Applies an `update` occurrence to an entry of the same kind, as its
/// kind defines ([`MergeMode`]).
fn update(
    current: &mut CatalogEntry,
    occurrence: &CatalogEntry,
) -> CatalogResult<()> {
    match (current, occurrence) {
        (CatalogEntry::Namespace(_), CatalogEntry::Namespace(group)) => {
            if group.description.is_some() || !group.join.is_empty() || !group.requires.is_empty() {
                return Err(CatalogError::new(
                    CatalogErrorKind::Structure,
                    "an extending occurrence carries only its `$kind`, a `$facet` to expect, \
                     and children",
                ));
            }
            Ok(())
        }
        (CatalogEntry::Resource(current), CatalogEntry::Resource(update)) => {
            let differs = update.file != current.file
                || update.format != current.format
                || update.sha256 != current.sha256
                || update.size != current.size
                || update.dtype != current.dtype
                || update.shards != current.shards
                || update
                    .description
                    .as_ref()
                    .is_some_and(|d| Some(d) != current.description.as_ref());
            if differs {
                return Err(CatalogError::new(
                    CatalogErrorKind::Merge,
                    "a resource update matches every field but `$sources`; to change the \
                     artifact, replace it",
                ));
            }
            let new: Vec<_> = update
                .sources
                .iter()
                .filter(|s| !current.sources.contains(s))
                .cloned()
                .collect();
            current.sources.splice(0..0, new);
            Ok(())
        }
        (CatalogEntry::Alias(current), CatalogEntry::Alias(update)) => {
            current.target = update.target.clone();
            if update.description.is_some() {
                current.description = update.description.clone();
            }
            Ok(())
        }
        (CatalogEntry::Contract(_), _) => Err(CatalogError::new(
            CatalogErrorKind::Merge,
            "a contract can't be updated: it is the interface its loader is built against; \
             replace it",
        )),
        (other, _) => Err(CatalogError::new(
            CatalogErrorKind::Merge,
            format!(
                "{} can't be updated; replace it",
                with_article(other.kind().as_str())
            ),
        )),
    }
}

/// Tags, axes and facets name tags; the contract's implied tag takes one
/// name from them.
fn reserved_tag_name(name: &CatalogName) -> CatalogResult<()> {
    if name.to_string() == crate::data::pretrained::build_catalog::resolve::CONTRACT_TAG {
        Err(CatalogError::new(
            CatalogErrorKind::Conflict,
            format!(
                "`{name}` is the tag every build's contract implies; nothing else may declare it"
            ),
        ))
    } else {
        Ok(())
    }
}

/// The prefixes that name a bool by its negative.
const NEGATIVE_PREFIXES: [&str; 4] = ["no", "non", "not", "without"];

/// A bool tag or axis is named by its positive: `timestamps`, not
/// `no_timestamps`. A bool axis `X` generates its own negative, `no-X`.
fn positive_bool_name(
    what: &str,
    name: &CatalogName,
) -> CatalogResult<()> {
    let leaf = name.leaf().to_ascii_lowercase();
    let negative = NEGATIVE_PREFIXES.iter().find_map(|prefix| {
        leaf.strip_prefix(prefix)
            .and_then(|rest| rest.strip_prefix(['_', '-']))
            .filter(|rest| !rest.is_empty())
    });
    match negative {
        Some(positive) => Err(CatalogError::new(
            CatalogErrorKind::Name,
            format!(
                "a bool {what} is named by its positive: `{positive}`, not `{}`",
                name.leaf()
            ),
        )),
        None => Ok(()),
    }
}

/// `word` with `a` or `an` before it.
pub(crate) fn with_article(word: &str) -> String {
    let an = word
        .chars()
        .next()
        .is_some_and(|c| "aeiou".contains(c.to_ascii_lowercase()));
    format!("{} {word}", if an { "an" } else { "a" })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::pretrained::build_catalog::{
        fixtures,
        lookup::LeafMatch,
    };

    fn n(s: &str) -> CatalogName {
        CatalogName::parse(s).unwrap()
    }

    fn names(refs: Vec<EntryRef<'_>>) -> Vec<String> {
        refs.iter().map(|e| e.name().to_string()).collect()
    }

    /// Pushes `yaml` over the built-in stack, expecting it to be refused,
    /// and checks the stack was left as it was.
    fn refused(yaml: &str) -> CatalogErrors {
        let mut stack = fixtures::example_stack().unwrap();
        let before = stack.layers().len();
        let errors = stack.push_yaml("bad", yaml).unwrap_err();
        assert_eq!(stack.layers().len(), before);
        assert!(errors.0.iter().all(|e| e.layer.as_deref() == Some("bad")));
        errors
    }

    #[test]
    fn test_fixtures_load() {
        let stack = fixtures::supplemented_example_stack().unwrap();
        let labels: Vec<&str> = stack.layers().iter().map(|l| l.label()).collect();
        assert_eq!(
            labels,
            ["prelude", "commons", "qwen3", "whisper", "supplemental"]
        );
    }

    #[test]
    fn test_facet_groups_are_the_groups_not_their_members() {
        let stack = fixtures::example_stack().unwrap();
        assert_eq!(
            names(stack.facet_groups(&n("family"))),
            ["qwen/qwen3", "openai/whisper"]
        );
        assert_eq!(
            names(stack.facet_groups(&n("publisher"))),
            ["qwen", "openai"]
        );
        let in_families = stack.builds_in_facet(&n("family"));
        assert_eq!(in_families.len(), stack.of_kind(EntryKind::Build).len());
    }

    #[test]
    fn test_occurrences_fall_through_and_children_union() {
        let stack = fixtures::supplemented_example_stack().unwrap();
        let qwen = n("qwen");
        let found: Vec<(usize, bool)> = stack
            .occurrences(&qwen)
            .map(|e| (e.layer, e.entry.merge() == MergeMode::Update))
            .collect();
        assert_eq!(found, [(1, false), (2, true), (4, true)]);
        let decl = stack.declaration(&qwen).unwrap();
        assert_eq!(decl.declared_in(), 1);
        // Each import that touched it is in its history.
        let history: Vec<(usize, MergeMode)> =
            decl.history().iter().map(|c| (c.layer, c.merge)).collect();
        assert_eq!(
            history,
            [
                (1, MergeMode::Deny),
                (2, MergeMode::Update),
                (4, MergeMode::Update)
            ]
        );
        // Children declared in three layers, collected by one walk.
        let kids = names(stack.children(&n("qwen/qwen3")));
        assert!(kids.contains(&"qwen/qwen3/qwen3-0.6b".to_string()));
        assert!(kids.contains(&"qwen/qwen3/qwen3-1.7b".to_string()));
        assert_eq!(names(stack.children(&qwen)), ["qwen/qwen3"]);
    }

    #[test]
    fn test_members_come_through_groups_and_template_classes() {
        let stack = fixtures::example_stack().unwrap();
        // Through the class chain: the class names the contract.
        let qwen3 = stack.builds_of_contract(&n("$contracts/qwen3-lm"));
        assert_eq!(qwen3.len(), 3, "{qwen3:?}");
        let family = stack.members_of(&n("qwen/qwen3"));
        assert_eq!(family.len(), 3, "{family:?}");
        let refresh = stack.members_of(&n("qwen/qwen3/qwen3-2507"));
        assert_eq!(refresh, [n("qwen/qwen3/qwen3-4b-thinking-2507")]);
        // The release group is placed in its family, so its members are too.
        assert!(
            stack
                .build_memberships(&n("openai/whisper/large/large-v1"))
                .contains(&n("openai/whisper/large/release"))
        );
    }

    #[test]
    fn test_releases_order_by_time() {
        let stack = fixtures::example_stack().unwrap();
        let group = n("openai/whisper/large/release");
        let order: Vec<String> = stack
            .releases(&group)
            .unwrap()
            .iter()
            .map(|(b, t)| format!("{} {t}", b.leaf()))
            .collect();
        assert_eq!(
            order,
            [
                "large-v1 2022-09",
                "large-v2 2022-12",
                "large-v3 2023-11-06"
            ]
        );
        assert_eq!(
            stack.latest(&group).unwrap(),
            Some(n("openai/whisper/large/large-v3"))
        );
        assert!(stack.releases(&n("openai/whisper")).is_err());
    }

    #[test]
    fn test_redeclaring_is_a_conflict() {
        let errors = refused("qwen: { $kind: namespace, $description: \"Again.\" }\n");
        assert!(errors.has(CatalogErrorKind::Conflict), "{errors}");
        assert!(errors.mentions("already declared in `commons`"));

        // A child name already taken in the scope, from another layer.
        let errors = refused(
            r#"
qwen:
  $kind: namespace
  $merge: update
  qwen3:
    $kind: namespace
    $merge: update
    qwen3-0.6b: { $kind: namespace, $description: "Collides with a build." }
"#,
        );
        assert!(errors.has(CatalogErrorKind::Conflict), "{errors}");
    }

    #[test]
    fn test_extensions_must_match_their_declarations() {
        let undeclared = refused("mistral: { $kind: namespace, $merge: update }\n");
        assert!(
            undeclared.mentions("no earlier catalog or this one declares"),
            "{undeclared}"
        );

        let wrong_facet = refused("qwen: { $kind: namespace, $merge: update, $facet: family }\n");
        assert!(wrong_facet.has(CatalogErrorKind::Merge), "{wrong_facet}");
        assert!(wrong_facet.mentions("declared with `publisher`"));

        let wrong_kind =
            refused("qwen: { $kind: release, $merge: update, $description: \"R.\" }\n");
        assert!(
            wrong_kind.mentions("it is declared a namespace"),
            "{wrong_kind}"
        );

        let carries =
            refused("qwen: { $kind: namespace, $merge: update, $description: \"No.\" }\n");
        assert!(carries.mentions("carries only its `$kind`"), "{carries}");
    }

    #[test]
    fn test_facet_rules() {
        let nested = refused(
            "qwen:\n  $kind: namespace\n  $merge: update\n  qwen3:\n    $kind: namespace\n    $merge: update\n    sub: { $kind: namespace, $facet: family, $description: \"D.\" }\n",
        );
        assert!(
            nested.mentions("can't be within another `family` group"),
            "{nested}"
        );

        let two = refused(
            r#"
qwen:
  $kind: namespace
  $merge: update
  qwen3:
    $kind: namespace
    $merge: update
    both-families:
      $kind: build
      $description: "In two families."
      $join: [openai/whisper]
      parameters: 1
"#,
        );
        assert!(two.mentions("more than one `family` group"), "{two}");

        let none = refused(
            r#"
loose:
  $kind: namespace
  $description: "No facet."
  orphan: { $kind: build, $description: "In no family." }
"#,
        );
        assert!(none.mentions("in no `family` group"), "{none}");
        assert!(none.mentions("in no `publisher` group"), "{none}");
    }

    #[test]
    fn test_names_repeat_in_other_scopes() {
        // A name is unique in its scope only: a later catalog may reuse a
        // leaf above, below or beside an existing one, so long as no
        // earlier reference changes.
        let mut stack = fixtures::example_stack().unwrap();
        stack
            .push_yaml(
                "reuse",
                r#"
timestamps: { $kind: namespace, $description: "Above openai/whisper/timestamps." }
qwen:
  $kind: namespace
  $merge: update
  qwen3:
    $kind: namespace
    $merge: update
    qwen3-0.6b-extras:
      $kind: namespace
      $description: "Beside the family's class."
      hybrid: { $kind: template_class, $description: "A namesake." }
"#,
            )
            .unwrap();
        // Earlier references still resolve where they did.
        let b = stack.resolve_build(&n("qwen/qwen3/qwen3-0.6b")).unwrap();
        assert_eq!(b.template_chain[1], n("qwen/qwen3/hybrid"));
        // A lookup takes the least unique spelling: a full name, or a
        // partial one that only one entry matches.
        let classes = [EntryKind::TemplateClass];
        assert_eq!(
            stack
                .find("qwen/qwen3/hybrid", &classes, LeafMatch::Exact)
                .unwrap(),
            n("qwen/qwen3/hybrid")
        );
        assert_eq!(
            stack
                .find("qwen3-0.6b-extras/hybrid", &classes, LeafMatch::Exact)
                .unwrap(),
            n("qwen/qwen3/qwen3-0.6b-extras/hybrid")
        );
        let both = stack
            .find("hybrid", &classes, LeafMatch::Exact)
            .unwrap_err();
        assert!(both.message.contains("is ambiguous"), "{both}");
        assert_eq!(
            stack.display_name(&n("timestamps")).short,
            "timestamps",
            "the root's is its full name"
        );
    }

    #[test]
    fn test_a_later_catalog_updates_and_replaces() {
        // One catalog states a name once; a later one updates or replaces it.
        let mut stack = fixtures::example_stack().unwrap();
        stack
            .push_yaml(
                "first",
                r#"
local:
  $kind: namespace
  $description: "First."
  notes: { $kind: resource, $file: notes.txt, $format: text, $sources: [{ $url: "https://a/notes.txt" }] }
"#,
            )
            .unwrap();
        stack
            .push_yaml(
                "second",
                r#"
local:
  $kind: namespace
  $merge: replace
  $description: "Second."
  notes:
    $kind: resource
    $merge: update
    $file: notes.txt
    $format: text
    $sources: [{ $url: "https://b/notes.txt" }]
"#,
            )
            .unwrap();
        let local = stack.declaration(&n("local")).unwrap();
        assert_eq!(
            local.group().unwrap().description.as_deref(),
            Some("Second.")
        );
        let notes = stack.resource(&n("local/notes")).unwrap();
        assert_eq!(notes.sources.len(), 2);
    }

    #[test]
    fn test_a_refused_import_leaves_the_state_as_it_was() {
        let mut stack = fixtures::example_stack().unwrap();
        let before: Vec<String> = stack.declarations().map(|e| e.name().to_string()).collect();
        let errors = stack
            .push_yaml(
                "half-good",
                r#"
fine: { $kind: namespace, $description: "Would be fine on its own." }
qwen: { $kind: namespace, $description: "A conflict." }
"#,
            )
            .unwrap_err();
        assert!(errors.has(CatalogErrorKind::Conflict));
        let after: Vec<String> = stack.declarations().map(|e| e.name().to_string()).collect();
        assert_eq!(before, after);
        assert!(stack.declaration(&n("fine")).is_none());
    }

    #[test]
    fn test_replace_substitutes_for_every_reference() {
        let mut stack = fixtures::supplemented_example_stack().unwrap();
        stack
            .push_yaml(
                "rename",
                r#"
qwen: { $kind: namespace, $merge: replace, $facet: publisher, $description: "Qwen, renamed." }
"#,
            )
            .unwrap();
        let qwen = stack.declaration(&n("qwen")).unwrap();
        assert_eq!(
            qwen.history().last().map(|c| (c.layer, c.merge)),
            Some((5, MergeMode::Replace))
        );
        assert_eq!(qwen.declared_in(), 1);
        assert_eq!(
            qwen.group().unwrap().description.as_deref(),
            Some("Qwen, renamed.")
        );
        // Builds in catalogs below it see the replacement.
        let b = stack.resolve_build(&n("qwen/qwen3/qwen3-0.6b")).unwrap();
        assert_eq!(b.facets[&n("publisher")], n("qwen"));
        // Its children are entries of their own, untouched.
        assert_eq!(stack.children(&n("qwen")).len(), 1);
    }

    #[test]
    fn test_a_replacement_that_breaks_builds_is_refused() {
        // Dropping the facet takes every Qwen build out of its publisher.
        let errors =
            refused("qwen: { $kind: namespace, $merge: replace, $description: \"No facet.\" }\n");
        assert!(errors.mentions("in no `publisher` group"), "{errors}");
        let kind = refused("qwen: { $kind: release, $merge: replace, $description: \"R.\" }\n");
        assert!(kind.mentions("a replacement keeps the kind"), "{kind}");
        let missing = refused("country: { $kind: facet, $merge: replace, $description: \"C.\" }\n");
        assert!(
            missing.mentions("replaces `country`, which no earlier catalog or this one declares"),
            "{missing}"
        );
    }

    #[test]
    fn test_update_is_defined_per_kind() {
        // A resource update matches every field but its sources, and puts
        // its new sources first.
        let mut stack = fixtures::example_stack().unwrap();
        stack
            .push_yaml(
                "mirror",
                r#"
openai:
  $kind: namespace
  $merge: update
  whisper:
    $kind: namespace
    $merge: update
    vocabulary:
      $kind: resource
      $merge: update
      $file: multilingual.tiktoken
      $format: tiktoken
      $sha256: "2000000000000000000000000000000000000000000000000000000000000001"
      $sources: [{ $url: "https://mirror.example/multilingual.tiktoken" }]
"#,
            )
            .unwrap();
        let v1 = stack
            .resolve_build(&n("openai/whisper/large/large-v1"))
            .unwrap();
        let sources = &v1.resources["vocabulary"].entry.sources;
        assert_eq!(sources.len(), 2);
        assert_eq!(
            sources[0],
            crate::data::pretrained::build_catalog::document::ResourceSource::Url(
                "https://mirror.example/multilingual.tiktoken".into()
            )
        );

        let changed = refused(
            r#"
openai:
  $kind: namespace
  $merge: update
  whisper:
    $kind: namespace
    $merge: update
    vocabulary:
      $kind: resource
      $merge: update
      $file: other.tiktoken
      $format: tiktoken
      $sources: [{ $url: "https://mirror.example/other.tiktoken" }]
"#,
        );
        assert!(
            changed.mentions("matches every field but `$sources`"),
            "{changed}"
        );

        // Contracts are the base catalog's alone: a later one can't write
        // `$contracts`, so a contract never meets an update.
        let contract = refused(
            "$contracts:\n  qwen3-lm: { $kind: contract, $merge: update, $description: \"C.\" }\n",
        );
        assert!(
            contract.mentions("only the base catalog declares `$contracts`"),
            "{contract}"
        );
        let build = refused(
            "qwen:\n  $kind: namespace\n  $merge: update\n  qwen3:\n    $kind: namespace\n    $merge: update\n    qwen3-0.6b: { $kind: build, $merge: update, $description: \"B.\" }\n",
        );
        assert!(build.mentions("a build can't be updated"), "{build}");
    }

    #[test]
    fn test_sharded_resources_and_place_sources() {
        let mut stack = fixtures::example_stack().unwrap();
        stack
            .push_yaml(
                "files",
                r#"
local:
  $kind: namespace
  $description: "Local files."
  sharded:
    $kind: resource
    $file: model.safetensors.index.json
    $format: safetensors-index
    $sha256: "5000000000000000000000000000000000000000000000000000000000000001"
    $size: 27000
    $sources:
      - { $url: "https://exact.example/model.safetensors.index.json" }
      - { $base_url: "https://mirror.example/repo" }
    $shards:
      - { $file: model-00001-of-00002.safetensors, $sha256: "5000000000000000000000000000000000000000000000000000000000000002", $size: 4000000000 }
      - $file: model-00002-of-00002.safetensors
        $sha256: "5000000000000000000000000000000000000000000000000000000000000003"
        $size: 3000000000
        $sources: [{ $url: "https://exact.example/model-00002-of-00002.safetensors" }]
  nested:
    $kind: resource
    $file: encoder/config.json
    $format: json
    $sources: [{ $local_dir: { $name: other-tool, $default: ["~/.cache/other-tool"] } }]
"#,
            )
            .unwrap();
        let sharded = stack.resource(&n("local/sharded")).unwrap();
        // A shard has its own sources first, then the resource's places: the
        // index's exact `$url` is the index's alone.
        let second: Vec<String> = sharded
            .shard_sources(&sharded.shards[1])
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            second,
            [
                "https://exact.example/model-00002-of-00002.safetensors",
                "https://mirror.example/repo/*"
            ]
        );
        assert_eq!(sharded.shard_sources(&sharded.shards[0]).count(), 1);
        assert_eq!(sharded.size, Some(27000));

        // A shard needs a source, a pin and a size; a path stays relative.
        let only_urls = refused(
            r#"
bad-shards:
  $kind: resource
  $file: index.json
  $format: safetensors-index
  $sources: [{ $url: "https://x/index.json" }]
  $shards: [{ $file: a.safetensors, $sha256: "5000000000000000000000000000000000000000000000000000000000000004", $size: 1 }]
"#,
        );
        assert!(
            only_urls.mentions("the shard `a.safetensors` has no source"),
            "{only_urls}"
        );
        let pinless = refused(
            "p: { $kind: resource, $file: i.json, $format: f, $sources: [{ $base_url: \"https://x\" }], $shards: [{ $file: a, $sha256: nope, $size: 1 }] }\n",
        );
        assert!(
            pinless.mentions("the shard `a`: `nope` is not a lowercase hex SHA-256"),
            "{pinless}"
        );
        let sizeless = refused(
            "p: { $kind: resource, $file: i.json, $format: f, $sources: [{ $base_url: \"https://x\" }], $shards: [{ $file: a, $sha256: \"5000000000000000000000000000000000000000000000000000000000000004\" }] }\n",
        );
        assert!(sizeless.mentions("missing field `$size`"), "{sizeless}");
        let rooted = refused(
            "p: { $kind: resource, $file: /etc/passwd, $format: f, $sources: [{ $url: \"https://x\" }] }\n",
        );
        assert!(
            rooted.mentions("`/etc/passwd` is not a file path"),
            "{rooted}"
        );

        // An update adds sources, and nothing else.
        let reshard = stack
            .clone()
            .push_yaml(
                "reshard",
                r#"
local:
  $kind: namespace
  $merge: update
  sharded:
    $kind: resource
    $merge: update
    $file: model.safetensors.index.json
    $format: safetensors-index
    $sha256: "5000000000000000000000000000000000000000000000000000000000000001"
    $size: 27000
    $sources: [{ $base_url: "https://other.example/repo" }]
"#,
            )
            .unwrap_err();
        assert!(
            reshard.mentions("matches every field but `$sources`"),
            "{reshard}"
        );
    }

    #[test]
    fn test_references_resolve_from_where_they_are_written() {
        let stack = fixtures::example_stack().unwrap();
        // A build's `checkpoint` is its own; the family's `vocabulary` is
        // found by climbing; the class's `vocabulary` resolves at the class.
        let v1 = stack
            .resolve_build(&n("openai/whisper/large/large-v1"))
            .unwrap();
        assert_eq!(
            v1.resources["checkpoint"].resource,
            n("openai/whisper/large/large-v1/checkpoint")
        );
        assert_eq!(
            v1.resources["vocabulary"].resource,
            n("openai/whisper/vocabulary")
        );
        assert_eq!(
            v1.resources["vocabulary"].set_by,
            n("openai/whisper/multilingual")
        );
        assert_eq!(v1.release.unwrap().group, n("openai/whisper/large/release"));
        // A leading `/` resolves from the root only.
        let errors = refused(
            r#"
openai:
  $kind: namespace
  $merge: update
  whisper:
    $kind: namespace
    $merge: update
    large:
      $kind: namespace
      $merge: update
      rooted:
        $kind: build
        $description: "Asks for a root `release`."
        $template_class: multilingual
        $join: [{ $group: /release, $released: "2030" }]
        $resources: { checkpoint: large-v1/checkpoint }
        parameters: 1
"#,
        );
        assert!(
            errors.mentions("no release group `/release` is visible"),
            "{errors}"
        );
        // The nearest name is taken whatever its kind; a wrong kind is an
        // error, not a reason to keep climbing.
        let errors = refused(
            r#"
openai:
  $kind: namespace
  $merge: update
  whisper:
    $kind: namespace
    $merge: update
    large:
      $kind: namespace
      $merge: update
      wrong-kind:
        $kind: build
        $description: "Its `vocabulary` dependency names a group."
        $template_class: multilingual
        $resources: { checkpoint: large-v1/checkpoint, vocabulary: release }
        parameters: 1
"#,
        );
        assert!(
            errors.mentions(
                "`release` resolves to `openai/whisper/large/release`, which is a release"
            ),
            "{errors}"
        );
    }

    #[test]
    fn test_aliases_name_entries_anywhere_and_merge() {
        let mut stack = fixtures::example_stack().unwrap();
        stack
            .push_yaml(
                "aliases",
                r#"
large:
  $kind: alias
  $description: "The newest large Whisper."
  $ref: { $path: openai/whisper/large/large-v3, $kind: build }
openai:
  $kind: namespace
  $merge: update
  whisper:
    $kind: namespace
    $merge: update
    turbo: { $kind: alias, $ref: { $path: large/large-v3-turbo, $kind: build } }
"#,
            )
            .unwrap();
        let v3 = n("openai/whisper/large/large-v3");
        let turbo = n("openai/whisper/large/large-v3-turbo");
        // A lookup for the target's kind resolves through the alias, by any
        // spelling that finds it; a lookup for aliases finds the alias.
        assert_eq!(stack.find_build("large").unwrap(), v3);
        assert_eq!(stack.find_build("turbo").unwrap(), turbo);
        assert_eq!(stack.find_build("whisper/turbo").unwrap(), turbo);
        assert_eq!(stack.find_build("openai/whisper/turbo").unwrap(), turbo);
        assert_eq!(
            stack
                .find("turbo", &[EntryKind::Alias], LeafMatch::Exact)
                .unwrap(),
            n("openai/whisper/turbo")
        );
        assert_eq!(stack.aliases_of(&v3), [n("large")]);
        // The template-class lookup doesn't see a build's alias.
        assert!(
            stack
                .find("large", &[EntryKind::TemplateClass], LeafMatch::Exact)
                .is_err()
        );
        // A later catalog retargets an alias by updating or replacing it.
        stack
            .push_yaml(
                "retarget",
                r#"
large:
  $kind: alias
  $merge: update
  $ref: { $path: openai/whisper/large/large-v2, $kind: build }
"#,
            )
            .unwrap();
        assert_eq!(
            stack.find_build("large").unwrap(),
            n("openai/whisper/large/large-v2")
        );
        assert_eq!(
            stack.declaration(&n("large")).unwrap().history().len(),
            2,
            "declared, then updated"
        );
        // An alias is checked like any reference, names no alias, and sits
        // where a namespace could hold it.
        let wrong = refused(
            "big: { $kind: alias, $ref: { $path: openai/whisper/large/large-v3, $kind: template_class } }\n",
        );
        assert!(
            wrong.mentions("which is a build, not a template_class"),
            "{wrong}"
        );
        let missing = refused("big: { $kind: alias, $ref: { $path: large-v9, $kind: build } }\n");
        assert!(
            missing.mentions("no build `large-v9` is visible from `big`"),
            "{missing}"
        );
        let chained = refused(
            r#"
big: { $kind: alias, $ref: { $path: openai/whisper/large/large-v3, $kind: build } }
bigger: { $kind: alias, $ref: { $path: big, $kind: alias } }
"#,
        );
        assert!(chained.mentions("not another alias"), "{chained}");
        let in_build = refused(
            r#"
openai:
  $kind: namespace
  $merge: update
  whisper:
    $kind: namespace
    $merge: update
    large:
      $kind: namespace
      $merge: update
      large-v4:
        $kind: build
        $description: "Not yet."
        $template_class: multilingual
        nick: { $kind: alias, $ref: { $path: large-v3, $kind: build } }
"#,
        );
        assert!(
            in_build.mentions("a build can't hold an alias"),
            "{in_build}"
        );
        // It survives the round trip through the keyed form.
        let doc = PretrainedCatalog::from_yaml(
            "big: { $kind: alias, $ref: { $path: a/b, $kind: build } }\n",
        )
        .unwrap();
        let again = PretrainedCatalog::from_yaml(&doc.to_yaml().unwrap()).unwrap();
        assert_eq!(again, doc);
    }

    #[test]
    fn test_a_later_catalog_cannot_change_an_earlier_reference() {
        // A name nearer to an earlier catalog's reference than what it
        // resolves to would capture it, and is refused.
        let tag = refused(
            r#"
qwen:
  $kind: namespace
  $merge: update
  qwen3:
    $kind: namespace
    $merge: update
    parameters: { $kind: tag, $type: $count, $description: "Nearer than the root's." }
"#,
        );
        assert!(
            tag.mentions("it would capture `parameters`, written in `qwen/qwen3/"),
            "{tag}"
        );
        assert!(tag.mentions("which resolves to `parameters`"), "{tag}");
        // A contract is named in `$contracts`, which no other catalog
        // writes: a namesake in the namespace changes nothing.
        let mut stack = fixtures::example_stack().unwrap();
        stack
            .push_yaml(
                "namesake",
                r#"
qwen:
  $kind: namespace
  $merge: update
  qwen3-lm: { $kind: namespace, $description: "Not the contract." }
"#,
            )
            .unwrap();
        let b = stack.resolve_build(&n("qwen/qwen3/qwen3-0.6b")).unwrap();
        assert_eq!(b.contract, Some(n("$contracts/qwen3-lm")));
    }

    #[test]
    fn test_template_classes_nest_without_conferring_membership() {
        let stack = fixtures::example_stack().unwrap();
        let class_scope = n("qwen/qwen3");
        // The class sits in `qwen/qwen3`, and its builds are members of
        // `qwen/qwen3` because they are placed there, not because of it.
        let mut stack2 = stack.clone();
        stack2
            .push_yaml(
                "elsewhere",
                r#"
elsewhere:
  $kind: namespace
  $facet: family
  $description: "Another family."
  borrowed-class:
    $kind: build
    $description: "Uses a Qwen3 class from another branch."
    $template_class: /qwen/qwen3/hybrid
    $join: [qwen]
    $resources: { checkpoint: /qwen/qwen3/qwen3-0.6b/checkpoint, config: /qwen/qwen3/qwen3-0.6b/config, tokenizer: /qwen/qwen3/qwen3-0.6b/tokenizer }
    parameters: 1
"#,
            )
            .unwrap();
        let memberships = stack2.build_memberships(&n("elsewhere/borrowed-class"));
        assert!(!memberships.contains(&class_scope), "{memberships:?}");
        // The class still gives it its contract.
        let b = stack2
            .resolve_build(&n("elsewhere/borrowed-class"))
            .unwrap();
        assert_eq!(b.contract, Some(n("$contracts/qwen3-lm")));
    }

    #[test]
    fn test_join_rules() {
        let cycle = refused(
            r#"
a: { $kind: namespace, $description: "A.", $join: [b] }
b: { $kind: namespace, $description: "B.", $join: [a] }
"#,
        );
        assert!(cycle.mentions("a member of itself"), "{cycle}");

        let plain_release = refused(
            "x: { $kind: namespace, $description: \"X.\", $join: [openai/whisper/large/release] }\n",
        );
        assert!(
            plain_release.has(CatalogErrorKind::Release),
            "{plain_release}"
        );

        let not_a_group =
            refused("x: { $kind: namespace, $description: \"X.\", $join: [parameters] }\n");
        assert!(
            not_a_group.mentions("is a tag, not a group"),
            "{not_a_group}"
        );
    }

    #[test]
    fn test_release_overlap_is_refused() {
        let errors = refused(
            r#"
openai:
  $kind: namespace
  $merge: update
  whisper:
    $kind: namespace
    $merge: update
    large-v4:
      $kind: build
      $description: "An imprecisely dated release."
      $template_class: multilingual
      $join: [{ $group: openai/whisper/large/release, $released: "2023" }]
      $resources: { checkpoint: openai/whisper/large/large-v4/checkpoint, vocabulary: openai/whisper/vocabulary }
      parameters: 1
      checkpoint:
        $kind: resource
        $file: v4.pt
        $format: pytorch
        $sources: [{ $url: "https://x/v4.pt" }]
"#,
        );
        assert!(
            errors.mentions("overlap in `openai/whisper/large/release`"),
            "{errors}"
        );
    }

    #[test]
    fn test_template_class_rules() {
        let mut empty = PretrainedCatalogStack::new();
        let errors = empty
            .push_yaml(
                "only",
                "$contracts: {}\nb: { $kind: build, $description: \"B.\" }\n",
            )
            .unwrap_err();
        assert!(errors.mentions("the prelude declares it"), "{errors}");

        let unknown = refused("t: { $kind: template_class, $base: nope, $description: \"T.\" }\n");
        assert!(
            unknown.mentions("no template class `nope` is visible from `t`"),
            "{unknown}"
        );

        let looped = refused(
            r#"
t1: { $kind: template_class, $base: t2, $description: "T1." }
t2: { $kind: template_class, $base: t1, $description: "T2." }
"#,
        );
        assert!(looped.mentions("loops"), "{looped}");
    }

    #[test]
    fn test_declaration_checks() {
        let errors = refused(
            r#"
country: { $kind: facet, $description: "" }
mood: { $kind: axis, $axis: bool, $members: [happy], $description: "M." }
size: { $kind: axis, $axis: categorical, $members: [one], $description: "S." }
contract: { $kind: tag, $type: $text, $description: "Shadows the implied tag." }
no_timestamps: { $kind: tag, $type: $bool, $description: "A negative bool tag." }
non-streaming: { $kind: axis, $axis: bool, $description: "A negative bool axis." }
notes: { $kind: tag, $type: $text, $description: "Not a bool, and not a negative." }
nothing: { $kind: tag, $type: $bool, $description: "`no` without a separator is a word." }
r: { $kind: resource, $file: a/../b, $format: "", $sha256: ABC, $sources: [] }
"#,
        );
        for text in [
            "no description",
            "members are generated",
            "two or more members",
            "not a file path",
            "names its format",
            "lowercase hex SHA-256",
            "at least one source",
            "the tag every build's contract implies",
            "a bool tag is named by its positive: `timestamps`, not `no_timestamps`",
            "a bool axis is named by its positive: `streaming`, not `non-streaming`",
        ] {
            assert!(errors.mentions(text), "{text}: {errors}");
        }
        assert!(!errors.mentions("`notes`"), "{errors}");
        assert!(!errors.mentions("`nothing`"), "{errors}");
    }
}
