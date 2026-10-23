//! Mapping a checkpoint's tensor names onto a module's paths.
use std::{
    collections::{
        BTreeMap,
        BTreeSet,
        HashMap,
        HashSet,
    },
    fmt,
    sync::Arc,
};

use burn::tensor::TensorData;

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// A map from a checkpoint's tensor names (a `transformers` or `NeMo` state
/// dict's, say) to the module paths they load into, built from rules.
///
/// A checkpoint names its tensors after its own module tree; bunsen's tree
/// differs. A model's map has two halves, kept apart so each changes in one
/// place: **routing**, which prefix of the checkpoint is which field of the
/// model, owned by the kit; and **component maps**, how one block's tensors
/// are named in one checkpoint format, owned by the block and
/// [mounted](Self::mount) by the kit wherever the block sits. Nesting a block
/// under a new field changes one mount; renaming a block's field changes its
/// own map, beside it.
///
/// [`apply`](Self::apply) tries the rules in order, and the first that
/// claims a name decides it: a path, or `None` to drop the tensor (a buffer
/// the module has no parameter for). A name no rule claims takes the
/// fallback: kept as it is ([`keep_rest`](Self::keep_rest)), dropped
/// ([`drop_rest`](Self::drop_rest)), or, for a mounted map, handed back to
/// the map it is mounted in, as if the mount had not claimed it
/// ([`pass_rest`](Self::pass_rest)).
///
/// A map can also say how a tensor's data changes on the way
/// ([`transform`](Self::transform)): for a weight whose axes follow a layout
/// the checkpoint's module chose differently from the block that loads it.
/// [`target`](Self::target) gives the path and that transform together; a
/// mounted map's transforms come along with its paths.
///
/// ```rust
/// use bunsen::burner::module::Rekey;
///
/// let block = Rekey::keep_rest()
///     .rename("self_attn.q_proj.", "attn.query.")
///     .rename("fc1.", "mlp.linear1.");
/// let model = Rekey::drop_rest()
///     .drop_suffix(".num_batches_tracked")
///     .mount_indexed("model.layers.", "encoder.layers.", block)
///     .rename("model.proj.", "projection.");
///
/// assert_eq!(
///     model
///         .apply("model.layers.3.self_attn.q_proj.weight")
///         .as_deref(),
///     Some("encoder.layers.3.attn.query.weight"),
/// );
/// assert_eq!(
///     model.apply("model.layers.0.norm.bias").as_deref(),
///     Some("encoder.layers.0.norm.bias"),
/// );
/// assert_eq!(
///     model.apply("model.proj.weight").as_deref(),
///     Some("projection.weight")
/// );
/// assert_eq!(model.apply("model.rotary.inv_freq"), None);
/// ```
#[derive(Clone)]
pub struct Rekey {
    rules: Vec<Rule>,
    transforms: Vec<(String, TensorTransform)>,
    unclaimed: Unclaimed,
}

/// What a [`Rekey`] does with a name none of its rules claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unclaimed {
    Keep,
    Drop,
    Pass,
}

/// How a [`Rekey::mount_indexed_with`] numbers the stack it mounts: how a
/// checkpoint's index `i` under the mount becomes the module's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Indices {
    /// The module's `i`: the checkpoint numbers the stack as the module does.
    AsIs,

    /// The module's `i - first`: the checkpoint counts from `first`
    /// (torchvision's `layer1` … `layer4`, a stack of `conv1`, `conv2`). An
    /// index below `first` is dropped.
    From(usize),

    /// Matched to the module's own indices under the mount, once the map
    /// sees both ([`Rekey::plan_for`]): when every index the checkpoint has
    /// is one the module has, each is kept, so a missing entry stays missing;
    /// otherwise, when the two have as many indices, the checkpoint's are
    /// laid onto the module's in order (an `nn.Sequential` whose
    /// parameter-free entries leave gaps: `0, 2, 4` onto `0, 1, 2`); anything
    /// else is a conflict the plan refuses. Without the module's paths
    /// ([`Rekey::target`]), as [`AsIs`](Self::AsIs).
    Module,
}

/// What a [`Rekey::custom`] rule says of a name.
pub type Claim = Option<Option<String>>;

/// A change to a tensor's data as it loads: a permute or reshape on the
/// host, applied before the store's own adapter.
pub type TensorTransform = Arc<dyn Fn(TensorData) -> TensorData + Send + Sync>;

/// Where a checkpoint's tensor loads, and how its data changes on the way.
#[derive(Clone)]
pub struct Target {
    /// The module path.
    pub path: String,

    /// The change to its data, if any.
    pub transform: Option<TensorTransform>,
}

impl fmt::Debug for Target {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("Target")
            .field("path", &self.path)
            .field("transform", &self.transform.is_some())
            .finish()
    }
}

/// A [`Rekey`] resolved over a whole checkpoint's names
/// ([`Rekey::plan`]).
#[derive(Debug)]
pub struct RekeyPlan {
    /// Each name, in order, and where it loads; `None` for a dropped name.
    pub targets: Vec<(String, Option<Target>)>,

    /// Each module path more than one name loads into, with those names.
    pub collisions: Vec<(String, Vec<String>)>,

    /// The module paths of transforms no name reached: a map that names a
    /// path its own rules never produce.
    pub orphan_transforms: Vec<String>,

    /// Rules that claimed no name, as written: shadowed by an earlier rule,
    /// or for a part this checkpoint lacks. A mount that claimed nothing is
    /// listed alone, not its map's rules.
    pub unused_rules: Vec<String>,

    /// [`Indices::Module`] stacks whose indices could not be matched to the
    /// module's, each with both sets; their names are dropped. Only a plan
    /// that sees the module ([`Rekey::plan_for`]) has any.
    pub index_conflicts: Vec<String>,

    /// Names whose target lies under no module of the model, with that
    /// target: kept as they are by a map, or mapped onto a part the model
    /// does not have. Only a plan that sees the module
    /// ([`Rekey::plan_for`]) has any; they are reported, not refused (the
    /// store reports them unused).
    pub unknown_paths: Vec<(String, String)>,
}

impl RekeyPlan {
    /// Refuses a plan that cannot load as written: two names onto one path,
    /// a transform no name reaches, or a stack whose indices do not match
    /// the module's. Unused rules and unknown paths are not refused; a
    /// block's map may cover parts a checkpoint lacks.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// naming the collisions, orphaned transforms and index conflicts,
    /// listed in the details.
    pub fn check(&self) -> BunsenResult<()> {
        if self.collisions.is_empty()
            && self.orphan_transforms.is_empty()
            && self.index_conflicts.is_empty()
        {
            return Ok(());
        }
        let mut details: Vec<String> = self
            .collisions
            .iter()
            .map(|(path, names)| format!("{path} <- {}", names.join(", ")))
            .collect();
        details.extend(
            self.orphan_transforms
                .iter()
                .map(|path| format!("transform for {path}: no name loads there")),
        );
        details.extend(
            self.index_conflicts
                .iter()
                .map(|conflict| format!("indices: {conflict}")),
        );
        Err(BunsenError::invalid_resource(format!(
            "the key map loads {} paths from more than one name, has {} transforms no name \
             reaches, and {} stacks whose indices do not match the module's",
            self.collisions.len(),
            self.orphan_transforms.len(),
            self.index_conflicts.len()
        ))
        .with_details(details.join("\n")))
    }
}

/// `first`, then `then`.
fn compose(
    first: Option<TensorTransform>,
    then: Option<TensorTransform>,
) -> Option<TensorTransform> {
    match (first, then) {
        (Some(a), Some(b)) => Some(Arc::new(move |data| b(a(data)))),
        (a, b) => a.or(b),
    }
}

/// `target`, its path under `prefix`.
fn prefixed(
    prefix: &str,
    target: Target,
) -> Target {
    Target {
        path: format!("{prefix}{}", target.path),
        transform: target.transform,
    }
}

fn plain(path: String) -> Target {
    Target {
        path,
        transform: None,
    }
}

#[derive(Clone)]
enum Rule {
    DropSuffix(String),
    Exact(String, String),
    Rename(String, String),
    Mount(String, String, Rekey),
    MountIndexed(String, String, Indices, Rekey),
    Custom(Arc<dyn Fn(&str) -> Claim + Send + Sync>),
}

/// The rules and transforms a set of names used, by address: what
/// [`Rekey::plan`] reports the rest of.
#[derive(Default)]
struct Trace {
    rules: HashSet<usize>,
    transforms: HashSet<usize>,
}

/// A stack's key in a plan: its rule, by address, and the checkpoint-side
/// prefix it sits under (an enclosing stack's index included).
type StackKey = (usize, String);

/// What a claim carries down the map: the trace, and, for
/// [`Indices::Module`], the module's paths, the indices each stack saw
/// (from a first pass over the names), and the index maps made from them.
#[derive(Default)]
struct Cx<'a> {
    trace: Trace,
    module: Option<&'a BTreeSet<String>>,
    seen: BTreeMap<StackKey, BTreeSet<usize>>,
    resolving: bool,
    index_maps: HashMap<StackKey, Option<BTreeMap<usize, usize>>>,
    index_conflicts: BTreeSet<String>,
}

impl Cx<'_> {
    /// The module index of a [`Indices::Module`] stack's checkpoint index
    /// `index`, the stack keyed by `key` and mounted at the module path
    /// `to`: `None` to drop the name, the stack's indices matching none of
    /// the module's.
    fn module_index(
        &mut self,
        key: &StackKey,
        to: &str,
        index: usize,
    ) -> Option<usize> {
        let (Some(module), true) = (self.module, self.resolving) else {
            return Some(index);
        };
        if !self.index_maps.contains_key(key) {
            let seen = self.seen.get(key).cloned().unwrap_or_default();
            let theirs: BTreeSet<usize> = module
                .range(to.to_string()..)
                .take_while(|path| path.starts_with(to))
                .filter_map(|path| path[to.len()..].split_once('.')?.0.parse().ok())
                .collect();
            let map = if seen.is_subset(&theirs) {
                Some(seen.iter().map(|&i| (i, i)).collect())
            } else if seen.len() == theirs.len() {
                Some(seen.iter().copied().zip(theirs.iter().copied()).collect())
            } else {
                self.index_conflicts.insert(format!(
                    "{}N.: the checkpoint has {seen:?}, the module ({to}N.) {theirs:?}",
                    key.1
                ));
                None
            };
            self.index_maps.insert(key.clone(), map);
        }
        self.index_maps[key].as_ref()?.get(&index).copied()
    }
}

fn rule_addr(rule: &Rule) -> usize {
    std::ptr::from_ref(rule) as usize
}

fn transform_addr(f: &TensorTransform) -> usize {
    Arc::as_ptr(f).cast::<()>() as usize
}

impl Rule {
    /// The rule as it was written, for a report.
    fn describe(&self) -> String {
        match self {
            Rule::DropSuffix(suffix) => format!("drop_suffix({suffix:?})"),
            Rule::Exact(from, to) => format!("exact({from:?} -> {to:?})"),
            Rule::Rename(from, to) => format!("rename({from:?} -> {to:?})"),
            Rule::Mount(from, to, _) => format!("mount({from:?} -> {to:?})"),
            Rule::MountIndexed(from, to, Indices::AsIs, _) => {
                format!("mount_indexed({from:?} -> {to:?})")
            }
            Rule::MountIndexed(from, to, indices, _) => {
                format!("mount_indexed({from:?} -> {to:?}, {indices:?})")
            }
            Rule::Custom(_) => "custom".to_string(),
        }
    }

    /// `None` when the rule does not claim `name`; `Some(target)` when it
    /// does, `target` being `None` for a dropped tensor. `from_prefix` and
    /// `to_prefix` are where the enclosing mounts put the map this rule is
    /// in, on the checkpoint's side and the module's.
    fn claim(
        &self,
        name: &str,
        from_prefix: &str,
        to_prefix: &str,
        cx: &mut Cx,
    ) -> Option<Option<Target>> {
        match self {
            Rule::DropSuffix(suffix) => name.ends_with(suffix.as_str()).then_some(None),
            Rule::Exact(from, to) => (name == from).then(|| Some(plain(to.clone()))),
            Rule::Rename(from, to) => name
                .strip_prefix(from.as_str())
                .map(|rest| Some(plain(format!("{to}{rest}")))),
            Rule::Mount(from, to, sub) => sub
                .claim(
                    name.strip_prefix(from.as_str())?,
                    &format!("{from_prefix}{from}"),
                    &format!("{to_prefix}{to}"),
                    cx,
                )
                .map(|t| t.map(|t| prefixed(to, t))),
            Rule::MountIndexed(from, to, indices, sub) => {
                let (digits, rest) = name.strip_prefix(from.as_str())?.split_once('.')?;
                if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                let index: usize = digits.parse().ok()?;
                let key = (rule_addr(self), format!("{from_prefix}{from}"));
                cx.seen.entry(key.clone()).or_default().insert(index);
                let mapped = match indices {
                    Indices::AsIs => Some(index),
                    Indices::From(first) => index.checked_sub(*first),
                    Indices::Module => cx.module_index(&key, &format!("{to_prefix}{to}"), index),
                };
                let Some(mapped) = mapped else {
                    // Claimed, and dropped: below the stack's first index, or
                    // matching none of the module's.
                    return Some(None);
                };
                let to = format!("{to}{mapped}.");
                sub.claim(
                    rest,
                    &format!("{from_prefix}{from}{digits}."),
                    &format!("{to_prefix}{to}"),
                    cx,
                )
                .map(|t| t.map(|t| prefixed(&to, t)))
            }
            Rule::Custom(f) => f(name).map(|path| path.map(plain)),
        }
    }
}

impl Rekey {
    /// A map whose unclaimed names load as they are: for a checkpoint that
    /// mostly shares the module's names.
    pub fn keep_rest() -> Self {
        Self {
            rules: Vec::new(),
            transforms: Vec::new(),
            unclaimed: Unclaimed::Keep,
        }
    }

    /// A map whose unclaimed names are dropped: for a checkpoint whose every
    /// loaded name a rule accounts for.
    pub fn drop_rest() -> Self {
        Self {
            rules: Vec::new(),
            transforms: Vec::new(),
            unclaimed: Unclaimed::Drop,
        }
    }

    /// A map that claims only what its rules claim: mounted, a name none of
    /// them claims goes back to the enclosing map's next rule, as if the
    /// mount had not matched; for a component whose names share a prefix
    /// with others it does not own (`conv1`, `conv2` beside `conv_out`). At
    /// the top of a map, an unclaimed name is dropped.
    pub fn pass_rest() -> Self {
        Self {
            rules: Vec::new(),
            transforms: Vec::new(),
            unclaimed: Unclaimed::Pass,
        }
    }

    fn with(
        mut self,
        rule: Rule,
    ) -> Self {
        self.rules.push(rule);
        self
    }

    /// Drops every name ending in `suffix` (`.num_batches_tracked`).
    pub fn drop_suffix(
        self,
        suffix: impl Into<String>,
    ) -> Self {
        self.with(Rule::DropSuffix(suffix.into()))
    }

    /// Maps the one name `from` to `to`.
    pub fn exact(
        self,
        from: impl Into<String>,
        to: impl Into<String>,
    ) -> Self {
        self.with(Rule::Exact(from.into(), to.into()))
    }

    /// Replaces the prefix `from` with `to`, keeping the rest of the name.
    pub fn rename(
        self,
        from: impl Into<String>,
        to: impl Into<String>,
    ) -> Self {
        self.with(Rule::Rename(from.into(), to.into()))
    }

    /// Maps everything under the prefix `from` by `sub`, and puts the result
    /// under `to`: a component's map, wherever the component sits. What
    /// `sub` drops is dropped.
    pub fn mount(
        self,
        from: impl Into<String>,
        to: impl Into<String>,
        sub: Rekey,
    ) -> Self {
        self.with(Rule::Mount(from.into(), to.into(), sub))
    }

    /// [`mount`](Self::mount) for a stack: `{from}{i}.{rest}` maps to
    /// `{to}{i}.{sub(rest)}`, the layer index carried across. Only a decimal
    /// index is claimed; `{from}norm.{rest}` is left to the next rule.
    pub fn mount_indexed(
        self,
        from: impl Into<String>,
        to: impl Into<String>,
        sub: Rekey,
    ) -> Self {
        self.mount_indexed_with(from, to, Indices::AsIs, sub)
    }

    /// [`mount_indexed`](Self::mount_indexed), the checkpoint's index `i`
    /// becoming the module's as `indices` says: `{from}{i}.{rest}` maps to
    /// `{to}{j}.{sub(rest)}`.
    ///
    /// ```rust
    /// use bunsen::burner::module::{
    ///     Indices,
    ///     Rekey,
    /// };
    ///
    /// // torchvision's `layer1` … `layer4` onto `layers.0` … `layers.3`.
    /// let rekey = Rekey::drop_rest().mount_indexed_with(
    ///     "layer",
    ///     "layers.",
    ///     Indices::From(1),
    ///     Rekey::keep_rest(),
    /// );
    /// assert_eq!(
    ///     rekey.apply("layer1.0.conv1.weight").as_deref(),
    ///     Some("layers.0.0.conv1.weight")
    /// );
    /// assert_eq!(rekey.apply("layer0.0.conv1.weight"), None);
    /// ```
    pub fn mount_indexed_with(
        self,
        from: impl Into<String>,
        to: impl Into<String>,
        indices: Indices,
        sub: Rekey,
    ) -> Self {
        self.with(Rule::MountIndexed(from.into(), to.into(), indices, sub))
    }

    /// A rule the others cannot say: `f` returns `None` to leave a name to
    /// the next rule, `Some(None)` to drop it, `Some(Some(path))` to map it.
    /// For arithmetic on names, such as a flat `ModuleList`'s indices onto
    /// nested modules.
    pub fn custom(
        self,
        f: impl Fn(&str) -> Claim + Send + Sync + 'static,
    ) -> Self {
        self.with(Rule::Custom(Arc::new(f)))
    }

    /// Changes the data of the tensor this map sends to `path` (a module
    /// path, relative to this map) by `f`: after any change a mounted map
    /// made to it.
    pub fn transform(
        mut self,
        path: impl Into<String>,
        f: impl Fn(TensorData) -> TensorData + Send + Sync + 'static,
    ) -> Self {
        self.transforms.push((path.into(), Arc::new(f)));
        self
    }

    /// Where `name` loads and how its data changes on the way, or `None` to
    /// drop it.
    pub fn target(
        &self,
        name: &str,
    ) -> Option<Target> {
        self.claim(name, "", "", &mut Cx::default()).flatten()
    }

    /// `None` when this map does not claim `name` (only a
    /// [`pass_rest`](Self::pass_rest) map leaves one unclaimed); `Some(target)`
    /// when it does, `target` being `None` for a dropped tensor.
    fn claim(
        &self,
        name: &str,
        from_prefix: &str,
        to_prefix: &str,
        cx: &mut Cx,
    ) -> Option<Option<Target>> {
        let claimed = self
            .rules
            .iter()
            .find_map(|rule| {
                let claim = rule.claim(name, from_prefix, to_prefix, cx)?;
                cx.trace.rules.insert(rule_addr(rule));
                Some(claim)
            })
            .or_else(|| match self.unclaimed {
                Unclaimed::Keep => Some(Some(plain(name.to_string()))),
                Unclaimed::Drop => Some(None),
                Unclaimed::Pass => None,
            })?;
        Some(claimed.map(|target| self.with_own_transform(target, &mut cx.trace)))
    }

    /// `target`, with this map's transform for its path composed after any
    /// it already carries.
    fn with_own_transform(
        &self,
        target: Target,
        trace: &mut Trace,
    ) -> Target {
        let own = self
            .transforms
            .iter()
            .find(|(path, _)| *path == target.path)
            .map(|(_, f)| {
                trace.transforms.insert(transform_addr(f));
                f.clone()
            });
        Target {
            transform: compose(target.transform, own),
            path: target.path,
        }
    }

    /// The module path `name` loads into, or `None` to drop it.
    pub fn apply(
        &self,
        name: &str,
    ) -> Option<String> {
        self.target(name).map(|t| t.path)
    }

    /// The map resolved over a whole checkpoint's names: each name's target,
    /// and what only the whole list shows (two names onto one path, a
    /// transform no name reached, a rule that claimed nothing).
    ///
    /// ```rust
    /// use bunsen::burner::module::Rekey;
    ///
    /// let rekey = Rekey::keep_rest().rename("model.", "").mount_indexed(
    ///     "model.layers.",
    ///     "blocks.",
    ///     Rekey::keep_rest(),
    /// );
    /// let plan = rekey.plan(["model.layers.0.w", "layers.0.w"]);
    /// // `rename("model.", "")` claims every `model.` name first.
    /// assert_eq!(
    ///     plan.unused_rules,
    ///     ["mount_indexed(\"model.layers.\" -> \"blocks.\")"]
    /// );
    /// assert_eq!(plan.collisions.len(), 1);
    /// assert!(plan.check().is_err());
    /// ```
    pub fn plan<S: AsRef<str>>(
        &self,
        names: impl IntoIterator<Item = S>,
    ) -> RekeyPlan {
        self.plan_in(names, None)
    }

    /// [`plan`](Self::plan), seeing the module the names load into, by its
    /// parameter paths: [`Indices::Module`] stacks are matched to the
    /// module's indices, and the plan reports the names whose target lies
    /// under no module of the model
    /// ([`unknown_paths`](RekeyPlan::unknown_paths)). A path's last segment
    /// is not compared (a norm's `weight` loads as `gamma`).
    ///
    /// ```rust
    /// use std::collections::BTreeSet;
    ///
    /// use bunsen::burner::module::{
    ///     Indices,
    ///     Rekey,
    /// };
    ///
    /// // An `nn.Sequential` whose `ReLU`s sat at 1 and 3.
    /// let rekey = Rekey::keep_rest().mount_indexed_with(
    ///     "fc.",
    ///     "fc.",
    ///     Indices::Module,
    ///     Rekey::keep_rest(),
    /// );
    /// let module: BTreeSet<String> =
    ///     ["fc.0.weight", "fc.1.weight", "fc.2.weight"]
    ///         .map(String::from)
    ///         .into();
    /// let plan =
    ///     rekey.plan_for(["fc.0.weight", "fc.2.weight", "fc.4.weight"], &module);
    /// let paths: Vec<_> = plan
    ///     .targets
    ///     .iter()
    ///     .map(|(_, t)| t.as_ref().unwrap().path.as_str())
    ///     .collect();
    /// assert_eq!(paths, ["fc.0.weight", "fc.1.weight", "fc.2.weight"]);
    ///
    /// // A missing entry stays missing.
    /// let plan = rekey.plan_for(["fc.0.weight", "fc.2.weight"], &module);
    /// assert_eq!(plan.targets[1].1.as_ref().unwrap().path, "fc.2.weight");
    /// ```
    pub fn plan_for<S: AsRef<str>>(
        &self,
        names: impl IntoIterator<Item = S>,
        module_paths: &BTreeSet<String>,
    ) -> RekeyPlan {
        self.plan_in(names, Some(module_paths))
    }

    fn plan_in<S: AsRef<str>>(
        &self,
        names: impl IntoIterator<Item = S>,
        module: Option<&BTreeSet<String>>,
    ) -> RekeyPlan {
        let names: Vec<String> = names.into_iter().map(|n| n.as_ref().to_string()).collect();
        let mut cx = Cx {
            module,
            ..Cx::default()
        };
        if module.is_some() {
            // Which indices each stack sees depends only on the names, not on
            // how any stack maps them: a first pass collects them.
            for name in &names {
                self.claim(name, "", "", &mut cx);
            }
            cx.trace = Trace::default();
            cx.resolving = true;
        }
        let targets: Vec<(String, Option<Target>)> = names
            .into_iter()
            .map(|name| {
                let target = self.claim(&name, "", "", &mut cx).flatten();
                (name, target)
            })
            .collect();
        let trace = std::mem::take(&mut cx.trace);
        let index_conflicts = cx.index_conflicts.into_iter().collect();

        let unknown_paths = match module {
            None => Vec::new(),
            Some(module) => {
                let parents: BTreeSet<&str> = module
                    .iter()
                    .flat_map(|path| path.match_indices('.').map(move |(i, _)| &path[..i]))
                    .collect();
                targets
                    .iter()
                    .filter_map(|(name, target)| {
                        let path = &target.as_ref()?.path;
                        let parent = path.rsplit_once('.').map_or("", |(p, _)| p);
                        (!parents.contains(parent)).then(|| (name.clone(), path.clone()))
                    })
                    .collect()
            }
        };

        let mut by_path: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        for (name, target) in &targets {
            if let Some(target) = target {
                by_path.entry(&target.path).or_default().push(name.clone());
            }
        }
        let collisions = by_path
            .into_iter()
            .filter(|(_, names)| names.len() > 1)
            .map(|(path, names)| (path.to_string(), names))
            .collect();

        let mut unused_rules = Vec::new();
        let mut orphan_transforms = Vec::new();
        self.unused(&trace, "", "", &mut unused_rules, &mut orphan_transforms);
        RekeyPlan {
            targets,
            collisions,
            orphan_transforms,
            unused_rules,
            index_conflicts,
            unknown_paths,
        }
    }

    /// The rules and transforms of this map, and of the maps mounted by its
    /// used mounts, that `trace` never saw: rules by their checkpoint-side
    /// prefix (`from`), transforms by their module path (`to`).
    fn unused(
        &self,
        trace: &Trace,
        from_prefix: &str,
        to_prefix: &str,
        rules: &mut Vec<String>,
        transforms: &mut Vec<String>,
    ) {
        for (path, f) in &self.transforms {
            if !trace.transforms.contains(&transform_addr(f)) {
                transforms.push(format!("{to_prefix}{path}"));
            }
        }
        for rule in &self.rules {
            if !trace.rules.contains(&rule_addr(rule)) {
                let context = if from_prefix.is_empty() {
                    String::new()
                } else {
                    format!("under {from_prefix:?}: ")
                };
                rules.push(format!("{context}{}", rule.describe()));
                continue;
            }
            match rule {
                Rule::Mount(from, to, sub) => sub.unused(
                    trace,
                    &format!("{from_prefix}{from}"),
                    &format!("{to_prefix}{to}"),
                    rules,
                    transforms,
                ),
                Rule::MountIndexed(from, to, _, sub) => sub.unused(
                    trace,
                    &format!("{from_prefix}{from}N."),
                    &format!("{to_prefix}{to}N."),
                    rules,
                    transforms,
                ),
                _ => {}
            }
        }
    }

    /// [`apply`](Self::apply) as a closure, for
    /// [`apply_named_tensors`](crate::data::pretrained::apply_named_tensors).
    pub fn as_fn(&self) -> impl Fn(&str) -> Option<String> + '_ {
        move |name| self.apply(name)
    }
}

impl fmt::Debug for Rekey {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("Rekey")
            .field("rules", &self.rules.len())
            .field("transforms", &self.transforms.len())
            .field("unclaimed", &self.unclaimed)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first rule that claims a name decides it, and the fallback takes
    /// what none claims.
    #[test]
    fn test_first_claim_decides() {
        let keep = Rekey::keep_rest()
            .drop_suffix(".num_batches_tracked")
            .rename("a.", "x.")
            .rename("a.b.", "never.");
        assert_eq!(keep.apply("a.b.weight").as_deref(), Some("x.b.weight"));
        assert_eq!(keep.apply("a.num_batches_tracked"), None);
        assert_eq!(keep.apply("other.bias").as_deref(), Some("other.bias"));

        let drop = Rekey::drop_rest().exact("silence_embeds", "silence");
        assert_eq!(drop.apply("silence_embeds").as_deref(), Some("silence"));
        assert_eq!(drop.apply("silence_embeds.x"), None);
        assert_eq!(drop.apply("other.bias"), None);
    }

    /// A mount maps only what is under its prefix, by its own rules and its
    /// own fallback, and what it drops stays dropped.
    #[test]
    fn test_mounts() {
        let block = Rekey::drop_rest().rename("fc1.", "mlp.linear1.");
        let model = Rekey::keep_rest()
            .mount("tower.block.", "encoder.block.", block.clone())
            .mount_indexed("tower.layers.", "encoder.transformer.layers.", block);
        assert_eq!(
            model.apply("tower.block.fc1.weight").as_deref(),
            Some("encoder.block.mlp.linear1.weight")
        );
        assert_eq!(model.apply("tower.block.other.weight"), None);
        assert_eq!(
            model.apply("tower.layers.11.fc1.bias").as_deref(),
            Some("encoder.transformer.layers.11.mlp.linear1.bias")
        );
        assert_eq!(model.apply("tower.layers.11.other.bias"), None);
        assert_eq!(model.apply("head.weight").as_deref(), Some("head.weight"));
    }

    /// `mount_indexed` claims only a decimal index; anything else under its
    /// prefix is left to the next rule.
    #[test]
    fn test_mount_indexed_claims_only_indices() {
        let rekey = Rekey::keep_rest()
            .mount_indexed("layers.", "blocks.", Rekey::keep_rest())
            .rename("layers.", "stack.");
        assert_eq!(rekey.apply("layers.12.w").as_deref(), Some("blocks.12.w"));
        assert_eq!(
            rekey.apply("layers.norm.weight").as_deref(),
            Some("stack.norm.weight")
        );
        assert_eq!(rekey.apply("layers..w").as_deref(), Some("stack..w"));
        assert_eq!(rekey.apply("layers.3").as_deref(), Some("stack.3"));
    }

    /// A mounted `pass_rest` map hands back what it does not claim, to the
    /// enclosing map's next rule; what it drops stays dropped; at the top it
    /// drops what it does not claim.
    #[test]
    fn test_pass_rest() {
        let numbered = Rekey::pass_rest()
            .custom(|name| {
                let (n, rest) = name.split_once('.')?;
                let n: usize = n.parse().ok()?;
                Some((n > 0).then(|| format!("blocks.{}.{rest}", n - 1)))
            })
            .transform("blocks.0.weight", |data| data);
        let model = Rekey::keep_rest().mount("conv", "head.", numbered.clone());
        assert_eq!(
            model.apply("conv1.weight").as_deref(),
            Some("head.blocks.0.weight")
        );
        assert!(model.target("conv1.weight").unwrap().transform.is_some());
        assert_eq!(model.apply("conv0.weight"), None);
        assert_eq!(
            model.apply("conv_out.weight").as_deref(),
            Some("conv_out.weight")
        );
        assert_eq!(numbered.apply("x.weight"), None);
    }

    /// A plan reports two names onto one path, a transform no name reaches,
    /// and rules that claimed nothing, a used mount's by their prefix; a
    /// transform reached through a mount is not an orphan.
    #[test]
    fn test_plan_reports() {
        let block = Rekey::keep_rest()
            .rename("fc1.", "mlp.linear1.")
            .rename("fc2.", "mlp.linear2.")
            .transform("mlp.linear1.weight", |data| data)
            .transform("mlp.nothing.weight", |data| data);
        let model = Rekey::drop_rest()
            .mount_indexed("layers.", "blocks.", block)
            .exact("head", "out.weight")
            .exact("tied_head", "out.weight")
            .mount("unused.", "never.", Rekey::keep_rest().rename("a.", "b."));
        let plan = model.plan(["layers.0.fc1.weight", "head", "tied_head", "other"]);

        assert_eq!(
            plan.targets
                .iter()
                .map(|(n, t)| (n.as_str(), t.as_ref().map(|t| t.path.as_str())))
                .collect::<Vec<_>>(),
            [
                ("layers.0.fc1.weight", Some("blocks.0.mlp.linear1.weight")),
                ("head", Some("out.weight")),
                ("tied_head", Some("out.weight")),
                ("other", None),
            ]
        );
        assert_eq!(
            plan.collisions,
            [(
                "out.weight".to_string(),
                vec!["head".to_string(), "tied_head".to_string()]
            )]
        );
        assert_eq!(plan.orphan_transforms, ["blocks.N.mlp.nothing.weight"]);
        assert_eq!(
            plan.unused_rules,
            [
                "under \"layers.N.\": rename(\"fc2.\" -> \"mlp.linear2.\")",
                "mount(\"unused.\" -> \"never.\")",
            ]
        );
        assert!(plan.check().is_err());

        let clean = Rekey::keep_rest().rename("a.", "b.").plan(["a.w", "c.w"]);
        assert!(clean.collisions.is_empty() && clean.unused_rules.is_empty());
        clean.check().unwrap();
    }

    /// `Indices::Module` matches each stack instance on its own, inside a
    /// stack numbered from 1: a gapped inner stack is laid onto the module's
    /// indices, a complete one kept, a mismatched one refused; without the
    /// module it is `AsIs`. A target under no module is reported.
    #[test]
    fn test_module_indices() {
        let inner = Rekey::keep_rest().mount_indexed_with(
            "seq.",
            "seq.",
            Indices::Module,
            Rekey::keep_rest(),
        );
        let rekey =
            Rekey::keep_rest().mount_indexed_with("stage", "stages.", Indices::From(1), inner);
        let module: BTreeSet<String> = [
            "stages.0.seq.0.w",
            "stages.0.seq.1.w",
            "stages.1.seq.0.w",
            "stages.1.seq.1.w",
            "stages.2.seq.0.w",
            "head.w",
        ]
        .map(String::from)
        .into();
        let names = [
            // Stage 1: `nn.Sequential` gaps, onto 0, 1.
            "stage1.seq.0.w",
            "stage1.seq.2.w",
            // Stage 2: already the module's.
            "stage2.seq.0.w",
            "stage2.seq.1.w",
            // Stage 3: two entries where the module has one.
            "stage3.seq.0.w",
            "stage3.seq.3.w",
            "tail.w",
        ];
        let plan = rekey.plan_for(names, &module);
        let paths: Vec<Option<&str>> = plan
            .targets
            .iter()
            .map(|(_, t)| t.as_ref().map(|t| t.path.as_str()))
            .collect();
        assert_eq!(
            paths,
            [
                Some("stages.0.seq.0.w"),
                Some("stages.0.seq.1.w"),
                Some("stages.1.seq.0.w"),
                Some("stages.1.seq.1.w"),
                None,
                None,
                Some("tail.w"),
            ]
        );
        assert_eq!(
            plan.index_conflicts,
            ["stage3.seq.N.: the checkpoint has {0, 3}, the module (stages.2.seq.N.) {0}"]
        );
        assert_eq!(
            plan.unknown_paths,
            [("tail.w".to_string(), "tail.w".to_string())]
        );
        assert!(plan.check().is_err());

        // Without the module, `Module` is `AsIs`.
        assert_eq!(
            rekey.apply("stage1.seq.2.w").as_deref(),
            Some("stages.0.seq.2.w")
        );
        assert_eq!(rekey.plan(names).index_conflicts, Vec::<String>::new());
    }

    /// A custom rule may pass a name on, drop it, or map it.
    #[test]
    fn test_custom() {
        let rekey = Rekey::drop_rest()
            .custom(|name| {
                let k: usize = name
                    .strip_prefix("conv.")?
                    .split_once('.')?
                    .0
                    .parse()
                    .ok()?;
                Some(
                    k.is_multiple_of(2)
                        .then(|| format!("stages.{}.conv", k / 2)),
                )
            })
            .rename("conv.", "kept.");
        assert_eq!(
            rekey.apply("conv.4.weight").as_deref(),
            Some("stages.2.conv")
        );
        assert_eq!(rekey.apply("conv.3.weight"), None);
        assert_eq!(
            rekey.apply("conv.x.weight").as_deref(),
            Some("kept.x.weight")
        );
    }

    /// A transform rides with its path through mounts, after the mounted
    /// map's own, and `apply` sees only the path.
    #[test]
    fn test_transforms_compose_through_mounts() {
        use burn::tensor::TensorData;

        let scale = |k: f32| {
            move |data: TensorData| {
                let v: Vec<f32> = data.try_to_vec::<f32>().unwrap();
                TensorData::new(v.into_iter().map(|x| x * k).collect::<Vec<_>>(), [2])
            }
        };
        let block = Rekey::keep_rest()
            .rename("fc1.", "mlp.linear1.")
            .transform("mlp.linear1.weight", scale(2.0));
        let model = Rekey::drop_rest()
            .mount("tower.", "encoder.", block)
            .transform("encoder.mlp.linear1.weight", |data| {
                let v: Vec<f32> = data.try_to_vec::<f32>().unwrap();
                TensorData::new(v.into_iter().map(|x| x + 1.0).collect::<Vec<_>>(), [2])
            });

        let target = model.target("tower.fc1.weight").unwrap();
        assert_eq!(target.path, "encoder.mlp.linear1.weight");
        let out = (target.transform.unwrap())(TensorData::new(vec![1.0f32, 3.0], [2]));
        // Doubled by the block's, then incremented by the model's.
        assert_eq!(out.try_to_vec::<f32>().unwrap(), vec![3.0, 7.0]);

        let plain = model.target("tower.fc1.bias").unwrap();
        assert_eq!(plain.path, "encoder.mlp.linear1.bias");
        assert!(plain.transform.is_none());
        assert_eq!(
            model.apply("tower.fc1.weight").as_deref(),
            Some("encoder.mlp.linear1.weight")
        );
    }
}
