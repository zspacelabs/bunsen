//! Mapping a checkpoint's tensor names onto a module's paths.
use std::{
    fmt,
    sync::Arc,
};

use burn::tensor::TensorData;

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
/// fallback: kept as it is ([`keep_rest`](Self::keep_rest)), or dropped
/// ([`drop_rest`](Self::drop_rest)).
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
    keep_unclaimed: bool,
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
    MountIndexed(String, String, Rekey),
    Custom(Arc<dyn Fn(&str) -> Claim + Send + Sync>),
}

impl Rule {
    /// `None` when the rule does not claim `name`; `Some(target)` when it
    /// does, `target` being `None` for a dropped tensor.
    fn claim(
        &self,
        name: &str,
    ) -> Option<Option<Target>> {
        match self {
            Rule::DropSuffix(suffix) => name.ends_with(suffix.as_str()).then_some(None),
            Rule::Exact(from, to) => (name == from).then(|| Some(plain(to.clone()))),
            Rule::Rename(from, to) => name
                .strip_prefix(from.as_str())
                .map(|rest| Some(plain(format!("{to}{rest}")))),
            Rule::Mount(from, to, sub) => name
                .strip_prefix(from.as_str())
                .map(|rest| sub.target(rest).map(|t| prefixed(to, t))),
            Rule::MountIndexed(from, to, sub) => {
                let (index, rest) = name.strip_prefix(from.as_str())?.split_once('.')?;
                Some(
                    sub.target(rest)
                        .map(|t| prefixed(&format!("{to}{index}."), t)),
                )
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
            keep_unclaimed: true,
        }
    }

    /// A map whose unclaimed names are dropped: for a checkpoint whose every
    /// loaded name a rule accounts for.
    pub fn drop_rest() -> Self {
        Self {
            rules: Vec::new(),
            transforms: Vec::new(),
            keep_unclaimed: false,
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
    /// `{to}{i}.{sub(rest)}`, the layer index carried across.
    pub fn mount_indexed(
        self,
        from: impl Into<String>,
        to: impl Into<String>,
        sub: Rekey,
    ) -> Self {
        self.with(Rule::MountIndexed(from.into(), to.into(), sub))
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
        let target = self
            .rules
            .iter()
            .find_map(|rule| rule.claim(name))
            .unwrap_or_else(|| self.keep_unclaimed.then(|| plain(name.to_string())))?;
        let own = self
            .transforms
            .iter()
            .find(|(path, _)| *path == target.path)
            .map(|(_, f)| f.clone());
        Some(Target {
            transform: compose(target.transform, own),
            path: target.path,
        })
    }

    /// The module path `name` loads into, or `None` to drop it.
    pub fn apply(
        &self,
        name: &str,
    ) -> Option<String> {
        self.target(name).map(|t| t.path)
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
            .field("keep_unclaimed", &self.keep_unclaimed)
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
