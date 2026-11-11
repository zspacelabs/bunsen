//! The selection core: weighted candidates, and the steps that derive,
//! weigh, order and cut them.
//!
//! A [`Selection`] holds candidates, each a resolved build with the tags a
//! selection has derived for it ([`SelectionCandidate`]), and a weight, 1 to
//! start with. Its steps run in the order a caller writes them:
//!
//! - [`derive`](Selection::derive) adds tags: a [`DeriveOp`] sees every
//!   candidate and writes into each one's derived tags. It declares the keys it
//!   writes ([`DerivedKey`]), so that an expression may name them.
//! - [`weigh`](Selection::weigh) weighs: a [`WeighOp`] scores every candidate,
//!   each weight is multiplied by its score, and a candidate whose weight falls
//!   below the epsilon, or whose score is 0, is dropped. A score between 0 and
//!   1 degrades a candidate, so a selection can prefer without refusing.
//! - [`order`](Selection::order) sorts, stably, by a tuple of keys, each
//!   ascending or descending ([`Order`]); a candidate that lacks a key sorts
//!   after those that have it.
//! - [`cutoff`](Selection::cutoff) drops the candidates whose weight is below a
//!   cutoff, and [`limit`](Selection::limit) keeps the first so many.
//!
//! A tag's lookup falls through: a candidate's derived tags first, then its
//! build's resolved tags, written and implied ([`SelectionCandidate::get`]).

use std::{
    cmp::Ordering,
    collections::BTreeMap,
};

use serde_json::Value;

use crate::{
    name::CatalogName,
    resolve::ResolvedBuild,
};

/// A build in a selection, with the tags the selection derived for it.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectionCandidate {
    /// The build, resolved.
    pub build: ResolvedBuild,
    /// Tags a derive step added, by key.
    pub derived: BTreeMap<String, Value>,
}

impl SelectionCandidate {
    /// A candidate of `build`, with no derived tags.
    pub fn new(build: ResolvedBuild) -> Self {
        Self {
            build,
            derived: BTreeMap::new(),
        }
    }

    /// The build's full name.
    pub fn name(&self) -> &CatalogName {
        &self.build.name
    }

    /// This candidate with the derived tag `key`.
    pub fn with_tag(
        mut self,
        key: impl Into<String>,
        value: Value,
    ) -> Self {
        self.derived.insert(key.into(), value);
        self
    }

    /// The value of the tag `key` (a tag's full name, or a derived key):
    /// derived first, then the build's.
    pub fn get(
        &self,
        key: &str,
    ) -> Option<&Value> {
        self.derived.get(key).or_else(|| {
            let name = CatalogName::parse(key).ok()?;
            self.build.tags.get(&name).map(|t| &t.value)
        })
    }

    /// Whether the tag `key` is `true`.
    pub fn is_true(
        &self,
        key: &str,
    ) -> bool {
        self.get(key).and_then(Value::as_bool) == Some(true)
    }

    /// The tag `key`, when it is text.
    pub fn text(
        &self,
        key: &str,
    ) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    /// The tag `key`, when it is a number.
    pub fn number(
        &self,
        key: &str,
    ) -> Option<f64> {
        self.get(key).and_then(Value::as_f64)
    }
}

/// A sort direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Least first.
    Ascending,
    /// Greatest first.
    Descending,
}

/// A weighing step: a score, 0 or more, for every candidate.
pub trait WeighOp {
    /// The scores, one per candidate, in order.
    fn scores(
        &self,
        items: &[(SelectionCandidate, f64)],
    ) -> Vec<f64>;
}

/// A [`WeighOp`] that scores one candidate at a time.
pub struct PerItem<F>(pub F);

impl<F: Fn(&SelectionCandidate) -> f64> WeighOp for PerItem<F> {
    fn scores(
        &self,
        items: &[(SelectionCandidate, f64)],
    ) -> Vec<f64> {
        items.iter().map(|(c, _)| (self.0)(c)).collect()
    }
}

/// A key a derive step writes, declared so an expression can name it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedKey {
    /// The key.
    pub name: String,
    /// What it holds, for a person.
    pub description: String,
}

/// A derive step: tags added to candidates.
pub trait DeriveOp {
    /// The keys it writes.
    fn keys(&self) -> Vec<DerivedKey>;

    /// Writes the tags into each candidate's derived tags.
    fn derive(
        &self,
        items: &mut [(SelectionCandidate, f64)],
    );
}

/// A [`DeriveOp`] that derives one candidate's tags at a time.
pub struct DeriveEach<F> {
    /// The keys it writes.
    pub keys: Vec<DerivedKey>,
    /// The tags, by key, for a candidate.
    pub tags: F,
}

impl<F: Fn(&SelectionCandidate) -> BTreeMap<String, Value>> DeriveOp for DeriveEach<F> {
    fn keys(&self) -> Vec<DerivedKey> {
        self.keys.clone()
    }

    fn derive(
        &self,
        items: &mut [(SelectionCandidate, f64)],
    ) {
        for (candidate, _) in items {
            let tags = (self.tags)(candidate);
            candidate.derived.extend(tags);
        }
    }
}

/// A set of candidates, each weighted, in an order: see the
/// [module](self).
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    items: Vec<(SelectionCandidate, f64)>,
    epsilon: f64,
}

impl Selection {
    /// The weight below which [`weigh`](Self::weigh) drops a candidate,
    /// unless [`with_epsilon`](Self::with_epsilon) sets another.
    pub const DEFAULT_EPSILON: f64 = 1e-9;

    /// `items`, in order, each of weight 1.
    pub fn new(items: impl IntoIterator<Item = SelectionCandidate>) -> Self {
        Self {
            items: items.into_iter().map(|c| (c, 1.0)).collect(),
            epsilon: Self::DEFAULT_EPSILON,
        }
    }

    /// This selection, dropping candidates whose weight falls below
    /// `epsilon`.
    pub fn with_epsilon(
        mut self,
        epsilon: f64,
    ) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// The candidates and their weights, in order.
    pub fn iter(&self) -> impl Iterator<Item = (&SelectionCandidate, f64)> {
        self.items.iter().map(|(c, w)| (c, *w))
    }

    /// How many candidates there are.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The candidates, in order, without their weights.
    pub fn into_candidates(self) -> Vec<SelectionCandidate> {
        self.items.into_iter().map(|(c, _)| c).collect()
    }

    /// The candidates' names, in order.
    pub fn names(&self) -> Vec<&CatalogName> {
        self.items.iter().map(|(c, _)| c.name()).collect()
    }

    /// Runs `op` over the candidates, adding tags.
    pub fn derive(
        mut self,
        op: &dyn DeriveOp,
    ) -> Self {
        op.derive(&mut self.items);
        self
    }

    /// Multiplies each weight by `op`'s score for it, and drops a candidate
    /// whose score is 0 or whose weight falls below the epsilon.
    ///
    /// # Panics
    /// When `op` gives a score count other than the candidates', or a score
    /// that is negative, NaN or infinite: the scorer's bug.
    pub fn weigh(
        mut self,
        op: &dyn WeighOp,
    ) -> Self {
        let scores = op.scores(&self.items);
        assert_eq!(
            scores.len(),
            self.items.len(),
            "a weighing step scores every candidate"
        );
        let epsilon = self.epsilon;
        self.items = std::mem::take(&mut self.items)
            .into_iter()
            .zip(scores)
            .filter_map(|((candidate, weight), score)| {
                assert!(
                    score.is_finite() && score >= 0.0,
                    "a score is finite and not negative: {score} for {}",
                    candidate.name()
                );
                let weight = weight * score;
                (score > 0.0 && weight >= epsilon).then_some((candidate, weight))
            })
            .collect();
        self
    }

    /// [`weigh`](Self::weigh) through [`PerItem`].
    pub fn weigh_each(
        self,
        score: impl Fn(&SelectionCandidate) -> f64,
    ) -> Self {
        self.weigh(&PerItem(score))
    }

    /// Sorts the candidates, stably, by the tuple `key` gives each (from the
    /// candidate and its current weight), most significant first, dimension
    /// `i` as `scheme[i]` says. Per dimension, a candidate with a key sorts
    /// before one without, whichever the direction.
    pub fn order<const D: usize>(
        mut self,
        scheme: &[Order; D],
        key: impl Fn(&SelectionCandidate, f64) -> [Option<f64>; D],
    ) -> Self {
        let mut keyed: Vec<_> = std::mem::take(&mut self.items)
            .into_iter()
            .map(|(c, w)| (key(&c, w), (c, w)))
            .collect();
        keyed.sort_by(|(a, _), (b, _)| compare(scheme, a, b));
        self.items = keyed.into_iter().map(|(_, item)| item).collect();
        self
    }

    /// The candidates whose weight is at least `cutoff`, in order.
    pub fn cutoff(
        mut self,
        cutoff: f64,
    ) -> Self {
        self.items.retain(|(_, w)| *w >= cutoff);
        self
    }

    /// The first `limit` candidates, in order.
    pub fn limit(
        mut self,
        limit: usize,
    ) -> Self {
        self.items.truncate(limit);
        self
    }

    /// The first candidate, with its weight.
    pub fn first(&self) -> Option<(&SelectionCandidate, f64)> {
        self.iter().next()
    }
}

/// `a` against `b`, dimension by dimension: a total order.
fn compare<const D: usize>(
    scheme: &[Order; D],
    a: &[Option<f64>; D],
    b: &[Option<f64>; D],
) -> Ordering {
    for i in 0..D {
        let ordering = match (a[i], b[i]) {
            (Some(x), Some(y)) => match scheme[i] {
                Order::Ascending => x.total_cmp(&y),
                Order::Descending => y.total_cmp(&x),
            },
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        };
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::select::testing::{
        selection,
        shown,
    };

    #[test]
    fn test_select_weighs_and_drops() {
        let s = selection(&["qwen3-0.6b", "qwen3-0.6b-base", "qwen3-4b-thinking-2507"])
            .weigh_each(|c| if c.is_true("base") { 0.0 } else { 1.0 })
            .weigh_each(|c| {
                if c.name().leaf().contains("4b") {
                    0.5
                } else {
                    1.0
                }
            });
        assert_eq!(shown(&s), ["qwen3-0.6b", "qwen3-4b-thinking-2507"]);
        let weights: Vec<f64> = s.iter().map(|(_, w)| w).collect();
        assert_eq!(weights, [1.0, 0.5]);
    }

    #[test]
    fn test_epsilon_drops_the_faint() {
        let s = selection(&["qwen3-0.6b", "qwen3-0.6b-base"])
            .with_epsilon(0.1)
            .weigh_each(|c| if c.is_true("base") { 0.05 } else { 1.0 });
        assert_eq!(shown(&s), ["qwen3-0.6b"]);
    }

    #[test]
    fn test_order_is_stable_and_puts_missing_last() {
        let s = selection(&[
            "large-v1",
            "qwen3-0.6b",
            "large-v3-turbo",
            "qwen3-4b-thinking-2507",
        ])
        .derive(&DeriveEach {
            keys: vec![],
            tags: |c: &SelectionCandidate| {
                let mut tags = BTreeMap::new();
                if c.name().leaf().starts_with("large") {
                    tags.insert("rank".to_string(), json!(1));
                }
                tags
            },
        })
        .order(&[Order::Descending], |c, _| [c.number("parameters")]);
        assert_eq!(
            shown(&s),
            [
                "qwen3-4b-thinking-2507",
                "large-v1",
                "large-v3-turbo",
                "qwen3-0.6b"
            ]
        );
        let s = s.order(&[Order::Ascending], |c, _| [c.number("rank")]);
        assert_eq!(
            shown(&s),
            [
                "large-v1",
                "large-v3-turbo",
                "qwen3-4b-thinking-2507",
                "qwen3-0.6b"
            ]
        );
    }

    #[test]
    fn test_cutoff_and_limit() {
        let s = selection(&["qwen3-0.6b", "qwen3-0.6b-base", "large-v3"]).weigh_each(|c| {
            if c.name().leaf() == "qwen3-0.6b" {
                0.4
            } else {
                1.0
            }
        });
        let cut = s.clone().cutoff(0.5);
        assert_eq!(shown(&cut), ["qwen3-0.6b-base", "large-v3"]);
        assert_eq!(cut.first().unwrap().1, 1.0);
        assert_eq!(shown(&cut.limit(1)), ["qwen3-0.6b-base"]);
        assert!(s.clone().cutoff(1.5).is_empty());
        assert_eq!(s.limit(5).len(), 3);
    }

    #[test]
    fn test_lookup_falls_through_derived_to_the_build() {
        let c = selection(&["qwen3-0.6b"])
            .into_candidates()
            .remove(0)
            .with_tag("est_size", json!(1200));
        assert_eq!(c.number("est_size"), Some(1200.0));
        assert_eq!(c.text("family"), Some("qwen/qwen3"));
        assert_eq!(c.text("thinking"), Some("switch"));
        assert_eq!(c.get("nope"), None);
    }
}
