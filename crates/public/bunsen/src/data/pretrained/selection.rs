//! # Selection: choosing among pretrained rows by their tags
//!
//! A [`Selection`] holds candidates, each a row with the tags the selection
//! has added to it ([`SelectionCandidate`]), and a weight, 1 to start with.
//! Its steps run in the order a caller writes them:
//!
//! - [`derive`](Selection::derive) adds tags: a [`DeriveOp`] sees every
//!   candidate and writes into each one's own tags ([`DeriveEach`] does it one
//!   candidate at a time; [`EstSize`] writes a model's estimated loaded size).
//! - [`select`](Selection::select) weighs: a [`SelectOp`] scores every
//!   candidate, each weight is multiplied by its score, and a candidate whose
//!   weight falls below the selection's epsilon, or whose score is 0, is
//!   dropped ([`PerItem`] scores one candidate at a time). A score of 1 keeps a
//!   candidate as it was; one between 0 and 1 degrades it, so a selection can
//!   prefer without refusing.
//! - [`order`](Selection::order) sorts, stably, by a tuple of keys, each
//!   ascending or descending ([`Order`]); a candidate that lacks a key sorts
//!   after those that have it.
//! - [`take`](Selection::take) returns the first candidate, in order, whose
//!   weight reaches a cutoff.
//!
//! A key's lookup falls through: a candidate's own tags first, then its
//! row's ([`SelectionCandidate::get`]).
//!
//! ```
//! use bunsen::data::pretrained::{
//!     Pretrained,
//!     ResourceBundle,
//!     Tags,
//!     selection::{
//!         Order,
//!         Selection,
//!         SelectionCandidate,
//!     },
//! };
//! use serde_json::json;
//!
//! let row = |name: &str, tags: serde_json::Value| Pretrained {
//!     name: name.to_string(),
//!     aliases: Vec::new(),
//!     description: String::new(),
//!     license: None,
//!     origin: None,
//!     prefab: None,
//!     resources: ResourceBundle::new(name),
//!     presets: Default::default(),
//!     tags: serde_json::from_value::<Tags>(tags).unwrap(),
//! };
//! let rows = [
//!     row("small", json!({"base": false, "parameters": 6e8})),
//!     row("large", json!({"base": false, "parameters": 8e9})),
//!     row("large-base", json!({"base": true, "parameters": 8e9})),
//! ];
//!
//! // Not a base model, then the largest.
//! let taken = Selection::new(rows.into_iter().map(SelectionCandidate::new))
//!     .select_each(|c| if c.is_true("base") { 0.0 } else { 1.0 })
//!     .order(&[Order::Descending], |c, _| [c.number("parameters")])
//!     .take(0.0)
//!     .unwrap();
//! assert_eq!(taken.pretrained.name, "large");
//! ```

use alloc::{
    boxed::Box,
    vec::Vec,
};
use core::cmp::Ordering;

use serde_json::Value;

use super::{
    Pretrained,
    Tags,
    tags::{
        DTYPE,
        PARAMETERS,
    },
};

/// The key [`EstSize`] writes: a model's estimated size once loaded, in
/// bytes. Derived, never a row's tag.
pub const EST_SIZE: &str = "est_size";

/// A selection's candidate: a row, and the tags the selection has added to
/// it.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectionCandidate {
    /// The row.
    pub pretrained: Pretrained,

    /// The candidate's own tags: what building the selection and its
    /// derivations added. A key here hides the row's.
    pub tags: Tags,
}

impl SelectionCandidate {
    /// A candidate of `pretrained`, with no tags of its own.
    pub fn new(pretrained: Pretrained) -> Self {
        Self {
            pretrained,
            tags: Tags::new(),
        }
    }

    /// This candidate with its own `key` set to `value`.
    pub fn with_tag(
        mut self,
        key: impl Into<alloc::string::String>,
        value: impl Into<Value>,
    ) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// The value of `key`: the candidate's own, else its row's.
    pub fn get(
        &self,
        key: impl AsRef<str>,
    ) -> Option<&Value> {
        let key = key.as_ref();
        self.tags.get(key).or_else(|| self.pretrained.tags.get(key))
    }

    /// Whether `key` is `true`.
    pub fn is_true(
        &self,
        key: impl AsRef<str>,
    ) -> bool {
        self.get(key) == Some(&Value::Bool(true))
    }

    /// The value of `key` as text, when it is a string.
    pub fn text(
        &self,
        key: impl AsRef<str>,
    ) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    /// The value of `key` as a number, when it is one.
    pub fn number(
        &self,
        key: impl AsRef<str>,
    ) -> Option<f64> {
        self.get(key).and_then(Value::as_f64)
    }
}

/// A sort direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Lowest first.
    Ascending,

    /// Highest first.
    Descending,
}

/// A select step: scores every candidate of a selection, seeing them all
/// (a candidate's rank among the others, say).
pub trait SelectOp {
    /// One score per candidate, in order: finite, and 0 or more.
    fn scores(
        &self,
        items: &[(SelectionCandidate, f64)],
    ) -> Vec<f64>;
}

/// A select step from a function of one candidate.
pub struct PerItem<F>(pub F);

impl<F: Fn(&SelectionCandidate) -> f64> SelectOp for PerItem<F> {
    fn scores(
        &self,
        items: &[(SelectionCandidate, f64)],
    ) -> Vec<f64> {
        items.iter().map(|(c, _)| (self.0)(c)).collect()
    }
}

/// A derivation: adds tags to the candidates of a selection, seeing them
/// all. It writes into each candidate's own tags, and decides what it writes
/// over.
pub trait DeriveOp {
    /// Adds the derived tags.
    fn derive(
        &self,
        items: &mut [(SelectionCandidate, f64)],
    );
}

/// A derivation from a function of one candidate: the tags it returns are set
/// in the candidate's own, over any of the same keys.
pub struct DeriveEach<F>(pub F);

impl<F: Fn(&SelectionCandidate) -> Tags> DeriveOp for DeriveEach<F> {
    fn derive(
        &self,
        items: &mut [(SelectionCandidate, f64)],
    ) {
        for (candidate, _) in items {
            let derived = (self.0)(candidate);
            for (key, value) in derived.iter() {
                candidate.tags.insert(key, value.clone());
            }
        }
    }
}

/// The bytes of a parameter count stored in a dtype, once loaded: what the
/// caller's backend makes of it (a `bf16` checkpoint widened to `f32` on a
/// backend without `bf16`, say). `None` when it cannot say.
pub type LoadedBytes = dyn Fn(&str, u64) -> Option<u64> + Send + Sync;

/// Derives [`EST_SIZE`]: a candidate's estimated loaded size in bytes, from
/// its `dtype` and `parameters` tags through the caller's [`LoadedBytes`]. A
/// candidate lacking either, or whose dtype the function cannot size, gets
/// none.
pub struct EstSize {
    bytes: Box<LoadedBytes>,
}

impl EstSize {
    /// Sizes through `bytes`.
    pub fn new(bytes: impl Fn(&str, u64) -> Option<u64> + Send + Sync + 'static) -> Self {
        Self {
            bytes: Box::new(bytes),
        }
    }
}

impl core::fmt::Debug for EstSize {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("EstSize").finish_non_exhaustive()
    }
}

impl DeriveOp for EstSize {
    fn derive(
        &self,
        items: &mut [(SelectionCandidate, f64)],
    ) {
        for (candidate, _) in items {
            let size = candidate
                .text(DTYPE)
                .zip(candidate.get(PARAMETERS).and_then(Value::as_u64));
            if let Some(bytes) =
                size.and_then(|(dtype, parameters)| (self.bytes)(dtype, parameters))
            {
                candidate.tags.insert(EST_SIZE, bytes);
            }
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
    /// The weight below which [`select`](Self::select) drops a candidate,
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

    /// The epsilon.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
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

    /// Runs `op` over the candidates, adding tags.
    pub fn derive(
        mut self,
        op: &dyn DeriveOp,
    ) -> Self {
        op.derive(&mut self.items);
        self
    }

    /// [`derive`](Self::derive) through [`DeriveEach`].
    pub fn derive_each(
        self,
        tags: impl Fn(&SelectionCandidate) -> Tags,
    ) -> Self {
        self.derive(&DeriveEach(tags))
    }

    /// Multiplies each weight by `op`'s score for it, and drops a candidate
    /// whose score is 0 or whose weight falls below the epsilon.
    ///
    /// # Panics
    /// When `op` gives a score count other than the candidates', or a score
    /// that is negative, NaN or infinite: the scorer's bug.
    pub fn select(
        mut self,
        op: &dyn SelectOp,
    ) -> Self {
        let scores = op.scores(&self.items);
        assert_eq!(
            scores.len(),
            self.items.len(),
            "a select step scores every candidate"
        );
        let epsilon = self.epsilon;
        self.items = core::mem::take(&mut self.items)
            .into_iter()
            .zip(scores)
            .filter_map(|((candidate, weight), score)| {
                assert!(
                    score.is_finite() && score >= 0.0,
                    "a score is finite and not negative: {score} for {:?}",
                    candidate.pretrained.name
                );
                let weight = weight * score;
                (score > 0.0 && weight >= epsilon).then_some((candidate, weight))
            })
            .collect();
        self
    }

    /// [`select`](Self::select) through [`PerItem`].
    pub fn select_each(
        self,
        score: impl Fn(&SelectionCandidate) -> f64,
    ) -> Self {
        self.select(&PerItem(score))
    }

    /// Sorts the candidates, stably, by the tuple `key` gives each (from the
    /// candidate and its current weight), most significant first, dimension
    /// `i` as `scheme[i]` says. Per dimension, a candidate with a key sorts
    /// before one without, whichever the direction; two without tie, and the
    /// next dimension decides.
    pub fn order<const D: usize>(
        mut self,
        scheme: &[Order; D],
        key: impl Fn(&SelectionCandidate, f64) -> [Option<f64>; D],
    ) -> Self {
        let mut keyed: Vec<_> = core::mem::take(&mut self.items)
            .into_iter()
            .map(|(c, w)| (key(&c, w), (c, w)))
            .collect();
        keyed.sort_by(|(a, _), (b, _)| compare(scheme, a, b));
        self.items = keyed.into_iter().map(|(_, item)| item).collect();
        self
    }

    /// The first candidate, in order, whose weight is at least `cutoff`.
    pub fn take(
        self,
        cutoff: f64,
    ) -> Option<SelectionCandidate> {
        self.items
            .into_iter()
            .find(|(_, w)| *w >= cutoff)
            .map(|(c, _)| c)
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
    use alloc::{
        string::ToString,
        vec,
    };

    use serde_json::json;

    use super::*;
    use crate::data::pretrained::ResourceBundle;

    fn row(
        name: &str,
        tags: Value,
    ) -> SelectionCandidate {
        SelectionCandidate::new(Pretrained {
            name: name.to_string(),
            aliases: Vec::new(),
            description: String::new(),
            license: None,
            origin: None,
            prefab: None,
            resources: ResourceBundle::new(name),
            presets: Default::default(),
            tags: serde_json::from_value(tags).unwrap(),
        })
    }

    fn names(selection: &Selection) -> Vec<&str> {
        selection
            .iter()
            .map(|(c, _)| c.pretrained.name.as_str())
            .collect()
    }

    fn toys() -> Selection {
        Selection::new([
            row("a", json!({"parameters": 2, "dtype": "bf16"})),
            row("b", json!({"parameters": 8, "dtype": "f32", "base": true})),
            row("c", json!({"dtype": "f16"})),
            row("d", json!({"parameters": 8, "dtype": "bf16"})),
        ])
    }

    /// A lookup falls through the candidate's own tags to its row's.
    #[test]
    fn test_lookup_falls_through() {
        let c =
            row("a", json!({"thinking": "never", "parameters": 2})).with_tag("thinking", "switch");
        assert_eq!(c.text("thinking"), Some("switch"));
        assert_eq!(c.number("parameters"), Some(2.0));
        assert_eq!(c.get("absent"), None);
        assert!(!c.is_true("thinking"));
    }

    /// Weights multiply; a score of 0, or a weight below the epsilon, drops a
    /// candidate.
    #[test]
    fn test_select() {
        let s = toys().select_each(|c| if c.is_true("base") { 0.0 } else { 0.5 });
        assert_eq!(names(&s), ["a", "c", "d"]);
        assert!(s.iter().all(|(_, w)| w == 0.5));
        let s = s.select_each(|c| if c.pretrained.name == "a" { 1e-9 } else { 1.0 });
        assert_eq!(names(&s), ["c", "d"], "0.5e-9 is below the default epsilon");
        let kept = toys()
            .with_epsilon(0.0)
            .select_each(|c| if c.pretrained.name == "a" { 1e-12 } else { 1.0 });
        assert_eq!(kept.len(), 4);
        assert_eq!(kept.epsilon(), 0.0);
    }

    #[test]
    #[should_panic(expected = "a score is finite and not negative")]
    fn test_a_negative_score_panics() {
        toys().select_each(|_| -1.0);
    }

    /// A whole-selection op sees every candidate: here, a candidate's rank by
    /// its parameters among them.
    #[test]
    fn test_a_whole_selection_op() {
        struct Rank;
        impl SelectOp for Rank {
            fn scores(
                &self,
                items: &[(SelectionCandidate, f64)],
            ) -> Vec<f64> {
                let most = items
                    .iter()
                    .filter_map(|(c, _)| c.number("parameters"))
                    .fold(0.0, f64::max);
                items
                    .iter()
                    .map(|(c, _)| c.number("parameters").map_or(1.0, |p| p / most))
                    .collect()
            }
        }
        let s = toys().select(&Rank);
        let weights: Vec<f64> = s.iter().map(|(_, w)| w).collect();
        assert_eq!(weights, [0.25, 1.0, 1.0, 1.0]);
    }

    /// The order is stable, by the tuple, each dimension in its direction; a
    /// candidate lacking a key sorts after those that have it either way.
    #[test]
    fn test_order() {
        let s = toys().order(&[Order::Descending], |c, _| [c.number("parameters")]);
        assert_eq!(names(&s), ["b", "d", "a", "c"]);
        let s = toys().order(&[Order::Ascending], |c, _| [c.number("parameters")]);
        assert_eq!(names(&s), ["a", "b", "d", "c"]);
        let s = toys().order(&[Order::Descending, Order::Ascending], |c, _| {
            [
                c.number("parameters"),
                c.text("dtype").map(|d| if d == "bf16" { 0.0 } else { 1.0 }),
            ]
        });
        assert_eq!(names(&s), ["d", "b", "a", "c"]);
        // The weight is a key too.
        let s = toys()
            .select_each(|c| if c.pretrained.name == "c" { 1.0 } else { 0.5 })
            .order(&[Order::Descending], |_, w| [Some(w)]);
        assert_eq!(names(&s), ["c", "a", "b", "d"]);
    }

    #[test]
    fn test_take() {
        let s = toys().select_each(|c| if c.pretrained.name == "a" { 0.5 } else { 0.25 });
        assert_eq!(s.clone().take(0.0).unwrap().pretrained.name, "a");
        assert_eq!(s.clone().take(0.3).unwrap().pretrained.name, "a");
        assert!(s.clone().take(0.6).is_none());
        let s = s.order(&[Order::Ascending], |_, w| [Some(w)]);
        assert_eq!(s.clone().take(0.3).unwrap().pretrained.name, "a");
        assert_eq!(s.take(0.0).unwrap().pretrained.name, "b");
        assert!(Selection::new(vec![]).take(0.0).is_none());
    }

    /// A derivation writes the candidates' own tags; the rows are untouched.
    #[test]
    fn test_derive() {
        let s = toys().derive_each(|c| {
            if c.get("base").is_none() {
                Tags::new().with("base", false)
            } else {
                Tags::new()
            }
        });
        let base: Vec<Option<&Value>> = s.iter().map(|(c, _)| c.get("base")).collect();
        assert_eq!(
            base,
            [
                Some(&json!(false)),
                Some(&json!(true)),
                Some(&json!(false)),
                Some(&json!(false))
            ]
        );
        assert!(
            s.iter()
                .all(|(c, _)| c.pretrained.tags.get("base") != Some(&json!(false)))
        );
    }

    /// `EstSize` sizes the candidates that carry both tags, through the
    /// caller's bytes per dtype.
    #[test]
    fn test_est_size() {
        let widened = EstSize::new(|dtype, parameters| match dtype {
            "f32" | "bf16" => Some(parameters * 4),
            _ => None,
        });
        let s = toys().derive(&widened);
        let sizes: Vec<Option<f64>> = s.iter().map(|(c, _)| c.number(EST_SIZE)).collect();
        assert_eq!(sizes, [Some(8.0), Some(32.0), None, Some(32.0)]);
        assert!(format!("{widened:?}").starts_with("EstSize"));
    }
}
