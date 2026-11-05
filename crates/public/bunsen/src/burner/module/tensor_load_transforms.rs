//! Named changes to a checkpoint's tensors as they load.

use std::{
    fmt,
    sync::Arc,
};

use burn::tensor::TensorData;

use crate::burner::tensor::{
    permute_data,
    reshape_data,
};

/// A change to a tensor's data as it loads: a permute or reshape on the
/// host, applied before the store's own adapter (burn-store's
/// `PyTorchToBurnAdapter`).
pub type TensorTransform = Arc<dyn Fn(TensorData) -> TensorData + Send + Sync>;

/// A named change to a checkpoint tensor's data as it loads, which a
/// [`Rekey`](super::Rekey) applies to the paths it
/// [`transform`](super::Rekey::transform)s.
///
/// The named constructors are what checkpoints ask for; each says what it
/// does to the data, and a [`RekeyPlan`](super::RekeyPlan) reports it by
/// that name. [`new`](Self::new) names a change of a block's own.
#[derive(Clone)]
pub struct TensorLoadTransform {
    name: String,
    f: TensorTransform,
}

impl fmt::Debug for TensorLoadTransform {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_tuple("TensorLoadTransform")
            .field(&self.name)
            .finish()
    }
}

/// `data`'s values as f32, with its shape.
fn f32_values(data: TensorData) -> (Vec<f32>, Vec<usize>) {
    let shape = data.shape().as_slice().to_vec();
    let values = data
        .convert::<f32>()
        .try_into_vec::<f32>()
        .expect("float values");
    (values, shape)
}

impl TensorLoadTransform {
    /// A load transform named `name` that changes the data by `f`.
    pub fn new(
        name: impl Into<String>,
        f: impl Fn(TensorData) -> TensorData + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            f: Arc::new(f),
        }
    }

    /// What it is called in a report.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The change, as a function.
    pub fn transform(&self) -> TensorTransform {
        self.f.clone()
    }

    /// `data`, changed.
    pub fn apply(
        &self,
        data: TensorData,
    ) -> TensorData {
        (self.f)(data)
    }

    /// This change, then `next`'s.
    pub fn then(
        self,
        next: TensorLoadTransform,
    ) -> Self {
        let (a, b) = (self.f, next.f);
        Self {
            name: format!("{} then {}", self.name, next.name),
            f: Arc::new(move |data| b(a(data))),
        }
    }

    /// `w → 1 + w`, in f32: a norm gain stored about zero and applied as
    /// `1 + w` (Gemma's, Qwen3.5's), for a norm that applies its gain as
    /// stored. The result is f32 whatever the stored dtype, so a bf16 `w`
    /// folds exactly; the loader then casts it to the parameter's dtype.
    pub fn zero_centered_gain() -> Self {
        Self::new("zero_centered_gain", |data| {
            let (values, shape) = f32_values(data);
            TensorData::new(values.into_iter().map(|w| 1.0 + w).collect(), shape)
        })
    }

    /// `w → k · w`, in f32.
    pub fn scale(k: f64) -> Self {
        Self::new(format!("scale({k})"), move |data| {
            let (values, shape) = f32_values(data);
            TensorData::new(
                values
                    .into_iter()
                    .map(|w| (f64::from(w) * k) as f32)
                    .collect(),
                shape,
            )
        })
    }

    /// A matrix's transpose: `[a, b]` to `[b, a]`.
    pub fn transpose() -> Self {
        Self::new("transpose", |data| permute_data(data, &[1, 0]))
    }

    /// A `[rows, a · b]` matrix whose columns are in `(a, b)` order, with them
    /// in `(b, a)` order: a projection whose input a checkpoint flattens the
    /// other way round from the block (channels and frequencies, frames and
    /// features).
    pub fn regroup_columns(
        a: usize,
        b: usize,
    ) -> Self {
        Self::new(format!("regroup_columns({a}, {b})"), move |data| {
            let rows = data.shape().num_elements() / (a * b);
            let data = reshape_data(data, vec![rows, a, b]);
            reshape_data(permute_data(data, &[0, 2, 1]), vec![rows, b * a])
        })
    }

    /// Rows in interleaved rotary pairs, `(x0, y0, x1, y1, …)` within each
    /// block of `block` rows, as half-split pairs, `(x0, x1, …, y0, y1, …)`:
    /// a projection onto heads of `block` rows that a checkpoint rotates as
    /// interleaved pairs (`DeepSeek`'s rope rows), for the half-split rotation
    /// (`rotate_half`). The rows are the first axis; any further axes ride
    /// along.
    ///
    /// # Panics
    /// If `block` is odd, or does not divide the rows.
    pub fn interleaved_to_halves(block: usize) -> Self {
        assert!(block.is_multiple_of(2), "an even block: {block}");
        Self::new(format!("interleaved_to_halves({block})"), move |data| {
            let shape = data.shape().as_slice().to_vec();
            let rows = shape[0];
            assert!(
                rows.is_multiple_of(block),
                "{block} does not divide {rows} rows"
            );
            let rest: usize = shape[1..].iter().product();
            let data = reshape_data(data, vec![rows / block, block / 2, 2, rest]);
            reshape_data(permute_data(data, &[0, 2, 1, 3]), shape)
        })
    }
}

/// Which of a map's module paths an [`TensorLoadTransform`] applies to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Select {
    /// The one path.
    Exact(String),

    /// Every path that ends so: every norm's `norm.weight`, say, across all
    /// the layers a map reaches.
    Suffix(String),
}

impl Select {
    /// Every path ending in `suffix`.
    pub fn suffix(suffix: impl Into<String>) -> Self {
        Self::Suffix(suffix.into())
    }

    /// Whether `path` is selected.
    pub fn matches(
        &self,
        path: &str,
    ) -> bool {
        match self {
            Self::Exact(p) => p == path,
            Self::Suffix(s) => path.ends_with(s.as_str()),
        }
    }
}

impl fmt::Display for Select {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Exact(p) => f.write_str(p),
            Self::Suffix(s) => write!(f, "*{s}"),
        }
    }
}

impl From<&str> for Select {
    fn from(path: &str) -> Self {
        Self::Exact(path.to_string())
    }
}

impl From<String> for Select {
    fn from(path: String) -> Self {
        Self::Exact(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(
        values: &[f32],
        shape: &[usize],
    ) -> TensorData {
        TensorData::new(values.to_vec(), shape.to_vec())
    }

    fn values(data: TensorData) -> Vec<f32> {
        data.convert::<f32>().try_into_vec().unwrap()
    }

    #[test]
    fn test_zero_centered_gain() {
        let a = TensorLoadTransform::zero_centered_gain();
        assert_eq!(a.name(), "zero_centered_gain");
        assert_eq!(
            values(a.apply(data(&[0.0, -0.25, 0.5], &[3]))),
            [1.0, 0.75, 1.5]
        );
        // A bf16 `w` folds in f32: `1 + 2^-9` is no bf16 value.
        let w = burn::tensor::bf16::from_f32(2f32.powi(-9));
        let folded = a.apply(TensorData::new(vec![w], [1]));
        assert_eq!(folded.dtype(), burn::tensor::DType::F32);
        assert_eq!(values(folded), [1.0 + 2f32.powi(-9)]);
    }

    #[test]
    fn test_scale() {
        let a = TensorLoadTransform::scale(0.5);
        assert_eq!(a.name(), "scale(0.5)");
        assert_eq!(values(a.apply(data(&[2.0, -4.0], &[2]))), [1.0, -2.0]);
    }

    /// A transpose, twice, is the identity.
    #[test]
    fn test_transpose() {
        let m = data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let t = TensorLoadTransform::transpose().apply(m.clone());
        assert_eq!(t.shape().as_slice(), [3, 2]);
        assert_eq!(values(t.clone()), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(values(TensorLoadTransform::transpose().apply(t)), values(m));
    }

    /// Columns in `(a, b)` order are in `(b, a)` order after; regrouping back
    /// undoes it.
    #[test]
    fn test_regroup_columns() {
        // One row, a = 2 by b = 3: (a0b0, a0b1, a0b2, a1b0, a1b1, a1b2).
        let m = data(&[0.0, 1.0, 2.0, 10.0, 11.0, 12.0], &[1, 6]);
        let r = TensorLoadTransform::regroup_columns(2, 3).apply(m.clone());
        assert_eq!(values(r.clone()), [0.0, 10.0, 1.0, 11.0, 2.0, 12.0]);
        assert_eq!(
            values(TensorLoadTransform::regroup_columns(3, 2).apply(r)),
            values(m)
        );
    }

    /// Interleaved pairs become halves within each block, the further axes
    /// riding along.
    #[test]
    fn test_interleaved_to_halves() {
        // Two blocks of four rows, each row two wide: x0 y0 x1 y1.
        let rows: Vec<f32> = [0.0, 100.0, 1.0, 101.0, 2.0, 102.0, 3.0, 103.0]
            .iter()
            .flat_map(|&r| [r, r + 0.5])
            .collect();
        let m = data(&rows, &[8, 2]);
        let h = TensorLoadTransform::interleaved_to_halves(4).apply(m);
        assert_eq!(h.shape().as_slice(), [8, 2]);
        let firsts: Vec<f32> = values(h).chunks(2).map(|r| r[0]).collect();
        assert_eq!(firsts, [0.0, 1.0, 100.0, 101.0, 2.0, 3.0, 102.0, 103.0]);
    }

    /// `then` applies in order, and is named for both.
    #[test]
    fn test_then() {
        let a = TensorLoadTransform::zero_centered_gain().then(TensorLoadTransform::scale(2.0));
        assert_eq!(a.name(), "zero_centered_gain then scale(2)");
        assert_eq!(values(a.apply(data(&[0.5], &[1]))), [3.0]);
    }

    #[test]
    fn test_select() {
        assert!(Select::from("a.norm.weight").matches("a.norm.weight"));
        assert!(!Select::from("a.norm.weight").matches("b.norm.weight"));
        let norms = Select::suffix("norm.weight");
        assert!(norms.matches("stack.layers.3.attn_post_norm.weight"));
        assert!(norms.matches("stack.norm.weight"));
        assert!(!norms.matches("stack.layers.3.attn.c_q.weight"));
        assert_eq!(norms.to_string(), "*norm.weight");
    }
}
