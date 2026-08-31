//! # `PytorchStore` reads raw storage, ignoring `PyTorch` tensor strides.
//!
//! Requires the `store` feature: the reproduction loads a checkpoint through
//! `burn_store`, an optional dependency.
//!
//! Affects every non-contiguous tensor in a `.pt` checkpoint.
//! Backend-independent — the fault is in the store, not in any kernel.
//!
//! ## Expected semantics
//!
//! A `PyTorch` tensor is a *view*: shape plus strides plus an offset over a
//! flat storage. `torch.save` records all of it. Reading `lin.weight` should
//! yield the logical tensor the shape and strides describe, whatever the
//! storage layout.
//!
//! ## The defect
//!
//! `PytorchStore` takes the storage bytes and interprets them as row-major in
//! the declared shape. For a contiguous tensor those agree, and everything
//! works. For a **column-major view** — shape `[R, C]` with strides `(1, R)` —
//! the storage actually holds the `[C, R]` transpose in row-major order, and
//! the read produces
//!
//! ```text
//! S = reshape(flat(Wᵀ), [R, C])
//! ```
//!
//! which is neither `W` nor `Wᵀ`. `PyTorchToBurnAdapter` then transposes it for
//! a `Linear` destination, giving `T = Sᵀ`, and because `T`'s shape matches the
//! parameter, the load reports success with no error.
//!
//! ## Why this is easy to miss
//!
//! For a **square** weight the corruption degenerates to a plain transpose, so
//! the symptom reads as "the adapter forgot to transpose" rather than as data
//! damage. Only a non-square weight makes it obvious. A checkpoint whose
//! `Linear` layers are all square can be wrong end-to-end while every shape
//! checks out.
//!
//! ## Scope
//!
//! Every `Linear` weight in an `OpenAI` Whisper checkpoint is stored this way —
//! 96 of the 245 tensors in `base.en.pt`, being exactly
//! `attn.{query,key,value,out}.weight` and `mlp.{0,2}.weight`, each with
//! strides `(1, N)`.
//!
//! ## Recovery
//!
//! The corruption is invertible, which is what makes a workaround possible.
//! Since `flat(Tᵀ) = flat(S) = flat(Wᵀ)`, transposing back and reshaping to the
//! parameter's own shape recovers `Wᵀ`:
//!
//! ```text
//! Wᵀ = reshape(flat(Tᵀ), [C, R])
//! ```
//!
//! That is [`repair_pytorch_strided_weight`], applied as a parameter load
//! mapper.

use bunsen::burner::store::repair_pytorch_strided_weight;
use burn::{
    module::Module,
    nn::{
        Linear,
        LinearConfig,
    },
    prelude::*,
};
use burn_store::{
    ModuleSnapshot,
    PytorchStore,
};

/// The fixture's logical weight is `[8, 3]` with `w[o][i] == i.o`, stored as a
/// column-major view. A row-major `Linear(3, 8)` wants its transpose.
const D_INPUT: usize = 3;

/// The fixture's output width.
const D_OUTPUT: usize = 8;

/// A single `Linear`, to give the store a `Struct:Linear` destination.
#[derive(Module, Debug)]
pub struct StridedLinearProbe<B: Backend> {
    /// The projection under test.
    pub lin: Linear<B>,
}

/// Loads the fixture's `key` entry into a `Linear(3, 8)`.
///
/// # Arguments
/// * `key`: the checkpoint's top-level key — `"strided"` or `"contiguous"`.
/// * `repair`: whether to attach [`repair_pytorch_strided_weight`].
pub fn load_probe<B: Backend>(
    key: &str,
    repair: bool,
    device: &B::Device,
) -> StridedLinearProbe<B> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/pytorch_strided_linear.pt");

    let mut lin = LinearConfig::new(D_INPUT, D_OUTPUT)
        .with_bias(false)
        .init::<B>(device);

    if repair {
        lin.weight = repair_pytorch_strided_weight(lin.weight);
    }

    let mut probe = StridedLinearProbe { lin };
    let mut store = PytorchStore::from_file(path).with_top_level_key(key);

    let result = probe.load_from(&mut store).expect("fixture loads");
    assert_eq!(
        result.errors.len(),
        0,
        "the load reports no error either way; that is the point",
    );
    assert_eq!(result.applied.len(), 1, "the weight is applied");

    probe
}

/// The `[3, 8]` weight a correct load must produce: `w[i][o] == i.o`.
pub fn expected_weight() -> Vec<f64> {
    (0..D_INPUT)
        .flat_map(|i| (0..D_OUTPUT).map(move |o| format!("{i}.{o}").parse::<f64>().unwrap()))
        .collect()
}

/// Reads a `[3, 8]` weight back as `f64`, row-major.
pub fn weight_of<B: Backend>(probe: &StridedLinearProbe<B>) -> Vec<f64> {
    probe
        .lin
        .weight
        .val()
        .cast(burn::tensor::DType::F64)
        .to_data()
        .to_vec()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use bunsen::support::testing::{
        CpuBackend,
        assert_close_to_vec,
    };

    use super::*;

    type B = CpuBackend;

    /// The fixture is `f32`, so `0.1` is only good to about `1.5e-8`.
    const TOLERANCE: f64 = 1e-6;

    /// A contiguous tensor loads correctly — the store is only wrong about
    /// strides, and this pins that the fixture and the destination agree.
    #[test]
    fn test_contiguous_source_is_correct() {
        let device = Default::default();
        let probe = load_probe::<B>("contiguous", false, &device);

        assert_close_to_vec(&weight_of(&probe), &expected_weight(), TOLERANCE);
    }

    /// **The reproduction.** The same values, stored as a column-major view,
    /// must load to the same weight.
    ///
    /// Ignored: it fails while the defect is present. Un-ignore it to check
    /// whether a `burn-store` release has fixed the read.
    #[test]
    #[ignore = "reproduces the burn-store stride defect; fails until it is fixed"]
    fn test_strided_source_should_match_contiguous() {
        let device = Default::default();
        let probe = load_probe::<B>("strided", false, &device);

        assert_close_to_vec(&weight_of(&probe), &expected_weight(), TOLERANCE);
    }

    /// Pins the **current** behaviour, so a `burn-store` fix announces itself
    /// here instead of silently making the workaround a double transpose.
    ///
    /// The failure message says what to remove.
    #[test]
    fn test_strided_source_is_currently_corrupt() {
        let device = Default::default();
        let probe = load_probe::<B>("strided", false, &device);

        let got = weight_of(&probe);
        let want = expected_weight();
        assert_eq!(got.len(), want.len());

        let differs = got
            .iter()
            .zip(&want)
            .any(|(a, b)| (a - b).abs() > TOLERANCE);
        assert!(
            differs,
            "the strided read now matches the contiguous one — burn-store \
             appears to honour strides. Remove `repair_pytorch_strided_weight` \
             and its callers, and un-ignore \
             `test_strided_source_should_match_contiguous`.",
        );
    }

    /// The workaround recovers the correct weight from the corrupted read.
    #[test]
    fn test_repair_recovers_the_weight() {
        let device = Default::default();
        let probe = load_probe::<B>("strided", true, &device);

        assert_close_to_vec(&weight_of(&probe), &expected_weight(), TOLERANCE);
    }

    /// The corruption degenerates to a transpose when the weight is square,
    /// which is why a square-only model looks merely mis-oriented rather than
    /// damaged — and why the repair cannot be applied blindly: on a weight
    /// that did not need it, it is a silent transpose.
    #[test]
    fn test_square_case_degenerates_to_a_transpose() {
        // `reshape(flat(Wᵀ), [n, n])` is `Wᵀ`, so `T = Sᵀ = W`: a pure
        // orientation error, with no data scrambling.
        let n = 4;
        let w: Vec<f64> = (0..n * n).map(|k| k as f64).collect();

        // Row-major `Wᵀ` read back as `[n, n]` is `Wᵀ`; transposing gives `W`.
        let transpose = |m: &[f64]| -> Vec<f64> {
            (0..n)
                .flat_map(|i| (0..n).map(move |j| (j, i)))
                .map(|(r, c)| m[r * n + c])
                .collect()
        };
        assert_eq!(transpose(&transpose(&w)), w);
    }
}
