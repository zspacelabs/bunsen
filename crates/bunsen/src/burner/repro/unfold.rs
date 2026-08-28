//! # `unfold` truncates its outer stride to the vectorization line width.
//!
//! Affects `CubeCL` backends; `burn::backend::Flex` is correct.
//!
//! ## Expected semantics
//!
//! `Tensor::unfold(dim, size, step)` should follow `PyTorch`'s
//! `Tensor.unfold`: a pure view that replaces `shape[dim]` with
//! `num = (shape[dim] - size) / step + 1`, appends a trailing axis of length
//! `size`, and sets
//!
//! ```rust,ignore
//! strides[dim]  = step * old_strides[dim]
//! strides.push(          old_strides[dim])
//! // every other stride is left ALONE
//! ```
//!
//! so element `[.., i, .., j]` reads `input[.., i * step + j, ..]`. The
//! load-bearing part is that the *other* dimensions' strides are never
//! recomputed — that is what keeps the view correct when the unfolded axis is
//! longer than the windows happen to cover.
//!
//! `burn-cubecl`'s `unfold` (`ops/base.rs`) computes exactly this, correctly.
//! The fault is downstream, in how the resulting view is read.
//!
//! ## The defect
//!
//! When `size` and `step` share a factor of two the access vectorizes, and the
//! **outer stride is truncated to a multiple of the line width `v`** —
//! `(len / v) * v` rather than `len`. Every row after the first is then read
//! `len % v` elements early.
//!
//! Row 0 is always correct, which is what makes it easy to miss: a batch-1
//! test passes, and the corruption appears only once a second row exists.
//!
//! ## Scope
//!
//! From [`sweep`] over 315 configurations (`size` 2..=8, `step` 1..=5,
//! `num` 2..=4, tails 0..=6):
//!
//! * 42 wrong, **all** with `size` and `step` both even.
//! * Zero wrong when either is odd — that is `v == 1`, the scalar path.
//! * Wrong exactly when `tail % v != 0`.
//!
//! The tail is the uncovered remainder, `len - ((num - 1) * step + size)`.
//!
//! ## Why the two rules needed separating
//!
//! "Uses the covered span `(num - 1) * step + size`" and "rounds the row down
//! to a whole line" predict the same offset in almost every configuration,
//! because `v` divides both `size` and `step` and therefore divides the
//! covered span. They differ at `size = 2, step = 4, len = 9`, which predicts
//! row 1 starting at flat `6` or `8` respectively. It starts at `8`, so the
//! truncation rule is the operative one — see
//! [`discriminate_truncation_rule`].
//!
//! That distinction matters for anyone fixing it: the outer stride is being
//! rounded, not substituted.

use burn::{
    prelude::*,
    tensor::Int,
};

/// The line width inferred for a `(size, step)` pair.
///
/// The largest power of two dividing both. Inferred from observed failures
/// rather than read out of the runtime, so treat it as a description of the
/// symptom rather than of the implementation.
pub fn inferred_line_width(
    size: usize,
    step: usize,
) -> usize {
    1usize << size.trailing_zeros().min(step.trailing_zeros())
}

/// The minimal failing case.
///
/// ```text
/// input            unfold(dim=1, size=2, step=2)
/// [[0,1,2,3,4],    [[[0,1],[2,3]],
///  [5,6,7,8,9]]     [[5,6],[7,8]]]
/// ```
///
/// `len = 5`, `v = 2`, so `5 % 2 = 1` and row 1 is read one element early,
/// coming back as `[[4,5],[6,7]]`.
///
/// `arange` is deliberate: every element is its own flat index, so a displaced
/// row reads as an off-by-one run rather than as arbitrary values.
///
/// # Panics
/// On an affected backend. That is the point.
pub fn minimal<B: Backend>(device: &B::Device) {
    let input = Tensor::<B, 1, Int>::arange(0..10, device).reshape([2, 5]);
    let unfolded = input.unfold::<3, _>(1, 2, 2);

    assert_eq!(unfolded.dims(), [2, 2, 2]);

    let got: Vec<i32> = unfolded.to_data().to_vec().unwrap();
    let want = vec![0, 1, 2, 3, /* row 1 */ 5, 6, 7, 8];

    assert_eq!(
        got, want,
        "\n  want {want:?}\n  got  {got:?}\n  \
         row 1 should start at flat index 5; it starts at 4, which is \
         (len / v) * v = (5 / 2) * 2.",
    );
}

/// Control: an odd `step` disables vectorization, same tail, correct result.
///
/// `size = 2`, `step = 3`, `len = 5` gives the same `num = 2` and the same
/// leftover tail of 1, but `v == 1`. Passing here is what rules out both
/// `unfold`'s stride computation and the mere presence of a tail.
///
/// # Panics
/// If this fails, the defect is *not* the one described here and the diagnosis
/// needs revisiting.
pub fn control_odd_step<B: Backend>(device: &B::Device) {
    let input = Tensor::<B, 1, Int>::arange(0..10, device).reshape([2, 5]);
    let unfolded = input.unfold::<3, _>(1, 2, 3);

    assert_eq!(unfolded.dims(), [2, 2, 2]);

    let got: Vec<i32> = unfolded.to_data().to_vec().unwrap();
    // Row 0: [0,1] [3,4]   Row 1: [5,6] [8,9]
    assert_eq!(got, vec![0, 1, 3, 4, 5, 6, 8, 9]);
}

/// Control: no leftover tail, same vectorization, correct result.
///
/// `size = 2`, `step = 2`, `len = 4` gives `tail = 0`. Passing here rules out
/// vectorization *per se*, and shows why trimming an input to the span its
/// windows cover is a sufficient workaround rather than merely a different
/// shape: `v` divides the covered span, so the truncation becomes a no-op.
///
/// # Panics
/// If this fails, the diagnosis needs revisiting.
pub fn control_no_tail<B: Backend>(device: &B::Device) {
    let input = Tensor::<B, 1, Int>::arange(0..8, device).reshape([2, 4]);
    let unfolded = input.unfold::<3, _>(1, 2, 2);

    assert_eq!(unfolded.dims(), [2, 2, 2]);

    let got: Vec<i32> = unfolded.to_data().to_vec().unwrap();
    // Row 0: [0,1] [2,3]   Row 1: [4,5] [6,7]
    assert_eq!(got, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

/// Separates the two candidate rules; see the module docs.
///
/// `size = 2, step = 4, len = 9`. Returns the flat index row 1 actually starts
/// at: `6` would mean the covered span is substituted, `8` that the row is
/// rounded down to a whole line, and `9` that the backend is correct.
pub fn discriminate_truncation_rule<B: Backend>(device: &B::Device) -> i32 {
    let input = Tensor::<B, 1, Int>::arange(0..18, device).reshape([2, 9]);
    let got: Vec<i32> = input.unfold::<3, _>(1, 2, 4).to_data().to_vec().unwrap();

    // Row 1's first window begins after row 0's `num * size` elements.
    let num = 2usize;
    got[num * 2]
}

/// One configuration's verdict, as reported by [`sweep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnfoldCase {
    /// The window length.
    pub size: usize,
    /// The window step.
    pub step: usize,
    /// The length of the unfolded axis.
    pub len: usize,
    /// How many windows fit.
    pub num: usize,
    /// Elements of the axis no window reaches.
    pub tail: usize,
    /// The line width inferred for this pair; see [`inferred_line_width`].
    pub line_width: usize,
}

impl UnfoldCase {
    /// Whether this case matches the predicted failure condition.
    pub fn predicted_wrong(&self) -> bool {
        !self.tail.is_multiple_of(self.line_width)
    }
}

/// Sweeps a neighbourhood of geometries and returns those that read wrong.
///
/// Reports rather than asserts, so it can be run against a candidate fix to
/// watch the failing set shrink to empty. Compare each entry's
/// [`predicted_wrong`](UnfoldCase::predicted_wrong) against its presence here:
/// on an affected backend the two agree exactly.
pub fn sweep<B: Backend>(device: &B::Device) -> Vec<UnfoldCase> {
    let mut wrong = Vec::new();

    for size in 2..=8usize {
        for step in 1..=5usize {
            for num in 2..=4usize {
                for extra in 0..=6usize {
                    let len = (num - 1) * step + size + extra;
                    if (len - size) / step + 1 != num {
                        continue;
                    }

                    let rows = 2usize;
                    let input = Tensor::<B, 1, Int>::arange(0..(rows * len) as i64, device)
                        .reshape([rows, len]);
                    let got: Vec<i32> = input
                        .unfold::<3, _>(1, size, step)
                        .to_data()
                        .to_vec()
                        .unwrap();

                    let want: Vec<i32> = (0..rows)
                        .flat_map(|b| {
                            (0..num).flat_map(move |i| {
                                (0..size).map(move |j| (b * len + i * step + j) as i32)
                            })
                        })
                        .collect();

                    if got != want {
                        wrong.push(UnfoldCase {
                            size,
                            step,
                            len,
                            num,
                            tail: len - ((num - 1) * step + size),
                            line_width: inferred_line_width(size, step),
                        });
                    }
                }
            }
        }
    }

    wrong
}

#[cfg(test)]
mod tests {
    use burn::tensor::backend::BackendTypes;

    use super::*;
    use crate::support::testing::{
        CpuBackend,
        PerformanceBackend,
    };

    /// The reproduction, against the performance backend.
    ///
    /// Ignored: it asserts the *correct* semantics, so on an affected backend
    /// it fails by design. Run it to check a candidate fix.
    #[test]
    #[ignore = "asserts correct semantics; fails on affected backends by design"]
    fn test_repro_on_performance_backend() {
        let device = <PerformanceBackend as BackendTypes>::Device::default();
        control_odd_step::<PerformanceBackend>(&device);
        control_no_tail::<PerformanceBackend>(&device);
        minimal::<PerformanceBackend>(&device);
    }

    /// The same reproduction against `Flex`, which is unaffected.
    ///
    /// Not ignored: it documents that the defect is backend-specific, and
    /// would catch a regression that made the CPU path match the broken one.
    #[test]
    fn test_cpu_backend_is_correct() {
        let device = <CpuBackend as BackendTypes>::Device::default();
        control_odd_step::<CpuBackend>(&device);
        control_no_tail::<CpuBackend>(&device);
        minimal::<CpuBackend>(&device);
        assert_eq!(discriminate_truncation_rule::<CpuBackend>(&device), 9);
        assert!(sweep::<CpuBackend>(&device).is_empty());
    }

    /// Pins the **current** behaviour, so a `CubeCL` fix announces itself here
    /// rather than leaving the covered-span trims as unexplained code.
    ///
    /// Gated on an accelerator feature: `PerformanceBackend` falls back to
    /// `Flex` when none is selected, and `Flex` is correct — so without the
    /// gate this would fail on a CPU-only run and report a fix that has not
    /// happened. The stride repro needs no such gate; that defect is in the
    /// store, this one is in a kernel.
    ///
    /// Asserts only that *something* still reads wrong. Pinning the exact
    /// count would break on a driver or hardware change that alters
    /// vectorization without meaning the defect is gone.
    #[test]
    #[cfg(any(feature = "wgpu", feature = "cuda", feature = "metal"))]
    fn test_performance_backend_is_currently_wrong() {
        let device = <PerformanceBackend as BackendTypes>::Device::default();

        assert!(
            !sweep::<PerformanceBackend>(&device).is_empty(),
            "`unfold` now reads every swept configuration correctly — CubeCL \
             appears to honour the outer stride. Un-ignore \
             `test_repro_on_performance_backend`, and re-check the \
             covered-span trims in `MelConverter::frame` and \
             `SlidingStft::analyze`. They are semantically free, so they may \
             be kept on their own merits — unlike a stride repair, leaving \
             them after a fix introduces nothing.",
        );
    }

    /// Prints the failing set for the performance backend.
    ///
    /// Ignored because it is a report, not an assertion.
    #[test]
    #[ignore = "diagnostic report, not an assertion"]
    fn test_report_sweep() {
        let device = <PerformanceBackend as BackendTypes>::Device::default();
        let wrong = sweep::<PerformanceBackend>(&device);

        eprintln!("{} configurations read wrong", wrong.len());
        eprintln!("size step  len num tail    v  predicted");
        for c in &wrong {
            eprintln!(
                "{:4} {:4} {:4} {:3} {:4} {:4}  {}",
                c.size,
                c.step,
                c.len,
                c.num,
                c.tail,
                c.line_width,
                c.predicted_wrong(),
            );
        }
        eprintln!(
            "row 1 starts at flat {} (6 = covered-span rule, 8 = truncation rule, 9 = correct)",
            discriminate_truncation_rule::<PerformanceBackend>(&device),
        );
        assert!(
            wrong.iter().all(UnfoldCase::predicted_wrong),
            "a failure fell outside the predicted condition",
        );
    }
}
