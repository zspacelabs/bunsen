//! # Pinned assumptions about burn's tensor behavior.
//!
//! Test-only. Each test here documents a behavior of an upstream burn op that
//! bunsen code works around, and fails when that behavior changes.
//!
//! These are not tests of bunsen. They exist because a workaround with no test
//! is indistinguishable from an accident: the next person to read the call site
//! cannot tell whether the awkward formulation is load-bearing or leftover, and
//! deletes it. Every test below is written to fail *when the bug is fixed*, and
//! says in its message which workaround has become redundant — so the upgrade
//! that fixes it also tells you what to go simplify.
//!
//! When one starts failing:
//!
//! 1. Confirm against the upstream change that the behavior really is fixed.
//! 2. Simplify the call sites the test names.
//! 3. Delete the test.
//!
//! Everything here is backend-visible behavior, so it runs on
//! `PerformanceBackend` like the rest of the device tests.

#[cfg(test)]
mod tests {
    use burn::{
        prelude::*,
        tensor::backend::BackendTypes,
    };

    use crate::{
        prelude::*,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;

    /// On `CubeCL`, `unfold` reads every row after the first too early when the
    /// unfolded axis is not a whole number of vectorization lines.
    ///
    /// `unfold` itself is correct: `burn-cubecl`'s implementation sets
    /// `strides[dim] = step * old` and appends `old`, leaving every other
    /// stride alone, which is exactly `PyTorch`'s view semantics. The fault is in
    /// the *read*. When `size` and `step` share a factor of two the access
    /// vectorizes, and the outer stride is truncated to a multiple of the line
    /// width `v` -- `(len / v) * v` rather than `len` -- so each subsequent row
    /// starts `len % v` elements early.
    ///
    /// Row 0 is always correct, which is what makes this easy to miss: a
    /// batch-1 test passes and the corruption appears only once a second row
    /// exists.
    ///
    /// Measured across 315 configurations: 42 wrong, all with `size` and `step`
    /// both even, and failing exactly when `tail % v != 0`. `Flex` is correct
    /// throughout, so this is `CubeCL`-specific.
    ///
    /// **Workaround:** trim the input to the span the windows actually cover,
    /// `(num - 1) * step + size`, before unfolding. That is sound because `v`
    /// divides both `size` and `step` and therefore divides the covered span,
    /// so the truncation becomes a no-op. Used by `ops::signal::DecimatingFir`
    /// and `ops::signal::LagSearch`.
    #[test]
    fn test_unfold_truncates_the_outer_stride_to_the_line_width() {
        let device = <B as BackendTypes>::Device::default();

        // [[0,1,2,3,4],      unfold(dim=1, size=2, step=2)  ->
        //  [5,6,7,8,9]]      [[[0,1],[2,3]], [[5,6],[7,8]]]
        //
        // len = 5, v = 2, so 5 % 2 = 1 and row 1 is read one element early.
        // `arange` so a displaced row reads as an off-by-one run rather than
        // as arbitrary values.
        let input = Tensor::<B, 1, Int>::arange(0..10, &device).reshape([2, 5]);
        let got: Vec<i32> = input.unfold::<3, _>(1, 2, 2).to_data().to_vec().unwrap();

        assert_eq!(
            got,
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            "expected the known-bad reading; [0,1,2,3,5,6,7,8] means CubeCL now              honours the outer stride, and every trim-before-unfold in the tree              is redundant",
        );
    }

    /// An odd `step` disables vectorization, and the same geometry is correct.
    ///
    /// The control that isolates the fault to the vectorized read: same input,
    /// same leftover tail, `v == 1`, right answer.
    #[test]
    fn test_unfold_is_correct_on_the_scalar_path() {
        let device = <B as BackendTypes>::Device::default();

        let input = Tensor::<B, 1, Int>::arange(0..10, &device).reshape([2, 5]);
        let got: Vec<i32> = input.unfold::<3, _>(1, 2, 3).to_data().to_vec().unwrap();

        // Row 0: [0,1] [3,4]   Row 1: [5,6] [8,9]
        assert_eq!(got, vec![0, 1, 3, 4, 5, 6, 8, 9]);
    }

    /// With no leftover tail there is nothing to truncate, and the vectorized
    /// path is correct.
    ///
    /// The second control, and the one that shows *why* trimming to the covered
    /// span is a sufficient workaround rather than merely a different shape.
    #[test]
    fn test_unfold_is_correct_without_a_tail() {
        let device = <B as BackendTypes>::Device::default();

        let input = Tensor::<B, 1, Int>::arange(0..8, &device).reshape([2, 4]);
        let got: Vec<i32> = input.unfold::<3, _>(1, 2, 2).to_data().to_vec().unwrap();

        assert_eq!(got, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    /// `matmul` fails outright when its left operand is an `unfold` view and
    /// the right operand is a column.
    ///
    /// Every autotune candidate returns `InvalidSamples` and the tuner panics
    /// rather than falling back, so this is a hard failure and not a slow path.
    /// The panic happens on a worker thread; the caller sees a `CallError`.
    ///
    /// The trigger is the **non-contiguous left operand**, which is worth
    /// stating because the autotune key claims otherwise -- it reports
    /// `matrix_layout_lhs: Contiguous` for exactly the view that fails.
    /// Narrowed by elimination: the same shapes with a contiguous left operand
    /// succeed on both random and constant data, and only the `unfold`-derived
    /// operand fails. Same family as the `unfold` row-stride bug above.
    ///
    /// **Workaround:** broadcast and reduce --
    /// `(lhs * rhs.squeeze_dim(2).unsqueeze_dim(1)).sum_dim(2)` -- which is the
    /// same arithmetic, avoids the mat-vec entirely, and fuses with any
    /// neighbouring reduction over the same operand. Used by
    /// `ops::signal::LagSearch`.
    ///
    /// Marked `#[should_panic]` because the failure *is* the behavior being
    /// pinned: when burn fixes this, the test fails by passing, and the
    /// workaround it names can go.
    #[test]
    #[should_panic(expected = "CallError")]
    fn test_matmul_rejects_an_unfold_view_against_a_column() {
        let device = <B as BackendTypes>::Device::default();
        let (rows, m, k) = (2usize, 32usize, 16usize);

        let buf = Tensor::<B, 2>::random(
            [rows, m + k - 1],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let lhs = buf.clone().unfold::<3, _>(1, k, 1);
        let rhs = buf.slice_dim(1, 0..k as isize).reshape([rows, k, 1]);

        // Force execution: the failure happens on the device, not at build.
        let _ = lhs.matmul(rhs).into_data();
    }

    /// A contiguous left operand at the same shape is fine.
    ///
    /// The control for
    /// [`test_matmul_rejects_an_unfold_view_against_a_column`]: without it the
    /// failure above reads as "mat-vec is broken", which would be both wrong
    /// and much more alarming.
    #[test]
    fn test_matmul_accepts_a_contiguous_column() {
        let device = <B as BackendTypes>::Device::default();
        let (rows, m, k) = (2usize, 32usize, 16usize);

        let lhs = Tensor::<B, 3>::random(
            [rows, m, k],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let rhs = Tensor::<B, 3>::random(
            [rows, k, 1],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        assert_eq!(lhs.matmul(rhs).dims(), [rows, m, 1]);
    }

    /// `gather` ignores strides on a non-contiguous *index* tensor.
    ///
    /// Given a transposed index view it reads element 0 of each row rather than
    /// the indexed element. The data tensor's strides are honoured; only the
    /// index's are not.
    ///
    /// **Workaround:** build index tensors with `stack`, which is contiguous,
    /// rather than `cat` + `swap_dims`, which is a view. Used by the ten-vad
    /// Viterbi backtrace.
    #[test]
    fn test_gather_ignores_strides_on_a_non_contiguous_index() {
        let device = <B as BackendTypes>::Device::default();
        let (steps, wide) = (3usize, 56usize);

        let mut v = vec![0i32; steps * wide];
        for t in 0..steps {
            for j in 0..wide {
                v[t * wide + j] = (t as i32) * 1000 + j as i32;
            }
        }
        let data = Tensor::<B, 3, Int>::from_data(TensorData::new(v, [steps, 1, wide]), &device);

        // The same logical index, built two ways.
        let contiguous = Tensor::<B, 3, Int>::full([steps, 1, 1], 37, &device);
        let rows: Vec<Tensor<B, 2, Int>> = (0..steps)
            .map(|_| Tensor::full([1, 1], 37, &device))
            .collect();
        let transposed = Tensor::cat(rows, 1).swap_dims(0, 1).unsqueeze_dim::<3>(2);

        let good: Vec<i32> = data
            .clone()
            .gather(2, contiguous)
            .to_data_as::<i32>()
            .to_vec_as::<i32>()
            .unwrap();
        let bad: Vec<i32> = data
            .gather(2, transposed)
            .to_data_as::<i32>()
            .to_vec_as::<i32>()
            .unwrap();

        assert_eq!(
            good,
            vec![37, 1037, 2037],
            "a contiguous index should read element 37 of each row",
        );
        assert_ne!(
            good, bad,
            "`gather` now honours strides on the index tensor; every \
             `stack`-instead-of-`cat`+`swap_dims` workaround in the tree is \
             no longer load-bearing.",
        );
    }
}
