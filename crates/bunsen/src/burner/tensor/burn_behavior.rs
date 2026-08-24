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

    /// `unfold` derives its batch-row stride from the span its windows cover
    /// rather than from the row's true length.
    ///
    /// A row with a leftover tail therefore places every *subsequent* row
    /// early, by exactly the tail length. Row 0 is always correct, which is
    /// what makes this so easy to miss: a batch-1 test passes, and the bug only
    /// appears once a second stream is added.
    ///
    /// **Workaround:** trim the input to the covered span before unfolding.
    /// Used by `ops::signal` and by the ten-vad pitch stages.
    #[test]
    fn test_unfold_derives_row_stride_from_the_covered_span() {
        let device = <B as BackendTypes>::Device::default();
        let (win, step, steps, tail) = (12usize, 4usize, 3usize, 3usize);
        let covered = (steps - 1) * step + win;
        let len = covered + tail;

        // Row 1 carries a marker at its very first element.
        let mut flat = vec![0.0f32; 2 * len];
        flat[len] = 1.0;
        let t = Tensor::<B, 2>::from_data(TensorData::new(flat, [2, len]), &device);

        let rows: Vec<f32> = t
            .unfold::<3, _>(1, win, step)
            .reshape([2 * steps, win])
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .unwrap();

        // Window `steps` is (batch 1, step 0), so the marker should sit at 0.
        let found = rows[steps * win..(steps + 1) * win]
            .iter()
            .position(|v| *v != 0.0);
        assert_eq!(
            found,
            Some(tail),
            "expected the row-1 marker displaced by the leftover tail ({tail}). \
             Some(0) means `unfold` now uses the true row stride, and every \
             trim-before-unfold in the tree is redundant.",
        );
    }

    /// With no leftover tail, `unfold`'s row stride is correct.
    ///
    /// The control for
    /// [`test_unfold_derives_row_stride_from_the_covered_span`]: it establishes
    /// that trimming to the covered span is a *sufficient* workaround, not just
    /// a different way to be wrong.
    #[test]
    fn test_unfold_row_stride_is_correct_without_a_tail() {
        let device = <B as BackendTypes>::Device::default();
        let (win, step, steps) = (12usize, 4usize, 3usize);
        let covered = (steps - 1) * step + win;

        let mut flat = vec![0.0f32; 2 * covered];
        flat[covered] = 1.0;
        let t = Tensor::<B, 2>::from_data(TensorData::new(flat, [2, covered]), &device);

        let rows: Vec<f32> = t
            .unfold::<3, _>(1, win, step)
            .reshape([2 * steps, win])
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .unwrap();

        assert_eq!(
            rows[steps * win..(steps + 1) * win]
                .iter()
                .position(|v| *v != 0.0),
            Some(0),
        );
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
