//! # Batched Viterbi decoding on device.
//!
//! Finds the highest-scoring path through a trellis:
//!
//! ```text
//! score[t][s] = emission[t][s] + max(score[t-1][c] + transition[c][s] for c)
//! ```
//!
//! Batched over independent sequences, streaming across calls, and with a
//! backtrace that does **not** walk step by step.
//!
//! ## The forward pass is sequential; the backtrace is not
//!
//! The recursion genuinely depends on its own previous step, so the forward
//! pass is one dependent iteration per timestep. That is irreducible.
//!
//! The backtrace is not. Walking back from every timestep independently looks
//! like `steps * depth` scalar hops, but at lookback `k` every timestep reads
//! the same *relative* position in the backpointer history — so one strided
//! slice serves all of them, and the whole backtrace is `depth` batched
//! gathers rather than `steps * depth` sequential ones. For a run of a few
//! thousand steps that is the difference between the backtrace dominating and
//! the backtrace being free.
//!
//! ## Renormalization
//!
//! Path scores grow without bound over a long stream, so
//! [`ViterbiState::score`] is kept with its peak at zero: each step subtracts
//! the running maximum. That changes nothing about which path wins — the same
//! constant is subtracted from every state — and it keeps `f32` from losing
//! the differences that matter to accumulated magnitude.
//!
//! ## Forbidden transitions
//!
//! Use a **large finite negative**, not `-inf`. A finite sentinel keeps the
//! `max` well-defined and lets a state with no reachable predecessor still
//! carry a comparable score, where `-inf` propagates and eventually produces
//! `NaN` under the renormalizing subtraction. [`FORBIDDEN`] is a suitable
//! value; anything that cannot be overcome by accumulated emissions works.

use burn::{
    config::Config,
    prelude::*,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        WithOkOrPanic,
    },
    prelude::TensorOpExt,
};

/// A transition score meaning "this transition may not be taken".
///
/// Large enough that no plausible accumulation of emission scores overcomes
/// it, finite so that `max` and the renormalizing subtraction stay defined.
pub const FORBIDDEN: f32 = -1e30;

/// Config for [`Viterbi`].
#[derive(Config, Debug, Copy)]
pub struct ViterbiConfig {
    /// The number of trellis states.
    pub n_states: usize,
}

impl ViterbiConfig {
    /// Validates the geometry.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the state count is zero.
    pub fn validate(&self) -> BunsenResult<()> {
        if self.n_states == 0 {
            return Err(BunsenError::Invalid(
                "Viterbi n_states must be non-zero".to_string(),
            ));
        }
        Ok(())
    }

    /// Builds a decoder around a transition table.
    ///
    /// # Arguments
    /// * `transition`: `[n_states * n_states]` row-major, indexed `[from][to]`.
    ///   Use [`FORBIDDEN`] for transitions that may not be taken.
    ///
    /// # Errors
    /// See [`validate`](Self::validate); also [`BunsenError::Invalid`] if
    /// `transition` is the wrong length.
    pub fn try_init<B: Backend>(
        &self,
        transition: &[f32],
        device: &B::Device,
    ) -> BunsenResult<Viterbi<B>> {
        self.validate()?;

        let want = self.n_states * self.n_states;
        if transition.len() != want {
            return Err(BunsenError::Invalid(format!(
                "Viterbi transition has {} entries, expected {want}",
                transition.len(),
            )));
        }

        Ok(Viterbi {
            n_states: self.n_states,
            transition: Tensor::from_data(
                TensorData::new(transition.to_vec(), [self.n_states, self.n_states]),
                device,
            ),
        })
    }

    /// Builds a decoder, panicking on error.
    pub fn init<B: Backend>(
        &self,
        transition: &[f32],
        device: &B::Device,
    ) -> Viterbi<B> {
        self.try_init(transition, device).ok_or_panic()
    }
}

/// A batched Viterbi decoder over a fixed transition table.
///
/// Built by [`ViterbiConfig::try_init`].
#[derive(Debug, Clone)]
pub struct Viterbi<B: Backend> {
    n_states: usize,

    /// `[n_states, n_states]` transition scores, indexed `[from][to]`.
    pub transition: Tensor<B, 2>,
}

/// The accumulator [`Viterbi`] carries between steps.
#[derive(Debug, Clone)]
pub struct ViterbiState<B: Backend> {
    /// `[batch, n_states]` path scores, renormalized so the peak is zero.
    pub score: Tensor<B, 2>,
}

impl<B: Backend> Viterbi<B> {
    /// The number of trellis states.
    pub fn n_states(&self) -> usize {
        self.n_states
    }

    /// A zeroed start-of-stream accumulator.
    ///
    /// All states start equally likely. A caller with a prior over initial
    /// states should build [`ViterbiState`] directly instead.
    pub fn init_state(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> ViterbiState<B> {
        ViterbiState {
            score: Tensor::zeros([batch_size, self.n_states], device),
        }
    }

    /// Advances one timestep.
    ///
    /// # Arguments
    /// * `state`: the accumulator.
    /// * `emission`: `[batch, n_states]` per-state scores for this step.
    ///
    /// # Returns
    /// `(next accumulator, `[batch, `n_states`]` backpointers)`, where entry
    /// `s` is the predecessor state the best path into `s` came from.
    pub fn step(
        &self,
        state: ViterbiState<B>,
        emission: Tensor<B, 2>,
    ) -> (ViterbiState<B>, Tensor<B, 2, Int>) {
        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["batch", "n_states"],
            &emission,
            &[("n_states", self.n_states)],
        );

        // [batch, to, from]: the score of arriving at `to` out of `from`.
        // `score` broadcasts over `to`; `transition` is [from, to], so it
        // transposes into place and broadcasts over the batch.
        let arrivals = state.score.unsqueeze_dim::<3>(1)
            + self
                .transition
                .clone()
                .swap_dims(0, 1)
                .unsqueeze_dim::<3>(0);

        let (best_in, arg_in) = arrivals.max_dim_with_indices(2);
        let scored = best_in.squeeze_dim::<2>(2) + emission;

        // Renormalize to peak zero; see the module docs.
        let peak = scored.clone().max_dim(1);

        (
            ViterbiState {
                score: scored - peak,
            },
            arg_in.squeeze_dim::<2>(2),
        )
    }

    /// Runs a sequence of timesteps.
    ///
    /// # Arguments
    /// * `emissions`: `[steps, batch, n_states]` per-step scores.
    /// * `state`: the accumulator.
    ///
    /// # Returns
    /// `(`[steps, batch, `n_states`]` backpointers, next accumulator)`.
    ///
    /// # Panics
    /// If `emissions` is empty.
    pub fn forward(
        &self,
        emissions: Tensor<B, 3>,
        state: ViterbiState<B>,
    ) -> (Tensor<B, 3, Int>, ViterbiState<B>) {
        let steps = emissions.dims()[0];
        assert_ne!(steps, 0, "Viterbi emissions must be non-empty");

        let mut state = state;
        let mut prev = Vec::with_capacity(steps);
        for step in 0..steps {
            let (next, back) = self.step(state, emissions.clone().select_dim::<2>(0, step));
            state = next;
            prev.push(back);
        }

        (Tensor::stack::<3>(prev, 0), state)
    }

    /// The best state per batch row, as `[batch, 1]`.
    pub fn best_state(
        &self,
        state: &ViterbiState<B>,
    ) -> Tensor<B, 2, Int> {
        state.score.clone().argmax(1)
    }

    /// Walks every timestep's path back `depth` states, all at once.
    ///
    /// At lookback `k` each timestep reads the same relative position in
    /// `prev`, so one strided slice serves them all and the cost is `depth`
    /// batched gathers rather than `steps * depth` sequential hops.
    ///
    /// # Arguments
    /// * `prev`: `[batch, depth - 1 + steps, n_states]` backpointers in stream
    ///   order. The final `steps` entries are this run's; the leading `depth -
    ///   1` are carried from earlier calls, so that early timesteps have
    ///   something to walk into.
    /// * `cursor`: `[steps, batch]` the state each timestep's path ends on.
    ///
    /// # Returns
    /// `[steps, batch, depth]`, where `[t][b][0]` is `cursor[t][b]` and
    /// `[t][b][k]` is the state `k` steps earlier on that path.
    ///
    /// # Panics
    /// If `prev` is not `depth - 1 + steps` long on its slot axis.
    pub fn backtrace(
        prev: Tensor<B, 3, Int>,
        cursor: Tensor<B, 2, Int>,
        depth: usize,
    ) -> Tensor<B, 3, Int> {
        assert_ne!(depth, 0, "Viterbi backtrace depth must be non-zero");
        let [steps, _] = cursor.dims();
        let [_, history, _] = prev.dims();
        assert_eq!(
            history,
            depth - 1 + steps,
            "Viterbi backtrace expects {} slots of history for {steps} steps at depth \
             {depth}",
            depth - 1 + steps,
        );

        let carry = depth - 1;
        let mut cursor = cursor;
        let mut path = Vec::with_capacity(depth);

        for k in 0..depth {
            path.push(cursor.clone().unsqueeze_dim::<3>(2));
            if k + 1 == depth {
                break;
            }

            // Timestep `t` looks back `k` steps, which is history slot
            // `carry + t - k`; over all `t` that is one contiguous slice.
            let lo = (carry - k) as isize;
            let rows = prev
                .clone()
                .slice_dim(1, lo..lo + steps as isize)
                .swap_dims(0, 1);

            // Built with `stack`, not `cat` + `swap_dims`: `gather` ignores
            // strides on a non-contiguous index, pinned by
            // `burner::tensor::burn_behavior`.
            let index = cursor.unsqueeze_dim::<3>(2);
            cursor = rows.gather(2, index).squeeze_dim::<2>(2);
        }

        Tensor::cat(path, 2)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::{
        prelude::*,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;

    const S: usize = 5;

    /// A deterministic, tie-free transition table.
    fn transition() -> Vec<f32> {
        (0..S * S)
            .map(|i| {
                let (from, to) = (i / S, i % S);
                let jump = (to as f32 - from as f32).abs();
                -0.3 * jump * jump + 0.017 * i as f32
            })
            .collect()
    }

    /// A deterministic, tie-free emission block.
    fn emissions(
        steps: usize,
        batch: usize,
    ) -> Vec<f32> {
        (0..steps * batch * S)
            .map(|i| ((i as f32) * 0.734).sin() + 0.11 * ((i as f32) * 0.31).cos())
            .collect()
    }

    /// A scalar Viterbi, written from the recursion, for one batch row.
    ///
    /// Returns the backpointers and the un-renormalized final scores.
    fn scalar_viterbi(
        emit: &[f32],
        trans: &[f32],
        steps: usize,
    ) -> (Vec<usize>, Vec<f32>) {
        let mut score = vec![0.0f64; S];
        let mut prev = vec![0usize; steps * S];

        for t in 0..steps {
            let mut next = vec![0.0f64; S];
            for to in 0..S {
                let mut best = f64::NEG_INFINITY;
                let mut arg = 0usize;
                for from in 0..S {
                    let v = score[from] + trans[from * S + to] as f64;
                    if v > best {
                        best = v;
                        arg = from;
                    }
                }
                prev[t * S + to] = arg;
                next[to] = best + emit[t * S + to] as f64;
            }
            score = next;
        }

        (prev, score.iter().map(|v| *v as f32).collect())
    }

    #[test]
    fn test_config_meta() {
        let cfg = ViterbiConfig::new(S);
        cfg.validate().unwrap();
        assert!(ViterbiConfig::new(0).validate().is_err());

        let device = Default::default();
        let v: Viterbi<B> = cfg.init(&transition(), &device);
        assert_eq!(v.n_states(), S);
        assert_eq!(v.transition.dims(), [S, S]);
    }

    #[test]
    fn test_init_rejects_a_mismatched_transition() {
        let device = Default::default();
        assert!(matches!(
            ViterbiConfig::new(S).try_init::<B>(&[0.0; S * S - 1], &device),
            Err(BunsenError::Invalid(_)),
        ));
    }

    #[test]
    fn test_matches_a_scalar_viterbi() {
        // The anchor: the device pass against the recursion written out by
        // hand, backpointers and all.
        let device = Default::default();
        let steps = 7;
        let trans = transition();
        let emit = emissions(steps, 1);

        let v: Viterbi<B> = ViterbiConfig::new(S).init(&trans, &device);
        let e = Tensor::<B, 3>::from_data(TensorData::new(emit.clone(), [steps, 1, S]), &device);
        let (prev, state) = v.forward(e, v.init_state(1, &device));

        let (want_prev, want_score) = scalar_viterbi(&emit, &trans, steps);

        let got_prev: Vec<i32> = prev.to_data_as::<i32>().to_vec_as::<i32>().ok_or_panic();
        assert_eq!(
            got_prev.iter().map(|v| *v as usize).collect::<Vec<_>>(),
            want_prev,
        );

        // Scores are renormalized to peak zero, so compare the differences.
        let got: Vec<f32> = state
            .score
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .ok_or_panic();
        let want_peak = want_score.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
        for s in 0..S {
            let want = want_score[s] - want_peak;
            assert!(
                (got[s] - want).abs() < 1e-4,
                "state {s}: got {}, want {want}",
                got[s],
            );
        }
    }

    #[test]
    fn test_score_peak_is_zero() {
        let device = Default::default();
        let steps = 6;
        let v: Viterbi<B> = ViterbiConfig::new(S).init(&transition(), &device);
        let e =
            Tensor::<B, 3>::from_data(TensorData::new(emissions(steps, 3), [steps, 3, S]), &device);
        let (_, state) = v.forward(e, v.init_state(3, &device));

        let peak: Vec<f32> = state
            .score
            .max_dim(1)
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .ok_or_panic();
        for (row, p) in peak.iter().enumerate() {
            assert!(p.abs() < 1e-5, "row {row} peak {p} should be zero");
        }
    }

    #[test]
    fn test_streaming_matches_a_single_call() {
        let device = Default::default();
        let steps = 8;
        let v: Viterbi<B> = ViterbiConfig::new(S).init(&transition(), &device);
        let emit = emissions(steps, 2);
        let all = Tensor::<B, 3>::from_data(TensorData::new(emit, [steps, 2, S]), &device);

        let (whole_prev, whole) = v.forward(all.clone(), v.init_state(2, &device));

        let (head_prev, mid) = v.forward(all.clone().slice_dim(0, ..3), v.init_state(2, &device));
        let (tail_prev, split) = v.forward(all.slice_dim(0, 3..), mid);

        let tol = Tolerance::<f32>::permissive();
        whole
            .score
            .to_data_as::<f32>()
            .assert_approx_eq::<f32>(&split.score.to_data_as::<f32>(), tol);
        whole_prev
            .to_data()
            .assert_eq(&Tensor::cat(vec![head_prev, tail_prev], 0).to_data(), true);
    }

    #[test]
    fn test_batch_rows_are_independent() {
        let device = Default::default();
        let steps = 5;
        let v: Viterbi<B> = ViterbiConfig::new(S).init(&transition(), &device);

        let a = emissions(steps, 1);
        // Row 1 gets a different, unrelated block.
        let b: Vec<f32> = a.iter().rev().copied().collect();

        let mut both = Vec::new();
        for t in 0..steps {
            both.extend_from_slice(&a[t * S..(t + 1) * S]);
            both.extend_from_slice(&b[t * S..(t + 1) * S]);
        }
        let paired = Tensor::<B, 3>::from_data(TensorData::new(both, [steps, 2, S]), &device);
        let (_, joint) = v.forward(paired, v.init_state(2, &device));

        let solo_a = Tensor::<B, 3>::from_data(TensorData::new(a, [steps, 1, S]), &device);
        let (_, alone) = v.forward(solo_a, v.init_state(1, &device));

        let joint_row: Vec<f32> = joint
            .score
            .slice_dim(0, 0..1)
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .ok_or_panic();
        let solo_row: Vec<f32> = alone
            .score
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .ok_or_panic();
        for s in 0..S {
            assert!(
                (joint_row[s] - solo_row[s]).abs() < 1e-5,
                "state {s}: batched {} vs alone {}",
                joint_row[s],
                solo_row[s],
            );
        }
    }

    #[test]
    fn test_backtrace_walks_the_recorded_pointers() {
        // Against a hand walk of the same history: the batched form must agree
        // with following one pointer at a time.
        let device = Default::default();
        let (steps, depth) = (5usize, 4usize);
        let history = depth - 1 + steps;

        let raw: Vec<i32> = (0..history * S).map(|i| ((i * 7 + 3) % S) as i32).collect();
        let prev =
            Tensor::<B, 3, Int>::from_data(TensorData::new(raw.clone(), [1, history, S]), &device);

        let cursor_raw: Vec<i32> = (0..steps).map(|t| ((t * 3 + 1) % S) as i32).collect();
        let cursor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(cursor_raw.clone(), [steps, 1]),
            &device,
        );

        let path = Viterbi::<B>::backtrace(prev, cursor, depth);
        assert_eq!(path.dims(), [steps, 1, depth]);
        let got: Vec<i32> = path.to_data_as::<i32>().to_vec_as::<i32>().ok_or_panic();

        for t in 0..steps {
            let mut c = cursor_raw[t] as usize;
            for k in 0..depth {
                assert_eq!(got[t * depth + k], c as i32, "step {t}, lookback {k}",);
                if k + 1 < depth {
                    // History slot `carry + t - k`.
                    let slot = (depth - 1) + t - k;
                    c = raw[slot * S + c] as usize;
                }
            }
        }
    }

    #[test]
    fn test_backtrace_depth_one_is_the_cursor() {
        let device = Default::default();
        let steps = 4;
        let prev = Tensor::<B, 3, Int>::zeros([1, steps, S], &device);
        let cursor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![2i32, 0, 4, 1], [steps, 1]),
            &device,
        );

        let path = Viterbi::<B>::backtrace(prev, cursor.clone(), 1);
        assert_eq!(path.dims(), [steps, 1, 1]);
        path.squeeze_dim::<2>(2)
            .to_data()
            .assert_eq(&cursor.to_data(), true);
    }

    #[test]
    #[should_panic(expected = "slots of history")]
    fn test_backtrace_rejects_short_history() {
        let device = Default::default();
        let prev = Tensor::<B, 3, Int>::zeros([1, 4, S], &device);
        let cursor = Tensor::<B, 2, Int>::zeros([4, 1], &device);
        Viterbi::<B>::backtrace(prev, cursor, 3);
    }

    #[test]
    fn test_forbidden_transitions_are_never_taken() {
        // Every route into state 0 is closed except from state 0 itself, so no
        // backpointer into 0 may name anything else -- however attractive the
        // emissions make it.
        let device = Default::default();
        let steps = 6;

        let mut trans = transition();
        for from in 1..S {
            trans[from * S] = FORBIDDEN;
        }

        let v: Viterbi<B> = ViterbiConfig::new(S).init(&trans, &device);
        let e =
            Tensor::<B, 3>::from_data(TensorData::new(emissions(steps, 1), [steps, 1, S]), &device);
        let (prev, _) = v.forward(e, v.init_state(1, &device));

        let got: Vec<i32> = prev.to_data_as::<i32>().to_vec_as::<i32>().ok_or_panic();
        for t in 0..steps {
            assert_eq!(
                got[t * S],
                0,
                "step {t}: state 0 reached from a closed edge"
            );
        }
    }

    #[test]
    #[should_panic(expected = "must be non-empty")]
    fn test_empty_input_is_rejected() {
        let device = Default::default();
        let v: Viterbi<B> = ViterbiConfig::new(S).init(&transition(), &device);
        let e = Tensor::<B, 3>::zeros([0, 1, S], &device);
        v.forward(e, v.init_state(1, &device));
    }

    #[test]
    fn test_random_sequences_match_the_scalar_reference() {
        // The deterministic fixtures above could conceivably flatter a shared
        // misreading of the recursion; random data cannot.
        let device = Default::default();
        let (steps, batch) = (9usize, 1usize);
        let trans = transition();
        let v: Viterbi<B> = ViterbiConfig::new(S).init(&trans, &device);

        let e = Tensor::<B, 3>::random([steps, batch, S], Distribution::Normal(0.0, 1.0), &device);
        let emit: Vec<f32> = e
            .clone()
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .ok_or_panic();

        let (prev, _) = v.forward(e, v.init_state(batch, &device));
        let (want_prev, _) = scalar_viterbi(&emit, &trans, steps);

        let got: Vec<i32> = prev.to_data_as::<i32>().to_vec_as::<i32>().ok_or_panic();
        assert_eq!(
            got.iter().map(|v| *v as usize).collect::<Vec<_>>(),
            want_prev,
        );
    }
}
