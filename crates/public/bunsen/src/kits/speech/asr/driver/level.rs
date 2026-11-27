//! # The level policy: what a window's features are scaled against.

use burn::Tensor;

use crate::kits::speech::support::LevelStatistic;

/// Which frames a window's [`LevelStatistic`] covers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LevelPolicy {
    /// The window's own frames. Right when a window is the whole clip, and
    /// what each model's reference does on one.
    #[default]
    PerWindow,

    /// Every frame the stream has seen so far, the window's included.
    ///
    /// Fed the whole clip before the first window is packaged, this is the
    /// whole-utterance summary; fed as audio arrives, it is the closest a
    /// live stream can come to it. In a stream opened per speech region, it
    /// is the per-region one.
    Running,
}

/// One stream's level: its [`LevelPolicy`] and, when running, the summary
/// of what it has seen.
///
/// [`observe`](Self::observe) takes `&mut self` and
/// [`reference`](Self::reference) takes `&self`, and the split is forced: a
/// provisional decode reads the stream through `&self`, so packaging a
/// window for a draft leaves the level as the commit will find it.
#[derive(Clone, Debug)]
pub struct LevelTracker<S: LevelStatistic> {
    policy: LevelPolicy,
    seen: Option<S>,
}

impl<S: LevelStatistic> LevelTracker<S> {
    /// A tracker under `policy` that has seen nothing.
    pub fn new(policy: LevelPolicy) -> Self {
        Self { policy, seen: None }
    }

    /// The policy.
    pub fn policy(&self) -> LevelPolicy {
        self.policy
    }

    /// The summary of everything observed, under
    /// [`Running`](LevelPolicy::Running); `None` before the first frame, and
    /// always under [`PerWindow`](LevelPolicy::PerWindow).
    pub fn seen(&self) -> Option<&S> {
        self.seen.as_ref()
    }

    /// Offers arriving frames: the arrival path, and the only place the
    /// tracker changes. An empty arrival changes nothing.
    ///
    /// # Arguments
    /// * `frames` - `[batch, frames, features]`, as they arrive.
    pub fn observe(
        &mut self,
        frames: Tensor<3>,
    ) {
        if self.policy == LevelPolicy::PerWindow || frames.dims()[1] == 0 {
            return;
        }
        let arriving = S::of(frames);
        self.seen = Some(match self.seen.take() {
            Some(seen) => seen.merge(arriving),
            None => arriving,
        });
    }

    /// The summary a window is packaged against.
    ///
    /// Under [`Running`](LevelPolicy::Running) a window's frames have been
    /// observed before it is packaged, so the summary of everything seen
    /// already covers them; with nothing seen, it is the window's own.
    ///
    /// # Arguments
    /// * `window` - `[batch, frames, features]`, at least one frame.
    pub fn reference(
        &self,
        window: Tensor<3>,
    ) -> S {
        match &self.seen {
            Some(seen) => seen.clone(),
            None => S::of(window),
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::TensorData,
        tensor::Tolerance,
    };

    use super::*;
    use crate::{
        kits::speech::support::MaxLevel,
        support::testing::{
            assert_tensor_close_to_vec,
            cpu_device,
        },
    };

    /// `[2, 2, 2]`: two rows, two frames, two features, from a flat list.
    fn frames(values: [f64; 8]) -> Tensor<3> {
        Tensor::from_data(TensorData::new(values.to_vec(), [2, 2, 2]), &cpu_device())
    }
    #[test]
    fn test_per_window_ignores_what_it_observes() {
        let mut tracker = LevelTracker::<MaxLevel>::new(LevelPolicy::PerWindow);
        tracker.observe(frames([99.0; 8]));
        assert!(tracker.seen().is_none());

        let window = frames([0.0, -3.0, -1.0, -20.0, 5.0, 4.0, -9.0, 1.0]);
        assert_tensor_close_to_vec(
            &tracker.reference(window).max,
            &[0.0, 5.0],
            Tolerance::default(),
        );
    }

    #[test]
    fn test_running_covers_everything_seen() {
        let mut tracker = LevelTracker::<MaxLevel>::new(LevelPolicy::Running);
        let quiet = frames([0.0, -3.0, -1.0, -20.0, 5.0, 4.0, -9.0, 1.0]);

        // Nothing seen: the window's own.
        assert_tensor_close_to_vec(
            &tracker.reference(quiet.clone()).max,
            &[0.0, 5.0],
            Tolerance::default(),
        );

        // An empty arrival changes nothing.
        tracker.observe(Tensor::zeros([2, 0, 2], &cpu_device()));
        assert!(tracker.seen().is_none());

        tracker.observe(frames([7.0, -3.0, -1.0, -20.0, -5.0, -4.0, -9.0, -1.0]));
        tracker.observe(quiet.clone());
        assert_tensor_close_to_vec(
            &tracker.reference(quiet).max,
            &[7.0, 5.0],
            Tolerance::default(),
        );
    }
}
