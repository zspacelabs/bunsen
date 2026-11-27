//! # Feature levels: what a front end scales a window against.

use core::fmt::Debug;

use burn::Tensor;

/// A summary of feature level, over some frames, that merges across
/// arrivals.
///
/// A front end packages a window against one of these: Whisper's and
/// Qwen3-ASR's floor the log-mels at a range below a [`MaxLevel`], and
/// Parakeet's normalizes each mel band by a [`MomentsLevel`]. Which frames
/// the summary covers is the stream's
/// [`LevelPolicy`](crate::kits::speech::asr::driver::LevelPolicy); what the summary is,
/// and how it is applied, is the model's.
///
/// Frames are `[batch, frames, features]`, one independent stream per row:
/// a summary never mixes rows.
pub trait LevelStatistic: Clone + Debug + Send + Sync {
    /// The summary of `frames`.
    ///
    /// # Arguments
    /// * `frames` - `[batch, frames, features]`, at least one frame.
    fn of(frames: Tensor<3>) -> Self;

    /// The summary of the frames of both, in order: `self`'s, then
    /// `later`'s.
    fn merge(
        self,
        later: Self,
    ) -> Self;
}

/// No level: for an engine that packages every window against fixed
/// constants, so no statistic of the stream enters. Summarizing costs
/// nothing.
impl LevelStatistic for () {
    fn of(_frames: Tensor<3>) -> Self {}

    fn merge(
        self,
        _later: Self,
    ) -> Self {
    }
}

/// The maximum over each row's frames and features: `[batch]`.
#[derive(Clone, Debug)]
pub struct MaxLevel {
    /// `[batch]`.
    pub max: Tensor<1>,
}

impl LevelStatistic for MaxLevel {
    fn of(frames: Tensor<3>) -> Self {
        Self {
            max: frames.max_dims(&[1, 2]).reshape([-1]),
        }
    }

    fn merge(
        self,
        later: Self,
    ) -> Self {
        Self {
            max: self.max.max_pair(later.max),
        }
    }
}

/// The count, mean and sum of squared deviations of each row's frames, per
/// feature.
///
/// Merged with the pairwise update of Chan, Golub and Le Veque, so a summary
/// built arrival by arrival matches one over all the frames at once, to
/// rounding.
#[derive(Clone, Debug)]
pub struct MomentsLevel {
    /// The frames summarized; the same for every row.
    pub count: usize,

    /// `[batch, features]`.
    pub mean: Tensor<2>,

    /// `[batch, features]`: the sum of squared deviations from the mean.
    pub m2: Tensor<2>,
}

impl MomentsLevel {
    /// The sample variance, `m2 / (count - 1)`, with the divisor clamped to
    /// at least 1: `[batch, features]`.
    pub fn sample_variance(&self) -> Tensor<2> {
        self.m2
            .clone()
            .div_scalar(self.count.saturating_sub(1).max(1) as f64)
    }

    /// `frames` normalized per feature by these moments, as `NeMo`'s front
    /// end normalizes an item: `(x - mean) / (std + epsilon)`, with the
    /// unbiased standard deviation ([`sample_variance`](Self::sample_variance)).
    ///
    /// # Arguments
    /// * `frames` - `[batch, frames, features]`, the batch and features these
    ///   moments are of.
    ///
    /// # Returns
    /// `[batch, frames, features]`.
    pub fn normalize(
        &self,
        frames: Tensor<3>,
        epsilon: f64,
    ) -> Tensor<3> {
        let [batch, features] = crate::contracts::unpack_shape_contract!(
            ["batch", "features"],
            &self.mean.dims(),
            &["batch", "features"]
        );
        #[cfg(any(debug_assertions, test))]
        let [count] = crate::contracts::unpack_shape_contract!(
            ["batch", "count", "features"],
            &frames.dims(),
            &["count"],
            &[("batch", batch), ("features", features)]
        );
        let std = self.sample_variance().sqrt().add_scalar(epsilon);
        let normalized = (frames - self.mean.clone().unsqueeze_dim(1)) / std.unsqueeze_dim(1);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "count", "features"],
            &normalized.dims(),
            &[("batch", batch), ("count", count), ("features", features)]
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = (batch, features);
        normalized
    }
}

impl LevelStatistic for MomentsLevel {
    fn of(frames: Tensor<3>) -> Self {
        let [batch, count, features] = frames.dims();
        let mean = frames.clone().mean_dim(1);
        let m2 = (frames - mean.clone()).powi_scalar(2).sum_dim(1);
        Self {
            count,
            mean: mean.reshape([batch, features]),
            m2: m2.reshape([batch, features]),
        }
    }

    fn merge(
        self,
        later: Self,
    ) -> Self {
        let (na, nb) = (self.count as f64, later.count as f64);
        let n = na + nb;
        let delta = later.mean.clone() - self.mean.clone();
        Self {
            count: self.count + later.count,
            mean: self.mean + delta.clone().mul_scalar(nb / n),
            m2: self.m2 + later.m2 + delta.powi_scalar(2).mul_scalar(na * nb / n),
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::TensorData,
        tensor::{
            Distribution,
            Tolerance,
        },
    };

    use super::*;
    use crate::support::testing::{
        assert_tensor_close_to_vec,
        assert_tensors_close,
        cpu_device,
        seeded_tensor,
    };

    /// `[2, 2, 2]`: two rows, two frames, two features, from a flat list.
    fn frames(values: [f64; 8]) -> Tensor<3> {
        Tensor::from_data(TensorData::new(values.to_vec(), [2, 2, 2]), &cpu_device())
    }
    #[test]
    fn test_max_is_each_rows_own() {
        let level = MaxLevel::of(frames([0.0, -3.0, -1.0, -20.0, 5.0, 4.0, -9.0, 1.0]));
        assert_tensor_close_to_vec(&level.max, &[0.0, 5.0], Tolerance::default());

        // A later, louder frame in row 0 raises row 0 only.
        let later = MaxLevel::of(frames([7.0, -3.0, -1.0, -20.0, -5.0, -4.0, -9.0, -1.0]));
        assert_tensor_close_to_vec(&level.merge(later).max, &[7.0, 5.0], Tolerance::default());
    }

    #[test]
    fn test_moments_of_a_known_clip() {
        // Row 0, feature 0: frames 1 and 3; mean 2, m2 2, sample variance 2.
        let level = MomentsLevel::of(frames([1.0, 10.0, 3.0, 10.0, 0.0, 2.0, 4.0, 6.0]));
        assert_eq!(level.count, 2);
        assert_tensor_close_to_vec(&level.mean, &[2.0, 10.0, 2.0, 4.0], Tolerance::default());
        assert_tensor_close_to_vec(&level.m2, &[2.0, 0.0, 8.0, 8.0], Tolerance::default());
        assert_tensor_close_to_vec(
            &level.sample_variance(),
            &[2.0, 0.0, 8.0, 8.0],
            Tolerance::default(),
        );
    }

    /// Merging arrival by arrival is the summary of all the frames at once.
    #[test]
    fn test_merges_match_the_whole() {
        let device = cpu_device();
        let clip = seeded_tensor(7, [2, 23, 5], Distribution::Normal(1.0, 3.0), &device);
        let cuts = [0, 1, 8, 9, 17, 23];

        let parts = |range: core::ops::Range<usize>| clip.clone().slice_dim(1, range);
        let whole_max = MaxLevel::of(clip.clone());
        let whole_moments = MomentsLevel::of(clip.clone());

        let mut max: Option<MaxLevel> = None;
        let mut moments: Option<MomentsLevel> = None;
        for pair in cuts.windows(2) {
            let part = parts(pair[0]..pair[1]);
            let (m, s) = (MaxLevel::of(part.clone()), MomentsLevel::of(part));
            max = Some(match max {
                Some(max) => max.merge(m),
                None => m,
            });
            moments = Some(match moments {
                Some(moments) => moments.merge(s),
                None => s,
            });
        }
        let (max, moments) = (max.unwrap(), moments.unwrap());

        assert_tensors_close(&max.max, &whole_max.max, Tolerance::default());
        assert_eq!(moments.count, whole_moments.count);
        assert_tensors_close(&moments.mean, &whole_moments.mean, Tolerance::default());
        assert_tensors_close(
            &moments.m2,
            &whole_moments.m2,
            Tolerance::rel_abs(1e-4, 1e-4),
        );
    }
}
