//! The TDT joint network.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
    },
    tensor::{
        Device,
        activation::relu,
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Common meta for [`TdtJoint`] and [`TdtJointConfig`].
pub trait TdtJointMeta {
    /// The joint's input width: the prediction network's.
    fn d_joint(&self) -> usize;

    /// The token logits: the vocabulary and the blank.
    fn n_vocab(&self) -> usize;

    /// The duration logits.
    fn n_durations(&self) -> usize;
}

/// Config for [`TdtJoint`].
///
/// Implements [`TdtJointMeta`].
#[derive(Config, Debug)]
pub struct TdtJointConfig {
    /// The joint's input width.
    pub d_joint: usize,

    /// The token logits: the vocabulary and the blank.
    pub n_vocab: usize,

    /// The duration logits.
    pub n_durations: usize,
}

impl TdtJointMeta for TdtJointConfig {
    fn d_joint(&self) -> usize {
        self.d_joint
    }

    fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    fn n_durations(&self) -> usize {
        self.n_durations
    }
}

impl ModuleInit<TdtJoint> for TdtJointConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<TdtJoint> {
        Ok(TdtJoint {
            head: LinearConfig::new(self.d_joint, self.n_vocab + self.n_durations).init(device),
            n_durations: self.n_durations,
        })
    }
}

/// The joint network of a token-and-duration transducer: `ReLU` of the sum of
/// a projected encoder frame and a prediction-network output, then one
/// projection to token logits followed by duration logits.
///
/// Built by [`TdtJointConfig`].
/// Implements [`TdtJointMeta`].
#[derive(Module, Debug)]
pub struct TdtJoint {
    /// `d_joint -> n_vocab + n_durations`.
    pub head: Linear,

    /// The duration logits at the end of the head's output.
    pub n_durations: usize,
}

impl TdtJointMeta for TdtJoint {
    fn d_joint(&self) -> usize {
        self.head.weight.dims()[0]
    }

    fn n_vocab(&self) -> usize {
        self.head.weight.dims()[1] - self.n_durations
    }

    fn n_durations(&self) -> usize {
        self.n_durations
    }
}

impl TdtJoint {
    /// Scores encoder frames against prediction outputs.
    ///
    /// The two inputs broadcast against each other, so one call scores a
    /// single pair, a frame against many outputs, or the full lattice.
    ///
    /// # Arguments
    /// * `encoder` - `[..., d_joint]`, projected encoder frames.
    /// * `prediction` - `[..., d_joint]`, prediction-network outputs.
    ///
    /// # Returns
    /// `[..., n_vocab + n_durations]`: token logits, then duration logits.
    pub fn forward<const D: usize>(
        &self,
        encoder: Tensor<D>,
        prediction: Tensor<D>,
    ) -> Tensor<D> {
        self.head.forward(relu(encoder + prediction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_meta() {
        let config = TdtJointConfig::new(6, 11, 5);
        let module: TdtJoint = config.init(&cpu_device());
        assert_eq!(module.d_joint(), config.d_joint());
        assert_eq!(module.n_vocab(), config.n_vocab());
        assert_eq!(module.n_durations(), config.n_durations());

        let device = cpu_device();
        let y = module.forward(
            Tensor::<4>::zeros([2, 7, 1, 6], &device),
            Tensor::<4>::zeros([2, 1, 4, 6], &device),
        );
        assert_eq!(y.dims(), [2, 7, 4, 16]);
    }
}
