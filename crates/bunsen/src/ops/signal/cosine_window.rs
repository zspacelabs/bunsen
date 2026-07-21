use burn::{
    Tensor,
    config::Config,
    prelude::Backend,
};

use crate::ops::{
    arange::tensor_arange_start_step,
    signal::window_builder::SamplingWindowBuilder,
};

/// Cosine Window:
/// `[ for i in range(n) | alpha - (1 - alpha) * cos(i * 2π / win_len) ]`
#[derive(Config, Copy, Debug, PartialEq)]
pub struct CosineWindowConfig {
    /// The alpha param.
    pub alpha: f64,

    /// Is this periodic?
    /// * `periodic`: `N = size`
    /// * `!periodic`: `N = size - 1`
    pub periodic: bool,
}

impl CosineWindowConfig {
    /// Construct a Hann Window: alpha = 0.5
    pub fn hann(periodic: bool) -> CosineWindowConfig {
        CosineWindowConfig::new(0.5, periodic)
    }

    /// Construct a Hamming Window: alpha = 0.54
    pub fn hamming(periodic: bool) -> CosineWindowConfig {
        CosineWindowConfig::new(0.54, periodic)
    }

    /// Get the alpha coeff.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the beta coeff (1.0 - alpha).
    pub fn beta(&self) -> f64 {
        1.0 - self.alpha
    }

    /// Is this a periodic window?
    pub fn is_periodic(&self) -> bool {
        self.periodic
    }

    /// Is this a periodic window?
    pub fn is_symmetric(&self) -> bool {
        !self.is_periodic()
    }
}

impl SamplingWindowBuilder for CosineWindowConfig {
    fn to_vec_window(
        &self,
        size: usize,
    ) -> Vec<f32> {
        let alpha = self.alpha();
        let beta = self.beta();

        match size {
            0 | 1 => return vec![1.0; size],
            _ => (),
        };

        let n = if self.is_periodic() { size } else { size - 1 };
        let step = core::f64::consts::TAU / n as f64;

        // n * (2π / win_len)
        (0..size)
            .map(|n| {
                let theta = (n as f64) * step;

                (alpha - beta * theta.cos()) as f32
            })
            .collect()
    }

    fn to_tensor_window<B: Backend>(
        &self,
        size: usize,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let alpha = self.alpha();
        let beta = self.beta();

        match size {
            0 | 1 => return Tensor::ones([size], device),
            _ => (),
        };

        let n = if self.is_periodic() { size } else { size - 1 };
        let step = core::f64::consts::TAU / n as f64;

        // n * (2π / win_len)
        let theta = tensor_arange_start_step(size, 0.0, Some(step), device);

        theta.cos().mul_scalar(-beta).add_scalar(alpha)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::TensorData,
        tensor::{
            Tolerance,
            backend::BackendTypes,
            signal::hann_window,
        },
    };

    use super::*;
    use crate::support::testing::{
        CpuBackend,
        assert_close_to_vec,
    };

    type B = CpuBackend;
    type F = <B as BackendTypes>::FloatElem;

    fn check_hann_matches<B: Backend>(
        periodic: &[f32],
        symmetric: &[f32],
        device: &B::Device,
    ) {
        assert_eq!(periodic.len(), symmetric.len());

        for (periodic, expected) in [(true, periodic), (false, symmetric)] {
            let cfg = CosineWindowConfig::hann(periodic);
            // println!("cfg: {cfg:?}");
            // println!("expected: {expected:?}");

            let size = expected.len();

            hann_window::<B>(size, periodic, device)
                .to_data()
                .assert_approx_eq::<F>(&TensorData::from(expected), Tolerance::default());

            assert_close_to_vec(&cfg.to_vec_window(size), expected, 0.001);

            cfg.to_tensor_window::<B>(size, &device)
                .to_data()
                .assert_approx_eq::<F>(&TensorData::from(expected), Tolerance::default());
        }
    }

    #[test]
    fn test_hann() {
        let device = Default::default();

        check_hann_matches::<B>(
            &[0.0, 0.146447, 0.5, 0.853553, 1.0, 0.853553, 0.5, 0.146447],
            &[
                0.0, 0.188255, 0.611260, 0.950484, 0.950484, 0.611260, 0.188255, 0.0,
            ],
            &device,
        );
    }

    #[test]
    fn test_empty() {
        let device = Default::default();
        check_hann_matches::<B>(&[], &[], &device);
    }

    #[test]
    fn test_size_1() {
        let device = Default::default();
        check_hann_matches::<B>(&[1.0], &[1.0], &device);
    }

    #[test]
    fn test_size_2() {
        let device = Default::default();

        check_hann_matches::<B>(&[0.0, 1.0], &[0.0, 0.0], &device);
    }

    #[test]
    fn test_size_3() {
        let device = Default::default();

        check_hann_matches::<B>(&[0.0, 0.75, 0.75], &[0.0, 1.0, 0.0], &device);
    }
}
