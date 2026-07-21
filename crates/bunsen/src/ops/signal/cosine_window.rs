use burn::{
    Tensor,
    config::Config,
    prelude::Backend,
    tensor::TensorCreationOptions,
};

use crate::ops::{
    arange::tensor_arange_start_step,
    signal::window_builder::SamplingWindowBuilder,
};

/// Cosine Window:
/// `[ for i in range(n) | alpha - beta * cos(i * 2π / win_len) ]`
#[derive(Config, Copy, Debug, PartialEq)]
pub struct CosineWindowConfig {
    /// The alpha param.
    pub alpha: f64,

    /// The beta param.
    pub beta: f64,

    /// Is this periodic?
    /// * `periodic`: `N = size`
    /// * `!periodic`: `N = size - 1`
    pub periodic: bool,
}

impl CosineWindowConfig {
    /// Construct a `CosineWindowConfig { alpha, beta: 1.0 - alpha }`.
    pub fn from_alpha_complement(
        alpha: f64,
        periodic: bool,
    ) -> CosineWindowConfig {
        let beta = 1.0 - alpha;
        CosineWindowConfig {
            alpha,
            beta,
            periodic,
        }
    }

    /// Construct a Hann Window: alpha = 0.5
    pub fn hann(periodic: bool) -> CosineWindowConfig {
        CosineWindowConfig::from_alpha_complement(0.5, periodic)
    }

    /// Construct a Hamming Window: alpha = 0.54
    pub fn hamming(periodic: bool) -> CosineWindowConfig {
        CosineWindowConfig::from_alpha_complement(0.54, periodic)
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
        let alpha = self.alpha;
        let beta = self.beta;

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
        options: impl Into<TensorCreationOptions<B>>,
    ) -> Tensor<B, 1> {
        let alpha = self.alpha;
        let beta = self.beta;

        match size {
            0 | 1 => return Tensor::ones([size], options),
            _ => (),
        };

        let n = if self.is_periodic() { size } else { size - 1 };
        let step = core::f64::consts::TAU / n as f64;

        // n * (2π / win_len)
        let theta = tensor_arange_start_step(size, 0.0, Some(step), options);

        theta.cos().mul_scalar(-beta).add_scalar(alpha)
    }
}

/// Cosine Window:
/// `[ for i in range(n)
///    | alpha - beta*cos(i * 2π/size) + gamma*cost(i * 4π/size) ]`
#[derive(Config, Copy, Debug, PartialEq)]
pub struct DualCosineWindow {
    /// The alpha param.
    pub alpha: f64,

    /// The beta param.
    pub beta: f64,

    /// The gamma param.
    pub gamma: f64,

    /// Is this periodic?
    /// * `periodic`: `N = size`
    /// * `!periodic`: `N = size - 1`
    pub periodic: bool,
}

impl DualCosineWindow {
    /// Construct a Blackman window.
    pub fn blackman(periodic: bool) -> DualCosineWindow {
        DualCosineWindow {
            alpha: 0.4,
            beta: 0.5,
            gamma: 0.08,
            periodic,
        }
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

impl SamplingWindowBuilder for DualCosineWindow {
    fn to_vec_window(
        &self,
        size: usize,
    ) -> Vec<f32> {
        let alpha = self.alpha;
        let beta = self.beta;
        let gamma = self.gamma;

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
                let b = beta * theta.cos();
                let g = gamma * (2.0 * theta).cos();

                (alpha - b + g) as f32
            })
            .collect()
    }

    fn to_tensor_window<B: Backend>(
        &self,
        size: usize,
        options: impl Into<TensorCreationOptions<B>>,
    ) -> Tensor<B, 1> {
        let alpha = self.alpha;
        let beta = self.beta;
        let gamma = self.gamma;

        match size {
            0 | 1 => return Tensor::ones([size], options),
            _ => (),
        };

        let n = if self.is_periodic() { size } else { size - 1 };
        let step = core::f64::consts::TAU / n as f64;

        // n * (2π / win_len)
        let theta = tensor_arange_start_step(size, 0.0, Some(step), options);

        let b = theta.clone().cos().mul_scalar(-beta);
        let g = theta.mul_scalar(2.0).cos().mul_scalar(gamma);
        (b + g).add_scalar(alpha)
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
        options: impl Into<TensorCreationOptions<B>>,
    ) {
        let options = options.into();
        assert_eq!(periodic.len(), symmetric.len());

        for (periodic, expected) in [(true, periodic), (false, symmetric)] {
            let cfg = CosineWindowConfig::hann(periodic);
            // println!("cfg: {cfg:?}");
            // println!("expected: {expected:?}");

            let size = expected.len();

            hann_window::<B>(size, periodic, options.clone())
                .to_data()
                .assert_approx_eq::<F>(&TensorData::from(expected), Tolerance::default());

            assert_close_to_vec(&cfg.to_vec_window(size), expected, 0.001);

            cfg.to_tensor_window::<B>(size, options.clone())
                .to_data()
                .assert_approx_eq::<F>(&TensorData::from(expected), Tolerance::default());
        }
    }

    #[test]
    fn test_hann_0() {
        let device = Default::default();
        check_hann_matches::<B>(&[], &[], &device);
    }

    #[test]
    fn test_hann_1() {
        let device = Default::default();
        check_hann_matches::<B>(&[1.0], &[1.0], &device);
    }

    #[test]
    fn test_hann_2() {
        let device = Default::default();
        check_hann_matches::<B>(&[0.0, 1.0], &[0.0, 0.0], &device);
    }

    #[test]
    fn test_hann_3() {
        let device = Default::default();
        check_hann_matches::<B>(&[0.0, 0.75, 0.75], &[0.0, 1.0, 0.0], &device);
    }

    #[test]
    fn test_hann_8() {
        let device = Default::default();
        check_hann_matches::<B>(
            &[0.0, 0.146447, 0.5, 0.853553, 1.0, 0.853553, 0.5, 0.146447],
            &[
                0.0, 0.188255, 0.611260, 0.950484, 0.950484, 0.611260, 0.188255, 0.0,
            ],
            &device,
        );
    }
}
