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
    ) -> Vec<f64> {
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

                self.alpha - self.beta * theta.cos()
            })
            .collect()
    }

    fn to_tensor_window<B: Backend>(
        &self,
        size: usize,
        options: impl Into<TensorCreationOptions<B>>,
    ) -> Tensor<B, 1> {
        match size {
            0 | 1 => return Tensor::ones([size], options),
            _ => (),
        };

        let n = if self.is_periodic() { size } else { size - 1 };
        let step = core::f64::consts::TAU / n as f64;

        // n * (2π / win_len)
        let theta = tensor_arange_start_step(size, 0.0, Some(step), options);

        theta.cos().mul_scalar(-self.beta).add_scalar(self.alpha)
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
    /// Construct: `DualCosineWindow { alpha, beta, gamma: 1.0 - (alpha + beta),
    /// periodic }`
    pub fn from_alpha_beta_complement(
        alpha: f64,
        beta: f64,
        periodic: bool,
    ) -> DualCosineWindow {
        let gamma = 1.0 - (alpha + beta);
        DualCosineWindow {
            alpha,
            beta,
            gamma,
            periodic,
        }
    }

    /// Construct a Blackman window.
    pub fn blackman(periodic: bool) -> DualCosineWindow {
        Self::from_alpha_beta_complement(0.42, 0.5, periodic)
    }

    /// Is this a periodic window?
    pub fn is_periodic(&self) -> bool {
        self.periodic
    }

    /// Is this a periodic window?
    pub fn is_symmetric(&self) -> bool {
        !self.periodic
    }

    /// Compute the angular increment.
    pub fn angular_increment(
        &self,
        size: usize,
    ) -> f64 {
        let n = if self.is_periodic() { size } else { size - 1 };
        core::f64::consts::TAU / n as f64
    }
}

impl SamplingWindowBuilder for DualCosineWindow {
    fn to_vec_window(
        &self,
        size: usize,
    ) -> Vec<f64> {
        match size {
            0 | 1 => return vec![1.0; size],
            _ => (),
        };
        let ang_inc = self.angular_increment(size);

        // n * (2π / win_len)
        (0..size)
            .map(|i| {
                let i = i as f64;
                // alpha - beta * cos(i * ai) + gamma * cos(i * 2 * ai)
                // => (alpha - gamma) - beta * cos(i * ai) + (2.0 * gamma) * (cos(i * ai)^2)
                let cos_val = (i * ang_inc).cos();
                (self.alpha - self.gamma) - self.beta * cos_val
                    + (2.0 * self.gamma) * cos_val.powi(2)
            })
            .collect()
    }

    fn to_tensor_window<B: Backend>(
        &self,
        size: usize,
        options: impl Into<TensorCreationOptions<B>>,
    ) -> Tensor<B, 1> {
        match size {
            0 | 1 => return Tensor::ones([size], options),
            _ => (),
        };
        let ang_inc = self.angular_increment(size);

        // alpha - beta * cos(i * ai) + gamma * cos(i * 2 * ai)
        // => (alpha - gamma) - beta * cos(i * ai) + (2.0 * gamma) * (cos(i * ai)^2)

        let cos_val = tensor_arange_start_step(size, 0.0, Some(ang_inc), options).cos();

        let b = cos_val.clone().mul_scalar(self.beta);
        let g = cos_val.square().mul_scalar(2.0 * self.gamma);

        (g - b).add_scalar(self.alpha - self.gamma)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::TensorData,
        tensor::{
            Tolerance,
            backend::BackendTypes,
            signal::{
                blackman_window,
                hann_window,
            },
        },
    };
    use tracing::{
        debug,
        info,
    };
    use tracing_test::traced_test;

    use super::*;
    use crate::{
        ops::signal::testing::assert_builder_impls_match,
        support::testing::CpuBackend,
    };

    type B = CpuBackend;
    type F = <B as BackendTypes>::FloatElem;

    fn check_hann_impl<B: Backend>(
        periodic: bool,
        expected: &[f64],
        options: impl Into<TensorCreationOptions<B>>,
    ) {
        let cfg = CosineWindowConfig::hann(periodic);
        debug!("cfg: {:?}", cfg);
        debug!("expected: {:?}", expected);

        let size = expected.len();
        let options = options.into();

        info!("checking hann_window reference implementation");
        hann_window::<B>(size, periodic, options.clone())
            .to_data()
            .assert_approx_eq::<F>(&TensorData::from(expected), Tolerance::default());

        info!("cross-checking vec/tensor impls");
        assert_builder_impls_match::<B>(&cfg, expected, options.clone());
    }

    #[test]
    #[traced_test]
    fn test_hann() {
        let device = Default::default();

        // size = 0
        check_hann_impl::<B>(true, &[], &device);
        check_hann_impl::<B>(false, &[], &device);

        // size = 1
        check_hann_impl::<B>(true, &[1.0], &device);
        check_hann_impl::<B>(false, &[1.0], &device);

        // size = 2
        check_hann_impl::<B>(true, &[0.0, 1.0], &device);
        check_hann_impl::<B>(false, &[0.0, 0.0], &device);

        // size = 3
        check_hann_impl::<B>(true, &[0.0, 0.75, 0.75], &device);
        check_hann_impl::<B>(false, &[0.0, 1.0, 0.0], &device);

        // size = 8
        check_hann_impl::<B>(
            true,
            &[0.0, 0.146447, 0.5, 0.853553, 1.0, 0.853553, 0.5, 0.146447],
            &device,
        );
        check_hann_impl::<B>(
            false,
            &[
                0.0, 0.188255, 0.611260, 0.950484, 0.950484, 0.611260, 0.188255, 0.0,
            ],
            &device,
        );
    }

    fn check_blackman_impl<B: Backend>(
        periodic: bool,
        expected: &[f64],
        options: impl Into<TensorCreationOptions<B>>,
    ) {
        let cfg = DualCosineWindow::blackman(periodic);
        debug!("cfg: {:?}", cfg);
        debug!("expected: {:?}", expected);

        let size = expected.len();
        let options = options.into();

        info!("checking blackman_window reference implementation");
        blackman_window::<B>(size, periodic, options.clone())
            .to_data()
            .assert_approx_eq::<F>(&TensorData::from(expected), Tolerance::default());

        info!("cross-checking vec/tensor impls");
        assert_builder_impls_match::<B>(&cfg, expected, options.clone());
    }

    #[test]
    #[traced_test]
    fn test_blackman() {
        let device = Default::default();

        // size = 0
        check_blackman_impl::<B>(true, &[], &device);
        check_blackman_impl::<B>(false, &[], &device);

        // size = 1
        check_blackman_impl::<B>(true, &[1.0], &device);
        check_blackman_impl::<B>(false, &[1.0], &device);

        // size = 2
        check_blackman_impl::<B>(true, &[0.0, 1.0], &device);
        check_blackman_impl::<B>(false, &[0.0, 0.0], &device);

        // size = 3
        check_blackman_impl::<B>(true, &[0.0, 0.63, 0.63], &device);
        check_blackman_impl::<B>(false, &[0.0, 1.0, 0.0], &device);

        // size = 8
        check_blackman_impl::<B>(
            true,
            &[0.0, 0.0664466, 0.34, 0.77355, 1.0, 0.77355, 0.34, 0.0664466],
            &device,
        );
        check_blackman_impl::<B>(
            false,
            &[
                0.0, 0.09045343, 0.45918, 0.92036, 0.92036, 0.45918, 0.09045343, 0.0,
            ],
            &device,
        );
    }
}
