use burn::{
    Tensor,
    config::Config,
    prelude::Backend,
    tensor::TensorCreationOptions,
};

use crate::ops::signal::{
    DualCosineWindow,
    cosine_window::CosineWindowConfig,
    window_builder::SamplingWindowBuilder,
};

/// Analysis window function for [`SlidingStft`].
#[derive(Config, Copy, Debug, PartialEq)]
pub enum StftWindowConfig {
    /// All Ones (rectangular) window.
    /// The reference analyzer default when no coefficient table is provided.
    Ones,

    /// Cosine Window:
    /// `[ for i in range(n) | alpha - beta * cos(i * 2π / win_len) ]`
    CosineWindow(CosineWindowConfig),

    /// Hann window:
    /// `CosineWindow { alpha: 0.5, beta: 0.5, periodic }`
    Hann {
        /// Is this a periodic or symmetric window?
        periodic: bool,
    },

    /// Hamming window:
    /// `CosineWindow { alpha: 0.54, beta: 0.46, periodic }`
    Hamming {
        /// Is this a periodic or symmetric window?
        periodic: bool,
    },

    /// Dual Cosine Window:
    /// `[ for i in range(n)
    ///    | alpha - beta*cos(i * 2π/win_len) + gamma*cos(i * 4π/win_len) ]`
    DualCosineWindow(DualCosineWindow),

    /// Blackman Window:
    /// `DualCosineWindow { alpha: 0.42, beta: 0.5, gamma: 0.08, periodic } `
    Blackman {
        /// Is this a periodic or symmetric window?
        periodic: bool,
    },
}

impl SamplingWindowBuilder for StftWindowConfig {
    fn to_vec_window(
        &self,
        size: usize,
    ) -> Vec<f64> {
        match self {
            Self::Ones => vec![1.0; size],
            Self::CosineWindow(cfg) => cfg.to_vec_window(size),
            Self::Hann { periodic } => CosineWindowConfig::hann(*periodic).to_vec_window(size),
            Self::Hamming { periodic } => {
                CosineWindowConfig::hamming(*periodic).to_vec_window(size)
            }
            Self::DualCosineWindow(cfg) => cfg.to_vec_window(size),
            Self::Blackman { periodic } => {
                DualCosineWindow::blackman(*periodic).to_vec_window(size)
            }
        }
    }

    fn to_tensor_window<B: Backend>(
        &self,
        size: usize,
        options: impl Into<TensorCreationOptions<B>>,
    ) -> Tensor<B, 1> {
        match self {
            Self::Ones => Tensor::ones([size], options),
            Self::CosineWindow(cfg) => cfg.to_tensor_window(size, options),
            Self::Hann { periodic } => {
                CosineWindowConfig::hann(*periodic).to_tensor_window(size, options)
            }
            Self::Hamming { periodic } => {
                CosineWindowConfig::hamming(*periodic).to_tensor_window(size, options)
            }
            Self::DualCosineWindow(cfg) => cfg.to_tensor_window(size, options),
            Self::Blackman { periodic } => {
                DualCosineWindow::blackman(*periodic).to_tensor_window(size, options)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        prelude::TensorData,
        tensor::{
            Tolerance,
            backend::BackendTypes,
        },
    };

    use super::*;
    use crate::support::testing::CpuBackend;

    type B = CpuBackend;
    type F = <B as BackendTypes>::FloatElem;

    #[test]
    fn test_hamming_coefficients() {
        let periodic = true;

        let cfg = StftWindowConfig::Hamming { periodic };
        let hamming = cfg.to_vec_window(768);

        // Periodic Hamming anchors: `0.54 - 0.46` at n = 0, and the peak
        // `0.54 + 0.46` at the (even) midpoint.
        assert!((hamming[0] - 0.08).abs() <= 1e-7);
        assert!((hamming[384] - 1.0).abs() <= 1e-7);

        // The periodic window satisfies w[n] == w[win_len - n].
        for n in 1..768 {
            assert!((hamming[n] - hamming[768 - n]).abs() <= 1e-7);
        }

        // Anchor against the standard table value at win_len = 8, n = 1:
        // 0.54 - 0.46 * cos(π / 4).
        let hamming8 = cfg.to_vec_window(8);
        assert!((hamming8[1] - 0.21473088).abs() <= 1e-7);
    }

    #[test]
    fn test_coefficients_tensor_matches_host() {
        let device = Default::default();
        for periodic in [true, false] {
            for window in [
                StftWindowConfig::Ones,
                StftWindowConfig::Hann { periodic },
                StftWindowConfig::Hamming { periodic },
            ] {
                let host = window.to_vec_window(48);
                let tensor: Tensor<B, 1> = window.to_tensor_window(48, &device);
                tensor
                    .to_data()
                    .assert_approx_eq::<F>(&TensorData::new(host, [48]), Tolerance::permissive());
            }
        }
    }
}
