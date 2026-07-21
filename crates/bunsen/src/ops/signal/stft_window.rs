use burn::{
    Tensor,
    config::Config,
    prelude::Backend,
};

use crate::ops::signal::{
    cosine_window::CosineWindowConfig,
    window_builder::SamplingWindowBuilder,
};

/// Analysis window function for [`SlidingStft`].
#[derive(Config, Copy, Debug, PartialEq)]
pub enum StftWindowConfig {
    /// All-ones (rectangular) window.
    ///
    /// The reference analyzer default when no coefficient table is provided.
    Ones,

    /// Periodic Hann window: `CosineWindow { alpha: 0.5, periodic }`
    Hann {
        /// Is this a periodic or symmetric window?
        periodic: bool,
    },

    /// Periodic Hamming window: `CosineWindow { alpha: 0.54, periodic }`
    Hamming {
        /// Is this a periodic or symmetric window?
        periodic: bool,
    },

    /// Cosine Window:
    /// `[ for i in range(n) | alpha - (1 - alpha) * cos(i * 2π / win_len) ]`
    CosineWindow {
        /// What is the alpha parameter?
        alpha: f64,

        /// Is this a periodic or symmetric window?
        periodic: bool,
    },
}

impl StftWindowConfig {
    /// The window coefficient table for a `win_len`-sample window.
    pub fn to_vec_window(
        &self,
        size: usize,
    ) -> Vec<f32> {
        let cfg = match self {
            Self::Ones => return vec![1.0; size],
            Self::Hann { periodic } => CosineWindowConfig::hann(*periodic),
            Self::Hamming { periodic } => CosineWindowConfig::hamming(*periodic),
            Self::CosineWindow { alpha, periodic } => CosineWindowConfig::new(*alpha, *periodic),
        };
        cfg.to_vec_window(size)
    }

    /// The window coefficient table for a `win_len`-sample window,
    /// materialized on `device`.
    pub fn to_tensor_window<B: Backend>(
        &self,
        size: usize,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let cfg = match self {
            Self::Ones => return Tensor::ones([size], device),
            Self::Hann { periodic } => CosineWindowConfig::hann(*periodic),
            Self::Hamming { periodic } => CosineWindowConfig::hamming(*periodic),
            Self::CosineWindow { alpha, periodic } => CosineWindowConfig::new(*alpha, *periodic),
        };
        cfg.to_tensor_window(size, device)
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
