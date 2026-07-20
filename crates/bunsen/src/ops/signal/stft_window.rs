use burn::{
    Tensor,
    config::Config,
    prelude::Backend,
};

use crate::ops::arange::tensor_arange_start_step;

/// Analysis window function for [`SlidingStft`].
#[derive(Config, Debug, PartialEq)]
pub enum StftWindowConfig {
    /// All-ones (rectangular) window.
    ///
    /// The reference analyzer default when no coefficient table is provided.
    Ones,

    /// Periodic Hann window: `CosineWindow(0.5)`
    Hann,

    /// Periodic Hamming window: `CosineWindow(0.54)`
    Hamming,

    /// Cosine Window:
    /// `[ for i in range(n) | alpha - (1 - alpha) * cos(i * 2π / win_len) ]`
    CosineWindow(f64),
}

impl StftWindowConfig {
    /// If this is a cos window; Get the alpha param for the window:
    /// `[ for i in range(n) | alpha - (1 - alpha) * cos(i * 2π / win_len) ]`
    pub fn cos_alpha(&self) -> Option<f64> {
        Some(match self {
            Self::Ones => 1.0,
            Self::Hann => 0.5,
            Self::Hamming => 0.54,
            Self::CosineWindow(alpha) => *alpha,
        })
    }

    /// If this is the cos window; Get the alpha and beta (`1 - alpha`) params
    /// for the window: `[ for i in range(n) | alpha - (1 - alpha) * cos(i *
    /// 2π / win_len) ]`
    pub fn cos_alpha_beta(&self) -> Option<(f64, f64)> {
        self.cos_alpha().map(|alpha| (alpha, 1.0 - alpha))
    }

    /// The window coefficient table for a `win_len`-sample window.
    pub fn to_vec_window(
        &self,
        win_len: usize,
    ) -> Vec<f32> {
        match self {
            Self::Ones => vec![1.0; win_len],
            _ => {
                let (alpha, beta) = self.cos_alpha_beta().unwrap();
                let step = core::f64::consts::TAU / win_len as f64;

                // n * (2π / win_len)
                (0..win_len)
                    .map(|n| {
                        let theta = (n as f64) * step;

                        (alpha - beta * theta.cos()) as f32
                    })
                    .collect()
            }
        }
    }

    /// The window coefficient table for a `win_len`-sample window,
    /// materialized on `device`.
    pub fn to_tensor_window<B: Backend>(
        &self,
        win_len: usize,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        match self {
            Self::Ones => Tensor::ones([win_len], device),
            _ => {
                let (alpha, beta) = self.cos_alpha_beta().unwrap();
                let step = core::f64::consts::TAU / win_len as f64;

                // n * (2π / win_len)
                let theta = tensor_arange_start_step(win_len, 0.0, Some(step), device);

                theta.cos().mul_scalar(-beta).add_scalar(alpha)
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
    fn test_cos_alpha_beta() {
        assert_eq!(
            StftWindowConfig::CosineWindow(0.2).cos_alpha_beta(),
            Some((0.2, 0.8))
        );

        assert_eq!(StftWindowConfig::Ones.cos_alpha_beta(), Some((1.0, 0.0)));
        assert_eq!(StftWindowConfig::Hann.cos_alpha_beta(), Some((0.5, 0.5)));
        assert_eq!(
            StftWindowConfig::Hamming.cos_alpha_beta(),
            Some((0.54, 1.0 - 0.54))
        );
    }

    #[test]
    fn test_window_coefficients() {
        assert_eq!(StftWindowConfig::Ones.to_vec_window(4), vec![1.0; 4]);

        let hann = StftWindowConfig::Hann.to_vec_window(768);

        // Spot-check against the reference `AUP_AED_STFTWindow_Hann768`
        // table (`coeff.h`): 0.0, 1.6733041e-05, 6.6931045e-05, 1.5059065e-04.
        let expected = [0.0000000e+00f32, 1.6733041e-05, 6.693104e-05, 1.5059065e-04];
        for (n, e) in expected.into_iter().enumerate() {
            assert!((hann[n] - e).abs() <= 1e-8, "hann[{n}]: {} vs {e}", hann[n]);
        }

        // The periodic window satisfies w[n] == w[win_len - n].
        for n in 1..768 {
            assert!((hann[n] - hann[768 - n]).abs() <= 1e-7);
        }
    }

    #[test]
    fn test_hamming_coefficients() {
        let hamming = StftWindowConfig::Hamming.to_vec_window(768);

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
        let hamming8 = StftWindowConfig::Hamming.to_vec_window(8);
        assert!((hamming8[1] - 0.21473088).abs() <= 1e-7);
    }

    #[test]
    fn test_coefficients_tensor_matches_host() {
        let device = Default::default();
        for window in [
            StftWindowConfig::Ones,
            StftWindowConfig::Hann,
            StftWindowConfig::Hamming,
        ] {
            let host = window.to_vec_window(48);
            let tensor: Tensor<B, 1> = window.to_tensor_window(48, &device);
            tensor
                .to_data()
                .assert_approx_eq::<F>(&TensorData::new(host, [48]), Tolerance::permissive());
        }
    }
}
