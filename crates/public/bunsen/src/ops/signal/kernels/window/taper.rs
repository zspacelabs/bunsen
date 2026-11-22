//! [`WindowTaper`]: a window as a function on `[-1, 1]`.

use core::f64::consts::PI;

use burn::Tensor;
// `std` has these float methods inherently; `core` has them through `Float`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use super::{
    CosineWindowConfig,
    DualCosineWindow,
    StftWindowConfig,
};

/// A window as a function on `[-1, 1]`, edge to edge, its centre at `0`;
/// zero outside.
///
/// An `N`-point table ([`SamplingWindowBuilder`](super::SamplingWindowBuilder))
/// is a sampling of it: a periodic table at `x_i = 2i/N - 1`, a symmetric one
/// at `x_i = 2i/(N - 1) - 1`; whether a table is periodic does not change the
/// taper. A windowed-sinc kernel reads it at fractional positions, which no
/// table holds.
///
/// [`taper`](Self::taper) is the host's (`f64`, the reference) and
/// [`taper_tensor`](Self::taper_tensor) a device's, element-wise; the two are
/// cross-tested here.
pub trait WindowTaper {
    /// The window at `x`; zero for `|x| > 1`.
    fn taper(
        &self,
        x: f64,
    ) -> f64;

    /// The window at each of `x`'s values, on its device; zero where `|x| >
    /// 1`.
    fn taper_tensor<const D: usize>(
        &self,
        x: Tensor<D>,
    ) -> Tensor<D>;
}

/// `y`, zero where `|x| > 1`.
fn within_unit<const D: usize>(
    x: Tensor<D>,
    y: Tensor<D>,
) -> Tensor<D> {
    y.mask_fill(x.abs().greater_elem(1.0), 0.0)
}

impl WindowTaper for CosineWindowConfig {
    /// `alpha - beta·cos(2πi/N)` at `x = 2i/N - 1` is `alpha + beta·cos(πx)`,
    /// computed as `(alpha - beta) + 2·beta·cos²(πx/2)`: Hann's product form,
    /// exactly (`cos²(πx/2)`, as `torchaudio` computes it), zero at the edges
    /// only to `cos(π/2)`'s rounding.
    fn taper(
        &self,
        x: f64,
    ) -> f64 {
        if x.abs() > 1.0 {
            return 0.0;
        }
        (self.alpha - self.beta) + 2.0 * self.beta * (x * PI / 2.0).cos().powi(2)
    }

    fn taper_tensor<const D: usize>(
        &self,
        x: Tensor<D>,
    ) -> Tensor<D> {
        let y = x
            .clone()
            .mul_scalar(PI / 2.0)
            .cos()
            .square()
            .mul_scalar(2.0 * self.beta)
            .add_scalar(self.alpha - self.beta);
        within_unit(x, y)
    }
}

impl WindowTaper for DualCosineWindow {
    /// `alpha - beta·cos(2πi/N) + gamma·cos(4πi/N)` at `x = 2i/N - 1` is
    /// `alpha + beta·cos(πx) + gamma·cos(2πx)`.
    fn taper(
        &self,
        x: f64,
    ) -> f64 {
        if x.abs() > 1.0 {
            return 0.0;
        }
        self.alpha + self.beta * (PI * x).cos() + self.gamma * (2.0 * PI * x).cos()
    }

    fn taper_tensor<const D: usize>(
        &self,
        x: Tensor<D>,
    ) -> Tensor<D> {
        let one = x.clone().mul_scalar(PI).cos().mul_scalar(self.beta);
        let two = x.clone().mul_scalar(2.0 * PI).cos().mul_scalar(self.gamma);
        within_unit(x, one.add(two).add_scalar(self.alpha))
    }
}

impl WindowTaper for StftWindowConfig {
    fn taper(
        &self,
        x: f64,
    ) -> f64 {
        match self {
            Self::Ones => {
                if x.abs() > 1.0 {
                    0.0
                } else {
                    1.0
                }
            }
            Self::CosineWindow(cfg) => cfg.taper(x),
            Self::Hann { periodic } => CosineWindowConfig::hann(*periodic).taper(x),
            Self::Hamming { periodic } => CosineWindowConfig::hamming(*periodic).taper(x),
            Self::DualCosineWindow(cfg) => cfg.taper(x),
            Self::Blackman { periodic } => DualCosineWindow::blackman(*periodic).taper(x),
        }
    }

    fn taper_tensor<const D: usize>(
        &self,
        x: Tensor<D>,
    ) -> Tensor<D> {
        match self {
            Self::Ones => within_unit(x.clone(), x.ones_like()),
            Self::CosineWindow(cfg) => cfg.taper_tensor(x),
            Self::Hann { periodic } => CosineWindowConfig::hann(*periodic).taper_tensor(x),
            Self::Hamming { periodic } => CosineWindowConfig::hamming(*periodic).taper_tensor(x),
            Self::DualCosineWindow(cfg) => cfg.taper_tensor(x),
            Self::Blackman { periodic } => DualCosineWindow::blackman(*periodic).taper_tensor(x),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use burn::tensor::Tolerance;

    use super::*;
    use crate::{
        ops::signal::kernels::window::SamplingWindowBuilder,
        support::testing::{
            assert_close_to_vec,
            assert_tensor_close_to_vec,
            performance_device,
        },
    };

    /// The windows a table and a taper are held equal for.
    fn windows() -> Vec<StftWindowConfig> {
        let mut all = Vec::new();
        for periodic in [true, false] {
            all.extend([
                StftWindowConfig::Ones,
                StftWindowConfig::Hann { periodic },
                StftWindowConfig::Hamming { periodic },
                StftWindowConfig::Blackman { periodic },
                StftWindowConfig::CosineWindow(CosineWindowConfig {
                    alpha: 0.6,
                    beta: 0.3,
                    periodic,
                }),
            ]);
        }
        all
    }

    /// Each table is its taper sampled: periodic at `2i/N - 1`, symmetric at
    /// `2i/(N - 1) - 1`.
    #[test]
    fn test_tables_sample_the_taper() {
        for window in windows() {
            let periodic = match window {
                StftWindowConfig::Ones => true,
                StftWindowConfig::Hann { periodic }
                | StftWindowConfig::Hamming { periodic }
                | StftWindowConfig::Blackman { periodic } => periodic,
                StftWindowConfig::CosineWindow(cfg) => cfg.periodic,
                StftWindowConfig::DualCosineWindow(cfg) => cfg.periodic,
            };
            for size in [2, 5, 16, 400] {
                let n = if periodic { size } else { size - 1 } as f64;
                let sampled: Vec<f64> = (0..size)
                    .map(|i| window.taper(2.0 * i as f64 / n - 1.0))
                    .collect();
                assert_close_to_vec(&window.to_vec_window(size), &sampled, 1e-14);
            }
        }
    }

    /// Zero outside `[-1, 1]`; Hann's product form is `cos²(πx/2)` itself,
    /// and Blackman at its centre is 1.
    #[test]
    fn test_taper_shape() {
        for window in windows() {
            assert_eq!(window.taper(1.5), 0.0);
            assert_eq!(window.taper(-1.000_001), 0.0);
        }
        let hann = CosineWindowConfig::hann(true);
        for x in [-0.7, -0.1, 0.0, 0.3, 0.9] {
            assert_eq!(hann.taper(x), (x * PI / 2.0).cos().powi(2));
        }
        assert!((DualCosineWindow::blackman(false).taper(0.0) - 1.0).abs() < 1e-15);
    }

    /// A device's taper is the host's, element-wise, inside and outside
    /// `[-1, 1]`.
    #[test]
    fn test_tensor_taper_is_the_host_taper() {
        let device = performance_device();
        let xs: Vec<f64> = (-130..=130).map(|i| i as f64 / 100.0).collect();
        for window in windows() {
            let x = Tensor::<1>::from_data(xs.as_slice(), &device);
            let expected: Vec<f64> = xs.iter().map(|&x| window.taper(x)).collect();
            assert_tensor_close_to_vec(
                &window.taper_tensor(x),
                &expected,
                Tolerance::rel_abs(1e-5, 1e-6),
            );
        }
    }
}
