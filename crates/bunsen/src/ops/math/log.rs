use burn::{
    Tensor,
    config::Config,
    prelude::Backend,
};

/// The logarithm applied during compression.
#[derive(Config, Copy, Debug, PartialEq)]
pub enum LogBase {
    /// `log10`. The Whisper / `librosa` default.
    Ten,

    /// Natural log, as used by Kaldi-flavoured frontends.
    E,

    /// Custom base.
    Custom {
        /// The natural logarithm of the base.
        base_ln: f64,
    },
}

impl LogBase {
    /// Custom base.
    pub fn custom(base: f64) -> Self {
        Self::Custom { base_ln: base.ln() }
    }

    /// The base of the logarithm.
    pub fn base(&self) -> f64 {
        match self {
            Self::Ten => 10.0,
            Self::E => core::f64::consts::E,
            Self::Custom { base_ln } => base_ln.exp(),
        }
    }

    /// The natural logarithm of the base.
    pub fn base_ln(&self) -> f64 {
        match self {
            Self::Ten => core::f64::consts::LN_10,
            Self::E => 1.0,
            Self::Custom { base_ln } => *base_ln,
        }
    }

    /// Applies the logarithm elementwise.
    ///
    /// `burn` exposes only the natural log, so base ten is `ln(x) / ln(10)`.
    pub fn apply<B: Backend, const D: usize>(
        &self,
        x: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match self {
            Self::E => x.log(),
            _ => x.log().div_scalar(self.base_ln()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::CpuBackend;

    type B = CpuBackend;

    #[test]
    fn test_log_e() {
        let log_base = LogBase::E;

        assert_eq!(log_base.base(), core::f64::consts::E);
        assert_eq!(log_base.base_ln(), 1.0);

        let device = Default::default();
        let x: Tensor<B, 1> = Tensor::from_floats([1.0, 10.0, 100.0], &device);
        let result = log_base.apply(x);
        let expected: Tensor<B, 1> =
            Tensor::from_floats([1.0_f64.ln(), 10.0_f64.ln(), 100.0_f64.ln()], &device);
        result
            .to_data()
            .assert_approx_eq::<f64>(&expected.to_data(), Default::default());
    }

    #[test]
    fn test_log_ten() {
        let log_base = LogBase::Ten;

        assert_eq!(log_base.base(), 10.0);
        assert_eq!(log_base.base_ln(), 10.0_f64.ln());

        let device = Default::default();
        let x: Tensor<B, 1> = Tensor::from_floats([1.0, 10.0, 100.0], &device);
        let result = log_base.apply(x);
        let expected: Tensor<B, 1> = Tensor::from_floats([0.0, 1.0, 2.0], &device);
        result.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn test_log_custom() {
        let log_base = LogBase::custom(2.0);

        assert_eq!(log_base.base(), 2.0);
        assert_eq!(log_base.base_ln(), 2.0_f64.ln());

        let device = Default::default();
        let x: Tensor<B, 1> = Tensor::from_floats([1.0, 10.0, 100.0], &device);
        let result = log_base.apply(x);

        let lc = |x: f64| x.ln() / 2.0_f64.ln();

        let expected: Tensor<B, 1> = Tensor::from_floats([lc(1.0), lc(10.0), lc(100.0)], &device);
        result
            .to_data()
            .assert_approx_eq::<f64>(&expected.to_data(), Default::default());
    }
}
