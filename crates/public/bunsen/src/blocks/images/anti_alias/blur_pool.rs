//! [`BlurPool2d`]: a binomial blur, then a stride.

use alloc::vec::Vec;

use burn::{
    config::Config,
    module::Module,
    prelude::Tensor,
    tensor::{
        TensorData,
        module::conv2d,
        ops::{
            ConvOptions,
            PadMode,
        },
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// The view shared by [`BlurPool2dConfig`] and [`BlurPool2d`].
pub trait BlurPool2dMeta {
    /// The blur's side: 3 or 5 in timm's nets.
    fn filt_size(&self) -> usize;

    /// The stride after the blur.
    fn stride(&self) -> usize;

    /// The reflect padding on each side: `((stride - 1) + (filt_size - 1)) /
    /// 2`, timm's `get_padding`.
    fn padding(&self) -> usize {
        (self.stride() - 1 + (self.filt_size() - 1)) / 2
    }
}

/// The config of a [`BlurPool2d`]; timm's `BlurPool2d` defaults.
#[derive(Config, Debug)]
pub struct BlurPool2dConfig {
    /// The blur's side, at least 2.
    #[config(default = 3)]
    pub filt_size: usize,

    /// The stride after the blur.
    #[config(default = 2)]
    pub stride: usize,
}

impl BlurPool2dMeta for BlurPool2dConfig {
    fn filt_size(&self) -> usize {
        self.filt_size
    }

    fn stride(&self) -> usize {
        self.stride
    }
}

/// The pool; it has no parameters, so `device` is unused.
///
/// # Errors
///
/// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a side under 2
/// or a zero stride.
impl ModuleInit<BlurPool2d> for BlurPool2dConfig {
    fn try_init(
        &self,
        _device: &burn::tensor::Device,
    ) -> BunsenResult<BlurPool2d> {
        if self.filt_size < 2 || self.stride == 0 {
            return Err(BunsenError::illegal(format!(
                "BlurPool2dConfig: filt_size {} and stride {}; the side is at least 2, the stride \
                 at least 1",
                self.filt_size, self.stride
            )));
        }
        Ok(BlurPool2d {
            filt_size: self.filt_size,
            stride: self.stride,
        })
    }
}

/// A binomial blur of each channel, then a stride: timm's `BlurPool2d` with
/// its default reflect padding. It has no parameters.
#[derive(Module, Debug)]
pub struct BlurPool2d {
    /// The blur's side.
    pub filt_size: usize,

    /// The stride after the blur.
    pub stride: usize,
}

impl BlurPool2dMeta for BlurPool2d {
    fn filt_size(&self) -> usize {
        self.filt_size
    }

    fn stride(&self) -> usize {
        self.stride
    }
}

impl BlurPool2d {
    /// The blur's taps: `C(n - 1, k) / 2^(n - 1)` for `k` in `0..n`, which
    /// sum to one.
    pub fn taps(&self) -> Vec<f64> {
        let n = self.filt_size;
        let mut row = alloc::vec![1u64; n];
        for k in 1..n.saturating_sub(1) {
            row[k] = row[k - 1] * (n - k) as u64 / k as u64;
        }
        let total = (1u64 << (n - 1)) as f64;
        row.into_iter().map(|c| c as f64 / total).collect()
    }

    /// The input, blurred and strided.
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, channels, height, width]`, each side above the
    ///   padding (reflect padding needs it).
    ///
    /// # Returns
    ///
    /// `[batch, channels, out_height, out_width]`.
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        let [batch, channels] = crate::contracts::unpack_shape_contract!(
            ["batch", "channels", _, _],
            &input.dims(),
            &["batch", "channels"]
        );
        let n = self.filt_size;
        let taps = self.taps();
        let filter: Vec<f32> = taps
            .iter()
            .flat_map(|a| taps.iter().map(move |b| (a * b) as f32))
            .collect();
        let weight = Tensor::<1>::from_data(
            TensorData::new(filter, [n * n]),
            (&input.device(), input.dtype()),
        )
        .reshape([1, 1, n, n])
        .expand([channels, 1, n, n]);
        let pad = self.padding();
        let padded = input.pad([(pad, pad), (pad, pad)], PadMode::Reflect);
        let y = conv2d(
            padded,
            weight,
            None,
            ConvOptions::new([self.stride, self.stride], [0, 0], [1, 1], channels),
        );
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "channels", _, _],
            &y.dims(),
            &[("batch", batch), ("channels", channels)]
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = batch;
        y
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;
    use crate::support::testing::cpu_device;

    /// The binomial taps.
    #[test]
    fn test_taps() {
        let device = cpu_device();
        assert_eq!(
            BlurPool2dConfig::new().init(&device).taps(),
            [0.25, 0.5, 0.25]
        );
        assert_eq!(
            BlurPool2dConfig::new()
                .with_filt_size(5)
                .init(&device)
                .taps(),
            [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0]
        );
    }

    /// A constant map stays constant (the taps sum to one, the padding
    /// reflects); the stride halves each side, rounding up.
    #[test]
    fn test_forward_keeps_a_constant_and_strides() {
        let device = cpu_device();
        let pool = BlurPool2dConfig::new().init(&device);
        let x = Tensor::<4>::ones([2, 3, 7, 6], &device) * 3.0;
        let y = pool.forward(x);
        assert_eq!(y.dims(), [2, 3, 4, 3]);
        y.into_data().assert_approx_eq::<f32>(
            &Tensor::<4>::full([2, 3, 4, 3], 3.0, &device).into_data(),
            Tolerance::absolute(1e-6),
        );
    }

    /// An impulse spreads as the outer product of the taps.
    #[test]
    fn test_impulse_response() {
        let device = cpu_device();
        let pool = BlurPool2dConfig::new().with_stride(1).init(&device);
        let mut values = alloc::vec![0f32; 25];
        values[12] = 16.0;
        let x = Tensor::<4>::from_data(TensorData::new(values, [1, 1, 5, 5]), &device);
        let y = pool.forward(x).narrow(2, 1, 3).narrow(3, 1, 3);
        y.into_data().assert_approx_eq::<f32>(
            &TensorData::new(
                alloc::vec![1f32, 2., 1., 2., 4., 2., 1., 2., 1.],
                [1, 1, 3, 3],
            ),
            Tolerance::absolute(1e-6),
        );
    }
}
