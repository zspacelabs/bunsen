use burn::{
    config::Config,
    module::Module,
    nn::pool::{
        AvgPool2d,
        AvgPool2dConfig,
    },
    prelude::{
        Backend,
        Tensor,
    },
};

use crate::ops::conv::pad_same;

// `AvgPool2dSame`
//
// # Reference Python
//
// ```python,ignore
// def avg_pool2d_same(
//         x: torch.Tensor,
//         kernel_size: List[int],
//         stride: List[int],
//         padding: List[int] = (0, 0),
//         ceil_mode: bool = False,
//         count_include_pad: bool = True,
//     ):
//     # FIXME how to deal with count_include_pad vs not for external padding?
//     x = pad_same(x, kernel_size, stride)
//     return F.avg_pool2d(x, kernel_size, stride, (0, 0), ceil_mode, count_include_pad)
//
// class AvgPool2d(conv.Module):
//     def __init__(
//         self,
//         kernel_size: _size_2_t,
//         stride: Optional[_size_2_t] = None,
//         padding: _size_2_t = 0,
//         ceil_mode: bool = False,
//         count_include_pad: bool = True,
//         divisor_override: Optional[int] = None,
//     ) -> None:
//         super().__init__()
//         self.kernel_size = kernel_size
//         self.stride = stride if (stride is not None) else kernel_size
//         self.padding = padding
//         self.ceil_mode = ceil_mode
//         self.count_include_pad = count_include_pad
//         self.divisor_override = divisor_override
//
//     def forward(self, input: Tensor) -> Tensor:
//         return F.avg_pool2d(
//             input,
//             self.kernel_size,
//             self.stride,
//             self.padding,
//             self.ceil_mode,
//             self.count_include_pad,
//             self.divisor_override,
//         )
//
// class AvgPool2dSame(conv.AvgPool2d):
//     """Tensorflow like 'SAME' wrapper for 2D average pooling."""
//     def __init__(
//             self,
//             kernel_size: _size_2_t,
//             stride: Optional[_size_2_t] = None,
//             padding: _size_2_t = 0,
//             ceil_mode=False,
//             count_include_pad=True,
//     ):
//         super(AvgPool2dSame, self).__init__(
//             kernel_size=kernel_size,
//             stride=stride,
//             padding=(0, 0), # padding is dropped, is this a bug?
//             ceil_mode=ceil_mode,
//             count_include_pad=count_include_pad,
//         )
//
//     def forward(self, x):
//         x = pad_same(x, self.kernel_size, self.stride)
//         return F.avg_pool2d(
//             x,
//             self.kernel_size,
//             self.stride,
//             self.padding,
//             self.ceil_mode,
//             self.count_include_pad,
//         )
// ```

/// [`AvgPool2dSame`] Configuration.
#[derive(Config, Debug)]
pub struct AvgPool2dSameConfig {
    pool: AvgPool2dConfig,
}

impl AvgPool2dSameConfig {
    /// Initializes [`AvgPool2dSame`].
    pub fn init(self) -> AvgPool2dSame {
        AvgPool2dSame {
            pool: self.pool.init(),
        }
    }
}

/// 2D average pooling with TensorFlow-style "SAME" padding.
///
/// Constructs via [`AvgPool2dSameConfig`] and `.init()`. The [`Self::forward`]
/// pass dynamically pads the input with [`pad_same`] so the output spatial size
/// matches TensorFlow's "SAME" convention, then applies the wrapped
/// [`AvgPool2d`].
///
/// Built by [`AvgPool2dSameConfig`].
#[derive(Module, Clone, Debug)]
pub struct AvgPool2dSame {
    pool: AvgPool2d,
}

impl AvgPool2dSame {
    /// Forward Pass.
    pub fn forward<B: Backend>(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let x = pad_same(input, self.pool.kernel_size, self.pool.stride, [1, 1], 0.0);
        self.pool.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        PerformanceBackend,
        performance_device,
    };

    #[test]
    #[serial]
    fn test_avg_pool_2d_same_output_is_ceil_of_size_over_stride() {
        type B = PerformanceBackend;
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        // Kernel 3, stride 3 (burn's default stride is the kernel size).
        let pool = AvgPool2dSameConfig::new(AvgPool2dConfig::new([3, 3])).init();

        // `[ceil(10 / 3), ceil(8 / 3)] == [4, 3]`.
        let input = Tensor::<B, 4>::ones([1, 1, 10, 8], &device);
        assert_eq!(pool.forward(input).dims(), [1, 1, 4, 3]);
    }
}
