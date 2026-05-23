# Convolution Support

[`bunsen::ops::conv`](https://docs.rs/bunsen/latest/bunsen/ops/conv/index.html)
collects everything around convolution that isn't itself a trainable
layer: shape arithmetic, a functional convolution kernel, and a few
helper filters.

## Shape arithmetic

Given a `Conv*d`'s kernel size, stride, padding, and dilation, what
output shape does it produce? `burn` answers this implicitly when you
call `forward`; `bunsen::ops::conv` gives you the same answer as a
pure function, which is what you reach for inside `Module::Config`
methods that need to advertise output sizes before any tensors
exist.

### General N-D

- **`maybe_conv_output_shape::<D>(input_shape, kernel, stride, padding, dilation)`**
  &mdash; const-generic over the spatial rank `D`. Returns
  `Option<[usize; D]>` &mdash; `None` if the configuration is
  impossible (e.g., kernel larger than the padded input).
- **`expect_conv_output_shape::<D>(...)`** &mdash; same but panics on
  impossible configurations, with a message naming the offending
  dimension.

The `_dyn` siblings (`maybe_conv_output_shape_dyn`,
`expect_conv_output_shape_dyn`) take slices instead of const-generic
arrays, for cases where the rank isn't known at compile time.

### 1-D scalar case

- **`maybe_conv1d_output_size(input, kernel, stride, padding, dilation)`**
  &mdash; the per-axis math broken out as a single scalar function,
  for when you only care about one dimension.
- **`expect_conv1d_output_size(...)`** &mdash; same, panicking.

### Common shortcuts

- **`stride_div_output_resolution(input_resolution, stride)`**
  &mdash; the "downsample by stride" arithmetic that comes up in
  ResNet-style blocks, with a check that the input is a multiple of
  the stride.
- **`get_square_conv2d_padding(kernel)`** &mdash; the "*same*-padding
  for a square odd kernel" formula.
- **`build_square_conv2d_padding_config(kernel)`** &mdash; the same
  result wrapped in `burn::nn::PaddingConfig2d`, for handing
  straight to a `Conv2dConfig`.

## Functional convolution

### `convolve_func_2d`

[`convolve_func_2d`](https://docs.rs/bunsen/latest/bunsen/ops/conv/fn.convolve_func_2d.html)
folds a user-supplied closure across the windows of a 2-D
convolution-shaped iteration:

```rust,ignore
// Pseudocode of the contract.
fn convolve_func_2d<B, KIn, KOut, F>(
    input: Tensor<B, 4, KIn>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    f: F,                          // window -> per-window output
) -> Tensor<B, 4, KOut>
where
    F: FnMut(Tensor<B, 4, KIn>) -> Tensor<B, 4, KOut>;
```

The closure receives one conv window at a time and returns the
corresponding output. This is the starting point for kernels that
aren't expressible as a linear `Conv2d`: median filters, mode
filters, custom per-window scoring, and so on. If your closure is
linear, you almost certainly want `burn::nn::conv::Conv2d` instead
&mdash; this function intentionally trades performance for
flexibility.

## Filters

### `conv2d_kernel_midpoint_filter`

[`conv2d_kernel_midpoint_filter`](https://docs.rs/bunsen/latest/bunsen/ops/conv/fn.conv2d_kernel_midpoint_filter.html)
is a kernel that picks the spatial mid-point sample from each
window &mdash; cheap to compute, useful as a stride-aware
"downsample to the centre pixel" filter and as a sanity-check
baseline for more sophisticated kernels.
