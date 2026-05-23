# Tensor Functions

The element-wise, shape-level, and generation helpers in
`bunsen::ops`. Organized by intent rather than alphabet.

## Tensor generation

### `arange` &mdash; floating-point ranges

[`arange`](https://docs.rs/bunsen/latest/bunsen/ops/arange/index.html)
fills the gap between integer `Tensor::arange` and the
`numpy.arange` / `numpy.linspace` family. The functions come in two
flavours:

- **Host-side `Vec<f64>` builders** when you want values to feed
  into other configuration:
  - `float_vec_arange(start, end, step)`
  - `float_vec_linspace(start, end, n)`
- **Device-side `Tensor` builders** when you want the values on the
  same backend as the rest of your computation:
  - `float_arange::<B>(start, end, step, &device)`
  - `float_linspace::<B>(start, end, n, &device)`

`step` is optional on `arange`; it defaults to `1.0` for ascending
ranges and `-1.0` for descending.

### `noise` &mdash; distribution + clamp value-objects

[`NoiseConfig`](https://docs.rs/bunsen/latest/bunsen/ops/noise/struct.NoiseConfig.html)
bundles a `burn::tensor::Distribution` with an optional
[`ClampOp`](./tensor-functions.md#clamp--optional-min-and-max) into
a single reusable value:

```rust,ignore
use bunsen::ops::{clamp::ClampOp, noise::NoiseConfig};
use burn::tensor::Distribution;

let cfg = NoiseConfig::default()
    .with_distribution(Distribution::Normal(0.0, 1.0))
    .with_clamp(ClampOp::new(Some(-3.0), Some(3.0)));

// Materialize:
let t  = cfg.noise::<B, _, 3>([batch, channels, length], &device);
let t2 = cfg.noise_like(&existing_tensor);
```

`noise()` takes a shape and a device; `noise_like(t)` takes another
tensor and matches its shape and device. The clamp, when present, is
applied in the same pass.

## Element-wise transforms

### `clamp` &mdash; optional min and max

[`ClampOp`](https://docs.rs/bunsen/latest/bunsen/ops/clamp/struct.ClampOp.html)
captures an optional minimum and an optional maximum as a single
serializable value, with `.clamp(tensor)` to apply it. It's designed
for places clamping shows up as a *setting* &mdash; a knob on a noise
generator, an entry in a serialized config &mdash; not just a
one-shot call:

```rust,ignore
let op = ClampOp::new(Some(0.0), None);   // ReLU-style: clamp below at 0
let y  = op.clamp(x);
```

`ClampOp` is `Config`-friendly: it implements `Serialize`,
`Deserialize`, and `ModuleDisplay`, so it drops cleanly into
upstream `#[derive(Config)]` blocks.

### `drop` &mdash; functional dropout

[`dropout(prob, input)`](https://docs.rs/bunsen/latest/bunsen/ops/drop/fn.dropout.html)
is the functional dropout op: at `prob == 0.0` it short-circuits to
the input unchanged; otherwise it samples a Bernoulli mask and
rescales by `1 / (1 - prob)`. Use this when you need a single call
inside a free function; for a trainable layer with its own toggle
state, use the modules in
[`bunsen::blocks::images::drop`](../blocks/images.md#stochastic-regularization).

## Normalization

### `rms_norm` &mdash; parameter-free RMS normalization

[`rms_norm(input, &options)`](https://docs.rs/bunsen/latest/bunsen/ops/norm/fn.rms_norm.html)
applies RMSNorm without a trainable gain. Configured via
[`RmsNormOptions`](https://docs.rs/bunsen/latest/bunsen/ops/norm/struct.RmsNormOptions.html),
which carries the epsilon (`with_eps(...)`).

This is the right call when you need RMSNorm inside a free function
or unit test. The parametric layer with a learned gain lives in
`burn::nn::norm::RmsNorm`; the two share the same numerics.

## Shape transforms

### `repeat_interleave` &mdash; NumPy-style interleaving

[`repeat_interleave(input, repeats, dim)`](https://docs.rs/bunsen/latest/bunsen/ops/repeat/fn.repeat_interleave.html)
repeats elements along a single axis, in-place rather than tiled.
Given `[a, b, c]` and `repeats = 2`, it returns `[a, a, b, b, c, c]`
&mdash; matching NumPy / PyTorch semantics, including negative
indexing for `dim`.
