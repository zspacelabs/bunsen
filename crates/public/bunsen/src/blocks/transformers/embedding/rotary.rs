//! # Rotary Embedding

use core::ops::Range;

use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::s,
    tensor::{
        DType,
        Device,
        FloatDType,
        bf16,
        f16,
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::embedding::{
        RotaryTableDTypePolicy,
        frequency_table,
        inverse_frequency_table,
    },
};

/// Common meta for [`RotaryEmbedding`] and [`RotaryEmbeddingConfig`].
pub trait RotaryEmbeddingMeta {
    /// Returns the sequence length.
    fn seq_len(&self) -> usize;

    /// Returns the head dimension.
    fn head_dim(&self) -> usize;

    /// Returns the number of leading head dimensions rotated; the rest pass
    /// through unrotated.
    fn rotary_dim(&self) -> usize {
        self.head_dim()
    }
}

/// Config for [`RotaryEmbedding`].
#[derive(Config, Debug)]
pub struct RotaryEmbeddingConfig {
    /// Sequence Length.
    pub seq_len: usize,

    /// Head Dimension.
    ///
    /// This must be even.
    pub head_dim: usize,

    /// The leading head dimensions rotated, `[..rotary_dim]`; the rest,
    /// `[rotary_dim..]`, pass through unrotated (partial rotary: Qwen3.5's
    /// `partial_rotary_factor` of 0.25 rotates 64 of 256). The frequencies
    /// are spread over `rotary_dim`, as `transformers`' are over
    /// `head_dim * partial_rotary_factor`. `None` rotates all of
    /// `head_dim`. It must be even and at most `head_dim`.
    #[config(default = "None")]
    pub rotary_dim: Option<usize>,

    /// Base.
    #[config(default = 10000)]
    pub base: usize,

    /// The rotation's sign convention.
    #[config(default = "RotaryDirection::Reverse")]
    pub direction: RotaryDirection,

    /// The dtype the `cos` / `sin` tables are kept in. They are computed in
    /// f32 whatever the device's default float, then rounded to this: for a
    /// model trained with tables in a narrower dtype, as nanochat's 2025
    /// model keeps them in bf16. `None` keeps them in f32, exact. On a device
    /// without the dtype (wgpu has no bf16), a narrower one's rounded values
    /// are kept in f32, which holds them exactly, and a wider one is not
    /// kept ([`RotaryTableDTypePolicy`]).
    /// [`apply`](RotaryEmbedding::apply) casts them to its input's dtype, as
    /// `transformers` casts `cos` and `sin` to `x.dtype`.
    #[config(default = "None")]
    pub table_dtype: Option<FloatDType>,

    /// How the frequencies are scaled for a longer context.
    #[config(default = "RopeScaling::None")]
    pub scaling: RopeScaling,
}

/// How a rotary embedding's frequencies are scaled, as `transformers`'
/// `rope_type` (`rope_scaling` in older configs) says.
#[derive(Config, Copy, Debug, PartialEq)]
pub enum RopeScaling {
    /// None: `rope_type: default`.
    None,

    /// Every inverse frequency divided by `factor` (`rope_type: linear`):
    /// the positions stretched `factor` times. Gemma 3's global table, at 8.
    Linear {
        /// The factor.
        factor: f64,
    },
}

impl RopeScaling {
    /// `inv_freq`, scaled.
    pub fn scale_inverse_frequencies(
        self,
        inv_freq: Tensor<1>,
    ) -> Tensor<1> {
        match self {
            Self::None => inv_freq,
            Self::Linear { factor } => inv_freq.div_scalar(factor),
        }
    }
}

/// The sign convention of a rotary embedding: which way each pair of head
/// dimensions turns with position.
///
/// The two halves of a head, `x1` and `x2`, are a pair per frequency. A
/// checkpoint is trained under one convention; applying the other rotates
/// every position backwards.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum RotaryDirection {
    /// `y1 = x1 cos + x2 sin`, `y2 = x2 cos - x1 sin`: `nanochat`'s.
    Reverse,

    /// `y1 = x1 cos - x2 sin`, `y2 = x2 cos + x1 sin`: the `RoFormer`
    /// convention, `transformers`' `rotate_half` (Llama, Qwen).
    Forward,
}

impl RotaryDirection {
    /// Rotates the halves `x1` and `x2` of `[B, T, H, D/2]` by angles whose
    /// `cos` and `sin` are given, returning the rotated halves.
    pub fn rotate(
        self,
        x1: Tensor<4>,
        x2: Tensor<4>,
        cos: Tensor<4>,
        sin: Tensor<4>,
    ) -> (Tensor<4>, Tensor<4>) {
        match self {
            Self::Reverse => (
                x1.clone() * cos.clone() + x2.clone() * sin.clone(),
                x2 * cos - x1 * sin,
            ),
            Self::Forward => (
                x1.clone() * cos.clone() - x2.clone() * sin.clone(),
                x2 * cos + x1 * sin,
            ),
        }
    }
}

impl RotaryEmbeddingMeta for RotaryEmbeddingConfig {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn rotary_dim(&self) -> usize {
        self.rotary_dim.unwrap_or(self.head_dim)
    }
}

impl ModuleInit<RotaryEmbedding> for RotaryEmbeddingConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<RotaryEmbedding> {
        if !self.head_dim.is_multiple_of(2) {
            return Err(ConstraintError::new(
                "RotaryEmbeddingConfig",
                "head_dim",
                Rule::NotMultiple {
                    value: self.head_dim,
                    of: "2".into(),
                },
            )
            .into());
        }
        let rotary_dim = self.rotary_dim();
        if !rotary_dim.is_multiple_of(2) {
            return Err(ConstraintError::new(
                "RotaryEmbeddingConfig",
                "rotary_dim",
                Rule::NotMultiple {
                    value: rotary_dim,
                    of: "2".into(),
                },
            )
            .into());
        }
        if rotary_dim == 0 || rotary_dim > self.head_dim {
            return Err(ConstraintError::new(
                "RotaryEmbeddingConfig",
                "rotary_dim",
                Rule::OutOfRange {
                    value: rotary_dim.to_string(),
                    range: format!("1..={}", self.head_dim),
                },
            )
            .into());
        }

        let freq_matrix = frequency_table(
            self.seq_len,
            self.scaling
                .scale_inverse_frequencies(inverse_frequency_table(self.base, rotary_dim, device)),
        );

        // Rounded on the host, as the reference rounds, and kept as the
        // policy says.
        let policy = RotaryTableDTypePolicy::init(self.table_dtype, |d| device.supports_dtype(d));
        let round = |t: Tensor<2>| {
            let data = t.into_data();
            let data = match policy.round_to {
                Some(FloatDType::BF16) => data.convert::<bf16>(),
                Some(FloatDType::F16) => data.convert::<f16>(),
                _ => data,
            };
            Tensor::<2>::from_data(data, (device, DType::from(policy.keep_in)))
        };

        let cos = round(freq_matrix.clone().cos())
            .set_require_grad(false)
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim(0);

        let sin = round(freq_matrix.sin())
            .set_require_grad(false)
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim(0);

        // [1, T, 1, R/2]

        Ok(RotaryEmbedding {
            head_dim: self.head_dim,
            direction: self.direction,
            cos,
            sin,
        })
    }
}

/// Rotary Embedding Module
///
/// Precomputed rotary positional embedding (`RoPE`). It holds per-position
/// `cos`/`sin` tables and applies position-dependent rotations to the
/// query/key head dimensions via [`apply`](RotaryEmbedding::apply); use
/// [`clip_range`](RotaryEmbedding::clip_range) to restrict it to a
/// sub-range of positions. Construct via [`RotaryEmbeddingConfig`] and
/// `.init(device)`, then call [`apply`](RotaryEmbedding::apply) on a
/// `[B, T, H, D]` tensor. A partial rotary rotates the leading `R` head
/// dimensions ([`rotary_dim`](RotaryEmbeddingMeta::rotary_dim)) and passes
/// the rest through.
///
/// Built by [`RotaryEmbeddingConfig`].
#[derive(Module, Debug)]
pub struct RotaryEmbedding {
    /// Head Dimension, D
    pub head_dim: usize,

    /// The rotation's sign convention.
    #[module(skip)]
    pub direction: RotaryDirection,

    /// a `[1, T, 1, R/2]` tensor, `R` the rotated dimensions.
    pub cos: Tensor<4>,

    /// a `[1, T, 1, R/2]` tensor, `R` the rotated dimensions.
    pub sin: Tensor<4>,
}

impl RotaryEmbeddingMeta for RotaryEmbedding {
    fn seq_len(&self) -> usize {
        self.cos.dims()[1]
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn rotary_dim(&self) -> usize {
        2 * self.cos.dims()[3]
    }
}

impl crate::burner::module::InputDTypes for RotaryEmbedding {
    /// Its tables' dtype.
    fn preferred_dtype(&self) -> FloatDType {
        FloatDType::from(self.cos.dtype())
    }

    /// Every float dtype: [`apply`](Self::apply) casts its tables to the
    /// input's.
    fn accepts_dtype(
        &self,
        _dtype: FloatDType,
    ) -> bool {
        true
    }
}

impl crate::burner::module::ResultDType for RotaryEmbedding {
    /// The input's.
    fn result_dtype(
        &self,
        input: FloatDType,
    ) -> FloatDType {
        input
    }
}

impl RotaryEmbedding {
    /// Casts the embedding to a different dtype.
    pub fn cast(
        self,
        dtype: DType,
    ) -> Self {
        Self {
            cos: self.cos.cast(dtype),
            sin: self.sin.cast(dtype),
            ..self
        }
    }

    /// Clip the embedding to cover only the given range.
    ///
    /// # Arguments
    /// - `range`: the ``start..end`` range to cover.
    ///
    /// # Returns
    /// - a clipped [`RotaryEmbedding`].
    pub fn clip_range(
        &self,
        range: Range<usize>,
    ) -> Self {
        Self {
            head_dim: self.head_dim,
            direction: self.direction,
            cos: self.cos.clone().slice_dim(1, range.clone()),
            sin: self.sin.clone().slice_dim(1, range),
        }
    }

    /// Applies the rotary embedding to the input, in the input's dtype: the
    /// tables are cast to it (a no-op when they are in it already). The
    /// leading [`rotary_dim`](RotaryEmbeddingMeta::rotary_dim) head
    /// dimensions are rotated, the rest passed through.
    ///
    /// # Arguments
    /// - `input`: a `[B, T, H, D]` tensor, in any float dtype.
    ///
    /// # Returns
    /// - a `[B, T, H, D]` tensor.
    pub fn apply(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        #[cfg(debug_assertions)]
        let [b, h] = crate::contracts::unpack_shape_contract!(
            ["B", "T", "H", "D"],
            &input.dims(),
            &["B", "H"],
            &[("T", self.seq_len()), ("D", self.head_dim())]
        );

        let rotary_dim = self.rotary_dim();
        let pass =
            (rotary_dim < self.head_dim()).then(|| input.clone().slice_dim(3, s![rotary_dim..]));
        let pivot = rotary_dim / 2;
        let x1 = input.clone().slice_dim(3, s![..pivot]);
        let x2 = input.clone().slice_dim(3, s![pivot..rotary_dim]);

        let dtype = input.dtype();
        let (y1, y2) = self.direction.rotate(
            x1,
            x2,
            self.cos.clone().cast(dtype),
            self.sin.clone().cast(dtype),
        );

        let output = Tensor::cat([vec![y1, y2], pass.into_iter().collect()].concat(), 3);

        #[cfg(debug_assertions)]
        crate::contracts::assert_shape_contract_periodically!(
            ["B", "T", "H", "D"],
            &output.dims(),
            &[
                ("B", b),
                ("T", self.seq_len()),
                ("H", h),
                ("D", self.head_dim())
            ]
        );

        output
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };
    use serial_test::serial;

    use super::*;
    use crate::{
        contracts::assert_shape_contract,
        support::testing::{
            DeviceMemoryGuard,
            performance_device,
        },
    };

    /// A linear factor divides every inverse frequency: position `p` of the
    /// scaled table is position `p / factor` of the plain one.
    #[test]
    fn test_linear_scaling() {
        let device = crate::support::testing::cpu_device();
        let plain: RotaryEmbedding = RotaryEmbeddingConfig::new(8, 16).init(&device);
        let scaled: RotaryEmbedding = RotaryEmbeddingConfig::new(4, 16)
            .with_scaling(RopeScaling::Linear { factor: 2.0 })
            .init(&device);
        // The angles are the positions times the halved inverse frequencies.
        let inv = inverse_frequency_table(10000, 16, &device);
        let expected = frequency_table(4, inv.div_scalar(2.0)).cos();
        scaled
            .cos
            .clone()
            .reshape([4, 8])
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
        // So the scaled table's position 2 is the plain table's position 1.
        plain
            .cos
            .clone()
            .slice_dim(1, 1..2)
            .into_data()
            .assert_approx_eq::<f32>(
                &scaled.cos.clone().slice_dim(1, 2..3).into_data(),
                Tolerance::default(),
            );
        assert_eq!(
            RotaryEmbeddingConfig::new(1, 2).scaling,
            RopeScaling::None,
            "unscaled by default"
        );
    }

    /// Tables kept in bf16 hold the exact f32 tables' values rounded to bf16:
    /// within bf16's half-ulp, kept in bf16 where the device has it (else in
    /// f32). `apply` casts them to its input's dtype, so an f32 input gives an
    /// f32 output.
    #[test]
    #[serial]
    fn test_tables_rounded_to_bf16() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let exact: RotaryEmbedding = RotaryEmbeddingConfig::new(64, 16).init(&device);
        let rounded: RotaryEmbedding = RotaryEmbeddingConfig::new(64, 16)
            .with_table_dtype(Some(FloatDType::BF16))
            .init(&device);
        let has_bf16 = device.supports_dtype(DType::BF16);
        assert_eq!(exact.cos.dtype(), DType::F32);
        assert_eq!(
            rounded.cos.dtype(),
            if has_bf16 { DType::BF16 } else { DType::F32 },
            "kept in bf16 where the device has it"
        );
        let x: Tensor<4> = Tensor::ones([1, 64, 2, 16], &device);
        assert_eq!(rounded.apply(x.clone()).dtype(), DType::F32);
        if has_bf16 {
            assert_eq!(rounded.apply(x.cast(DType::BF16)).dtype(), DType::BF16);
        }
        for (e, r) in [(exact.cos, rounded.cos), (exact.sin, rounded.sin)] {
            let e = e.into_data().try_to_vec::<f32>().unwrap();
            let r = r.cast(DType::F32).into_data().try_to_vec::<f32>().unwrap();
            assert!(e != r, "rounding changed nothing");
            for (e, r) in e.iter().zip(&r) {
                assert_eq!(bf16::from_f32(*r).to_f32(), *r, "{r} is not a bf16 value");
                assert!(
                    (e - r).abs() <= e.abs() * 2f32.powi(-8) + 1e-30,
                    "{e} -> {r}"
                );
            }
        }
    }

    #[test]
    #[serial]
    fn test_clip_range() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let config = RotaryEmbeddingConfig::new(1024, 64);
        let re: RotaryEmbedding = config.init(&device);
        assert_eq!(re.seq_len(), 1024);
        assert_eq!(re.head_dim(), 64);

        let clip_re = re.clip_range(10..20);
        assert_eq!(clip_re.seq_len(), 10);
        clip_re
            .sin
            .clone()
            .to_data()
            .assert_eq(&re.sin.clone().slice_dim(1, 10..20).to_data(), true);
        clip_re
            .cos
            .clone()
            .to_data()
            .assert_eq(&re.cos.clone().slice_dim(1, 10..20).to_data(), true);
    }

    #[test]
    #[serial]
    fn test_rotary_embedding() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let batch = 1;
        let heads = 2;
        let seq_len = 1024;
        let head_dim = 64;

        let config = RotaryEmbeddingConfig::new(seq_len, head_dim);
        assert_eq!(config.seq_len(), seq_len);
        assert_eq!(config.head_dim(), head_dim);
        assert_eq!(config.base, 10000);

        let re: RotaryEmbedding = config.init(&device);
        assert_eq!(re.seq_len(), seq_len);
        assert_eq!(re.head_dim(), head_dim);

        let input: Tensor<4> = Tensor::random(
            [batch, seq_len, heads, head_dim],
            Distribution::Default,
            &device,
        );

        let output = re.apply(input.clone());
        assert_shape_contract!(
            ["B", "T", "H", "D"],
            &output.dims(),
            &[("B", batch), ("T", seq_len), ("H", heads), ("D", head_dim)]
        );

        let x1 = input.clone().slice_dim(3, s![..head_dim / 2]);
        let x2 = input.clone().slice_dim(3, s![head_dim / 2..]);
        let y1 = x1.clone() * re.cos.clone() + x2.clone() * re.sin.clone();
        let y2 = x1 * (-re.sin.clone()) + x2 * re.cos.clone();
        let expected = Tensor::cat(vec![y1, y2], 3);

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::default());
    }

    /// A partial rotary rotates the leading `rotary_dim` head dimensions as a
    /// full rotary of that width would, frequencies and all, and passes the
    /// rest through unchanged.
    #[test]
    fn test_partial_rotary() {
        let device = crate::support::testing::cpu_device();
        let (seq_len, head_dim, rotary_dim) = (6, 12, 4);
        for direction in [RotaryDirection::Forward, RotaryDirection::Reverse] {
            let cfg = RotaryEmbeddingConfig::new(seq_len, head_dim)
                .with_rotary_dim(Some(rotary_dim))
                .with_direction(direction);
            assert_eq!(cfg.rotary_dim(), rotary_dim);
            let partial: RotaryEmbedding = cfg.init(&device);
            assert_eq!(partial.head_dim(), head_dim);
            assert_eq!(partial.rotary_dim(), rotary_dim);
            assert_eq!(partial.clip_range(1..3).rotary_dim(), rotary_dim);
            let full: RotaryEmbedding = RotaryEmbeddingConfig::new(seq_len, rotary_dim)
                .with_direction(direction)
                .init(&device);

            let x = crate::support::testing::seeded_tensor::<4>(
                2,
                [2, seq_len, 3, head_dim],
                Distribution::Default,
                &device,
            );
            let y = partial.apply(x.clone());
            assert_eq!(y.dims(), [2, seq_len, 3, head_dim]);
            y.clone()
                .slice_dim(3, s![..rotary_dim])
                .into_data()
                .assert_approx_eq::<f32>(
                    &full
                        .apply(x.clone().slice_dim(3, s![..rotary_dim]))
                        .into_data(),
                    Tolerance::default(),
                );
            assert_eq!(
                y.slice_dim(3, s![rotary_dim..]).into_data(),
                x.slice_dim(3, s![rotary_dim..]).into_data(),
                "the rest passes through"
            );
        }

        let unset = RotaryEmbeddingConfig::new(2, 8);
        assert_eq!(unset.rotary_dim(), 8, "all of head_dim by default");
        for bad in [3, 0, 10] {
            assert!(
                RotaryEmbeddingConfig::new(2, 8)
                    .with_rotary_dim(Some(bad))
                    .try_init(&device)
                    .is_err(),
                "rotary_dim {bad}"
            );
        }
    }

    /// At position 1 the first frequency is one radian: `Forward` turns
    /// `(1, 0)` to `(cos 1, sin 1)`, `Reverse` to `(cos 1, -sin 1)`.
    #[test]
    fn test_directions() {
        use crate::support::testing::cpu_device;
        let device = cpu_device();
        let x = Tensor::<4>::from_data([[[[0.0f32, 0.0]]], [[[1.0, 0.0]]]], &device)
            .reshape([1, 2, 1, 2]);
        for (direction, sign) in [
            (RotaryDirection::Forward, 1.0f32),
            (RotaryDirection::Reverse, -1.0),
        ] {
            let re: RotaryEmbedding = RotaryEmbeddingConfig::new(2, 2)
                .with_direction(direction)
                .init(&device);
            let y: Vec<f32> = re
                .apply(x.clone())
                .into_data()
                .try_into_vec_as::<f32>()
                .unwrap();
            assert!((y[2] - 1f32.cos()).abs() < 1e-6, "{direction:?}: {y:?}");
            assert!(
                (y[3] - sign * 1f32.sin()).abs() < 1e-6,
                "{direction:?}: {y:?}"
            );
        }
    }
}
