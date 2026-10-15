//! # Rotary Embedding

use std::ops::Range;

use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::s,
    tensor::{
        DType,
        Device,
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::embedding::positional_frequency_table,
};

/// Common meta for [`RotaryEmbedding`] and [`RotaryEmbeddingConfig`].
pub trait RotaryEmbeddingMeta {
    /// Returns the sequence length.
    fn seq_len(&self) -> usize;

    /// Returns the head dimension.
    fn head_dim(&self) -> usize;
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

    /// Base.
    #[config(default = 10000)]
    pub base: usize,

    /// The rotation's sign convention.
    #[config(default = "RotaryDirection::Reverse")]
    pub direction: RotaryDirection,
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

        let freq_matrix =
            positional_frequency_table(self.seq_len, self.base, self.head_dim, device);

        // TODO: possibly down-cast to the smallest available dtype.

        let cos = freq_matrix
            .clone()
            .cos()
            .set_require_grad(false)
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim(0);

        let sin = freq_matrix
            .sin()
            .set_require_grad(false)
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim(0);

        // [1, T, 1, D/2]

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
/// `[B, T, H, D]` tensor.
///
/// Built by [`RotaryEmbeddingConfig`].
#[derive(Module, Debug)]
pub struct RotaryEmbedding {
    /// Head Dimension, D
    pub head_dim: usize,

    /// The rotation's sign convention.
    #[module(skip)]
    pub direction: RotaryDirection,

    /// a `[1, T, 1, D/2]` tensor.
    pub cos: Tensor<4>,

    /// a `[1, T, 1, D/2]` tensor.
    pub sin: Tensor<4>,
}

impl RotaryEmbeddingMeta for RotaryEmbedding {
    fn seq_len(&self) -> usize {
        self.cos.dims()[1]
    }

    fn head_dim(&self) -> usize {
        self.head_dim
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

    /// Applies the rotary embedding to the input.
    ///
    /// # Arguments
    /// - `input`: a `[B, T, H, D]` tensor.
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

        let pivot = self.head_dim() / 2;
        let x1 = input.clone().slice_dim(3, s![..pivot]);
        let x2 = input.clone().slice_dim(3, s![pivot..]);

        let (y1, y2) = self
            .direction
            .rotate(x1, x2, self.cos.clone(), self.sin.clone());

        let output = Tensor::cat(vec![y1, y2], 3);

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
