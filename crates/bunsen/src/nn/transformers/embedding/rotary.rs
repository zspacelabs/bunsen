//! # Rotary Embedding

use std::ops::Range;

use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::{
        Backend,
        s,
    },
    tensor::{
        DType,
        linalg,
    },
};

/// Common meta for [`RotaryEmbedding`] and [`RotaryEmbeddingConfig`].
pub trait RotaryEmbeddingMeta {
    /// Return the sequence length.
    fn seq_len(&self) -> usize;

    /// Return the head dimension.
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
}

impl RotaryEmbeddingMeta for RotaryEmbeddingConfig {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl RotaryEmbeddingConfig {
    /// Initialize the rotary embedding.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> RotaryEmbedding<B> {
        assert!(
            self.head_dim.is_multiple_of(2),
            "Head dimension must be even: {}",
            self.head_dim
        );

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

        RotaryEmbedding {
            head_dim: self.head_dim,
            cos,
            sin,
        }
    }
}

/// Rotary Embedding Module
#[derive(Module, Debug)]
pub struct RotaryEmbedding<B: Backend> {
    /// Head Dimension, D
    pub head_dim: usize,

    /// a ``[1, T, 1, D/2]`` tensor.
    pub cos: Tensor<B, 4>,

    /// a ``[1, T, 1, D/2]`` tensor.
    pub sin: Tensor<B, 4>,
}

impl<B: Backend> RotaryEmbeddingMeta for RotaryEmbedding<B> {
    fn seq_len(&self) -> usize {
        self.cos.dims()[1]
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Cast the embedding to a different dtype.
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
            cos: self.cos.clone().slice_dim(1, range.clone()),
            sin: self.sin.clone().slice_dim(1, range),
        }
    }

    /// Apply the rotary embedding to the input.
    ///
    /// # Arguments
    /// - `input`: a ``[B, T, H, D]`` tensor.
    ///
    /// # Returns
    /// - a ``[B, T, H, D]`` tensor.
    pub fn apply(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        #[cfg(debug_assertions)]
        let [b, h] = bunsen_contracts::unpack_shape_contract!(
            ["B", "T", "H", "D"],
            &input.dims(),
            &["B", "H"],
            &[("T", self.seq_len()), ("D", self.head_dim())]
        );

        let pivot = self.head_dim() / 2;
        let x1 = input.clone().slice_dim(3, s![..pivot]);
        let x2 = input.clone().slice_dim(3, s![pivot..]);

        let y1 = x1.clone() * self.cos.clone() + x2.clone() * self.sin.clone();
        let y2 = x1 * (-self.sin.clone()) + x2 * self.cos.clone();

        let output = Tensor::cat(vec![y1, y2], 3);

        #[cfg(debug_assertions)]
        bunsen_contracts::assert_shape_contract_periodically!(
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

/// Compute the rotary embedding inverse frequency table.
///
/// # Arguments
/// - `base`: the base.
/// - `head_dim`: the number of head dimensions.
/// - `device`: the target device.
///
/// # Returns
/// - ``[(1.0 / (base**(d / head_dim))) for d in 0:head_dim:2]``
pub fn inverse_frequency_table<B: Backend>(
    base: usize,
    head_dim: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    Tensor::from_data([base as f32], device).powf(
        -Tensor::arange_step(0..head_dim as i64, 2, device)
            .float()
            .div_scalar(head_dim as f32),
    )
}

/// Compute the positionally shifted frequency table.
///
/// # Arguments
/// - `seq_len`: the sequence length.
/// - `base`: the base.
/// - `head_dim`: the number of head dimensions.
/// - `device`: the target device.
///
/// # Returns
/// - ``[T, F=D/2]`` sequence x inverse frequency table.
pub fn positional_frequency_table<B: Backend>(
    seq_len: usize,
    base: usize,
    head_dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let inv_freq = inverse_frequency_table::<B>(base, head_dim, device);

    let t: Tensor<B, 1> = Tensor::arange(0..seq_len as i64, device).float();

    linalg::outer::<_, 1, 2, _>(t, inv_freq)
}

#[cfg(test)]
mod tests {
    use bunsen_contracts::assert_shape_contract;
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::BunsenTestBackend;

    #[test]
    fn test_inverse_frequency_table() {
        type B = BunsenTestBackend;
        let device = Default::default();

        let base = 10000;
        let head_dim = 4;

        let base_f = base as f32;
        let head_dim_f = head_dim as f32;

        inverse_frequency_table::<B>(base, head_dim, &device)
            .to_data()
            .assert_approx_eq(
                &Tensor::<B, 1>::from_data(
                    [
                        1.0 / base_f.powf(0.0 / head_dim_f),
                        1.0 / base_f.powf(2.0 / head_dim_f),
                    ],
                    &device,
                )
                .to_data(),
                Tolerance::<f32>::default(),
            );
    }

    #[test]
    fn test_frequency_matrix() {
        type B = BunsenTestBackend;
        let device = Default::default();

        let base = 10000;
        let head_dim = 4;

        let base_f = base as f32;
        let head_dim_f = head_dim as f32;

        positional_frequency_table::<B>(3, base, head_dim, &device)
            .to_data()
            .assert_approx_eq(
                &Tensor::<B, 2>::from_data(
                    [
                        [0.0, 0.0],
                        [
                            1.0 / base_f.powf(0.0 / head_dim_f),
                            1.0 / base_f.powf(2.0 / head_dim_f),
                        ],
                        [
                            2.0 / base_f.powf(0.0 / head_dim_f),
                            2.0 / base_f.powf(2.0 / head_dim_f),
                        ],
                    ],
                    &device,
                )
                .to_data(),
                Tolerance::<f32>::default(),
            );
    }

    #[test]
    fn test_clip_range() {
        type B = BunsenTestBackend;
        let device = Default::default();

        let config = RotaryEmbeddingConfig::new(1024, 64);
        let re: RotaryEmbedding<B> = config.init(&device);
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
    fn test_rotary_embedding() {
        type B = BunsenTestBackend;
        let device = Default::default();

        let batch = 1;
        let heads = 2;
        let seq_len = 1024;
        let head_dim = 64;

        let config = RotaryEmbeddingConfig::new(seq_len, head_dim);
        assert_eq!(config.seq_len(), seq_len);
        assert_eq!(config.head_dim(), head_dim);
        assert_eq!(config.base, 10000);

        let re: RotaryEmbedding<B> = config.init(&device);
        assert_eq!(re.seq_len(), seq_len);
        assert_eq!(re.head_dim(), head_dim);

        let input: Tensor<B, 4> = Tensor::random(
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
}
