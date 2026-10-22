//! # Activation layout
//!
//! [`AxisOrder`]: the axis permutes a block applies at its boundary, so that
//! a caller's layout and the block's own need not agree.
//!
//! A layout is a block's choice of axis order. It is config, never a
//! parameter: an `AxisOrder` holds no tensors, so it is never part of a
//! record or a checkpoint's keys. When a block's internal layout differs from
//! the one a checkpoint's module used, the weight transform that implies
//! belongs in the block's own key map, not here.
use burn::Tensor;

/// The permutes a block applies to its input before it runs, and to its
/// output after, each optional.
///
/// A permute is the new order of the old axes, as
/// [`Tensor::permute`](burn::Tensor::permute) takes it: `[0, 2, 1]` swaps
/// the last two axes, `[batch, n_mels, time]` to `[batch, time, n_mels]`.
/// A block that masks by length counts frames along its own time axis,
/// after `pre`; `post` only reorders what it returns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AxisOrder {
    /// Applied to the input, before the block runs.
    pub pre: Option<[usize; 3]>,

    /// Applied to the output, after the block runs.
    pub post: Option<[usize; 3]>,
}

impl AxisOrder {
    /// No permutes: the caller's layout is the block's.
    pub const IDENTITY: Self = Self {
        pre: None,
        post: None,
    };
    /// The last two axes swapped on the way in: a `[batch, n_mels, time]`
    /// block fed `[batch, time, n_mels]`, or the reverse.
    pub const SWAP_IN: Self = Self {
        pre: Some([0, 2, 1]),
        post: None,
    };

    /// `x` as the block reads it.
    pub fn pre_apply(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        match self.pre {
            Some(order) => x.permute(order),
            None => x,
        }
    }

    /// `y` as the caller reads it.
    pub fn post_apply(
        &self,
        y: Tensor<3>,
    ) -> Tensor<3> {
        match self.post {
            Some(order) => y.permute(order),
            None => y,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_permutes() {
        let device = cpu_device();
        let x = Tensor::<3>::random([2, 3, 5], Distribution::Default, &device);

        let same = AxisOrder::IDENTITY.pre_apply(x.clone());
        assert_eq!(same.dims(), [2, 3, 5]);

        let swapped = AxisOrder::SWAP_IN.pre_apply(x.clone());
        assert_eq!(swapped.dims(), [2, 5, 3]);
        swapped
            .swap_dims(1, 2)
            .to_data()
            .assert_eq(&x.to_data(), true);
        assert_eq!(AxisOrder::SWAP_IN.post_apply(x.clone()).dims(), [2, 3, 5]);

        let order = AxisOrder {
            pre: None,
            post: Some([1, 0, 2]),
        };
        assert_eq!(order.post_apply(x).dims(), [3, 2, 5]);
    }
}
