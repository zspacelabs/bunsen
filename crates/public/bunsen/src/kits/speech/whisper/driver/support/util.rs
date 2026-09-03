use burn::{
    Tensor,
    prelude::Backend,
    tensor::BasicOps,
};

/// Drops the trailing frame of a whole stream's log-perceptive_audio.
///
/// # Arguments
/// * `stream` - `[batch, frames, n_mels]`.
///
/// # Returns
/// `[batch, frames - 1, n_mels]`.
///
/// # Panics
/// If `frames` is less than 2: one frame leaves nothing to package.
pub fn drop_last_frame<B, K>(stream: Tensor<B, 3, K>) -> Tensor<B, 3, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    let frames = stream.dims()[1];
    assert!(
        frames >= 2,
        "package_mels needs at least 2 frames, got {frames}: one is dropped, \
         and the clamp reduces over what remains",
    );
    stream.slice_dim(1, 0..frames as isize - 1)
}

#[cfg(test)]
mod test {
    use burn::{
        Tensor,
        prelude::s,
        tensor::Int,
    };

    use super::*;
    use crate::{
        burner::tensor::*,
        support::testing::CpuBackend,
    };

    #[test]
    fn test_drop_last_frame() {
        let device = Default::default();
        type B = CpuBackend;
        let stream: Tensor<B, 3, Int> =
            Tensor::<B, 1, Int>::arange(0..24, &device).reshape([2, 3, 4]);

        let actual: Tensor<B, 3, Int> = drop_last_frame(stream.clone());

        let expected: Tensor<B, 3, Int> = stream.slice(s![.., ..-1, ..]);

        actual
            .to_data_as::<i32>()
            .assert_eq(&expected.to_data_as::<i32>(), true);
    }
}
