//! Per-channel normalisation, as torchvision's `Normalize`.

use alloc::format;

use burn::{
    Tensor,
    tensor::TensorData,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// Normalises each channel of each image: `(x - mean[c]) / std[c]`, as
/// torchvision's `Normalize` computes it (a subtraction, then a division, in
/// the images' dtype).
///
/// See the [errors convention](crate::errors#convention-try_x-and-x);
/// [`normalize`] is the panicking twin.
///
/// # Arguments
/// - `images`: a `[batch, channels, height, width]` batch.
/// - `mean`, `std`: one value per channel.
///
/// # Returns
/// The `[batch, channels, height, width]` normalised batch.
///
/// # Errors
/// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when `mean` or `std`
/// does not hold one value per channel, or a `std` value is not a positive
/// finite number.
pub fn try_normalize(
    images: Tensor<4>,
    mean: &[f32],
    std: &[f32],
) -> BunsenResult<Tensor<4>> {
    let [_batch, channels, _height, _width] = crate::contracts::unpack_shape_contract!(
        ["batch", "channels", "height", "width"],
        &images,
        &["batch", "channels", "height", "width"],
    );
    if mean.len() != channels || std.len() != channels {
        return Err(BunsenError::illegal(format!(
            "a mean of {} and a std of {} values for images of {channels} channels",
            mean.len(),
            std.len()
        )));
    }
    if let Some(s) = std.iter().find(|s| !(s.is_finite() && **s > 0.0)) {
        return Err(BunsenError::illegal(format!("a normalisation std of {s}")));
    }
    let (device, dtype) = (images.device(), images.dtype());
    let per_channel = |values: &[f32]| {
        Tensor::<1>::from_data(TensorData::new(values.to_vec(), [channels]), &device)
            .cast(dtype)
            .reshape([1, channels, 1, 1])
    };
    Ok((images - per_channel(mean)) / per_channel(std))
}

/// Normalises each channel of each image: `(x - mean[c]) / std[c]`.
///
/// The panicking twin of [`try_normalize`].
///
/// # Panics
/// With [`try_normalize`]'s error report.
pub fn normalize(
    images: Tensor<4>,
    mean: &[f32],
    std: &[f32],
) -> Tensor<4> {
    try_normalize(images, mean, std).unwrap_or_else(|e| panic!("{e:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::cpu_device,
    };

    /// Each channel by its own mean and std, bit for bit as the two f32
    /// operations give it.
    #[test]
    fn test_normalize_is_per_channel() {
        let device = cpu_device();
        let pixels = [[0.25f32, 1.0], [0.5, 0.0]];
        let images = Tensor::<3>::from_floats([pixels], &device).reshape([1, 2, 1, 2]);
        let (mean, std) = ([0.485f32, 0.456], [0.229f32, 0.224]);
        let out = normalize(images, &mean, &std);
        let want: Vec<f32> = (0..2)
            .flat_map(|c| pixels[c].map(|x| (x - mean[c]) / std[c]))
            .collect();
        out.to_data()
            .assert_eq(&TensorData::new(want, [1, 2, 1, 2]), true);
    }

    #[test]
    fn test_try_normalize_rejects_a_mismatched_mean() {
        let device = cpu_device();
        let images = Tensor::<4>::zeros([1, 3, 2, 2], &device);
        ErrorMatcher::illegal()
            .message_contains("a mean of 1 and a std of 3 values")
            .assert_err(&try_normalize(images, &[0.5], &[0.5; 3]));
    }

    #[test]
    fn test_try_normalize_rejects_a_zero_std() {
        let device = cpu_device();
        let images = Tensor::<4>::zeros([1, 1, 2, 2], &device);
        ErrorMatcher::illegal()
            .message_contains("std of 0")
            .assert_err(&try_normalize(images, &[0.5], &[0.0]));
    }
}
