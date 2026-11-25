//! Center crops, with and without padding, at torchvision's offsets.

use alloc::format;

use burn::{
    Tensor,
    tensor::{
        TensorData,
        ops::PadMode,
    },
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// The offset of a centred crop of `crop` out of `size`, as torchvision's
/// `center_crop` rounds it: `int(round((size - crop) / 2.0))`, with Python's
/// round-half-to-even.
///
/// An odd margin `2m + 1` puts the crop at `m` when `m` is even and at `m + 1`
/// when `m` is odd: a 225-pixel side cropped to 224 starts at 0, a 227-pixel
/// side at 2.
///
/// ```rust
/// use bunsen::ops::images::center_crop_offset;
///
/// assert_eq!(center_crop_offset(256, 224), 16);
/// assert_eq!(center_crop_offset(225, 224), 0);
/// assert_eq!(center_crop_offset(227, 224), 2);
/// ```
///
/// # Panics
/// When `crop` exceeds `size`.
pub fn center_crop_offset(
    size: usize,
    crop: usize,
) -> usize {
    assert!(
        crop <= size,
        "center_crop_offset: a crop of {crop} out of {size}"
    );
    let margin = size - crop;
    let m = margin / 2;
    if margin.is_multiple_of(2) || m.is_multiple_of(2) {
        m
    } else {
        m + 1
    }
}

/// Crops the centre `size` (`[height, width]`) out of each image, at
/// torchvision's offsets ([`center_crop_offset`]).
///
/// See the [errors convention](crate::errors#convention-try_x-and-x);
/// [`center_crop`] is the panicking twin.
///
/// # Arguments
/// - `images`: a `[batch, channels, height, width]` batch.
/// - `size`: the crop's `[crop_height, crop_width]`.
///
/// # Returns
/// The `[batch, channels, crop_height, crop_width]` crops.
///
/// # Errors
/// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when the crop is
/// larger than the images on either side; [`center_crop_or_pad`] pads
/// instead.
pub fn try_center_crop(
    images: Tensor<4>,
    size: [usize; 2],
) -> BunsenResult<Tensor<4>> {
    let [batch, channels, height, width] = crate::contracts::unpack_shape_contract!(
        ["batch", "channels", "height", "width"],
        &images,
        &["batch", "channels", "height", "width"],
    );
    let [crop_height, crop_width] = size;
    if crop_height > height || crop_width > width {
        return Err(BunsenError::illegal(format!(
            "a {crop_height}x{crop_width} center crop of {height}x{width} images"
        )));
    }
    let top = center_crop_offset(height, crop_height);
    let left = center_crop_offset(width, crop_width);
    Ok(images.slice([
        0..batch,
        0..channels,
        top..top + crop_height,
        left..left + crop_width,
    ]))
}

/// Crops the centre `size` (`[height, width]`) out of each image.
///
/// The panicking twin of [`try_center_crop`].
///
/// # Panics
/// With [`try_center_crop`]'s error report, when the crop is larger than the
/// images.
pub fn center_crop(
    images: Tensor<4>,
    size: [usize; 2],
) -> Tensor<4> {
    try_center_crop(images, size).unwrap_or_else(|e| panic!("{e:#}"))
}

/// Crops the centre `size` (`[height, width]`) out of each image, padding
/// first where the images are smaller, as timm's `center_crop_or_pad`.
///
/// A side shorter than the crop is padded to it with `fill`, the larger half
/// of the padding after the image (`(n - side) / 2` before, the rest after);
/// a longer side is cropped at [`center_crop_offset`].
///
/// See the [errors convention](crate::errors#convention-try_x-and-x);
/// [`center_crop_or_pad`] is the panicking twin.
///
/// # Arguments
/// - `images`: a `[batch, channels, height, width]` batch.
/// - `size`: the output's `[out_height, out_width]`.
/// - `fill`: the padding's value in each channel, in the images' own units.
///
/// # Returns
/// The `[batch, channels, out_height, out_width]` batch.
///
/// # Errors
/// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when `fill` does not
/// hold one value per channel.
pub fn try_center_crop_or_pad(
    images: Tensor<4>,
    size: [usize; 2],
    fill: &[f32],
) -> BunsenResult<Tensor<4>> {
    let [_batch, channels, height, width] = crate::contracts::unpack_shape_contract!(
        ["batch", "channels", "height", "width"],
        &images,
        &["batch", "channels", "height", "width"],
    );
    if fill.len() != channels {
        return Err(BunsenError::illegal(format!(
            "a fill of {} values for images of {channels} channels",
            fill.len()
        )));
    }
    let [out_height, out_width] = size;
    let pad = |side: usize, out: usize| {
        if out > side {
            ((out - side) / 2, (out - side).div_ceil(2))
        } else {
            (0, 0)
        }
    };
    let (top, bottom) = pad(height, out_height);
    let (left, right) = pad(width, out_width);
    let images = if top + bottom + left + right == 0 {
        images
    } else {
        // Zero padding, then the fill where the padding is: exact, as
        // `x + fill * 0` is `x` and `0 + fill * 1` is `fill`.
        // Both constants are made in the images' own dtype.
        let (device, dtype) = (images.device(), images.dtype());
        let padding = [(top, bottom), (left, right)];
        let border = Tensor::<4>::ones([1, 1, height, width], &device)
            .cast(dtype)
            .pad(padding, PadMode::Constant(0.0))
            .neg()
            + 1.0;
        let fill = Tensor::<1>::from_data(TensorData::new(fill.to_vec(), [channels]), &device)
            .cast(dtype)
            .reshape([1, channels, 1, 1]);
        images.pad(padding, PadMode::Constant(0.0)) + fill * border
    };
    try_center_crop(images, size)
}

/// Crops the centre `size` (`[height, width]`) out of each image, padding
/// first where the images are smaller.
///
/// The panicking twin of [`try_center_crop_or_pad`].
///
/// # Panics
/// With [`try_center_crop_or_pad`]'s error report, when `fill` does not hold
/// one value per channel.
pub fn center_crop_or_pad(
    images: Tensor<4>,
    size: [usize; 2],
    fill: &[f32],
) -> Tensor<4> {
    try_center_crop_or_pad(images, size, fill).unwrap_or_else(|e| panic!("{e:#}"))
}

#[cfg(test)]
mod tests {
    use burn::tensor::Int;

    use super::*;
    use crate::{
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        support::testing::cpu_device,
    };

    /// Python's `int(round((size - crop) / 2.0))`, by its values: even
    /// margins halve, odd margins round half to even.
    #[test]
    fn test_center_crop_offset_rounds_half_to_even() {
        for (size, crop, want) in [
            (256, 224, 16),
            (224, 224, 0),
            (225, 224, 0),
            (226, 224, 1),
            (227, 224, 2),
            (229, 224, 2),
            (231, 224, 4),
            (233, 224, 4),
        ] {
            assert_eq!(center_crop_offset(size, crop), want, "{size} -> {crop}");
        }
    }

    /// The crop is the window at the offsets, every channel and batch row.
    #[test]
    fn test_center_crop_takes_the_offset_window() {
        let device = cpu_device();
        // [2, 2, 5, 4]: each value its own flat index.
        let images = Tensor::<1, Int>::arange(0..80, &device)
            .float()
            .reshape([2, 2, 5, 4]);
        let crop = center_crop(images.clone(), [2, 2]);
        // Height 5 -> 2: margin 3, m 1 (odd) -> top 2. Width 4 -> 2: left 1.
        let want = images.slice([0..2, 0..2, 2..4, 1..3]);
        crop.to_data().assert_eq(&want.to_data(), true);
    }

    #[test]
    fn test_try_center_crop_rejects_a_larger_crop() {
        let device = cpu_device();
        let images = Tensor::<4>::zeros([1, 3, 4, 4], &device);
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("5x4 center crop of 4x4")
            .assert_err(&try_center_crop(images, [5, 4]));
    }

    /// Padding puts the smaller half before, fills each channel with its own
    /// value, and keeps the pixels as they were.
    #[test]
    fn test_center_crop_or_pad_pads_with_the_fill() {
        let device = cpu_device();
        let images = Tensor::<4>::from_floats([[[[1.0, 2.0]], [[3.0, 4.0]]]], &device);
        // [1, 2, 1, 2] -> [1, 2, 3, 3]: height pads (1, 1), width (0, 1).
        let out = center_crop_or_pad(images, [3, 3], &[-1.0, -2.0]);
        let want = Tensor::<4>::from_floats(
            [[
                [[-1.0, -1.0, -1.0], [1.0, 2.0, -1.0], [-1.0, -1.0, -1.0]],
                [[-2.0, -2.0, -2.0], [3.0, 4.0, -2.0], [-2.0, -2.0, -2.0]],
            ]],
            &device,
        );
        out.to_data().assert_eq(&want.to_data(), true);
    }

    /// A side longer than the output is cropped while the other is padded.
    #[test]
    fn test_center_crop_or_pad_crops_and_pads() {
        let device = cpu_device();
        let images = Tensor::<1, Int>::arange(0..5, &device)
            .float()
            .reshape([1, 1, 1, 5]);
        let out = center_crop_or_pad(images, [2, 3], &[9.0]);
        // Height 1 -> 2 pads (0, 1); width 5 -> 3 crops at 1.
        let want = Tensor::<4>::from_floats([[[[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]]]], &device);
        out.to_data().assert_eq(&want.to_data(), true);
    }

    #[test]
    fn test_try_center_crop_or_pad_rejects_a_short_fill() {
        let device = cpu_device();
        let images = Tensor::<4>::zeros([1, 3, 2, 2], &device);
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("1 values for images of 3 channels")
            .assert_err(&try_center_crop_or_pad(images, [3, 3], &[0.0]));
    }
}
