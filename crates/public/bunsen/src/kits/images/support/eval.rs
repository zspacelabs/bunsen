//! # The eval transform, from an image file to a model's input
//!
//! The half of an [`ImageEvalTransformConfig`] that needs the `image` crate
//! (feature `images`), on [`support::images`](crate::support::images): the
//! `image` filter for timm's names ([`try_filter_for`]), a resize by them
//! ([`try_resize_image`]), and the whole transform ([`try_eval_input`],
//! [`try_load_eval_batch`]).

use std::path::Path;

use burn::{
    Tensor,
    prelude::Device,
};
use image::{
    DynamicImage,
    imageops::FilterType,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
    },
    kits::images::support::transform::{
        ImageEvalTransformConfig,
        ImageInterpolation,
    },
    support::images::{
        image_to_f32_tensor,
        resize_image_exact,
        try_open_image,
    },
};

/// The `image` filter that computes timm's (`PIL`'s) `interpolation`.
///
/// See the [errors convention](crate::errors#convention-try_x-and-x).
///
/// | timm | `image` |
/// |---|---|
/// | `nearest` | `Nearest` |
/// | `bilinear` | `Triangle` |
/// | `bicubic` | `CatmullRom` (the `a = -0.5` cubic) |
/// | `lanczos` | `Lanczos3` |
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for `box` and
/// `hamming`, which `image` does not have.
pub fn try_filter_for(interpolation: ImageInterpolation) -> BunsenResult<FilterType> {
    match interpolation {
        ImageInterpolation::Nearest => Ok(FilterType::Nearest),
        ImageInterpolation::Bilinear => Ok(FilterType::Triangle),
        ImageInterpolation::Bicubic => Ok(FilterType::CatmullRom),
        ImageInterpolation::Lanczos => Ok(FilterType::Lanczos3),
        ImageInterpolation::Box | ImageInterpolation::Hamming => Err(BunsenError::unsupported(
            format!("resizing with the {} filter", interpolation.timm_name()),
        )),
    }
}

/// Resizes `image` to exactly `size` (`[height, width]`) with timm's
/// `interpolation`; an image already that size is returned as it is, as
/// torchvision's `Resize` returns it.
///
/// See the [errors convention](crate::errors#convention-try_x-and-x).
///
/// # Errors
/// [`try_filter_for`]'s, for a filter `image` does not have.
pub fn try_resize_image(
    image: &DynamicImage,
    size: [usize; 2],
    interpolation: ImageInterpolation,
) -> BunsenResult<DynamicImage> {
    if [image.height() as usize, image.width() as usize] == size {
        return Ok(image.clone());
    }
    Ok(resize_image_exact(
        image,
        size,
        try_filter_for(interpolation)?,
    ))
}

/// An image as a model's input, through its eval `transform`, as timm's
/// `transforms_imagenet_eval` takes a `PIL` image converted to RGB: resize to
/// [`resize_size`](ImageEvalTransformConfig::resize_size) with the
/// transform's filter, scale to `[0, 1]`, crop (or pad), normalise.
///
/// See the [errors convention](crate::errors#convention-try_x-and-x).
///
/// # Returns
/// The `[channels, height, width]` input, `height` and `width` the
/// transform's [`size`](ImageEvalTransformConfig::size).
///
/// # Errors
/// - [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a transform that
///   does not validate, or an empty image;
/// - [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
///   transform of other than 3 channels (an image is converted to RGB, as timm
///   converts it), or a filter `image` does not have.
pub fn try_eval_input(
    transform: &ImageEvalTransformConfig,
    image: &DynamicImage,
    device: &Device,
) -> BunsenResult<Tensor<3>> {
    transform.try_validate()?;
    if transform.channels != 3 {
        return Err(BunsenError::unsupported(format!(
            "an eval transform of {} channels: images are read as RGB",
            transform.channels
        )));
    }
    let rgb = DynamicImage::ImageRgb8(image.to_rgb8());
    let size = transform.try_resize_size([rgb.height() as usize, rgb.width() as usize])?;
    let resized = try_resize_image(&rgb, size, transform.interpolation)?;
    let chw = image_to_f32_tensor(&resized, device)
        .permute([2, 0, 1])
        .unsqueeze::<4>();
    Ok(transform.try_crop_and_normalize(chw)?.squeeze_dim::<3>(0))
}

/// The image files at `paths` as one batch of a model's input, each through
/// [`try_eval_input`].
///
/// See the [errors convention](crate::errors#convention-try_x-and-x).
///
/// # Returns
/// The `[paths, channels, height, width]` batch.
///
/// # Errors
/// [`try_open_image`]'s for a file, with its path in a frame, and
/// [`try_eval_input`]'s; [`Illegal`](crate::errors::BunsenErrorKind::Illegal)
/// for no paths.
pub fn try_load_eval_batch(
    transform: &ImageEvalTransformConfig,
    paths: &[impl AsRef<Path>],
    device: &Device,
) -> BunsenResult<Tensor<4>> {
    if paths.is_empty() {
        return Err(ConstraintError::zero_or_empty("try_load_eval_batch", "paths").into());
    }
    let inputs = paths
        .iter()
        .map(|path| {
            let path = path.as_ref();
            try_open_image(path)
                .and_then(|image| try_eval_input(transform, &image, device))
                .map_err(|e| e.context(format!("the eval input of {}", path.display())))
        })
        .collect::<BunsenResult<Vec<_>>>()?;
    Ok(Tensor::stack(inputs, 0))
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::cpu_device,
    };

    /// A pixel's value as torchvision computes it: `u8 / 255`, then
    /// `(x - mean) / std`, in f32.
    fn normalized(
        v: u8,
        mean: f32,
        std: f32,
    ) -> f32 {
        (v as f32 / 255.0 - mean) / std
    }

    /// At the scale size already, the image is not resized, and the input is
    /// the centre crop's pixels, normalised exactly.
    #[test]
    fn test_try_eval_input_crops_and_normalises_exactly() {
        let device = cpu_device();
        let transform = ImageEvalTransformConfig::new()
            .with_size([4, 4])
            .with_crop_pct(0.5);
        // scale 8x8, crop 4x4 at offset 2.
        let pixel = |x: u32, y: u32| image::Rgb([(x * 30) as u8, (y * 30) as u8, (x + y) as u8]);
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_fn(8, 8, pixel));
        let input = try_eval_input(&transform, &image, &device).unwrap();

        let mut want = Vec::new();
        for c in 0..3 {
            for y in 2..6 {
                for x in 2..6 {
                    want.push(normalized(
                        pixel(x, y).0[c],
                        transform.mean[c],
                        transform.std[c],
                    ));
                }
            }
        }
        input
            .to_data()
            .assert_eq(&TensorData::new(want, [3, 4, 4]), true);
    }

    /// A larger image is resized (shorter side to the scale size) and
    /// cropped to the transform's size; grey stays grey.
    #[test]
    fn test_try_eval_input_resizes_to_the_input_size() {
        let device = cpu_device();
        let transform = ImageEvalTransformConfig::new();
        let image =
            DynamicImage::ImageLuma8(image::GrayImage::from_pixel(300, 200, image::Luma([128])));
        let input = try_eval_input(&transform, &image, &device).unwrap();
        assert_eq!(input.dims(), [3, 224, 224]);
        let data = input.to_data();
        let values = data.as_slice::<f32>().unwrap();
        for c in 0..3 {
            let want = normalized(128, transform.mean[c], transform.std[c]);
            let plane = &values[c * 224 * 224..(c + 1) * 224 * 224];
            assert!(plane.iter().all(|v| (v - want).abs() < 1e-6), "channel {c}");
        }
    }

    #[test]
    fn test_try_eval_input_rejects_a_non_rgb_transform() {
        let device = cpu_device();
        let transform = ImageEvalTransformConfig::new()
            .with_channels(1)
            .with_mean(vec![0.5])
            .with_std(vec![0.5]);
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));
        ErrorMatcher::unsupported()
            .message_contains("1 channels")
            .assert_err(&try_eval_input(&transform, &image, &device));
    }

    /// An image at the size already is not filtered.
    #[test]
    fn test_try_resize_image_keeps_an_image_at_the_size() {
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_fn(4, 3, |x, y| {
            image::Rgb([(x * 60) as u8, (y * 80) as u8, 9])
        }));
        let out = try_resize_image(&image, [3, 4], ImageInterpolation::Bicubic).unwrap();
        assert_eq!(out, image);
    }

    #[test]
    fn test_try_filter_for_rejects_box_and_hamming() {
        for f in [ImageInterpolation::Box, ImageInterpolation::Hamming] {
            ErrorMatcher::unsupported()
                .message_contains(f.timm_name())
                .assert_err(&try_filter_for(f));
        }
    }

    /// The open test photos, each a different size and aspect, through the
    /// default transform: one `[6, 3, 224, 224]` batch of finite values.
    #[test]
    fn test_the_test_images_transform() {
        let device = cpu_device();
        let paths: Vec<_> = bunsen_test_images::ALL.iter().map(|i| i.path()).collect();
        let batch = try_load_eval_batch(&ImageEvalTransformConfig::new(), &paths, &device).unwrap();
        assert_eq!(batch.dims(), [6, 3, 224, 224]);
        let values = batch
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
    }

    /// Files to a batch; a bad file names its path.
    #[test]
    fn test_try_load_eval_batch() {
        let device = cpu_device();
        let temp_dir = tempfile::tempdir().unwrap();
        let paths: Vec<_> = (0..2)
            .map(|i| {
                let path = temp_dir.path().join(format!("{i}.png"));
                image::RgbImage::from_pixel(10 + i, 12, image::Rgb([i as u8, 0, 0]))
                    .save(&path)
                    .unwrap();
                path
            })
            .collect();
        let transform = ImageEvalTransformConfig::new().with_size([8, 8]);
        let batch = try_load_eval_batch(&transform, &paths, &device).unwrap();
        assert_eq!(batch.dims(), [2, 3, 8, 8]);

        let missing = [temp_dir.path().join("missing.png")];
        ErrorMatcher::lookup()
            .frame_contains("missing.png")
            .assert_err(&try_load_eval_batch(&transform, &missing, &device));
    }
}
