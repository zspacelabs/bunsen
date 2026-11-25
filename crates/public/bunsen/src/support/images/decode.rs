//! Decoding image files, and their pixels as a tensor.

use std::path::Path;

use burn::{
    Tensor,
    prelude::Device,
    tensor::{
        DType,
        TensorCreationOptions,
        TensorData,
    },
};
use image::DynamicImage;

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    sys_at,
};

/// Opens and decodes the image file at `path`, in whatever format its bytes
/// are (JPEG, PNG and GIF are built in).
///
/// See the [errors convention](crate::errors#convention-try_x-and-x).
///
/// # Errors
/// - [`Lookup`](BunsenErrorKind::Lookup) (or another
///   [`Sys`](BunsenErrorKind::Sys)-class kind, by the `io::Error`) when the
///   file cannot be read;
/// - [`Unsupported`](BunsenErrorKind::Unsupported) for a format not built in;
/// - [`InvalidResource`](BunsenErrorKind::InvalidResource) for bytes that do
///   not decode. The cause is the `image::ImageError`.
pub fn try_open_image(path: impl AsRef<Path>) -> BunsenResult<DynamicImage> {
    let path = path.as_ref();
    image::open(path).map_err(|e| open_error(path, e))
}

/// The error of opening the image at `path`, by what `image` reports.
fn open_error(
    path: &Path,
    error: image::ImageError,
) -> BunsenError {
    match error {
        image::ImageError::IoError(e) => sys_at("open image", path)(e),
        image::ImageError::Unsupported(_) => {
            BunsenError::from_cause(BunsenErrorKind::Unsupported, error)
                .context(format!("opening image {}", path.display()))
        }
        _ => BunsenError::from_cause(BunsenErrorKind::InvalidResource, error)
            .context(format!("opening image {}", path.display())),
    }
}

/// An image's pixels as an `f32` `[height, width, channels]` tensor, in the
/// image's own channels: 8-bit values over 255, 16-bit values over 65535,
/// float values as they are.
///
/// Each value is one division (`v as f32 / 255.0`), as torchvision's
/// `ToTensor` computes it.
pub fn image_to_f32_tensor(
    image: &DynamicImage,
    device: &Device,
) -> Tensor<3> {
    let shape = [
        image.height() as usize,
        image.width() as usize,
        image.color().channel_count() as usize,
    ];
    let over = |raw: &[u16]| raw.iter().map(|&v| v as f32 / 65535.0).collect();
    let data: Vec<f32> = match image {
        DynamicImage::ImageLuma8(_)
        | DynamicImage::ImageLumaA8(_)
        | DynamicImage::ImageRgb8(_)
        | DynamicImage::ImageRgba8(_) => {
            image.as_bytes().iter().map(|&v| v as f32 / 255.0).collect()
        }
        DynamicImage::ImageLuma16(b) => over(b.as_raw()),
        DynamicImage::ImageLumaA16(b) => over(b.as_raw()),
        DynamicImage::ImageRgb16(b) => over(b.as_raw()),
        DynamicImage::ImageRgba16(b) => over(b.as_raw()),
        DynamicImage::ImageRgb32F(b) => b.as_raw().clone(),
        DynamicImage::ImageRgba32F(b) => b.as_raw().clone(),
        // A layout `image` adds later: its float RGBA, which keeps every
        // value of the narrower layouts.
        other => {
            let rgba = other.to_rgba32f();
            return Tensor::from_data(
                TensorData::new(rgba.into_raw(), [shape[0], shape[1], 4]),
                TensorCreationOptions::new(device.clone()).with_dtype(DType::F32),
            );
        }
    };
    Tensor::from_data(
        TensorData::new(data, shape),
        TensorCreationOptions::new(device.clone()).with_dtype(DType::F32),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::cpu_device,
    };

    #[test]
    fn test_open_error_kinds() {
        let temp_dir = tempfile::tempdir().unwrap();

        let missing = temp_dir.path().join("missing.png");
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .has_cause::<std::io::Error>()
            .assert_err(&try_open_image(&missing));

        let junk = temp_dir.path().join("junk.png");
        std::fs::write(&junk, b"not a png").unwrap();
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .frame_contains("opening image")
            .has_cause::<image::ImageError>()
            .assert_err(&try_open_image(&junk));
    }

    /// A saved PNG opens to the same pixels.
    #[test]
    fn test_try_open_image_round_trips_a_png() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("pixels.png");
        let source = image::RgbImage::from_fn(3, 2, |x, y| image::Rgb([x as u8, y as u8, 7]));
        source.save(&path).unwrap();
        let opened = try_open_image(&path).unwrap();
        assert_eq!(opened.to_rgb8(), source);
    }

    /// 8-bit values over 255, 16-bit over 65535, `[height, width, channels]`.
    #[test]
    fn test_image_to_f32_tensor_scales_by_depth() {
        let device = cpu_device();

        let rgb = DynamicImage::ImageRgb8(image::RgbImage::from_fn(2, 1, |x, _| {
            image::Rgb([x as u8 * 255, 51, 0])
        }));
        image_to_f32_tensor(&rgb, &device).to_data().assert_eq(
            &TensorData::new(
                vec![0.0f32, 51.0 / 255.0, 0.0, 1.0, 51.0 / 255.0, 0.0],
                [1, 2, 3],
            ),
            true,
        );

        let luma16 = DynamicImage::ImageLuma16(
            image::ImageBuffer::from_raw(1, 2, vec![0u16, 65535]).unwrap(),
        );
        image_to_f32_tensor(&luma16, &device)
            .to_data()
            .assert_eq(&TensorData::new(vec![0.0f32, 1.0], [2, 1, 1]), true);
    }
}
