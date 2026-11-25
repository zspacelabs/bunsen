//! Resizing images with `image`'s filters.

use image::{
    DynamicImage,
    imageops::FilterType,
};

/// Resizes `image` to exactly `size` (`[height, width]`) with `filter`,
/// ignoring its aspect ratio.
///
/// The filters are antialiased: each widens its support by the downscale
/// ratio, as `PIL`'s do.
pub fn resize_image_exact(
    image: &DynamicImage,
    size: [usize; 2],
    filter: FilterType,
) -> DynamicImage {
    let [height, width] = size;
    image.resize_exact(width as u32, height as u32, filter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_image_exact_takes_height_then_width() {
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(10, 20));
        let out = resize_image_exact(&image, [5, 8], FilterType::Triangle);
        assert_eq!((out.height(), out.width()), (5, 8));
    }
}
