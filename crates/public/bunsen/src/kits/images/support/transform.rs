//! An image model's eval transform: its settings and its size arithmetic.

use burn::{
    Tensor,
    config::Config,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    ops::images::{
        try_center_crop_or_pad,
        try_normalize,
    },
};

/// The `ImageNet` mean, per RGB channel: timm's `IMAGENET_DEFAULT_MEAN`.
pub const IMAGENET_DEFAULT_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// The `ImageNet` standard deviation, per RGB channel: timm's
/// `IMAGENET_DEFAULT_STD`.
pub const IMAGENET_DEFAULT_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The resampling filter an image is resized with, named as timm names it.
///
/// The filters are `PIL`'s: each is antialiased, its support widened by the
/// downscale ratio.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ImageInterpolation {
    /// Nearest neighbour.
    Nearest,
    /// A triangle (linear) filter.
    Bilinear,
    /// A cubic filter (`a = -0.5`, Catmull-Rom). timm's default.
    Bicubic,
    /// A 3-lobe Lanczos filter.
    Lanczos,
    /// A box filter.
    Box,
    /// A Hamming-windowed filter.
    Hamming,
}

impl ImageInterpolation {
    /// The name timm's `pretrained_cfg` gives the filter: `"bicubic"`.
    pub fn timm_name(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Bilinear => "bilinear",
            Self::Bicubic => "bicubic",
            Self::Lanczos => "lanczos",
            Self::Box => "box",
            Self::Hamming => "hamming",
        }
    }

    /// The filter timm's `pretrained_cfg` names, or `None` for a name it does
    /// not use.
    pub fn from_timm_name(name: &str) -> Option<Self> {
        [
            Self::Nearest,
            Self::Bilinear,
            Self::Bicubic,
            Self::Lanczos,
            Self::Box,
            Self::Hamming,
        ]
        .into_iter()
        .find(|f| f.timm_name() == name)
    }
}

/// How an eval transform fits an image to the model's size, as timm's
/// `crop_mode`.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ImageCropMode {
    /// Resize, keeping the aspect ratio, so the shorter side is the scale
    /// size; then crop the centre. Part of the image is lost. timm's default.
    Center,
    /// Resize each side to the scale size, ignoring the aspect ratio; then
    /// crop the centre.
    Squash,
    /// Resize, keeping the aspect ratio, so the longer side is the scale size;
    /// then pad to the crop with the mean colour, and crop the centre. No
    /// image is lost at a `crop_pct` of 1.
    Border,
}

/// An image model's eval transform, as timm's `transforms_imagenet_eval`
/// applies it: resize to the scale size, crop (or pad) to the input size,
/// scale to `[0, 1]`, normalise.
///
/// The settings are a timm `pretrained_cfg`'s (`input_size`, `interpolation`,
/// `crop_pct`, `crop_mode`, `mean`, `std`), and the defaults
/// (`ImageEvalTransformConfig::new()`) are those timm's `resolve_data_config`
/// falls back on: a `3x224x224` input, bicubic, a `crop_pct` of 0.875,
/// center crop, the `ImageNet` mean and std.
///
/// The transform is in two halves. Decoding and resizing need the `image`
/// crate, and are the `eval` module's (feature `images`), sized by
/// [`try_resize_size`](Self::try_resize_size). The rest works on the resized
/// `[batch, channels, height, width]` pixels in `[0, 1]`:
/// [`try_crop_and_normalize`](Self::try_crop_and_normalize).
///
/// ```rust
/// use bunsen::kits::images::support::transform::ImageEvalTransformConfig;
///
/// let transform = ImageEvalTransformConfig::new();
/// // 224 / 0.875 = 256: the shorter side goes to 256, then a 224 crop.
/// assert_eq!(transform.scale_size(), [256, 256]);
/// assert_eq!(transform.resize_size([480, 640]), [256, 341]);
/// assert_eq!(transform.crop_size(), [224, 224]);
/// ```
#[derive(Config, Debug, PartialEq)]
pub struct ImageEvalTransformConfig {
    /// The channel count the model takes.
    #[config(default = 3)]
    pub channels: usize,

    /// The model's input `[height, width]`: the crop.
    #[config(default = "[224, 224]")]
    pub size: [usize; 2],

    /// The resize filter.
    #[config(default = "ImageInterpolation::Bicubic")]
    pub interpolation: ImageInterpolation,

    /// The fraction of the scale size the crop keeps: the scale size is
    /// `floor(size / crop_pct)`.
    #[config(default = 0.875)]
    pub crop_pct: f64,

    /// How the image is fitted to the scale size.
    #[config(default = "ImageCropMode::Center")]
    pub crop_mode: ImageCropMode,

    /// The normalisation mean, per channel, of `[0, 1]` pixels.
    #[config(default = "IMAGENET_DEFAULT_MEAN.to_vec()")]
    pub mean: Vec<f32>,

    /// The normalisation standard deviation, per channel.
    #[config(default = "IMAGENET_DEFAULT_STD.to_vec()")]
    pub std: Vec<f32>,
}

impl ImageEvalTransformConfig {
    /// The size the image is resized toward, `[height, width]`: each side of
    /// [`size`](Self::size) over [`crop_pct`](Self::crop_pct), floored, as
    /// timm's `math.floor(x / crop_pct)`.
    pub fn scale_size(&self) -> [usize; 2] {
        self.size
            .map(|side| (side as f64 / self.crop_pct).floor() as usize)
    }

    /// The model's input `[height, width]`, the crop: [`size`](Self::size).
    pub fn crop_size(&self) -> [usize; 2] {
        self.size
    }

    /// The `[height, width]` a `[height, width]` image is resized to, as timm
    /// and torchvision compute it.
    ///
    /// - [`Center`](ImageCropMode::Center) with a square scale size: the
    ///   shorter side to the scale size, the longer to `int(scale * long /
    ///   short)` (torchvision's `Resize(int)`).
    /// - [`Center`](ImageCropMode::Center) with a non-square scale size: both
    ///   sides over the smaller of the two ratios, rounded half to even (timm's
    ///   `ResizeKeepRatio`).
    /// - [`Squash`](ImageCropMode::Squash): the scale size.
    /// - [`Border`](ImageCropMode::Border): both sides over the larger ratio,
    ///   rounded half to even.
    ///
    /// See the [errors convention](crate::errors#convention-try_x-and-x);
    /// [`resize_size`](Self::resize_size) is the panicking twin.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for an image with
    /// an empty side, or a `crop_pct` that is not in `(0, 1]`.
    pub fn try_resize_size(
        &self,
        image: [usize; 2],
    ) -> BunsenResult<[usize; 2]> {
        self.try_validate()?;
        let [height, width] = image;
        if height == 0 || width == 0 {
            return Err(BunsenError::illegal(format!(
                "resizing a {height}x{width} image"
            )));
        }
        let scale = self.scale_size();
        let keep_ratio = |longest: f64| {
            let ratio_h = height as f64 / scale[0] as f64;
            let ratio_w = width as f64 / scale[1] as f64;
            let ratio = ratio_h.max(ratio_w) * longest + ratio_h.min(ratio_w) * (1.0 - longest);
            [height, width].map(|side| (side as f64 / ratio).round_ties_even() as usize)
        };
        Ok(match self.crop_mode {
            ImageCropMode::Center if scale[0] == scale[1] => {
                let s = scale[0];
                let (short, long) = (height.min(width), height.max(width));
                let new_long = ((s * long) as f64 / short as f64) as usize;
                if width <= height {
                    [new_long, s]
                } else {
                    [s, new_long]
                }
            }
            ImageCropMode::Center => keep_ratio(0.0),
            ImageCropMode::Squash => scale,
            ImageCropMode::Border => keep_ratio(1.0),
        })
    }

    /// The `[height, width]` a `[height, width]` image is resized to.
    ///
    /// The panicking twin of [`try_resize_size`](Self::try_resize_size).
    ///
    /// # Panics
    /// With [`try_resize_size`](Self::try_resize_size)'s error report.
    pub fn resize_size(
        &self,
        image: [usize; 2],
    ) -> [usize; 2] {
        self.try_resize_size(image)
            .unwrap_or_else(|e| panic!("{e:#}"))
    }

    /// The padding colour of [`Border`](ImageCropMode::Border) mode, per
    /// channel, in `[0, 1]` pixels: timm's `round(255 * mean)`, as an 8-bit
    /// value scaled back.
    pub fn border_fill(&self) -> Vec<f32> {
        self.mean
            .iter()
            .map(|m| (255.0 * *m as f64).round_ties_even() as f32 / 255.0)
            .collect()
    }

    /// The second half of the transform: crops the resized images to
    /// [`size`](Self::size) at torchvision's offsets (padding with 0 where a
    /// side is short, or with the [`border_fill`](Self::border_fill) in
    /// [`Border`](ImageCropMode::Border) mode), then normalises them.
    ///
    /// See the [errors convention](crate::errors#convention-try_x-and-x);
    /// [`crop_and_normalize`](Self::crop_and_normalize) is the panicking twin.
    ///
    /// # Arguments
    /// - `resized`: a `[batch, channels, height, width]` batch of `[0, 1]`
    ///   pixels, each image resized to [`resize_size`](Self::resize_size) of
    ///   its own.
    ///
    /// # Returns
    /// The `[batch, channels, size_h, size_w]` model input.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when the images'
    /// channels are not [`channels`](Self::channels), or the mean or std do
    /// not hold one value per channel.
    pub fn try_crop_and_normalize(
        &self,
        resized: Tensor<4>,
    ) -> BunsenResult<Tensor<4>> {
        self.try_validate()?;
        let [_batch, channels, _height, _width] = crate::contracts::unpack_shape_contract!(
            ["batch", "channels", "height", "width"],
            &resized,
            &["batch", "channels", "height", "width"],
        );
        if channels != self.channels {
            return Err(BunsenError::illegal(format!(
                "{channels}-channel images for a {}-channel transform",
                self.channels
            )));
        }
        let fill = match self.crop_mode {
            ImageCropMode::Border => self.border_fill(),
            ImageCropMode::Center | ImageCropMode::Squash => vec![0.0; channels],
        };
        let cropped = try_center_crop_or_pad(resized, self.size, &fill)?;
        try_normalize(cropped, &self.mean, &self.std)
    }

    /// The second half of the transform: crop, then normalise.
    ///
    /// The panicking twin of
    /// [`try_crop_and_normalize`](Self::try_crop_and_normalize).
    ///
    /// # Panics
    /// With [`try_crop_and_normalize`](Self::try_crop_and_normalize)'s error
    /// report.
    pub fn crop_and_normalize(
        &self,
        resized: Tensor<4>,
    ) -> Tensor<4> {
        self.try_crop_and_normalize(resized)
            .unwrap_or_else(|e| panic!("{e:#}"))
    }

    /// Checks the settings that every use reads.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a `crop_pct`
    /// not in `(0, 1]`, an empty size, or a mean or std that does not hold
    /// one value per channel.
    pub fn try_validate(&self) -> BunsenResult<()> {
        if !(self.crop_pct > 0.0 && self.crop_pct <= 1.0) {
            return Err(BunsenError::illegal(format!(
                "an eval transform's crop_pct of {}",
                self.crop_pct
            )));
        }
        if self.size.contains(&0) {
            return Err(BunsenError::illegal(format!(
                "an eval transform of size {:?}",
                self.size
            )));
        }
        if self.mean.len() != self.channels || self.std.len() != self.channels {
            return Err(BunsenError::illegal(format!(
                "an eval transform of {} channels with a mean of {} and a std of {} values",
                self.channels,
                self.mean.len(),
                self.std.len()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::cpu_device,
    };

    /// The defaults are `resolve_data_config`'s fallbacks.
    #[test]
    fn test_defaults_are_timms() {
        let t = ImageEvalTransformConfig::new();
        assert_eq!(t.channels, 3);
        assert_eq!(t.size, [224, 224]);
        assert_eq!(t.interpolation, ImageInterpolation::Bicubic);
        assert_eq!(t.crop_pct, 0.875);
        assert_eq!(t.crop_mode, ImageCropMode::Center);
        assert_eq!(t.mean, IMAGENET_DEFAULT_MEAN.to_vec());
        assert_eq!(t.std, IMAGENET_DEFAULT_STD.to_vec());
    }

    /// The scale size floors: `math.floor(x / crop_pct)`.
    #[test]
    fn test_scale_size_floors() {
        for (size, crop_pct, want) in [
            (224, 0.875, 256),
            (224, 0.95, 235),
            (224, 0.9, 248),
            (288, 1.0, 288),
            (160, 0.95, 168),
            (384, 1.0, 384),
        ] {
            let t = ImageEvalTransformConfig::new()
                .with_size([size, size])
                .with_crop_pct(crop_pct);
            assert_eq!(t.scale_size(), [want, want], "{size} at {crop_pct}");
        }
    }

    /// torchvision's `Resize(int)`: the shorter side to the scale, the
    /// longer to `int(s * long / short)`, either orientation.
    #[test]
    fn test_center_square_resizes_the_shorter_side() {
        let t = ImageEvalTransformConfig::new();
        // 256 * 640 / 480 = 341.33.. -> 341.
        assert_eq!(t.resize_size([480, 640]), [256, 341]);
        assert_eq!(t.resize_size([640, 480]), [341, 256]);
        assert_eq!(t.resize_size([300, 300]), [256, 256]);
        // 256 * 1001 / 500 = 512.512 -> 512.
        assert_eq!(t.resize_size([500, 1001]), [256, 512]);
    }

    /// A non-square scale size takes timm's `ResizeKeepRatio(longest=0)`:
    /// both sides over the smaller ratio, rounded half to even.
    #[test]
    fn test_center_non_square_keeps_the_ratio() {
        let t = ImageEvalTransformConfig::new()
            .with_size([200, 100])
            .with_crop_pct(1.0);
        // ratios 400/200 = 2, 400/100 = 4: over 2 -> [200, 200].
        assert_eq!(t.resize_size([400, 400]), [200, 200]);
        // ratios 3/200, 5/100: min 0.015 -> 3/0.015 = 200, 5/0.015 = 333.3.
        assert_eq!(t.resize_size([3, 5]), [200, 333]);
    }

    #[test]
    fn test_squash_ignores_the_ratio() {
        let t = ImageEvalTransformConfig::new().with_crop_mode(ImageCropMode::Squash);
        assert_eq!(t.resize_size([480, 640]), [256, 256]);
    }

    /// Border fits the longer side: both sides over the larger ratio.
    #[test]
    fn test_border_fits_the_longer_side() {
        let t = ImageEvalTransformConfig::new()
            .with_crop_mode(ImageCropMode::Border)
            .with_crop_pct(1.0);
        // ratios 480/224, 640/224: max 2.857.. -> 168, 224.
        assert_eq!(t.resize_size([480, 640]), [168, 224]);
    }

    /// The border fill is the mean as an 8-bit colour.
    #[test]
    fn test_border_fill_rounds_the_mean_to_8_bits() {
        let t = ImageEvalTransformConfig::new();
        // round(255 * [0.485, 0.456, 0.406]) = [124, 116, 104].
        assert_eq!(
            t.border_fill(),
            vec![124.0 / 255.0, 116.0 / 255.0, 104.0 / 255.0]
        );
    }

    /// Crop then normalise, at the right window.
    #[test]
    fn test_crop_and_normalize() {
        let device = cpu_device();
        let t = ImageEvalTransformConfig::new()
            .with_channels(1)
            .with_size([2, 2])
            .with_mean(vec![0.5])
            .with_std(vec![0.25]);
        let resized = Tensor::<4>::from_floats(
            [[[
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.75, 0.0],
                [0.0, 0.5, 0.25, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]]],
            &device,
        );
        let out = t.crop_and_normalize(resized);
        let want = Tensor::<4>::from_floats([[[[2.0, 1.0], [0.0, -1.0]]]], &device);
        out.to_data().assert_eq(&want.to_data(), true);
    }

    #[test]
    fn test_try_crop_and_normalize_rejects_the_wrong_channels() {
        let device = cpu_device();
        let t = ImageEvalTransformConfig::new();
        ErrorMatcher::illegal()
            .message_contains("1-channel images for a 3-channel transform")
            .assert_err(&t.try_crop_and_normalize(Tensor::<4>::zeros([1, 1, 256, 256], &device)));
    }

    #[test]
    fn test_try_validate_rejects_a_bad_crop_pct() {
        for crop_pct in [0.0, -0.5, 1.5, f64::NAN] {
            ErrorMatcher::illegal()
                .message_contains("crop_pct")
                .assert_err(
                    &ImageEvalTransformConfig::new()
                        .with_crop_pct(crop_pct)
                        .try_validate(),
                );
        }
    }

    #[test]
    fn test_interpolation_timm_names_round_trip() {
        for f in [
            ImageInterpolation::Nearest,
            ImageInterpolation::Bilinear,
            ImageInterpolation::Bicubic,
            ImageInterpolation::Lanczos,
            ImageInterpolation::Box,
            ImageInterpolation::Hamming,
        ] {
            assert_eq!(ImageInterpolation::from_timm_name(f.timm_name()), Some(f));
        }
        assert_eq!(ImageInterpolation::from_timm_name("random"), None);
    }
}
