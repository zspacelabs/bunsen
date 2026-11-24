//! Image files through a classifier, to their top classes.

use std::path::Path;

use burn::prelude::Device;

use super::{
    ClassPrediction,
    ImageClassifier,
    top_k,
};
use crate::{
    errors::BunsenResult,
    kits::images::support::{
        card::ImageModelCard,
        eval::try_load_eval_batch,
    },
};

/// The top `k` classes of each image file at `paths`, through `card`'s eval
/// transform and `model`'s head, named by the card's labels.
///
/// See the [errors convention](crate::errors#convention-try_x-and-x).
///
/// # Returns
/// One list per path, in order, most probable class first.
///
/// # Errors
/// [`try_load_eval_batch`]'s, for a file that does not open or transform.
///
/// # Panics
/// When the card's labels do not have the model's class count.
pub fn classify_files(
    model: &dyn ImageClassifier,
    card: &ImageModelCard,
    paths: &[impl AsRef<Path>],
    k: usize,
    device: &Device,
) -> BunsenResult<Vec<Vec<ClassPrediction>>> {
    let images = try_load_eval_batch(&card.transform, paths, device)?;
    Ok(top_k(model.forward_logits(images), k, card.labels.as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::images::{
            classification::resnet::{
                ResNet,
                ResNetContractConfig,
            },
            support::{
                labels::ImageLabelSet,
                transform::ImageEvalTransformConfig,
            },
        },
        support::testing::cpu_device,
    };

    /// Files through the card's transform and the model's head, to `k`
    /// classes each, named by the card's labels.
    #[test]
    fn test_classify_files() {
        let device = cpu_device();
        let model: ResNet = ResNetContractConfig::new(vec![1], 3)
            .with_stem_width(8)
            .init(&device);
        let card = ImageModelCard::new(ImageEvalTransformConfig::new().with_size([32, 32]))
            .with_labels(Some(ImageLabelSet::try_parse("n0\nn1\nn2", None).unwrap()));
        let dir = tempfile::tempdir().unwrap();
        let paths: Vec<_> = (0..2)
            .map(|i| {
                let path = dir.path().join(format!("{i}.png"));
                image::RgbImage::from_pixel(40, 48, image::Rgb([i * 100, 50, 25]))
                    .save(&path)
                    .unwrap();
                path
            })
            .collect();
        let top = classify_files(&model, &card, &paths, 2, &device).unwrap();
        assert_eq!(top.len(), 2);
        for row in &top {
            assert_eq!(row.len(), 2);
            assert!(row[0].probability >= row[1].probability);
            assert_eq!(
                row[0].name.as_deref(),
                Some(row[0].synset.as_deref().unwrap())
            );
        }
    }
}
