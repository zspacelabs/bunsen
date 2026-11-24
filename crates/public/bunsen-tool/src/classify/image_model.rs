//! A pretrained image model by name, whatever its kit.

use std::sync::Arc;

use bunsen::{
    data::{
        builds::BuildFactory,
        formats::LoadPolicy,
        resources::ResourceCache,
    },
    errors::BunsenResult,
    kits::images::{
        classification::{
            driver::ImageClassifier,
            resnet::pretrained::{
                ResNetConstructConfig,
                default_resnet_factory,
            },
        },
        support::card::ImageModelCard,
    },
};
use burn::tensor::Device;

/// The checkpoint `classify` and `embed` load when none is named.
pub const DEFAULT_IMAGE_MODEL: &str = "timm/resnet50.a1_in1k";

/// The image kits' factories `classify` and `embed` load from: `ResNet`'s.
pub fn image_factory() -> BunsenResult<BuildFactory<ResNetConstructConfig>> {
    default_resnet_factory()
}

/// A loaded image model: the classifier and its card.
pub struct LoadedImageModel {
    /// The model, as any image kit's classifier.
    pub model: Box<dyn ImageClassifier>,

    /// Its eval transform and labels.
    pub card: ImageModelCard,
}

/// The checkpoint `name` (a catalog name, or `hf:timm/<repo>`), loaded on
/// `device` under `policy`.
///
/// # Errors
/// As [`BuildFactory::load_construct`].
pub fn load_image_model(
    name: &str,
    cache: &ResourceCache,
    device: &Device,
    policy: LoadPolicy,
) -> BunsenResult<LoadedImageModel> {
    let construct =
        Arc::unwrap_or_clone(image_factory()?.load_construct(name, cache, device, policy)?);
    Ok(LoadedImageModel {
        model: Box::new(construct.model),
        card: construct.card,
    })
}
