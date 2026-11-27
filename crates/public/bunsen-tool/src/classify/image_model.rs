//! A pretrained image model by name, whatever its kit.

use bunsen::{
    data::{
        formats::LoadPolicy,
        resources::{
            LoadedResourceSet,
            ResourceCache,
        },
    },
    errors::BunsenResult,
    kits::images::{
        classification::{
            driver::ImageClassifier,
            pretrained::{
                image_kit_factories,
                image_kit_for,
            },
        },
        support::card::ImageModelCard,
    },
};
use burn::tensor::Device;

/// The checkpoint `classify` and `embed` load when none is named.
pub const DEFAULT_IMAGE_MODEL: &str = "timm/resnet50.a1_in1k";

/// A loaded image model: the classifier and its card.
pub struct LoadedImageModel {
    /// The model, as any image kit's classifier.
    pub model: Box<dyn ImageClassifier>,

    /// Its eval transform and labels.
    pub card: ImageModelCard,
}

/// The checkpoint `name` (a catalog name, or `hf:timm/<repo>`), loaded on
/// `device` under `policy`, through its kit's factory
/// ([`image_kit_for`]).
///
/// # Errors
/// When a kit's factory doesn't build; as the kit's
/// [`load`](bunsen::kits::images::classification::pretrained::ImageKitFactory::load).
pub fn load_image_model(
    name: &str,
    cache: &ResourceCache,
    device: &Device,
    policy: LoadPolicy,
) -> BunsenResult<LoadedImageModel> {
    let kits = image_kit_factories()?;
    let (model, card) = image_kit_for(&kits, name).load(name, cache, device, policy)?;
    Ok(LoadedImageModel { model, card })
}

/// The checkpoint `name`'s files, fetched into `cache` through its kit's
/// factory: its id, and the loaded set.
///
/// # Errors
/// When a kit's factory doesn't build; as resolving, planning and loading
/// the build.
pub fn fetch_image_model(
    name: &str,
    cache: &ResourceCache,
) -> BunsenResult<(String, LoadedResourceSet)> {
    let kits = image_kit_factories()?;
    image_kit_for(&kits, name).fetch(name, cache)
}
