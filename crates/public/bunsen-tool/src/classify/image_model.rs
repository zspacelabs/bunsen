//! A pretrained image model by name, whatever its kit.

use std::sync::Arc;

use bunsen::{
    data::{
        builds::{
            BuildFactory,
            BuildProvider,
            ConstructInit,
        },
        catalog::ResolvedBuild,
        formats::LoadPolicy,
        pretrained::Prefabs,
        resources::{
            LoadedResourceSet,
            ResourceCache,
        },
    },
    errors::BunsenResult,
    kits::images::{
        classification::{
            driver::ImageClassifier,
            resnet::{
                ResNetContractConfig,
                pretrained::default_resnet_factory,
            },
            vit::{
                ViTContractConfig,
                pretrained::default_vit_factory,
            },
        },
        support::card::ImageModelCard,
    },
};
use burn::tensor::Device;

/// The checkpoint `classify` and `embed` load when none is named.
pub const DEFAULT_IMAGE_MODEL: &str = "timm/resnet50.a1_in1k";

/// The image kits `classify` and `embed` load from, each by its factory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFamily {
    /// `kits::images::classification::resnet`: the `ResNet`s.
    ResNet,

    /// `kits::images::classification::vit`: the `ViT`s, and the CLIP image
    /// towers.
    ViT,
}

impl ImageFamily {
    /// Every family, in the order `list` and the summary show them.
    pub const ALL: [Self; 2] = [Self::ResNet, Self::ViT];

    /// The family's kit name.
    pub fn name(self) -> &'static str {
        match self {
            Self::ResNet => "resnet",
            Self::ViT => "vit",
        }
    }

    /// The family's architectures, by timm's names: its prefabs'.
    fn architectures(self) -> Vec<&'static str> {
        match self {
            Self::ResNet => ResNetContractConfig::prefab_names(),
            Self::ViT => ViTContractConfig::prefab_names(),
        }
    }

    /// The family a checkpoint name loads through: the one whose catalog
    /// names it; else, for a hub repo (`hf:timm/vit_base_patch16_224.x`),
    /// the one with a prefab of the repo's architecture; else `ResNet`, whose
    /// factory then says what it is missing.
    ///
    /// # Errors
    /// When a family's factory doesn't build.
    pub fn of(name: &str) -> BunsenResult<Self> {
        for family in Self::ALL {
            if family.lookup_provider(name)?.as_deref() == Some("catalog") {
                return Ok(family);
            }
        }
        let repo = name.rsplit('/').next().unwrap_or(name);
        let architecture = repo.split('.').next().unwrap_or(repo);
        Ok(Self::ALL
            .into_iter()
            .find(|family| family.architectures().contains(&architecture))
            .unwrap_or(Self::ResNet))
    }

    /// The provider of the family's factory that names `name`, if one does.
    fn lookup_provider(
        self,
        name: &str,
    ) -> BunsenResult<Option<String>> {
        fn provider<C: ConstructInit>(
            factory: BuildFactory<C>,
            name: &str,
        ) -> Option<String> {
            factory.lookup(name).ok().map(|(provider, _)| provider)
        }
        Ok(match self {
            Self::ResNet => provider(default_resnet_factory()?, name),
            Self::ViT => provider(default_vit_factory()?, name),
        })
    }

    /// The family factory's providers, and its rows by id, in order.
    ///
    /// # Errors
    /// When the factory doesn't build, or a listed id doesn't look up.
    pub fn rows(self) -> BunsenResult<ImageFamilyRows> {
        fn rows<C: ConstructInit>(factory: BuildFactory<C>) -> BunsenResult<ImageFamilyRows> {
            let rows = factory
                .ids()
                .into_iter()
                .map(|id| Ok((id.clone(), factory.lookup(&id)?.1)))
                .collect::<BunsenResult<_>>()?;
            Ok(ImageFamilyRows {
                providers: factory.providers().to_vec(),
                rows,
            })
        }
        match self {
            Self::ResNet => rows(default_resnet_factory()?),
            Self::ViT => rows(default_vit_factory()?),
        }
    }
}

/// A family factory's providers and its rows, as `classify list` prints
/// them.
pub struct ImageFamilyRows {
    /// The providers, in search order.
    pub providers: Vec<Arc<dyn BuildProvider>>,

    /// The rows, by id, in order.
    pub rows: Vec<(String, ResolvedBuild)>,
}

/// A loaded image model: the classifier and its card.
pub struct LoadedImageModel {
    /// The model, as any image kit's classifier.
    pub model: Box<dyn ImageClassifier>,

    /// Its eval transform and labels.
    pub card: ImageModelCard,
}

/// The checkpoint `name` (a catalog name, or `hf:timm/<repo>`), loaded on
/// `device` under `policy`, through its family's factory
/// ([`ImageFamily::of`]).
///
/// # Errors
/// As [`ImageFamily::of`] and [`BuildFactory::load_construct`].
pub fn load_image_model(
    name: &str,
    cache: &ResourceCache,
    device: &Device,
    policy: LoadPolicy,
) -> BunsenResult<LoadedImageModel> {
    Ok(match ImageFamily::of(name)? {
        ImageFamily::ResNet => {
            let construct = Arc::unwrap_or_clone(
                default_resnet_factory()?.load_construct(name, cache, device, policy)?,
            );
            LoadedImageModel {
                model: Box::new(construct.model),
                card: construct.card,
            }
        }
        ImageFamily::ViT => {
            let construct = Arc::unwrap_or_clone(
                default_vit_factory()?.load_construct(name, cache, device, policy)?,
            );
            LoadedImageModel {
                model: Box::new(construct.model),
                card: construct.card,
            }
        }
    })
}

/// The checkpoint `name`'s files, fetched into `cache` through its family's
/// factory: its id, and the loaded set.
///
/// # Errors
/// As [`ImageFamily::of`], and as resolving, planning and loading the build.
pub fn fetch_image_model(
    name: &str,
    cache: &ResourceCache,
) -> BunsenResult<(String, LoadedResourceSet)> {
    fn fetch<C: ConstructInit>(
        factory: BuildFactory<C>,
        name: &str,
        cache: &ResourceCache,
    ) -> BunsenResult<(String, LoadedResourceSet)> {
        let model = factory.resolve(name, cache)?;
        let map = model.plan(cache)?;
        Ok((model.id().to_string(), cache.load(&map)?))
    }
    match ImageFamily::of(name)? {
        ImageFamily::ResNet => fetch(default_resnet_factory()?, name, cache),
        ImageFamily::ViT => fetch(default_vit_factory()?, name, cache),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A catalog name loads through the family whose catalog has it; a hub
    /// repo through the family of its architecture.
    #[test]
    fn test_family_of() {
        for (name, family) in [
            ("timm/resnet50.a1_in1k", ImageFamily::ResNet),
            ("resnet50", ImageFamily::ResNet),
            (
                "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
                ImageFamily::ViT,
            ),
            ("hf:timm/vit_base_patch16_224.mine", ImageFamily::ViT),
            ("hf:timm/resnet18.mine", ImageFamily::ResNet),
            ("hf:someone/unknown", ImageFamily::ResNet),
        ] {
            assert_eq!(ImageFamily::of(name).unwrap(), family, "{name}");
        }
    }
}
