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
            deit::{
                DeiTContractConfig,
                pretrained::default_deit_factory,
            },
            driver::ImageClassifier,
            regnet::{
                RegNetContractConfig,
                pretrained::default_regnet_factory,
            },
            res2net::{
                Res2NetContractConfig,
                pretrained::default_res2net_factory,
            },
            resnet::{
                ResNetContractConfig,
                pretrained::default_resnet_factory,
            },
            resnetv2::{
                ResNetV2ContractConfig,
                pretrained::default_resnetv2_factory,
            },
            swin::{
                v1::{
                    SwinV1ContractConfig,
                    pretrained::default_swin_v1_factory,
                },
                v2::{
                    SwinV2ContractConfig,
                    pretrained::default_swin_v2_factory,
                },
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

    /// `kits::images::classification::deit`: `DeiT` and `DeiT` III.
    DeiT,

    /// `kits::images::classification::swin::v1`: the Swin Transformer, and
    /// Swin-S3.
    SwinV1,

    /// `kits::images::classification::swin::v2`: Swin Transformer V2.
    SwinV2,

    /// `kits::images::classification::regnet`: `RegNetX` and `RegNetY`, and
    /// the `SEER` trunks.
    RegNet,

    /// `kits::images::classification::resnetv2`: the pre-activation
    /// `ResNetV2` and `BiT`.
    ResNetV2,

    /// `kits::images::classification::res2net`: `Res2Net` and `Res2NeXt`.
    Res2Net,
}

impl ImageFamily {
    /// Every family, in the order `list` and the summary show them.
    pub const ALL: [Self; 8] = [
        Self::ResNet,
        Self::ResNetV2,
        Self::Res2Net,
        Self::RegNet,
        Self::ViT,
        Self::DeiT,
        Self::SwinV1,
        Self::SwinV2,
    ];

    /// The family's kit name.
    pub fn name(self) -> &'static str {
        match self {
            Self::ResNet => "resnet",
            Self::ViT => "vit",
            Self::DeiT => "deit",
            Self::SwinV1 => "swin-v1",
            Self::SwinV2 => "swin-v2",
            Self::RegNet => "regnet",
            Self::ResNetV2 => "resnetv2",
            Self::Res2Net => "res2net",
        }
    }

    /// The family's architectures, by timm's names: its prefabs'.
    fn architectures(self) -> Vec<&'static str> {
        match self {
            Self::ResNet => ResNetContractConfig::prefab_names(),
            Self::ViT => ViTContractConfig::prefab_names(),
            Self::DeiT => DeiTContractConfig::prefab_names(),
            Self::SwinV1 => SwinV1ContractConfig::prefab_names(),
            Self::SwinV2 => SwinV2ContractConfig::prefab_names(),
            Self::RegNet => RegNetContractConfig::prefab_names(),
            Self::ResNetV2 => ResNetV2ContractConfig::prefab_names(),
            Self::Res2Net => Res2NetContractConfig::prefab_names(),
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
            Self::DeiT => provider(default_deit_factory()?, name),
            Self::SwinV1 => provider(default_swin_v1_factory()?, name),
            Self::SwinV2 => provider(default_swin_v2_factory()?, name),
            Self::RegNet => provider(default_regnet_factory()?, name),
            Self::ResNetV2 => provider(default_resnetv2_factory()?, name),
            Self::Res2Net => provider(default_res2net_factory()?, name),
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
            Self::DeiT => rows(default_deit_factory()?),
            Self::SwinV1 => rows(default_swin_v1_factory()?),
            Self::SwinV2 => rows(default_swin_v2_factory()?),
            Self::RegNet => rows(default_regnet_factory()?),
            Self::ResNetV2 => rows(default_resnetv2_factory()?),
            Self::Res2Net => rows(default_res2net_factory()?),
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
        ImageFamily::DeiT => {
            let construct = Arc::unwrap_or_clone(
                default_deit_factory()?.load_construct(name, cache, device, policy)?,
            );
            LoadedImageModel {
                model: Box::new(construct.model),
                card: construct.card,
            }
        }
        ImageFamily::SwinV1 => {
            let construct = Arc::unwrap_or_clone(
                default_swin_v1_factory()?.load_construct(name, cache, device, policy)?,
            );
            LoadedImageModel {
                model: Box::new(construct.model),
                card: construct.card,
            }
        }
        ImageFamily::SwinV2 => {
            let construct = Arc::unwrap_or_clone(
                default_swin_v2_factory()?.load_construct(name, cache, device, policy)?,
            );
            LoadedImageModel {
                model: Box::new(construct.model),
                card: construct.card,
            }
        }
        ImageFamily::RegNet => {
            let construct = Arc::unwrap_or_clone(
                default_regnet_factory()?.load_construct(name, cache, device, policy)?,
            );
            LoadedImageModel {
                model: Box::new(construct.model),
                card: construct.card,
            }
        }
        ImageFamily::ResNetV2 => {
            let construct = Arc::unwrap_or_clone(
                default_resnetv2_factory()?.load_construct(name, cache, device, policy)?,
            );
            LoadedImageModel {
                model: Box::new(construct.model),
                card: construct.card,
            }
        }
        ImageFamily::Res2Net => {
            let construct = Arc::unwrap_or_clone(
                default_res2net_factory()?.load_construct(name, cache, device, policy)?,
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
        ImageFamily::DeiT => fetch(default_deit_factory()?, name, cache),
        ImageFamily::SwinV1 => fetch(default_swin_v1_factory()?, name, cache),
        ImageFamily::SwinV2 => fetch(default_swin_v2_factory()?, name, cache),
        ImageFamily::RegNet => fetch(default_regnet_factory()?, name, cache),
        ImageFamily::ResNetV2 => fetch(default_resnetv2_factory()?, name, cache),
        ImageFamily::Res2Net => fetch(default_res2net_factory()?, name, cache),
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
            (
                "timm/deit_base_distilled_patch16_224.fb_in1k",
                ImageFamily::DeiT,
            ),
            ("hf:timm/deit3_small_patch16_224.mine", ImageFamily::DeiT),
            ("timm/swin_s3_tiny_224.ms_in1k", ImageFamily::SwinV1),
            (
                "hf:timm/swin_tiny_patch4_window7_224.mine",
                ImageFamily::SwinV1,
            ),
            ("timm/swinv2_tiny_window8_256.ms_in1k", ImageFamily::SwinV2),
            ("hf:timm/swinv2_base_window12_192.mine", ImageFamily::SwinV2),
            ("timm/regnety_032.ra_in1k", ImageFamily::RegNet),
            ("timm/regnety_320.seer", ImageFamily::RegNet),
            ("hf:timm/regnetx_002.mine", ImageFamily::RegNet),
            (
                "timm/resnetv2_50x1_bit.goog_in21k_ft_in1k",
                ImageFamily::ResNetV2,
            ),
            ("hf:timm/resnetv2_50.mine", ImageFamily::ResNetV2),
            ("timm/res2net50_26w_4s.in1k", ImageFamily::Res2Net),
            ("hf:timm/res2next50.mine", ImageFamily::Res2Net),
            ("hf:timm/resnet18.mine", ImageFamily::ResNet),
            ("hf:someone/unknown", ImageFamily::ResNet),
        ] {
            assert_eq!(ImageFamily::of(name).unwrap(), family, "{name}");
        }
    }
}
