//! # Constructing a `ResNet` pretrained

use burn::tensor::Device;

use crate::{
    blocks::images::models::conv_net::ConvNetModel,
    data::resources::LoadedResourceSet,
    errors::BunsenResult,
    kits::images::{
        classification::resnet::{
            ResNetContractConfig,
            pretrained::RESNET_KIT,
        },
        support::{
            pretrained::{
                TimmConstructConfig,
                TimmImageContract,
            },
            timm::interop::TimmModelConfigMirror,
        },
    },
};

/// How a `ResNet` pretrained is built: the config the model is
/// initialised from before the checkpoint is read into it.
///
/// `ResNet`'s [`ConstructInit`](crate::data::builds::ConstructInit) hook, behind
/// [`default_resnet_factory`](super::default_resnet_factory). It
/// builds a [`ConvNetModel`] from the config, reads the checkpoint into it with
/// [`load_resnet`](crate::kits::images::classification::resnet::interop::torchvision::load_resnet),
/// and reads the build's [`ImageModelCard`](crate::kits::images::support::card::ImageModelCard): its eval transform and labels.
///
/// A `PyTorch` checkpoint does not describe its own geometry, so the config
/// comes from [`with_config`](Self::with_config), which wins and is what a
/// given path needs; else from the geometry the row states; else, for a timm
/// hub repo, from its `config.json`'s architecture
/// ([`resnet_contract_config`](crate::kits::images::classification::resnet::interop::timm::resnet_contract_config)). To build a modified model, resolve the name, set the
/// config on the deferred model's `hook`, then load it;
/// `examples/resnet_tiny` swaps the activation this way before the
/// weights land.
pub type ResNetConstructConfig = TimmConstructConfig<ResNetContractConfig>;

impl TimmImageContract for ResNetContractConfig {
    type Model = ConvNetModel;

    const CONSTRUCT: &'static str = "ResNetConstructConfig";
    const KIT: &'static str = RESNET_KIT;

    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn from_timm_config(config: &TimmModelConfigMirror) -> BunsenResult<Self> {
        crate::kits::images::classification::resnet::interop::timm::resnet_contract_config(config)
    }

    fn load(
        loaded: &LoadedResourceSet,
        config: &Self,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<ConvNetModel> {
        crate::kits::images::classification::resnet::interop::torchvision::load_resnet(
            loaded, config, device, policy,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::{
            builds::GEOMETRY,
            pretrained::Prefabs,
        },
        errors::{
            ConstraintError,
            Rule,
        },
        kits::images::classification::resnet::{
            ResNetContractConfig,
            pretrained::{
                CHECKPOINT,
                default_resnet_factory,
            },
        },
        support::testing::assert_config_eq,
    };

    /// A torchvision row states its eval transform (bilinear, a 0.875 crop:
    /// timm's `tv_in1k` configs) and binds `ImageNet`-1k's labels.
    #[test]
    fn test_a_torchvision_row_states_its_card() {
        use crate::{
            data::{
                cache::BunsenDiskCacheOptions,
                resources::{
                    ResourceCache,
                    ResourceCacheOptions,
                },
            },
            kits::images::support::{
                card::{
                    EVAL_TRANSFORM,
                    LABEL_NAMES,
                    LABELS,
                },
                transform::{
                    ImageEvalTransformConfig,
                    ImageInterpolation,
                },
            },
        };

        let dir = tempfile::tempdir().unwrap();
        let cache = ResourceCache::new(
            ResourceCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap();
        let named = default_resnet_factory()
            .unwrap()
            .resolve("torchvision/resnet18", &cache)
            .unwrap()
            .model;
        let transform: ImageEvalTransformConfig = named.stated(EVAL_TRANSFORM).unwrap().unwrap();
        assert_eq!(
            transform,
            ImageEvalTransformConfig::new()
                .with_interpolation(ImageInterpolation::Bilinear)
                .with_crop_pct(0.875)
        );
        let set = named.to_resource_set();
        assert_eq!(set.get(LABELS).unwrap().file, "imagenet_synsets.txt");
        assert_eq!(
            set.get(LABEL_NAMES).unwrap().file,
            "imagenet_synset_to_lemma.txt"
        );
    }

    /// The hook builds from the prefab a row names, or from an explicit
    /// config, which wins; a given checkpoint with neither is refused
    /// before its bytes are read.
    #[test]
    fn test_the_hook_takes_the_prefab_or_an_explicit_config() {
        use std::fs;

        use crate::{
            data::{
                builds::{
                    BuildRef,
                    ConstructInit,
                    DeferredBuild,
                },
                cache::BunsenDiskCacheOptions,
                resources::{
                    ResourceCache,
                    ResourceCacheOptions,
                    ResourceSet,
                },
            },
            errors::{
                BunsenErrorKind,
                testing::{
                    ErrorMatcher,
                    predicate,
                    value,
                },
            },
            support::testing::cpu_device,
        };

        let dir = tempfile::tempdir().unwrap();
        let cache = ResourceCache::new(
            ResourceCacheOptions::default()
                .with_disk(
                    crate::data::cache::BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap();
        let factory = default_resnet_factory().unwrap();
        let named = factory.resolve("resnet50", &cache).unwrap();
        let hook = named.config.clone();
        let named = named.model;
        let resnet50 = ResNetContractConfig::expect_prefab("resnet50");
        assert_config_eq(
            "resnet50",
            &hook.config_for(&named, None).unwrap(),
            &resnet50,
        );

        let resnet18 = ResNetContractConfig::expect_prefab("resnet18");
        let explicit = hook.clone().with_config(resnet18.clone());
        assert_config_eq(
            "an explicit config wins over the prefab",
            &explicit.config_for(&named, None).unwrap(),
            &resnet18,
        );

        let file = dir.path().join("ckpt.pth");
        fs::write(&file, b"not a state dict").unwrap();
        assert!(
            factory.resolve(file.to_str().unwrap(), &cache).is_err(),
            "a path is not a name the factory knows"
        );
        let given = BuildRef::from(ResourceSet::given("mine", CHECKPOINT, &file));
        let missing_config = |kinds: &[BunsenErrorKind]| {
            ErrorMatcher::new()
                .with_kind_matching(value::one_of(kinds.to_vec()))
                .cause(predicate("a missing config", |e: &ConstraintError| {
                    e.field == "config" && e.rule == Rule::Missing
                }))
                .frame_contains("with_config")
        };
        missing_config(&[BunsenErrorKind::Illegal]).assert_err(&hook.config_for(&given, None));

        // A stated geometry that doesn't read as a config is an error, which a
        // timm `config.json` never replaces.
        let mut bad = named.clone();
        if let BuildRef::Named { build, .. } = &mut bad {
            build
                .args
                .get_mut(GEOMETRY)
                .expect("a stated geometry")
                .value = serde_json::json!({"layers": "not a list"});
        }
        ErrorMatcher::illegal()
            .message_contains("doesn't read as its config")
            .assert_err(&hook.config_for(&bad, None));
        assert_config_eq(
            "resnet18",
            &explicit.config_for(&given, None).unwrap(),
            &resnet18,
        );

        // Through the pathway: the refusal comes before the file is read.
        let cache = ResourceCache::new(
            ResourceCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap();
        let err = DeferredBuild::<ResNetConstructConfig>::from_resource_set(ResourceSet::given(
            "mine", CHECKPOINT, &file,
        ))
        .unwrap()
        .load(
            &cache,
            &cpu_device(),
            crate::data::formats::LoadPolicy::load_as_stored(),
        );
        // The loading pathway is an input boundary, which may re-mark the
        // hook's `Illegal` as `Policy`.
        missing_config(&[BunsenErrorKind::Policy]).assert_err(&err);
        assert_eq!(<ResNetConstructConfig as ConstructInit>::KIT, RESNET_KIT);
    }
}
