//! # Constructing a `ResNet` pretrained

use alloc::sync::Arc;

use burn::tensor::Device;

use crate::{
    blocks::images::models::conv_net::ConvNetModel,
    data::{
        builds::{
            BuildRef,
            ConstructInit,
            GEOMETRY,
        },
        resources::{
            LoadedResourceSet,
            ResourceSet,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
        ResultContext,
        Rule,
    },
    kits::images::{
        classification::resnet::{
            ResNetContractConfig,
            pretrained::RESNET_KIT,
        },
        support::card::{
            ImageModelCard,
            ImageModelConstruct,
        },
    },
};

/// How a `ResNet` pretrained is built: the config the model is
/// initialised from before the checkpoint is read into it.
///
/// `ResNet`'s [`ConstructInit`] hook, behind
/// [`default_resnet_factory`](super::default_resnet_factory). It
/// builds a [`ConvNetModel`] from the config, reads the checkpoint into it with
/// [`load_resnet`](crate::kits::images::classification::resnet::interop::torchvision::load_resnet),
/// and reads the build's [`ImageModelCard`]: its eval transform and labels.
///
/// A `PyTorch` checkpoint does not describe its own geometry, so the config
/// comes from [`with_config`](Self::with_config), which wins and is what a
/// given path needs; else from the geometry the row states; else, for a timm
/// hub repo, from its `config.json`'s architecture
/// ([`resnet_contract_config`](crate::kits::images::classification::resnet::interop::timm::resnet_contract_config)). To build a modified model, resolve the name, set the
/// config on the deferred model's `hook`, then load it;
/// `examples/resnet_tiny` swaps the activation this way before the
/// weights land.
#[derive(Clone, Debug, Default)]
pub struct ResNetConstructConfig {
    /// The config to build from, overriding the prefab's.
    pub config: Option<ResNetContractConfig>,
}

impl ResNetConstructConfig {
    /// Builds from the prefab the row names.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the config to build from, which wins over the build's stated
    /// geometry.
    pub fn with_config(
        mut self,
        config: ResNetContractConfig,
    ) -> Self {
        self.config = Some(config);
        self
    }

    /// The config `model` is built from: the explicit one, else the
    /// geometry the build states, else its loaded timm `config.json`'s.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when the stated
    /// geometry doesn't read as a config (never replaced by a `config.json`);
    /// as [`resnet_contract_config`](super::super::interop::timm::resnet_contract_config);
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause ([`Rule::Missing`] on `config`), when there
    /// is none: a given checkpoint needs [`with_config`](Self::with_config).
    pub fn config_for(
        &self,
        model: &BuildRef,
        loaded: Option<&LoadedResourceSet>,
    ) -> BunsenResult<ResNetContractConfig> {
        use crate::{
            data::builds::CONFIG,
            kits::images::{
                classification::resnet::interop::timm::resnet_contract_config,
                support::interop::timm::timm_model_config_from_path,
            },
        };
        if let Some(config) = &self.config {
            return Ok(config.clone());
        }
        if let Some(config) = model.stated(GEOMETRY)? {
            return Ok(config);
        }
        if let Some(timm) = loaded.and_then(|l| l.get(CONFIG)) {
            return resnet_contract_config(&timm_model_config_from_path(&timm.path)?);
        }
        Err(BunsenError::from(ConstraintError::new(
            "ResNetConstructConfig",
            "config",
            Rule::Missing,
        ))
        .context(format!(
            "{}: states no geometry to build from; a given checkpoint needs `with_config`",
            model.id()
        )))
    }
}

impl ConstructInit for ResNetConstructConfig {
    type Construct = ImageModelConstruct<ConvNetModel>;

    const KIT: &'static str = RESNET_KIT;

    /// Every row is a `PyTorch` state dict; the config comes at
    /// construction, from the row's prefab or the caller's.
    fn for_resource_set(_map: &ResourceSet) -> BunsenResult<Self> {
        Ok(Self::new())
    }

    /// Initialises the config's model, reads the checkpoint into it (a
    /// `PyTorch` state dict, or a timm safetensors file), and reads the
    /// build's card.
    fn try_init(
        &self,
        model: &BuildRef,
        loaded: &LoadedResourceSet,
        device: &Device,
        load: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<Arc<ImageModelConstruct<ConvNetModel>>> {
        use crate::{
            data::resources::SAFETENSORS,
            kits::images::classification::resnet::interop::{
                timm,
                torchvision,
            },
        };

        let config = self.config_for(model, Some(loaded))?;
        let card = ImageModelCard::try_from_build(model, loaded)?
            .try_for_classes(config.num_classes)
            .with_context(|| format!("{}'s labels", model.id()))?;
        let safetensors = loaded
            .map
            .get(super::CHECKPOINT)
            .and_then(|r| r.format.as_deref())
            == Some(SAFETENSORS);
        let model = if safetensors {
            timm::load_resnet(loaded, &config, device, load)?
        } else {
            torchvision::load_resnet(loaded, &config, device, load)?
        };
        Ok(Arc::new(ImageModelConstruct { model, card }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::pretrained::Prefabs,
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
