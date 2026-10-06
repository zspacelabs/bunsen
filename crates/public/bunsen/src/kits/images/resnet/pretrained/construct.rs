//! # Constructing a `ResNet` pretrained

use std::sync::Arc;

use burn::prelude::Backend;

use crate::{
    burner::module::ModuleInit,
    data::pretrained::{
        Construct,
        LoadedResources,
        PretrainedRef,
        ResourceMap,
    },
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
        Rule,
    },
    kits::images::resnet::{
        ResNet,
        ResNetContractConfig,
        pretrained::{
            CHECKPOINT,
            RESNET_KIT,
            RESNET_PREFABS,
        },
    },
};

/// How a `ResNet` pretrained is built: the config the model is
/// initialised from before the checkpoint is read into it.
///
/// `ResNet`'s [`Construct`] hook, behind
/// [`default_resnet_factory`](super::default_resnet_factory). It
/// builds a [`ResNet`] from the config, then reads the checkpoint into
/// it with [`ResNet::load_pytorch_weights`].
///
/// A checkpoint does not describe its own geometry, so the config comes
/// from the prefab the row names, or from
/// [`with_config`](Self::with_config), which wins and is what a given
/// path needs. To build a modified model, resolve the name, set the
/// config on the deferred model's `hook`, then load it;
/// `examples/resnet_tiny` swaps the activation this way before the
/// weights land.
#[derive(Clone, Debug, Default)]
pub struct ResNetConstruct {
    /// The config to build from, overriding the prefab's.
    pub config: Option<ResNetContractConfig>,
}

impl ResNetConstruct {
    /// Builds from the prefab the row names.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the config to build from, which wins over the prefab's.
    pub fn with_config(
        mut self,
        config: ResNetContractConfig,
    ) -> Self {
        self.config = Some(config);
        self
    }

    /// The config `model` is built from: the explicit one, else the
    /// prefab the row names.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause ([`Rule::Missing`] on `config`), when there
    /// is neither: a given checkpoint needs [`with_config`](Self::with_config).
    pub fn config_for(
        &self,
        model: &PretrainedRef,
    ) -> BunsenResult<ResNetContractConfig> {
        if let Some(config) = &self.config {
            return Ok(config.clone());
        }
        model
            .prefab(&RESNET_PREFABS)
            .map(|prefab| prefab.to_config())
            .ok_or_else(|| {
                BunsenError::from(ConstraintError::new(
                    "ResNetConstruct",
                    "config",
                    Rule::Missing,
                ))
                .context(format!(
                    "{}: names no prefab to build from; a given checkpoint needs \
                         `with_config`",
                    model.id()
                ))
            })
    }
}

impl Construct for ResNetConstruct {
    type Built<B: Backend> = ResNet<B>;

    const KIT: &'static str = RESNET_KIT;

    /// Every row is a `PyTorch` state dict; the config comes at
    /// construction, from the row's prefab or the caller's.
    fn for_map(_map: &ResourceMap) -> BunsenResult<Self> {
        Ok(Self::new())
    }

    /// Initialises the config's model and reads the checkpoint into it.
    fn construct<B: Backend>(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &B::Device,
    ) -> BunsenResult<Arc<ResNet<B>>> {
        let config = self.config_for(model)?;
        let resnet: ResNet<B> = config.try_init(device)?;
        Ok(Arc::new(
            resnet.load_pytorch_weights(loaded.expect(CHECKPOINT)?)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::resnet::pretrained::default_resnet_factory;

    /// The hook builds from the prefab a row names, or from an explicit
    /// config, which wins; a given checkpoint with neither is refused
    /// before its bytes are read.
    #[test]
    fn test_the_hook_takes_the_prefab_or_an_explicit_config() {
        use std::fs;

        use crate::{
            data::{
                cache::BunsenDiskCacheOptions,
                pretrained::{
                    Construct,
                    Deferred,
                    PretrainedCache,
                    PretrainedCacheOptions,
                    PretrainedRef,
                    ResourceMap,
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
            support::testing::{
                CpuBackend,
                default_device,
            },
        };

        let dir = tempfile::tempdir().unwrap();
        let cache = PretrainedCache::new(
            PretrainedCacheOptions::default()
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
        let hook = named.hook.clone();
        let named = named.model;
        let resnet50 = RESNET_PREFABS.expect_lookup_prefab("resnet50").to_config();
        assert_eq!(
            format!("{:?}", hook.config_for(&named).unwrap()),
            format!("{resnet50:?}")
        );

        let resnet18 = RESNET_PREFABS.expect_lookup_prefab("resnet18").to_config();
        let explicit = hook.clone().with_config(resnet18.clone());
        assert_eq!(
            format!("{:?}", explicit.config_for(&named).unwrap()),
            format!("{resnet18:?}"),
            "an explicit config wins over the prefab"
        );

        let file = dir.path().join("ckpt.pth");
        fs::write(&file, b"not a state dict").unwrap();
        assert!(
            factory.resolve(file.to_str().unwrap(), &cache).is_err(),
            "a path is not a name the factory knows"
        );
        let given = PretrainedRef::from(ResourceMap::given("mine", CHECKPOINT, &file));
        let missing_config = |kinds: &[BunsenErrorKind]| {
            ErrorMatcher::new()
                .with_kind_matching(value::one_of(kinds.to_vec()))
                .cause(predicate("a missing config", |e: &ConstraintError| {
                    e.field == "config" && e.rule == Rule::Missing
                }))
                .frame_contains("with_config")
        };
        missing_config(&[BunsenErrorKind::Illegal]).assert_err(&hook.config_for(&given));
        assert_eq!(
            format!("{:?}", explicit.config_for(&given).unwrap()),
            format!("{resnet18:?}")
        );

        // Through the pathway: the refusal comes before the file is read.
        let cache = PretrainedCache::new(
            PretrainedCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap();
        let err =
            Deferred::<ResNetConstruct>::from_map(ResourceMap::given("mine", CHECKPOINT, &file))
                .unwrap()
                .load::<CpuBackend>(&cache, &default_device());
        // The loading pathway is an input boundary, which may re-mark the
        // hook's `Illegal` as `Policy`.
        missing_config(&[BunsenErrorKind::Policy]).assert_err(&err);
        assert_eq!(<ResNetConstruct as Construct>::KIT, RESNET_KIT);
    }
}
