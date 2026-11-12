//! # Constructing a `ResNet` pretrained

use alloc::sync::Arc;

use burn::tensor::Device;

use crate::{
    burner::module::ModuleInit,
    data::{
        pretrained::{
            GEOMETRY,
            PretrainedRef,
        },
        resources::{
            LoadedResources,
            ResourceBundle,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
        Rule,
    },
    kits::{
        images::resnet::{
            ResNet,
            ResNetContractConfig,
            pretrained::{
                CHECKPOINT,
                RESNET_KIT,
                RESNET_PREFABS,
            },
        },
        support::construct::ConstructInit,
    },
};

/// How a `ResNet` pretrained is built: the config the model is
/// initialised from before the checkpoint is read into it.
///
/// `ResNet`'s [`ConstructInit`] hook, behind
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
pub struct ResNetConstructConfig {
    /// The config to build from, overriding the prefab's.
    pub config: Option<ResNetContractConfig>,
}

impl ResNetConstructConfig {
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
        if let Some(config) = model.stated(GEOMETRY)? {
            return Ok(config);
        }
        model
            .prefab(&RESNET_PREFABS)
            .map(|prefab| prefab.to_config())
            .ok_or_else(|| {
                BunsenError::from(ConstraintError::new(
                    "ResNetConstructConfig",
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

impl ConstructInit for ResNetConstructConfig {
    type Construct = ResNet;

    const KIT: &'static str = RESNET_KIT;

    /// Every row is a `PyTorch` state dict; the config comes at
    /// construction, from the row's prefab or the caller's.
    fn for_resource_bundle(_map: &ResourceBundle) -> BunsenResult<Self> {
        Ok(Self::new())
    }

    /// Initialises the config's model and reads the checkpoint into it.
    fn try_init(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
        load: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<Arc<ResNet>> {
        let config = self.config_for(model)?;
        let resnet: ResNet = config.try_init(device)?;
        let resnet = resnet.load_pytorch_weights(loaded.expect(CHECKPOINT)?)?;
        // The files' precision (f32), or wider as the policy asks.
        Ok(Arc::new(load.place(resnet, device, |_| true)?))
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
                pretrained::PretrainedRef,
                resources::{
                    PretrainedCache,
                    PretrainedCacheOptions,
                    ResourceBundle,
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
            kits::support::construct::{
                ConstructInit,
                Deferred,
            },
            support::testing::cpu_device,
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
        let hook = named.config.clone();
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
        let given = PretrainedRef::from(ResourceBundle::given("mine", CHECKPOINT, &file));
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
        let err = Deferred::<ResNetConstructConfig>::from_resource_bundle(ResourceBundle::given(
            "mine", CHECKPOINT, &file,
        ))
        .unwrap()
        .load(
            &cache,
            &cpu_device(),
            crate::data::formats::LoadPolicy::as_stored(),
        );
        // The loading pathway is an input boundary, which may re-mark the
        // hook's `Illegal` as `Policy`.
        missing_config(&[BunsenErrorKind::Policy]).assert_err(&err);
        assert_eq!(<ResNetConstructConfig as ConstructInit>::KIT, RESNET_KIT);
    }
}
