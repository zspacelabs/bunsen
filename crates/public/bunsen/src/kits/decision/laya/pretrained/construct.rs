//! # Constructing a Laya pretrained
use alloc::sync::Arc;

use burn::tensor::Device;

use super::{
    AGENT_CONFIG,
    CHECKPOINT,
    ENCODER_CONFIG,
    LAYA_KIT,
};
use crate::{
    burner::module::ModuleInit,
    data::pretrained::{
        LoadedResources,
        PretrainedRef,
        ResourceBundle,
        SAFETENSORS,
        SafetensorsCheckpoint,
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::{
        decision::laya::{
            LayaAgentConfig,
            LayaConstruct,
            LayaModel,
            weights::{
                laya_config,
                load_laya_tensors,
            },
        },
        support::construct::ConstructInit,
        text::encoders::modernbert::weights::modernbert_config_from_transformers,
    },
};

/// How a Laya pretrained is built: its two configs, the encoder's
/// `config.json` ([`modernbert_config_from_transformers`]) and the head's
/// `rl_agent_config.json`, give the model, into which its
/// `model.safetensors` loads ([`load_laya_tensors`]) in the dtype the load
/// policy resolves; bundled with the head's config, and every file of the
/// checkpoint by its key.
///
/// The Laya kit's construct config: its [`ConstructInit`] hook.
/// [`default_laya_factory`](super::default_laya_factory) attaches it; a
/// caller never builds one.
#[derive(Clone, Debug)]
pub struct LayaConstructConfig;

fn read(
    loaded: &LoadedResources,
    key: &str,
) -> BunsenResult<String> {
    let part = loaded.get(key).ok_or_else(|| {
        BunsenError::invalid_resource(format!("a Laya checkpoint without its {key}"))
    })?;
    std::fs::read_to_string(&part.path).map_err(sys_at("read", &part.path))
}

impl ConstructInit for LayaConstructConfig {
    type Construct = LayaConstruct;

    const KIT: &'static str = LAYA_KIT;

    /// The hook for a safetensors checkpoint.
    fn for_resource_bundle(map: &ResourceBundle) -> BunsenResult<Self> {
        match map.try_get(CHECKPOINT)?.kind.as_deref() {
            Some(SAFETENSORS) => Ok(Self),
            other => Err(BunsenError::unsupported(format!(
                "no Laya reader for a checkpoint of kind {other:?}"
            ))),
        }
    }

    fn try_init(
        &self,
        _model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
        policy: crate::kits::support::construct::LoadPolicy,
    ) -> BunsenResult<Arc<LayaConstruct>> {
        let encoder = modernbert_config_from_transformers(&read(loaded, ENCODER_CONFIG)?)?;
        let agent = LayaAgentConfig::parse(&read(loaded, AGENT_CONFIG)?)?;
        let tensors = SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
        let stored = crate::data::pretrained::stored_float_dtype(&tensors)
            .unwrap_or(burn::tensor::FloatDType::F32);
        let dtype = policy.resolve(stored, device, |_| true)?;
        let mut model: LayaModel = laya_config(encoder, &agent).try_init(device)?;
        load_laya_tensors(&mut model, tensors, Some(dtype))?;
        let mut bundle = LayaConstruct::new(model, agent);
        for (key, part) in &loaded.parts {
            bundle = bundle.with_file(key, &part.path);
        }
        Ok(Arc::new(bundle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::decision::laya::pretrained::LAYA_CATALOG;

    #[test]
    fn test_the_hook_follows_the_kind() {
        for name in [
            "laya/laya",
            "laya/laya-multilingual",
            "laya/laya-typed-decisions",
        ] {
            LayaConstructConfig::for_resource_bundle(&LAYA_CATALOG.resource_bundle(name)).unwrap();
        }
        let mut other = LAYA_CATALOG.resource_bundle("laya/laya");
        other.resources.get_mut(CHECKPOINT).unwrap().kind = Some("gguf".to_string());
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Unsupported)
            .message_contains("gguf")
            .assert_err(&LayaConstructConfig::for_resource_bundle(&other));
    }
}
