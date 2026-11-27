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
    data::{
        builds::{
            BuildRef,
            ConstructInit,
        },
        resources::{
            LoadedResourceSet,
            ResourceSet,
            SAFETENSORS,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        language::{
            decision::laya::{
                LayaConstruct,
                interop::laya::{
                    LayaAgentMirror,
                    laya_config,
                    load_laya,
                },
            },
            encoders::modernbert::interop::transformers::modernbert_config_from_path,
        },
        support::interop::json_mirror_from_path,
    },
};

/// How a Laya pretrained is built: its two configs, the encoder's
/// `config.json` ([`modernbert_config_from_path`]) and the head's
/// `rl_agent_config.json`, give the model, into which its
/// `model.safetensors` loads ([`load_laya`]) in the dtype the load
/// policy resolves; bundled with the head's config, and every file of the
/// checkpoint by its key.
///
/// The Laya kit's construct config: its [`ConstructInit`] hook.
/// [`default_laya_factory`](super::default_laya_factory) attaches it; a
/// caller never builds one.
#[derive(Clone, Debug)]
pub struct LayaConstructConfig;

/// The path of `loaded`'s `key`.
fn part<'a>(
    loaded: &'a LoadedResourceSet,
    key: &str,
) -> BunsenResult<&'a std::path::Path> {
    let part = loaded.get(key).ok_or_else(|| {
        BunsenError::invalid_resource(format!("a Laya checkpoint without its {key}"))
    })?;
    Ok(&part.path)
}

impl ConstructInit for LayaConstructConfig {
    type Construct = LayaConstruct;

    const KIT: &'static str = LAYA_KIT;

    /// The hook for a safetensors checkpoint.
    fn for_resource_set(map: &ResourceSet) -> BunsenResult<Self> {
        match map.try_get(CHECKPOINT)?.format.as_deref() {
            Some(SAFETENSORS) => Ok(Self),
            other => Err(crate::kits::support::pretrained::unsupported_kind(
                "Laya",
                "checkpoint",
                other,
            )),
        }
    }

    fn try_init(
        &self,
        _model: &BuildRef,
        loaded: &LoadedResourceSet,
        device: &Device,
        policy: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<Arc<LayaConstruct>> {
        let encoder = modernbert_config_from_path(part(loaded, ENCODER_CONFIG)?)?;
        let agent =
            json_mirror_from_path::<LayaAgentMirror>(part(loaded, AGENT_CONFIG)?)?.agent_config();
        let model = load_laya(loaded, &laya_config(encoder, &agent), device, policy)?;
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

    #[test]
    fn test_the_hook_follows_the_kind() {
        for name in [
            "laya/laya",
            "laya/laya-multilingual",
            "laya/laya-typed-decisions",
        ] {
            LayaConstructConfig::for_resource_set(
                &crate::data::catalog::BuiltinCatalog::get()
                    .unwrap()
                    .build(name)
                    .unwrap()
                    .resource_set(),
            )
            .unwrap();
        }
        let mut other = crate::data::catalog::BuiltinCatalog::get()
            .unwrap()
            .build("laya/laya")
            .unwrap()
            .resource_set();
        other.resources.get_mut(CHECKPOINT).unwrap().format = Some("gguf".to_string());
        crate::errors::testing::ErrorMatcher::unsupported()
            .message_contains("gguf")
            .assert_err(&LayaConstructConfig::for_resource_set(&other));
    }
}
