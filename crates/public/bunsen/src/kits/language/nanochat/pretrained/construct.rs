//! # Constructing a nanochat pretrained
use alloc::sync::Arc;

use burn::tensor::Device;

use super::{
    NANOCHAT_KIT,
    PYTORCH,
};
use crate::{
    data::{
        builds::{
            ConstructInit,
            PretrainedRef,
        },
        resources::{
            LoadedResourceSet,
            ResourceSet,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        language::nanochat::{
            CHECKPOINT,
            META,
            NanoChatConstruct,
            nanochat_default_policy,
            weights::{
                NanoChatMeta,
                load_pt_checkpoint,
            },
        },
        support::model_config::resolve_ref_presets,
    },
};

/// How a nanochat pretrained is built: its `model_<step>.pt` read into the
/// geometry its `meta_<step>.json` gives, as the generation its state dict
/// is ([`load_pt_checkpoint`]); bundled with the positions it was trained on
/// (the meta's `sequence_len`), the ref's presets over
/// [`nanochat_default_policy`], and every file of the checkpoint by its key.
///
/// The nanochat kit's construct config: its [`ConstructInit`] hook.
/// [`default_nanochat_factory`](super::default_nanochat_factory) attaches it;
/// a caller never builds one.
#[derive(Clone, Debug)]
pub struct NanoChatConstructConfig;

impl ConstructInit for NanoChatConstructConfig {
    type Construct = NanoChatConstruct;

    const KIT: &'static str = NANOCHAT_KIT;
    const RESOLVES_PRESETS: bool = true;

    /// The hook for a `PyTorch` state dict with its meta.
    fn for_resource_set(map: &ResourceSet) -> BunsenResult<Self> {
        map.try_get(META)?;
        match map.try_get(CHECKPOINT)?.format.as_deref() {
            Some(PYTORCH) => Ok(Self),
            other => Err(BunsenError::unsupported(format!(
                "no nanochat reader for a checkpoint of kind {other:?}"
            ))),
        }
    }

    fn try_init(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResourceSet,
        device: &Device,
        load: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<Arc<NanoChatConstruct>> {
        let meta = loaded.expect(META)?;
        let checkpoint = loaded.expect(CHECKPOINT)?;
        let sequence_len = NanoChatMeta::from_file(meta)?.model_config.sequence_len;
        let (gpt, _) = load_pt_checkpoint(meta, checkpoint, device, load)?;
        let policy = resolve_ref_presets(&nanochat_default_policy(), model, &Default::default())?;
        let mut bundle = NanoChatConstruct::new(gpt)
            .with_max_positions(Some(sequence_len))
            .with_policy(policy.config.clone(), policy.applied());
        for (key, part) in &loaded.parts {
            bundle = bundle.with_file(key, &part.path);
        }
        Ok(Arc::new(bundle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    #[test]
    fn test_the_hook_follows_the_kind() {
        NanoChatConstructConfig::for_resource_set(
            &crate::data::builds::builtin_build_bundle("nanochat-students/base-d20").unwrap(),
        )
        .unwrap();
        NanoChatConstructConfig::for_resource_set(
            &crate::data::builds::builtin_build_bundle("nitishpandey04/nanochat-d18-sft").unwrap(),
        )
        .unwrap();

        let mut safetensors =
            crate::data::builds::builtin_build_bundle("nanochat-students/base-d20").unwrap();
        safetensors.resources.get_mut(CHECKPOINT).unwrap().format = Some("safetensors".to_string());
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("safetensors")
            .assert_err(&NanoChatConstructConfig::for_resource_set(&safetensors));
    }
}
