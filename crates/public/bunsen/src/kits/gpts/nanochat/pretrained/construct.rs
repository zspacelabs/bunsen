//! # Constructing a nanochat pretrained
use std::sync::Arc;

use burn::tensor::Device;

use super::{
    NANOCHAT_KIT,
    PYTORCH,
};
use crate::{
    data::pretrained::{
        Construct,
        LoadedResources,
        PretrainedRef,
        ResourceMap,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::gpts::nanochat::{
        CHECKPOINT,
        META,
        NanoChatBundle,
        weights::{
            NanoChatMeta,
            load_pt_checkpoint,
        },
    },
};

/// How a nanochat pretrained is built: its `model_<step>.pt` read into the
/// geometry its `meta_<step>.json` gives, as the generation its state dict
/// is ([`load_pt_checkpoint`]); bundled with its meta and every file of the
/// checkpoint by its key.
///
/// The [`Construct`] hook for the nanochat kit.
/// [`default_nanochat_factory`](super::default_nanochat_factory) attaches it;
/// a caller never builds one.
#[derive(Clone, Debug)]
pub struct NanoChatConstruct;

impl Construct for NanoChatConstruct {
    type Built = NanoChatBundle;

    const KIT: &'static str = NANOCHAT_KIT;

    /// The hook for a `PyTorch` state dict with its meta.
    fn for_map(map: &ResourceMap) -> BunsenResult<Self> {
        map.try_get(META)?;
        match map.try_get(CHECKPOINT)?.kind.as_deref() {
            Some(PYTORCH) => Ok(Self),
            other => Err(BunsenError::unsupported(format!(
                "no nanochat reader for a checkpoint of kind {other:?}"
            ))),
        }
    }

    fn construct(
        &self,
        _model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
    ) -> BunsenResult<Arc<NanoChatBundle>> {
        let meta = loaded.expect(META)?;
        let checkpoint = loaded.expect(CHECKPOINT)?;
        let (model, generation) = load_pt_checkpoint(meta, checkpoint, device)?;
        let mut bundle = NanoChatBundle::new(model, generation, NanoChatMeta::from_file(meta)?);
        for (key, part) in &loaded.parts {
            bundle = bundle.with_file(key, &part.path);
        }
        Ok(Arc::new(bundle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        kits::gpts::nanochat::pretrained::{
            BASE_D20_FILES,
            D18_SFT_FILES,
        },
    };

    #[test]
    fn test_the_hook_follows_the_kind() {
        NanoChatConstruct::for_map(&BASE_D20_FILES.to_map()).unwrap();
        NanoChatConstruct::for_map(&D18_SFT_FILES.to_map()).unwrap();

        let mut safetensors = BASE_D20_FILES.to_map();
        safetensors.resources.get_mut(CHECKPOINT).unwrap().kind = Some("safetensors".to_string());
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("safetensors")
            .assert_err(&NanoChatConstruct::for_map(&safetensors));
    }
}
