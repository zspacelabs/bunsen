//! # Constructing a Qwen3 pretrained
use std::sync::Arc;

use burn::tensor::Device;

use super::{
    CHECKPOINT,
    QWEN3_KIT,
    QWEN3_PREFABS,
};
use crate::{
    burner::module::ModuleInit,
    data::pretrained::{
        CONFIG,
        Construct,
        HfConfigJson,
        LoadedResources,
        PretrainedRef,
        ResourceMap,
        SAFETENSORS,
        SAFETENSORS_INDEX,
        SafetensorsCheckpoint,
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::gpts::qwen3::{
        Qwen3Bundle,
        Qwen3Model,
        weights::{
            load_hf_tensors,
            qwen3_config_from_hf,
        },
    },
};

/// How a Qwen3 pretrained is built: its safetensors checkpoint (one file or
/// shards) read into the geometry its `config.json` gives
/// ([`qwen3_config_from_hf`]), or else the prefab its row names, in the
/// device's float dtype; bundled with the positions the `config.json` says it
/// was trained on, and every file of the checkpoint by its key.
///
/// The [`Construct`] hook for the Qwen3 kit.
/// [`default_qwen3_factory`](super::default_qwen3_factory) attaches it; a
/// caller never builds one.
#[derive(Clone, Debug)]
pub struct Qwen3Construct;

impl Construct for Qwen3Construct {
    type Built = Qwen3Bundle;

    const KIT: &'static str = QWEN3_KIT;

    /// The hook for a safetensors checkpoint.
    fn for_map(map: &ResourceMap) -> BunsenResult<Self> {
        let index = format!("{CHECKPOINT}.index");
        let checkpoint = match map.get(&index) {
            Some(index) => index,
            None => map.try_get(CHECKPOINT)?,
        };
        match checkpoint.kind.as_deref() {
            Some(SAFETENSORS | SAFETENSORS_INDEX) => Ok(Self),
            other => Err(BunsenError::unsupported(format!(
                "no Qwen3 reader for a checkpoint of kind {other:?}"
            ))),
        }
    }

    fn construct(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
    ) -> BunsenResult<Arc<Qwen3Bundle>> {
        let (config, max_positions) = match loaded.get(CONFIG) {
            Some(part) => {
                let json =
                    std::fs::read_to_string(&part.path).map_err(sys_at("read", &part.path))?;
                let config =
                    qwen3_config_from_hf(&json).map_err(|e| e.context(part.path.display()))?;
                let max_positions = HfConfigJson::parse(&json)?
                    .usize("max_position_embeddings")
                    .ok();
                (config, max_positions)
            }
            None => match model.prefab(&QWEN3_PREFABS) {
                Some(prefab) => (prefab.to_config(), None),
                None => {
                    return Err(BunsenError::invalid_resource(
                        "a Qwen3 checkpoint with neither a config.json nor a prefab",
                    )
                    .context(&loaded.map.name));
                }
            },
        };
        let tensors = SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
        let mut qwen3: Qwen3Model = config.try_init(device)?;
        load_hf_tensors(&mut qwen3, tensors)?;
        let mut bundle = Qwen3Bundle::new(qwen3).with_max_positions(max_positions);
        for (key, part) in &loaded.parts {
            bundle = bundle.with_file(key, &part.path);
        }
        Ok(Arc::new(bundle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::gpts::qwen3::pretrained::{
        QWEN3_0_6B_FILES,
        QWEN3_4B_FILES,
    };

    #[test]
    fn test_the_hook_follows_the_kind() {
        Qwen3Construct::for_map(&QWEN3_0_6B_FILES.to_map()).unwrap();
        Qwen3Construct::for_map(&QWEN3_4B_FILES.to_map()).unwrap();

        let mut gguf = QWEN3_0_6B_FILES.to_map();
        gguf.resources.get_mut(CHECKPOINT).unwrap().kind = Some("gguf".to_string());
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Unsupported)
            .message_contains("gguf")
            .assert_err(&Qwen3Construct::for_map(&gguf));
    }
}
