//! [`TransformersLmKit`] and [`load_lm_rekeyed`]: loading a `transformers` LM.
use alloc::{
    format,
    sync::Arc,
    vec::Vec,
};
use core::fmt::Debug;

use burn::{
    config::Config,
    store::burn_pack::Tensor as PackTensor,
    tensor::{
        Device,
        FloatDType,
    },
};

use super::{
    ConstructInit,
    LmConstruct,
    LmPolicy,
    LoadPolicy,
};
use crate::{
    blocks::transformers::models::language_decoder::LmDecoderHolder,
    burner::module::{
        CanonicalKeys,
        ModuleInit,
        Rekey,
    },
    data::{
        pretrained::{
            CONFIG,
            NamedTensorsApplied,
            PretrainedRef,
            ResolvedPresetSet,
            SafetensorsCheckpoint,
            StaticPreFabMap,
            load_rekeyed,
            stored_float_dtype,
        },
        resources::{
            LoadedResources,
            ResourceBundle,
            SAFETENSORS,
            SAFETENSORS_INDEX,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::support::model_config::{
        TransformersConfigJson,
        resolve_ref_presets,
    },
};

/// A language-model kit whose pretrained is a `transformers` checkpoint: its
/// safetensors (one file or shards) under [`CHECKPOINT`](Self::CHECKPOINT),
/// read into the geometry its `config.json` gives, or else the prefab its row
/// names; built as an [`LmConstruct`] with the positions the `config.json`
/// says it was trained on (its `text_config`'s, where it has one), an
/// [`LmPolicy`] from the ref's presets over the probed ones, and every file of
/// the checkpoint by its key.
///
/// What tells such kits apart is this trait's: the config, the model, the
/// prefabs, the `config.json` reader, the tensor loader and the probes. Each
/// kit's construct config implements it, and is its [`ConstructInit`] hook by
/// a blanket impl. The weights are placed in the stored dtype where the load
/// policy and the device allow it, never f16 (whose range a bf16-trained
/// model's activations exceed).
pub trait TransformersLmKit: Default {
    /// The kit's name ([`ConstructInit::KIT`]).
    const KIT: &'static str;

    /// The kit's name, in errors: `"Qwen3"`.
    const NAME: &'static str;

    /// The checkpoint's key in the resource bundle.
    const CHECKPOINT: &'static str;

    /// The model's config.
    type Config: 'static + Config + Debug + Clone + ModuleInit<Self::Model>;

    /// The model.
    type Model: burn::module::Module;

    /// The kit's prefabs, which a row without a `config.json` names.
    fn prefabs() -> &'static StaticPreFabMap<Self::Config>;

    /// The config of a `config.json`.
    ///
    /// # Errors
    /// The kit's: a file that does not describe its model.
    fn config_from_transformers(json: &str) -> BunsenResult<Self::Config>;

    /// Loads the checkpoint's tensors into `model`, every floating tensor in
    /// `dtype`.
    ///
    /// # Errors
    /// The kit's loader's.
    fn load_tensors(
        model: &mut Self::Model,
        tensors: Vec<PackTensor>,
        dtype: FloatDType,
    ) -> BunsenResult<NamedTensorsApplied>;

    /// The presets of a checkpoint no row describes (an `hf:` repo's), as the
    /// catalog builder probes a row's.
    ///
    /// # Errors
    /// As [`probe_loaded`](crate::kits::support::model_config::probe_loaded).
    fn probed(loaded: &LoadedResources) -> BunsenResult<ResolvedPresetSet>;
}

impl<K: TransformersLmKit> ConstructInit for K {
    type Construct = LmConstruct<K::Model>;

    const KIT: &'static str = K::KIT;
    const RESOLVES_PRESETS: bool = true;

    /// The hook for a safetensors checkpoint.
    fn for_resource_bundle(map: &ResourceBundle) -> BunsenResult<Self> {
        let index = format!("{}.index", K::CHECKPOINT);
        let checkpoint = match map.get(&index) {
            Some(index) => index,
            None => map.try_get(K::CHECKPOINT)?,
        };
        match checkpoint.kind.as_deref() {
            Some(SAFETENSORS | SAFETENSORS_INDEX) => Ok(Self::default()),
            other => Err(BunsenError::unsupported(format!(
                "no {} reader for a checkpoint of kind {other:?}",
                K::NAME
            ))),
        }
    }

    fn try_init(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
        policy: LoadPolicy,
    ) -> BunsenResult<Arc<LmConstruct<K::Model>>> {
        let (config, max_positions) = match loaded.get(CONFIG) {
            Some(part) => {
                let json =
                    std::fs::read_to_string(&part.path).map_err(sys_at("read", &part.path))?;
                let config = K::config_from_transformers(&json)
                    .map_err(|e| e.context(part.path.display()))?;
                let root = TransformersConfigJson::parse(&json)?;
                let text = root.section("text_config").unwrap_or(root);
                let max_positions = text.usize("max_position_embeddings").ok();
                (config, max_positions)
            }
            None => match model.prefab(K::prefabs()) {
                Some(prefab) => (prefab.to_config(), None),
                None => {
                    return Err(BunsenError::invalid_resource(format!(
                        "a {} checkpoint with neither a config.json nor a prefab",
                        K::NAME
                    ))
                    .context(&loaded.map.name));
                }
            },
        };
        let tensors = SafetensorsCheckpoint::from_loaded(loaded, K::CHECKPOINT)?.tensors()?;
        let stored = stored_float_dtype(&tensors).unwrap_or(FloatDType::F32);
        let dtype = policy.resolve(stored, device, |d| d != FloatDType::F16)?;
        let mut lm: K::Model = config.try_init(device)?;
        K::load_tensors(&mut lm, tensors, dtype)?;
        let policy = resolve_ref_presets(&LmPolicy::default(), model, &K::probed(loaded)?)?;
        let mut bundle = LmConstruct::new(lm)
            .with_max_positions(max_positions)
            .with_policy(policy.config.clone(), policy.applied());
        for (key, part) in &loaded.parts {
            bundle = bundle.with_file(key, &part.path);
        }
        Ok(Arc::new(bundle))
    }
}

/// Loads tensors under a checkpoint's names into `model`'s
/// [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel),
/// not its holder: [`load_rekeyed`] under `rekey` (onto the language model's
/// own paths, `embedding.`, `head.`, `stack.`) then its canonical keys; every
/// floating tensor in `dtype`, or, when `None`, the dtype it was built in.
///
/// # Errors
/// As [`load_rekeyed`].
pub fn load_lm_rekeyed<W: LmDecoderHolder>(
    model: &mut W,
    tensors: impl IntoIterator<Item = PackTensor>,
    rekey: Rekey,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let lm = model.lm_mut();
    let rekey = rekey.then(lm.canonical_rekey());
    load_rekeyed(lm, tensors, &rekey, dtype)
}
