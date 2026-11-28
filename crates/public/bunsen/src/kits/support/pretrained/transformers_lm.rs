//! The construct hook of a `transformers` language model.
use alloc::{
    format,
    sync::Arc,
};
use core::fmt::Debug;
use std::path::Path;

use burn::{
    config::Config,
    store::{
        BurnpackStore,
        ModuleSnapshot,
    },
    tensor::Device,
};

use crate::{
    burner::module::ModuleInit,
    data::{
        builds::{
            BuildRef,
            CONFIG,
            ConstructInit,
        },
        catalog::presets::ResolvedPresetSet,
        formats::LoadPolicy,
        resources::{
            LoadedResourceSet,
            ResourceSet,
            SAFETENSORS,
            SAFETENSORS_INDEX,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::support::{
        construct::{
            LmConstruct,
            LmPolicy,
        },
        model_config::resolve_ref_presets,
        transformers::interop::{
            TransformersConfigJson,
            config_from_path,
            lm_max_positions,
        },
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
/// What tells such kits apart is this trait's: the config, the model, and
/// the kit's interop rungs (its `config.json` conversion and its loader) and
/// probes. Its weights' converted copy, where the cache keeps one, is the
/// built model as a burnpack, loaded into the model `config.json` builds. Each
/// kit's construct config implements it, and is its [`ConstructInit`] hook by a
/// blanket impl. The hook reads `config.json` once, through
/// [`config_from_path`](crate::kits::support::transformers::interop::config_from_path).
pub trait TransformersLmKit: Default {
    /// The kit's name ([`ConstructInit::KIT`]).
    const KIT: &'static str;

    /// The kit's name, in errors: `"Qwen3"`.
    const NAME: &'static str;

    /// The checkpoint's key in the resource set.
    const CHECKPOINT: &'static str;

    /// The model's config.
    type Config: 'static + Config + Debug + Clone + ModuleInit<Self::Model>;

    /// The model.
    type Model: burn::module::Module;

    /// The config of a parsed `config.json`: the kit's
    /// `interop::transformers` conversion.
    ///
    /// # Errors
    /// The kit's: a file that does not describe its model.
    fn config(c: &TransformersConfigJson) -> BunsenResult<Self::Config>;

    /// The model of `loaded`'s checkpoint, built from `config`: the kit's
    /// `interop::transformers` loader (its weights in the stored dtype where
    /// `policy` and the device allow it, never f16, whose range a
    /// bf16-trained model's activations exceed).
    ///
    /// # Errors
    /// The kit's loader's.
    fn load(
        loaded: &LoadedResourceSet,
        config: &Self::Config,
        device: &Device,
        policy: LoadPolicy,
    ) -> BunsenResult<Self::Model>;

    /// The presets of a checkpoint no row describes (an `hf:` repo's), as the
    /// catalog builder probes a row's.
    ///
    /// # Errors
    /// As [`probe_loaded`](crate::kits::support::pretrained::probe_loaded).
    fn probed(loaded: &LoadedResourceSet) -> BunsenResult<ResolvedPresetSet>;
}

impl<K: TransformersLmKit> ConstructInit for K {
    type Construct = LmConstruct<K::Model>;

    const KIT: &'static str = K::KIT;
    const RESOLVES_PRESETS: bool = true;

    fn applied_presets(
        construct: &Self::Construct
    ) -> Option<crate::data::catalog::presets::PresetSelection> {
        Some(construct.presets.clone())
    }

    /// The hook for a safetensors checkpoint.
    fn for_resource_set(map: &ResourceSet) -> BunsenResult<Self> {
        let index = format!("{}.index", K::CHECKPOINT);
        let checkpoint = match map.get(&index) {
            Some(index) => index,
            None => map.try_get(K::CHECKPOINT)?,
        };
        match checkpoint.format.as_deref() {
            Some(SAFETENSORS | SAFETENSORS_INDEX) => Ok(Self::default()),
            other => Err(crate::kits::support::pretrained::unsupported_kind(
                K::NAME,
                "checkpoint",
                other,
            )),
        }
    }

    fn try_init(
        &self,
        model: &BuildRef,
        loaded: &LoadedResourceSet,
        device: &Device,
        policy: LoadPolicy,
    ) -> BunsenResult<Arc<LmConstruct<K::Model>>> {
        let (config, max_positions) = read_config::<K>(loaded)?;
        let lm = K::load(loaded, &config, device, policy)?;
        bundle::<K>(model, loaded, lm, max_positions)
    }

    /// The checkpoint: the one file, or the index and its shards.
    fn converted_weights(
        &self,
        map: &ResourceSet,
    ) -> Option<ResourceSet> {
        let weights = map.family(K::CHECKPOINT);
        (!weights.resources.is_empty()).then_some(weights)
    }

    /// The model built from `config.json`, its weights the burnpack at
    /// `path`, every parameter of the model in it.
    fn try_init_converted(
        &self,
        model: &BuildRef,
        loaded: &LoadedResourceSet,
        path: &Path,
        device: &Device,
        _policy: LoadPolicy,
    ) -> BunsenResult<Arc<LmConstruct<K::Model>>> {
        let (config, max_positions) = read_config::<K>(loaded)?;
        let mut lm: K::Model = config.try_init(device)?;
        let mut store = BurnpackStore::from_file(path).auto_extension(false);
        let result = lm.load_from(&mut store).map_err(|e| {
            BunsenError::invalid_resource(format!("cannot read {}", path.display())).with_cause(e)
        })?;
        if !result.errors.is_empty() || !result.missing.is_empty() {
            return Err(BunsenError::invalid_resource(format!(
                "{} does not load into the {} model: {} errors, {} parameters missing",
                path.display(),
                K::NAME,
                result.errors.len(),
                result.missing.len()
            )));
        }
        bundle::<K>(model, loaded, lm, max_positions)
    }

    fn save_converted(
        &self,
        construct: &LmConstruct<K::Model>,
        path: &Path,
    ) -> BunsenResult<()> {
        let mut store = BurnpackStore::from_file(path)
            .auto_extension(false)
            .overwrite(true);
        construct
            .model
            .save_into(&mut store)
            .map_err(|e| BunsenError::sys(format!("write {}", path.display())).with_cause(e))
    }
}

/// The model config and the trained positions of `loaded`'s `config.json`.
fn read_config<K: TransformersLmKit>(
    loaded: &LoadedResourceSet
) -> BunsenResult<(K::Config, Option<usize>)> {
    let Some(part) = loaded.get(CONFIG) else {
        return Err(BunsenError::invalid_resource(format!(
            "a {} checkpoint without a config.json",
            K::NAME
        ))
        .context(&loaded.map.name));
    };
    config_from_path(&part.path, |c| Ok((K::config(c)?, lm_max_positions(c))))
}

/// The construct of a built model: its positions, its policy (the ref's
/// presets over the probed), and every file of `loaded` by its key.
fn bundle<K: TransformersLmKit>(
    model: &BuildRef,
    loaded: &LoadedResourceSet,
    lm: K::Model,
    max_positions: Option<usize>,
) -> BunsenResult<Arc<LmConstruct<K::Model>>> {
    let policy = resolve_ref_presets(&LmPolicy::default(), model, &K::probed(loaded)?)?;
    let mut bundle = LmConstruct::new(lm)
        .with_max_positions(max_positions)
        .with_policy(policy.config.clone(), policy.applied());
    for (key, part) in &loaded.parts {
        bundle = bundle.with_file(key, &part.path);
    }
    Ok(Arc::new(bundle))
}

/// The `hf:` provider of a `transformers` language model's repo: its
/// checkpoint (one file or shards) and `config.json`, with its
/// [`TOKENIZER`](crate::kits::support::construct::TOKENIZER),
/// [`TOKENIZER_CONFIG`](crate::kits::support::construct::TOKENIZER_CONFIG) and
/// [`GENERATION_CONFIG`](crate::kits::support::construct::GENERATION_CONFIG)
/// files when it has them, and its
/// [`CHAT_TEMPLATE`](crate::kits::support::construct::CHAT_TEMPLATE) file
/// under `chat_template`.
pub fn transformers_lm_hf_provider(chat_template: bool) -> crate::data::builds::HfProvider {
    use crate::{
        data::builds::{
            CHAT_TEMPLATE_JINJA,
            GENERATION_CONFIG_JSON,
            HfProvider,
            TOKENIZER_CONFIG_JSON,
            TOKENIZER_JSON,
        },
        kits::support::construct::{
            CHAT_TEMPLATE,
            GENERATION_CONFIG,
            TOKENIZER,
            TOKENIZER_CONFIG,
        },
    };
    let provider = HfProvider::new()
        .with_extra_file(TOKENIZER, TOKENIZER_JSON, TOKENIZER_JSON)
        .with_extra_file(
            TOKENIZER_CONFIG,
            TOKENIZER_CONFIG_JSON,
            TOKENIZER_CONFIG_JSON,
        )
        .with_extra_file(
            GENERATION_CONFIG,
            GENERATION_CONFIG_JSON,
            GENERATION_CONFIG_JSON,
        );
    match chat_template {
        true => provider.with_extra_file(CHAT_TEMPLATE, CHAT_TEMPLATE_JINJA, CHAT_TEMPLATE_JINJA),
        false => provider,
    }
}
