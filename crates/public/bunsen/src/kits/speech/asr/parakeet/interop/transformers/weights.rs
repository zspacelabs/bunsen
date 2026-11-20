//! Loading a `ParakeetForTDT` checkpoint.
use burn::store::burn_pack::Tensor as PackTensor;

use super::parakeet_tdt_rekey;
use crate::{
    burner::module::CanonicalKeys,
    data::formats::NamedTensorsApplied,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::asr::parakeet::ParakeetTdt,
};

/// Loads tensors named as `transformers`' `ParakeetForTDT` state dict into
/// `model`: [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) under
/// [`parakeet_tdt_rekey`], at `model`'s subsampling.
///
/// # Errors
/// As [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to);
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) when
/// `model`'s encoder does not subsample by depthwise striding.
pub fn load_parakeet_tdt_tensors(
    model: &mut ParakeetTdt,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let striding = model.encoder.subsampling.dw_striding().ok_or_else(|| {
        BunsenError::invalid_resource("a Parakeet encoder subsamples by depthwise striding")
    })?;
    let rekey = parakeet_tdt_rekey(striding);
    let rekey = rekey.then(model.canonical_rekey());
    rekey.apply_to(model, tensors)
}

/// The model of `loaded`'s `transformers` checkpoint (its `checkpoint`, one
/// file or shards), built from `config`, and its `tokenizer.json` vocabulary
/// (its `vocabulary`): the weights in the files' precision (f32), or wider as
/// `policy` asks.
///
/// # Errors
/// When a file can't be read or parsed, and as [`load_parakeet_tdt_tensors`].
#[cfg(feature = "cache")]
pub fn load_parakeet_tdt(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::speech::asr::parakeet::ParakeetTdtConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<(
    ParakeetTdt,
    crate::kits::speech::asr::parakeet::ParakeetVocabulary,
)> {
    use crate::kits::speech::asr::parakeet::{
        ParakeetVocabulary,
        pretrained::{
            CHECKPOINT,
            VOCABULARY,
        },
    };
    let tokenizer = loaded.expect(VOCABULARY)?;
    let json =
        std::fs::read_to_string(tokenizer).map_err(crate::errors::sys_at("read", tokenizer))?;
    let vocab = ParakeetVocabulary::from_tokenizer_json(&json)
        .map_err(|e| e.context(tokenizer.display()))?;
    let tensors =
        crate::data::formats::SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
    Ok((build_parakeet_tdt(tensors, config, device, policy)?, vocab))
}

/// `config`'s model with `tensors` loaded, in the files' precision (f32) or
/// wider as `policy` asks: what each source's `load_parakeet_tdt` ends with.
///
/// # Errors
/// As [`load_parakeet_tdt_tensors`] and
/// [`LoadPolicy::place`](crate::data::formats::LoadPolicy::place).
#[cfg(feature = "cache")]
pub(crate) fn build_parakeet_tdt(
    tensors: Vec<PackTensor>,
    config: &crate::kits::speech::asr::parakeet::ParakeetTdtConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<ParakeetTdt> {
    use crate::burner::module::ModuleInit;
    let mut model: ParakeetTdt = config.try_init(device)?;
    load_parakeet_tdt_tensors(&mut model, tensors)?;
    policy.place(model, device, |_| true)
}
