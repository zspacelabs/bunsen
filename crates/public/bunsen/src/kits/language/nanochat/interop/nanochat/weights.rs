//! Loading an upstream nanochat checkpoint.
use std::path::Path;

use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::{
        Device,
        FloatDType,
    },
};

use super::{
    NanoChatGeneration,
    NanoChatMetaMirror,
    nanochat_gpt_rekey,
};
use crate::{
    burner::module::{
        ModuleInit,
        Rekey,
    },
    data::formats::{
        NamedTensorsApplied,
        read_pytorch_tensors,
        stored_float_dtype,
        unit_scale_tensors,
    },
    errors::{
        BunsenError,
        BunsenResult,
        ResultContext,
    },
    kits::language::nanochat::NanoChatGpt,
};

/// Loads an upstream nanochat state dict (`tensors`, under upstream's names)
/// into `model` by [`nanochat_gpt_rekey`], every floating tensor cast to
/// `dtype`: a bf16 checkpoint loaded at bf16 takes its tensors as they are.
///
/// Upstream's norms carry no learned scale, and bunsen's do: each norm's
/// scale (a [`BunsenNormalization`](crate::blocks::norm::BunsenNormalization)'s
/// `gamma`, found by type) loads as ones, which is upstream's norm exactly,
/// in any dtype. The ones are named by their record paths, which the map's
/// last stage loads them at as they are, ahead of the model's
/// [`canonical_rekey`](crate::burner::module::CanonicalKeys::canonical_rekey).
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) naming any
/// tensor the model has no parameter for; otherwise as
/// [`Rekey::apply_as`].
pub fn load_nanochat_gpt_tensors(
    model: &mut NanoChatGpt,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: FloatDType,
) -> BunsenResult<NamedTensorsApplied> {
    let mut tensors: Vec<PackTensor> = tensors.into_iter().collect();
    let scales = unit_scale_tensors(model, is_norm_scale);
    let last = scales
        .iter()
        .fold(Rekey::keep_rest(), |rekey, t| {
            rekey.exact(t.name.clone(), t.name.clone())
        })
        .mount_canonical("", model);
    tensors.extend(scales);
    let rekey = nanochat_gpt_rekey().then(last);
    let applied = rekey.apply_as(model, tensors, Some(dtype.into()))?;
    if !applied.unused.is_empty() {
        return Err(BunsenError::unsupported(format!(
            "a nanochat state dict with tensors the model has no parameter for: {}",
            applied.unused.join(", ")
        )));
    }
    Ok(applied)
}

/// Whether the parameter at `path`, under the module types `containers`
/// (`Struct:NanoChatGpt.…`), is a norm's scale.
fn is_norm_scale(
    path: &str,
    containers: &str,
) -> bool {
    path.ends_with(".gamma")
        && containers
            .split('.')
            .any(|c| c == "Enum:BunsenNormalization")
}

/// An upstream nanochat checkpoint, its `meta_<step>.json` at `meta` and its
/// `model_<step>.pt` at `model`, as a [`NanoChatGpt`] on `device`: its
/// config from the meta and the state dict's names
/// ([`NanoChatMetaMirror::generation`], [`NanoChatMetaMirror::gpt_config`]),
/// built, and loaded ([`load_nanochat_gpt_tensors`]) at the dtype `policy`
/// resolves the state dict's stored dtype to on `device`
/// ([`LoadPolicy::resolve`](crate::data::formats::LoadPolicy::resolve)).
///
/// # Errors
/// As [`NanoChatMetaMirror::from_path`],
/// [`read_pytorch_tensors`], [`NanoChatMetaMirror::generation`],
/// [`LoadPolicy::resolve`](crate::data::formats::LoadPolicy::resolve)
/// and [`load_nanochat_gpt_tensors`]; each names the file it is about.
pub fn load_nanochat_gpt_from_paths(
    meta: &Path,
    model: &Path,
    device: &Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<(NanoChatGpt, NanoChatGeneration)> {
    let meta_data = NanoChatMetaMirror::from_path(meta)?;
    let tensors = read_pytorch_tensors(model)?;
    let names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
    let generation = meta_data
        .generation(&names)
        .with_context(|| format!("{} and {}", meta.display(), model.display()))?;
    let stored = stored_float_dtype(&tensors).unwrap_or(FloatDType::F32);
    let dtype = policy
        .resolve(stored, device, |_| true)
        .with_context(|| model.display().to_string())?;
    let mut gpt: NanoChatGpt = meta_data.gpt_config(generation).try_init(device)?;
    load_nanochat_gpt_tensors(&mut gpt, tensors, dtype)
        .with_context(|| model.display().to_string())?;
    Ok((gpt, generation))
}

/// The model of `loaded`'s checkpoint, its `meta` and its `checkpoint`, as
/// [`load_nanochat_gpt_from_paths`] reads them: a checkpoint carries its own
/// geometry, in its meta.
///
/// # Errors
/// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a set without
/// either file; otherwise as [`load_nanochat_gpt_from_paths`].
#[cfg(feature = "cache")]
pub fn load_nanochat_gpt(
    loaded: &crate::data::resources::LoadedResourceSet,
    device: &Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<(NanoChatGpt, NanoChatGeneration)> {
    use crate::kits::language::nanochat::{
        CHECKPOINT,
        META,
    };
    load_nanochat_gpt_from_paths(
        loaded.expect(META)?,
        loaded.expect(CHECKPOINT)?,
        device,
        policy,
    )
}
