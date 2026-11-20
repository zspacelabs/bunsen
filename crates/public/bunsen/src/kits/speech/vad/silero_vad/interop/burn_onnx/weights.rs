//! Loading a `burn-onnx`-keyed burnpack.
use std::path::Path;

use burn::{
    store::{
        BurnpackStore,
        KeyRemapper,
        ModuleSnapshot,
        burn_pack::Error as BurnpackError,
    },
    tensor::Device,
};

use crate::{
    blocks::conv::ConvSeq1d,
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenErrorKind,
        BunsenResult,
        ResultContext,
        WithOkOrPanic,
        sys_at,
    },
    kits::speech::vad::silero_vad::{
        SileroVad,
        SileroVadCollection,
        SileroVadSignalConfig,
    },
};

/// The 16 kHz branch's `burn-onnx` node names onto a [`SileroVad`]'s module
/// paths.
pub fn burn_onnx_16khz_remapper() -> KeyRemapper {
    KeyRemapper::from_patterns(vec![
        ("conv1d37", "stft".to_string()),
        ("conv1d38", ConvSeq1d::block_conv_path("encoder", 0)),
        ("conv1d39", ConvSeq1d::block_conv_path("encoder", 1)),
        ("conv1d40", ConvSeq1d::block_conv_path("encoder", 2)),
        ("conv1d41", ConvSeq1d::block_conv_path("encoder", 3)),
        ("linear13", "lstm.hidden".to_string()),
        ("linear14", "lstm.features".to_string()),
        ("conv1d42", "decoder".to_string()),
    ])
    .ok_or_panic()
}

/// The 8 kHz branch's `burn-onnx` node names onto a [`SileroVad`]'s module
/// paths.
pub fn burn_onnx_8khz_remapper() -> KeyRemapper {
    KeyRemapper::from_patterns(vec![
        ("conv1d43", "stft".to_string()),
        ("conv1d44", ConvSeq1d::block_conv_path("encoder", 0)),
        ("conv1d45", ConvSeq1d::block_conv_path("encoder", 1)),
        ("conv1d46", ConvSeq1d::block_conv_path("encoder", 2)),
        ("conv1d47", ConvSeq1d::block_conv_path("encoder", 3)),
        ("linear15", "lstm.hidden".to_string()),
        ("linear16", "lstm.features".to_string()),
        ("conv1d48", "decoder".to_string()),
    ])
    .ok_or_panic()
}

/// `cfg`'s model, loaded from a burnpack store under `remapper`.
///
/// # Errors
/// - as `cfg`'s [`try_init`](ModuleInit::try_init);
/// - [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource), with
///   the [`BurnpackError`] as its cause, when the store is not a burnpack or
///   lacks a tensor the model needs;
///   [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
///   burnpack version this build does not read;
///   [`Sys`](crate::errors::BunsenErrorKind::Sys) when reading it fails.
pub fn load_silero_vad_from_burnpack<C>(
    store: BurnpackStore,
    cfg: C,
    remapper: KeyRemapper,
    device: &Device,
) -> BunsenResult<SileroVad>
where
    C: ModuleInit<SileroVad>,
{
    let mut store = store.remap(remapper);
    let mut module = cfg.try_init(device)?;
    module.load_from(&mut store).map_err(burnpack_load_error)?;
    Ok(module)
}

/// [`load_silero_vad_from_burnpack`] of the file at `path`, under a frame
/// naming it.
fn load_silero_vad_from_burnpack_path<C>(
    path: &Path,
    cfg: C,
    remapper: KeyRemapper,
    device: &Device,
) -> BunsenResult<SileroVad>
where
    C: ModuleInit<SileroVad>,
{
    std::fs::metadata(path).map_err(sys_at("open", path))?;
    load_silero_vad_from_burnpack(BurnpackStore::from_file(path), cfg, remapper, device)
        .with_context(|| format!("loading {}", path.display()))
}

/// Sorts a [`BurnpackError`] from loading a module by what it means.
fn burnpack_load_error(error: BurnpackError) -> BunsenError {
    let kind = match &error {
        BurnpackError::IoError(_) => BunsenErrorKind::Sys,
        BurnpackError::InvalidVersion => BunsenErrorKind::Unsupported,
        _ => BunsenErrorKind::InvalidResource,
    };
    BunsenError::from_cause(kind, error)
}

/// The 16 kHz model of a burnpack's bytes.
///
/// # Errors
/// As [`load_silero_vad_from_burnpack`].
pub fn load_silero_vad_16khz_from_burnpack_bytes(
    bytes: burn::tensor::Bytes,
    device: &Device,
) -> BunsenResult<SileroVad> {
    load_silero_vad_from_burnpack(
        BurnpackStore::from_bytes(Some(bytes)),
        SileroVadSignalConfig::standard_16khz(),
        burn_onnx_16khz_remapper(),
        device,
    )
}

/// The 16 kHz model of the burnpack at `path`.
///
/// # Errors
/// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) when `path` is
/// missing or forbidden; otherwise as [`load_silero_vad_from_burnpack`],
/// under a frame naming the file.
pub fn load_silero_vad_16khz_from_burnpack_path(
    path: impl AsRef<Path>,
    device: &Device,
) -> BunsenResult<SileroVad> {
    load_silero_vad_from_burnpack_path(
        path.as_ref(),
        SileroVadSignalConfig::standard_16khz(),
        burn_onnx_16khz_remapper(),
        device,
    )
}

/// The 8 kHz model of a burnpack's bytes.
///
/// # Errors
/// As [`load_silero_vad_from_burnpack`].
pub fn load_silero_vad_8khz_from_burnpack_bytes(
    bytes: burn::tensor::Bytes,
    device: &Device,
) -> BunsenResult<SileroVad> {
    load_silero_vad_from_burnpack(
        BurnpackStore::from_bytes(Some(bytes)),
        SileroVadSignalConfig::standard_8khz(),
        burn_onnx_8khz_remapper(),
        device,
    )
}

/// The 8 kHz model of the burnpack at `path`.
///
/// # Errors
/// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) when `path` is
/// missing or forbidden; otherwise as [`load_silero_vad_from_burnpack`],
/// under a frame naming the file.
pub fn load_silero_vad_8khz_from_burnpack_path(
    path: impl AsRef<Path>,
    device: &Device,
) -> BunsenResult<SileroVad> {
    load_silero_vad_from_burnpack_path(
        path.as_ref(),
        SileroVadSignalConfig::standard_8khz(),
        burn_onnx_8khz_remapper(),
        device,
    )
}

/// Both branches, 16 kHz and 8 kHz, of a burnpack's bytes.
///
/// # Errors
/// As [`load_silero_vad_from_burnpack`].
pub fn load_silero_vad_collection_from_burnpack_bytes(
    bytes: burn::tensor::Bytes,
    device: &Device,
) -> BunsenResult<SileroVadCollection> {
    Ok(SileroVadCollection::from_16khz_8khz(
        load_silero_vad_16khz_from_burnpack_bytes(bytes.clone(), device)?,
        load_silero_vad_8khz_from_burnpack_bytes(bytes, device)?,
    ))
}

/// Both branches, 16 kHz and 8 kHz, of the burnpack at `path`.
///
/// # Errors
/// As [`load_silero_vad_16khz_from_burnpack_path`].
pub fn load_silero_vad_collection_from_burnpack_path(
    path: impl AsRef<Path>,
    device: &Device,
) -> BunsenResult<SileroVadCollection> {
    let path = path.as_ref();
    Ok(SileroVadCollection::from_16khz_8khz(
        load_silero_vad_16khz_from_burnpack_path(path, device)?,
        load_silero_vad_8khz_from_burnpack_path(path, device)?,
    ))
}
