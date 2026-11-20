//! Loading a Qwen3-TTS model file and the codec's file.

use alloc::{
    format,
    string::String,
    vec::Vec,
};

use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::{
    QWEN3_TTS_CODE_PREDICTOR_PREFIX,
    QWEN3_TTS_CODEC_DECODER_PREFIX,
    QWEN3_TTS_TALKER_PREFIX,
    code_predictor_rekey,
    codec_decoder_rekey,
    talker_rekey,
};
use crate::{
    blocks::transformers::decoders::depth::DepthDecoder,
    burner::module::Rekey,
    data::formats::NamedTensorsApplied,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::tts::qwen3_tts::{
        Qwen3TtsModel,
        codec::{
            Qwen3TtsCodecDecoder,
            Qwen3TtsCodecDecoderContractConfig,
        },
        talker::Qwen3TtsTalker,
    },
};

/// Loads a model file's talker tensors (`talker.*` but the code
/// predictor's; any others are dropped) into `model`: the names mapped
/// ([`talker_rekey`]), every floating tensor in `dtype`, or, when
/// `None`, in the dtype `model` was built in.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) as
/// [`Rekey::load`]; and naming any talker tensor the model has no parameter
/// for.
pub fn load_talker_tensors(
    model: &mut Qwen3TtsTalker,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let tensors: Vec<PackTensor> = tensors
        .into_iter()
        .filter(|t| {
            t.name.starts_with(QWEN3_TTS_TALKER_PREFIX)
                && !t.name.starts_with(QWEN3_TTS_CODE_PREDICTOR_PREFIX)
        })
        .collect();
    let rekey = Rekey::drop_rest()
        .mount(QWEN3_TTS_TALKER_PREFIX, "", talker_rekey())
        .then(Rekey::keep_rest().mount_canonical("lm.", &model.lm));
    let applied = rekey.load(model, tensors, dtype)?;
    if !applied.unused.is_empty() {
        return Err(BunsenError::invalid_resource(format!(
            "Qwen3-TTS talker tensors the model has no parameter for: {}",
            applied.unused.join(", ")
        )));
    }
    Ok(applied)
}

/// Loads a model file's code predictor tensors (`talker.code_predictor.*`;
/// any others are dropped) into `model`: the names mapped
/// ([`code_predictor_rekey`]), every floating tensor in `dtype`,
/// or, when `None`, in the dtype `model` was built in.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) as
/// [`Rekey::load`]; and naming any predictor tensor the model has no
/// parameter for.
pub fn load_code_predictor_tensors(
    model: &mut DepthDecoder,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let tensors: Vec<PackTensor> = tensors
        .into_iter()
        .filter(|t| t.name.starts_with(QWEN3_TTS_CODE_PREDICTOR_PREFIX))
        .collect();
    let rekey = Rekey::drop_rest()
        .mount(QWEN3_TTS_CODE_PREDICTOR_PREFIX, "", code_predictor_rekey())
        .then(Rekey::keep_rest().mount_canonical("", model));
    let applied = rekey.load(model, tensors, dtype)?;
    if !applied.unused.is_empty() {
        return Err(BunsenError::invalid_resource(format!(
            "Qwen3-TTS code predictor tensors the model has no parameter for: {}",
            applied.unused.join(", ")
        )));
    }
    Ok(applied)
}

/// Whether a decoder tensor (its name without the prefix) is the encoder's
/// side, which the decoder does not read: a quantizer's input projection.
fn is_encode_side(name: &str) -> bool {
    name.starts_with("quantizer.") && name.ends_with(".input_proj.weight")
}

/// Loads the codec file's tensors (`decoder.*`; the encoder's and any others
/// are dropped) into `model`, built from `config`: the names mapped
/// ([`codec_decoder_rekey`]), every floating tensor in `dtype`, or, when
/// `None`, in the dtype `model` was built in.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) as
/// [`Rekey::load`]; and naming any `decoder.*` tensor the model has no
/// parameter for.
pub fn load_codec_decoder_tensors(
    model: &mut Qwen3TtsCodecDecoder,
    config: &Qwen3TtsCodecDecoderContractConfig,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let prefix = QWEN3_TTS_CODEC_DECODER_PREFIX;
    let tensors: Vec<PackTensor> = tensors
        .into_iter()
        .filter(|t| {
            t.name
                .strip_prefix(prefix)
                .is_some_and(|name| !is_encode_side(name))
        })
        .collect();
    let rekey = Rekey::drop_rest()
        .mount(prefix, "", codec_decoder_rekey(config))
        .then(Rekey::keep_rest().mount_canonical("", model));
    let applied = rekey.load(model, tensors, dtype)?;
    if !applied.unused.is_empty() {
        return Err(BunsenError::invalid_resource(format!(
            "Qwen3-TTS codec decoder tensors the model has no parameter for: {}",
            applied.unused.join(", ")
        )));
    }
    Ok(applied)
}

/// Loads a model file's tensors into `model`: the talker's
/// ([`load_talker_tensors`]) and the code predictor's
/// ([`load_code_predictor_tensors`]); any others (a Base model's
/// speaker encoder) are dropped. Every floating tensor is placed in `dtype`,
/// or, when `None`, in the dtype `model` was built in.
///
/// # Returns
/// Both loads' paths, under the model's (`talker.`, `code_predictor.`).
///
/// # Errors
/// As the two loaders.
pub fn load_model_tensors(
    model: &mut Qwen3TtsModel,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let (predictor, talker): (Vec<_>, Vec<_>) = tensors
        .into_iter()
        .partition(|t| t.name.starts_with(QWEN3_TTS_CODE_PREDICTOR_PREFIX));
    let talker = load_talker_tensors(&mut model.talker, talker, dtype)?;
    let predictor = load_code_predictor_tensors(&mut model.code_predictor, predictor, dtype)?;
    Ok(NamedTensorsApplied {
        applied: under("talker.", talker.applied)
            .chain(under("code_predictor.", predictor.applied))
            .collect(),
        unused: Vec::new(),
    })
}

/// `paths`, each under `prefix`.
fn under(
    prefix: &'static str,
    paths: Vec<String>,
) -> impl Iterator<Item = String> {
    paths.into_iter().map(move |p| format!("{prefix}{p}"))
}

/// The model of `loaded`'s model file (its `checkpoint`), built from
/// `config`: each part in the dtype `policy` resolves its stored dtype to
/// (the release stores bf16).
///
/// # Errors
/// When the file can't be read, as
/// [`LoadPolicy::resolve`](crate::data::formats::LoadPolicy::resolve),
/// and as [`load_model_tensors`].
#[cfg(feature = "cache")]
pub fn load_model(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::speech::tts::qwen3_tts::Qwen3TtsModelConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<Qwen3TtsModel> {
    use crate::{
        burner::module::ModuleInit,
        kits::speech::tts::qwen3_tts::pretrained::CHECKPOINT,
    };
    let tensors = crate::data::formats::SafetensorsCheckpoint::single(loaded.expect(CHECKPOINT)?)
        .tensors()?;
    let dtype = stored_dtype(&tensors, device, policy)?;
    let mut model: Qwen3TtsModel = config.try_init(device)?;
    load_model_tensors(&mut model, tensors, Some(dtype))?;
    Ok(model)
}

/// The codec decoder of `loaded`'s codec file (its `codec`), built from
/// `config`, in the dtype `policy` resolves its stored dtype to (the
/// release stores f32).
///
/// # Errors
/// As [`load_model`], and as [`load_codec_decoder_tensors`].
#[cfg(feature = "cache")]
pub fn load_codec_decoder(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &Qwen3TtsCodecDecoderContractConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<Qwen3TtsCodecDecoder> {
    use crate::{
        burner::module::ModuleInit,
        kits::speech::tts::qwen3_tts::pretrained::CODEC,
    };
    let tensors =
        crate::data::formats::SafetensorsCheckpoint::single(loaded.expect(CODEC)?).tensors()?;
    let dtype = stored_dtype(&tensors, device, policy)?;
    let mut codec: Qwen3TtsCodecDecoder = config.try_init(device)?;
    load_codec_decoder_tensors(&mut codec, config, tensors, Some(dtype))?;
    Ok(codec)
}

/// The dtype `policy` resolves `tensors`' stored float dtype to (f32 when
/// they store none).
#[cfg(feature = "cache")]
fn stored_dtype(
    tensors: &[PackTensor],
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<FloatDType> {
    policy.resolve(
        crate::data::formats::stored_float_dtype(tensors).unwrap_or(FloatDType::F32),
        device,
        |_| true,
    )
}
