//! Reading an `OpenAI` Whisper `.pt`.
use std::path::{
    Path,
    PathBuf,
};

use burn::{
    store::{
        ModuleStore,
        PytorchStore,
    },
    tensor::Device,
};

use crate::{
    blocks::conv::ConvSeq1d,
    burner::module::{
        CanonicalKeys,
        ModuleInit,
    },
    data::{
        builds::CHECKPOINT,
        formats::pytorch_store_error,
        resources::LoadedResourceSet,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::asr::whisper::{
        blocks::{
            WHISPER_ENCODER_STRIDE,
            Whisper,
            WhisperApiConfig,
        },
        interop::WhisperScanDefaults,
    },
};

/// The layers of the `kind` stack (`encoder` or `decoder`): one feed-forward
/// norm each, a name the two stacks' layers share (their attentions'
/// projections are named apart).
fn block_layers_from_keys<S: AsRef<str>>(
    kind: &str,
    keys: &[S],
) -> usize {
    keys.iter()
        .filter(|k| {
            let k = k.as_ref();
            k.starts_with(&format!("{kind}.blocks.")) && k.ends_with(".mlp_norm.weight")
        })
        .count()
}

/// The key of the state dict in an `OpenAI` Whisper `.pt`.
pub const OPENAI_WHISPER_STATE_DICT_KEY: &str = "model_state_dict";

/// The store over an `OpenAI` Whisper `.pt` at `path`: its state dict, under
/// remaps that name the tensors canonically ([`CanonicalKeys`]), which a
/// model's [`canonical_rekey`](CanonicalKeys::canonical_rekey) takes onto its
/// record paths.
fn open_state_dict(path: &Path) -> PytorchStore {
    let path: PathBuf = path.to_path_buf();
    PytorchStore::from_file(path.clone())
        .map_indices_contiguous(false)
        // PyTorch's 1-based `conv1`/`conv2` map to the 0-based
        // `ConvSeq1d` blocks of the encoder head.
        .with_key_remapping(
            r"encoder\.conv1\.",
            format!("{}.", ConvSeq1d::block_conv_path("encoder.head", 0)),
        )
        .with_key_remapping(
            r"encoder\.conv2\.",
            format!("{}.", ConvSeq1d::block_conv_path("encoder.head", 1)),
        )
        // The encoder's attention is a decoder block's (`c_{q,k,v,proj}`);
        // the decoder's, burn's multi-head attention's.
        .with_key_remapping(
            r"encoder\.blocks\.(\d+)\.attn\.query\.",
            "encoder.blocks.$1.attn.c_q.",
        )
        .with_key_remapping(
            r"encoder\.blocks\.(\d+)\.attn\.key\.",
            "encoder.blocks.$1.attn.c_k.",
        )
        .with_key_remapping(
            r"encoder\.blocks\.(\d+)\.attn\.value\.",
            "encoder.blocks.$1.attn.c_v.",
        )
        .with_key_remapping(
            r"encoder\.blocks\.(\d+)\.attn\.out\.",
            "encoder.blocks.$1.attn.c_proj.",
        )
        .with_key_remapping(r"\.attn\.out", ".attn.output")
        .with_key_remapping(r"\.cross_attn\.out", ".cross_attn.output")
        .with_key_remapping(r"\.mlp\.2", ".mlp.linear2")
        .with_key_remapping(r"\.mlp\.0", ".mlp.linear1")
        .with_key_remapping(r"\.attn_ln\.", ".attn_norm.")
        .with_key_remapping(r"\.cross_attn_ln\.", ".cross_attn_norm.")
        .with_key_remapping(r"\.mlp_ln\.", ".mlp_norm.")
        .with_top_level_key(OPENAI_WHISPER_STATE_DICT_KEY)
}

/// The config of an `OpenAI` Whisper `.pt` at `path`, scanned from its state
/// dict: the geometry from the tensors' shapes, the layer counts from their
/// names; `defaults` declares the rest.
///
/// # Errors
/// - [`Lookup`](crate::errors::BunsenErrorKind::Lookup) when `path` is missing
///   or forbidden;
/// - [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource), with
///   the [`PytorchStoreError`](burn::store::PytorchStoreError) as its cause,
///   when the file is not a `PyTorch` checkpoint; without one, naming a tensor
///   the layout requires and the checkpoint lacks, or has at another rank;
/// - [`Sys`](crate::errors::BunsenErrorKind::Sys) when reading it fails
///   otherwise.
pub fn whisper_config_from_path(
    path: impl AsRef<Path>,
    defaults: &WhisperScanDefaults,
) -> BunsenResult<WhisperApiConfig> {
    let path = path.as_ref();
    let store = open_state_dict(path);
    let mut store = store;

    let keys = store.keys().map_err(pytorch_store_error(path))?;

    // The shape of a tensor the layout requires, at the rank it
    // requires: the same refusals as the safetensors scanner's.
    let mut shape = |tensor: &str, rank: usize| -> BunsenResult<Vec<usize>> {
        let shape = store
            .get_tensor(tensor)
            .map_err(pytorch_store_error(path))?
            .ok_or_else(|| {
                BunsenError::invalid_resource(format!(
                    "{}: not an OpenAI Whisper checkpoint: no {tensor}",
                    path.display()
                ))
            })?
            .shape
            .to_vec();
        if shape.len() == rank {
            Ok(shape)
        } else {
            Err(BunsenError::invalid_resource(format!(
                "{}: {tensor} has shape {shape:?}, not rank {rank}",
                path.display()
            )))
        }
    };

    let conv1 = shape("encoder.head.blocks.0.conv.weight", 3)?;
    let (d_model, n_mels) = (conv1[0], conv1[1]);
    let vocab_size = shape("decoder.token_embedding.weight", 2)?[0];
    let max_audio_ctx = shape("encoder.positional_embedding", 2)?[0] * WHISPER_ENCODER_STRIDE;
    let max_text_ctx = shape("decoder.positional_embedding", 2)?[0];

    let encoder_layers = block_layers_from_keys("encoder", &keys);
    let decoder_layers = block_layers_from_keys("decoder", &keys);

    Ok(defaults.declare(WhisperApiConfig::new(
        n_mels,
        vocab_size,
        d_model,
        max_audio_ctx,
        encoder_layers,
        max_text_ctx,
        decoder_layers,
    )))
}

/// Loads the tensors of the `OpenAI` Whisper `.pt` at `path` into `module`.
///
/// # Errors
/// As [`whisper_config_from_path`] for the file, and
/// [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to).
pub fn load_whisper_tensors(
    module: &mut Whisper,
    path: impl AsRef<Path>,
) -> BunsenResult<()> {
    let path = path.as_ref();
    let mut store = open_state_dict(path);
    let tensors: Vec<_> = store
        .get_all_tensors()
        .map_err(pytorch_store_error(path))?
        .iter()
        .map(|(name, tensor)| {
            let mut tensor = tensor.clone();
            tensor.name = name.clone();
            tensor
        })
        .collect();
    let rekey = module.canonical_rekey();
    rekey.apply_to(module, tensors)?;
    Ok(())
}

/// The `.pt` in `loaded`, under [`CHECKPOINT`], as a model of `config` on
/// `device`: loaded at the file's precision (`OpenAI`'s ship f16), then
/// placed under `policy`.
///
/// # Errors
/// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) when `loaded` has no
/// checkpoint; as [`ModuleInit::try_init`] for `config`,
/// [`load_whisper_tensors`] and
/// [`LoadPolicy::place`](crate::data::formats::LoadPolicy::place).
pub fn load_whisper(
    loaded: &LoadedResourceSet,
    config: &WhisperApiConfig,
    device: &Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<Whisper> {
    let mut module: Whisper = config.try_init(device)?;
    load_whisper_tensors(&mut module, loaded.expect(CHECKPOINT)?)?;
    policy.place(module, device, |_| true)
}
