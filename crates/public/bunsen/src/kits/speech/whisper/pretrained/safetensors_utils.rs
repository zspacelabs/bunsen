//! # Reading a Hugging Face Whisper checkpoint
//!
//! `transformers` exports a Whisper model as `model.safetensors` under its
//! own parameter names: `model.encoder.layers.N.self_attn.q_proj.weight`
//! where `OpenAI`'s `.pt` has `encoder.blocks.N.attn.query.weight`, `fc1`
//! for `mlp.0`, `embed_positions.weight` for `positional_embedding`, and
//! so on. [`SafetensorsWhisperScanner`] reads that layout: the geometry
//! from the file's header alone, the weights through `burn-store` with
//! the names mapped to bunsen's. The tied output projection, when a repo
//! carries it, is skipped: bunsen's decoder projects through its token
//! embedding, as upstream's does.
//!
//! A safetensors file is contiguous and row-major, so the transposition
//! the `PyTorch` adapter applies to a `Linear` weight is right as it
//! stands; the strided-view repair `OpenAI`'s `.pt` files need does not
//! apply here.

use std::{
    collections::BTreeMap,
    io::Read,
    path::Path,
};

use burn::{
    config::Config,
    prelude::Backend,
};
use burn_store::{
    ModuleSnapshot,
    PyTorchToBurnAdapter,
    SafetensorsStore,
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::whisper::blocks::{
        AUDIO_ENCODER_STRIDE,
        WHISPER_DEFAULT_D_MODEL,
        Whisper,
        WhisperApiConfig,
        WhisperFrontEndConfig,
        WhisperTokenLayoutConfig,
    },
};

/// The key remappings from `transformers`' Whisper names to bunsen's, in
/// the order they apply.
///
/// `layers.N` is `blocks.N`; `self_attn.{q,k,v,out}_proj` is
/// `attn.{query,key,value,output}` and `encoder_attn` is `cross_attn`; the
/// three layer norms are `attn_ln`, `cross_attn_ln` and `mlp_ln`; `fc1`
/// and `fc2` are `mlp.linear1` and `mlp.linear2`; the two convolutions
/// are the encoder head's blocks; the positional tables lose their
/// `.weight`; the final norms are `ln_post` and `ln`; and the `model.`
/// prefix goes.
pub const HF_TO_BUNSEN: &[(&str, &str)] = &[
    (r"^model\.", ""),
    (r"^encoder\.conv1\.", "encoder.head.blocks.0.conv."),
    (r"^encoder\.conv2\.", "encoder.head.blocks.1.conv."),
    (r"\.embed_positions\.weight$", ".positional_embedding"),
    (r"\.embed_tokens\.", ".token_embedding."),
    (r"\.layers\.(\d+)\.", ".blocks.$1."),
    (r"\.self_attn\.q_proj\.", ".attn.query."),
    (r"\.self_attn\.k_proj\.", ".attn.key."),
    (r"\.self_attn\.v_proj\.", ".attn.value."),
    (r"\.self_attn\.out_proj\.", ".attn.output."),
    (r"\.encoder_attn\.q_proj\.", ".cross_attn.query."),
    (r"\.encoder_attn\.k_proj\.", ".cross_attn.key."),
    (r"\.encoder_attn\.v_proj\.", ".cross_attn.value."),
    (r"\.encoder_attn\.out_proj\.", ".cross_attn.output."),
    (r"\.self_attn_layer_norm\.", ".attn_ln."),
    (r"\.encoder_attn_layer_norm\.", ".cross_attn_ln."),
    (r"\.final_layer_norm\.", ".mlp_ln."),
    (r"\.fc1\.", ".mlp.linear1."),
    (r"\.fc2\.", ".mlp.linear2."),
    (r"^encoder\.layer_norm\.", "encoder.ln_post."),
    (r"^decoder\.layer_norm\.", "decoder.ln."),
];

/// The key remappings from bunsen's names to `transformers`': the inverse
/// of [`HF_TO_BUNSEN`], for writing a file in that layout.
pub const BUNSEN_TO_HF: &[(&str, &str)] = &[
    (r"^encoder\.head\.blocks\.0\.conv\.", "encoder.conv1."),
    (r"^encoder\.head\.blocks\.1\.conv\.", "encoder.conv2."),
    (r"\.positional_embedding$", ".embed_positions.weight"),
    (r"\.token_embedding\.", ".embed_tokens."),
    (r"\.blocks\.(\d+)\.", ".layers.$1."),
    (r"\.attn\.query\.", ".self_attn.q_proj."),
    (r"\.attn\.key\.", ".self_attn.k_proj."),
    (r"\.attn\.value\.", ".self_attn.v_proj."),
    (r"\.attn\.output\.", ".self_attn.out_proj."),
    (r"\.cross_attn\.query\.", ".encoder_attn.q_proj."),
    (r"\.cross_attn\.key\.", ".encoder_attn.k_proj."),
    (r"\.cross_attn\.value\.", ".encoder_attn.v_proj."),
    (r"\.cross_attn\.output\.", ".encoder_attn.out_proj."),
    (r"\.attn_ln\.", ".self_attn_layer_norm."),
    (r"\.cross_attn_ln\.", ".encoder_attn_layer_norm."),
    (r"\.mlp_ln\.", ".final_layer_norm."),
    (r"\.mlp\.linear1\.", ".fc1."),
    (r"\.mlp\.linear2\.", ".fc2."),
    (r"^encoder\.ln_post\.", "encoder.layer_norm."),
    (r"^decoder\.ln\.", "decoder.layer_norm."),
    (r"^", "model."),
];

/// One tensor as the safetensors header describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetensorsEntry {
    /// The element type, as the file spells it: `F16`, `F32`, `BF16`.
    pub dtype: String,

    /// The shape.
    pub shape: Vec<usize>,
}

/// The longest header accepted, `safetensors`' own bound: 100 MB.
const MAX_HEADER_LEN: u64 = 100_000_000;

/// The header of a safetensors file: every tensor's name, element type
/// and shape, read without touching the data.
///
/// # Errors
/// [`BunsenError::External`] for a file that cannot be read;
/// [`BunsenError::Invalid`] for a header that is not a safetensors one.
pub fn safetensors_header(path: &Path) -> BunsenResult<BTreeMap<String, SafetensorsEntry>> {
    let mut file = std::fs::File::open(path).map_err(BunsenError::external)?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len).map_err(BunsenError::external)?;
    let len = u64::from_le_bytes(len);
    // The header is JSON of a few hundred bytes per tensor, and never
    // longer than the file it heads: a length past either is not a header
    // and must not be allocated for.
    let file_len = file.metadata().map_err(BunsenError::external)?.len();
    if len > MAX_HEADER_LEN || len.saturating_add(8) > file_len {
        return Err(BunsenError::Invalid(format!(
            "{}: not a safetensors file: a header of {len} bytes",
            path.display()
        )));
    }
    let len = usize::try_from(len).map_err(BunsenError::external)?;
    let mut header = vec![0u8; len];
    file.read_exact(&mut header)
        .map_err(BunsenError::external)?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
        .map_err(|e| {
            BunsenError::Invalid(format!("{}: not a safetensors header: {e}", path.display()))
        })?;

    let mut entries = BTreeMap::new();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = value
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| BunsenError::Invalid(format!("{}: {name}: no dtype", path.display())))?
            .to_string();
        let shape = value
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| BunsenError::Invalid(format!("{}: {name}: no shape", path.display())))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .and_then(|d| usize::try_from(d).ok())
                    .ok_or_else(|| {
                        BunsenError::Invalid(format!(
                            "{}: {name}: a shape that is not usize",
                            path.display()
                        ))
                    })
            })
            .collect::<BunsenResult<Vec<usize>>>()?;
        entries.insert(name, SafetensorsEntry { dtype, shape });
    }
    Ok(entries)
}

/// The shape of `name` in `header`, which must be there at rank `rank`.
fn shape_of<'a>(
    header: &'a BTreeMap<String, SafetensorsEntry>,
    path: &Path,
    name: &str,
    rank: usize,
) -> BunsenResult<&'a [usize]> {
    let shape = header
        .get(name)
        .map(|e| e.shape.as_slice())
        .ok_or_else(|| {
            BunsenError::Invalid(format!(
                "{}: not a transformers Whisper checkpoint: no {name}",
                path.display()
            ))
        })?;
    if shape.len() == rank {
        Ok(shape)
    } else {
        Err(BunsenError::Invalid(format!(
            "{}: {name} has shape {shape:?}, not rank {rank}",
            path.display()
        )))
    }
}

/// Reads a `transformers` Whisper checkpoint, `model.safetensors`.
#[derive(Debug, Config)]
pub struct SafetensorsWhisperScanner {
    /// The audio front end to declare on the scanned config.
    ///
    /// A checkpoint does not record it; `OpenAI`'s were all trained with
    /// the default, and so is every conversion of them.
    #[config(default = "WhisperFrontEndConfig::new()")]
    pub front_end: WhisperFrontEndConfig,

    /// The token layout to declare on the scanned config: upstream's.
    #[config(default = "WhisperTokenLayoutConfig::new()")]
    pub token_layout: WhisperTokenLayoutConfig,

    /// Head dimensionality: every `OpenAI` checkpoint's is 64, and a
    /// checkpoint does not record it.
    #[config(default = "WHISPER_DEFAULT_D_MODEL")]
    pub d_head: usize,
}

impl SafetensorsWhisperScanner {
    /// The store the weights are read through: the file, `transformers`'
    /// names mapped to bunsen's, `PyTorch`'s `[out, in]` `Linear` weights
    /// transposed to burn's, and `weight`/`bias` on a norm read as
    /// `gamma`/`beta`.
    fn store(path: &Path) -> SafetensorsStore {
        let mut store = SafetensorsStore::from_file(path.to_path_buf())
            .with_from_adapter(PyTorchToBurnAdapter)
            .allow_partial(true);
        for (from, to) in HF_TO_BUNSEN {
            store = store.with_key_remapping(*from, *to);
        }
        store
    }

    /// Scans a checkpoint for its config from the file's header: the
    /// geometry from the tensors' shapes, the layer counts from their
    /// names. No tensor data is read.
    ///
    /// # Errors
    /// As [`safetensors_header`]; [`BunsenError::Invalid`] naming a tensor
    /// the layout requires and the file lacks, or has at another rank.
    pub fn scan_cfg<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> BunsenResult<WhisperApiConfig> {
        let path = path.as_ref();
        let header = safetensors_header(path)?;

        let conv1 = shape_of(&header, path, "model.encoder.conv1.weight", 3)?;
        let (d_model, n_mels) = (conv1[0], conv1[1]);
        let vocab_size = shape_of(&header, path, "model.decoder.embed_tokens.weight", 2)?[0];
        let max_audio_ctx = shape_of(&header, path, "model.encoder.embed_positions.weight", 2)?[0]
            * AUDIO_ENCODER_STRIDE;
        let max_text_ctx = shape_of(&header, path, "model.decoder.embed_positions.weight", 2)?[0];
        let layers = |side: &str| {
            let prefix = format!("model.{side}.layers.");
            header
                .keys()
                .filter(|k| k.starts_with(&prefix) && k.ends_with(".self_attn.k_proj.weight"))
                .count()
        };

        Ok(WhisperApiConfig::new(
            n_mels,
            vocab_size,
            d_model,
            max_audio_ctx,
            layers("encoder"),
            max_text_ctx,
            layers("decoder"),
        )
        .with_d_head(self.d_head)
        .with_front_end(self.front_end.clone())
        .with_token_layout(self.token_layout.clone()))
    }

    /// Loads a checkpoint into a model at the precision the file stores.
    ///
    /// # Errors
    /// As [`scan_cfg`](Self::scan_cfg); [`BunsenError::External`] from the
    /// store; [`BunsenError::Invalid`] naming a parameter the model has and
    /// the file does not.
    pub fn load<B: Backend, P: AsRef<Path>>(
        &self,
        path: P,
        device: &B::Device,
    ) -> BunsenResult<(Whisper<B>, WhisperApiConfig)> {
        let path = path.as_ref();
        let cfg = self.scan_cfg(path)?;
        let mut module: Whisper<B> = cfg.try_init(device)?;
        let mut store = Self::store(path);
        let result = module
            .load_from(&mut store)
            .map_err(BunsenError::external)?;
        if !result.missing.is_empty() {
            let missing: Vec<&str> = result.missing.iter().map(|(p, _)| p.as_str()).collect();
            return Err(BunsenError::Invalid(format!(
                "{}: the checkpoint lacks {} of the model's parameters: {}",
                path.display(),
                missing.len(),
                missing.join(", ")
            )));
        }
        Ok((module, cfg))
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tensor;
    use burn_store::BurnToPyTorchAdapter;

    use super::*;
    use crate::{
        kits::speech::whisper::WhisperMeta,
        support::testing::{
            CpuBackend,
            default_device,
        },
    };

    /// Eight-wide heads over a 16-wide model, which the scanner must be
    /// told: a checkpoint does not record its head size.
    const D_HEAD: usize = 8;

    /// A toy geometry, small enough to write and read in a test.
    fn toy() -> WhisperApiConfig {
        WhisperApiConfig::new(80, 64, 16, 40, 2, 12, 3).with_d_head(D_HEAD)
    }

    /// Writes `model` as `transformers` would: its names, `PyTorch`'s
    /// weight layout.
    fn write_hf(
        model: &Whisper<CpuBackend>,
        path: &Path,
    ) {
        let mut store =
            SafetensorsStore::from_file(path.to_path_buf()).with_to_adapter(BurnToPyTorchAdapter);
        for (from, to) in BUNSEN_TO_HF {
            store = store.with_key_remapping(*from, *to);
        }
        model.save_into(&mut store).unwrap();
    }

    /// A toy model written in `transformers`' layout comes back with the
    /// same geometry and the same weights, through the names and the
    /// transposition both ways.
    #[test]
    fn test_a_transformers_layout_round_trips() {
        let device = default_device();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let model: Whisper<CpuBackend> = toy().try_init(&device).unwrap();
        write_hf(&model, &path);

        let header = safetensors_header(&path).unwrap();
        assert!(header.contains_key("model.encoder.conv1.weight"));
        assert!(header.contains_key("model.encoder.layers.0.self_attn.q_proj.weight"));
        assert!(header.contains_key("model.decoder.layers.2.encoder_attn.out_proj.bias"));
        assert!(header.contains_key("model.decoder.layers.0.fc1.weight"));
        assert!(header.contains_key("model.encoder.layer_norm.weight"));
        assert!(header.contains_key("model.decoder.embed_positions.weight"));
        assert!(
            header.keys().all(|k| k.starts_with("model.")),
            "{:?}",
            header.keys().find(|k| !k.starts_with("model."))
        );
        assert_eq!(
            header["model.encoder.layers.0.fc1.weight"].shape,
            [64, 16],
            "PyTorch's [out, in]"
        );
        assert_eq!(header["model.decoder.embed_tokens.weight"].shape, [64, 16]);

        let scanner = SafetensorsWhisperScanner::new().with_d_head(D_HEAD);
        let cfg = scanner.scan_cfg(&path).unwrap();
        assert_eq!(cfg.geometry(), toy().geometry());

        let (loaded, cfg) = scanner.load::<CpuBackend, _>(&path, &device).unwrap();
        assert_eq!(cfg.geometry(), toy().geometry());
        assert_eq!(loaded.n_mels(), 80);
        assert_eq!(loaded.vocab_size(), 64);

        let same = |a: Tensor<CpuBackend, 2>, b: Tensor<CpuBackend, 2>| {
            assert_eq!(a.dims(), b.dims());
            a.into_data().assert_eq(&b.into_data(), false);
        };
        same(
            model.encoder.blocks[0].attn.query.weight.val(),
            loaded.encoder.blocks[0].attn.query.weight.val(),
        );
        same(
            model.decoder.blocks[2].cross_attn.output.weight.val(),
            loaded.decoder.blocks[2].cross_attn.output.weight.val(),
        );
        same(
            model.decoder.blocks[1].mlp.linear2.weight.val(),
            loaded.decoder.blocks[1].mlp.linear2.weight.val(),
        );
        same(
            model.encoder.positional_embedding.val(),
            loaded.encoder.positional_embedding.val(),
        );
        same(
            model.decoder.token_embedding.weight.val(),
            loaded.decoder.token_embedding.weight.val(),
        );
    }

    /// A safetensors file of something else is refused by name, from its
    /// header; a file that is not safetensors at all is refused too.
    #[test]
    fn test_a_file_of_another_shape_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("other.safetensors");
        let header = br#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 24]);
        std::fs::write(&path, bytes).unwrap();

        let parsed = safetensors_header(&path).unwrap();
        assert_eq!(
            parsed["weight"],
            SafetensorsEntry {
                dtype: "F32".to_string(),
                shape: vec![2, 3]
            }
        );
        let err = SafetensorsWhisperScanner::new()
            .scan_cfg(&path)
            .unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("no model.encoder.conv1.weight")),
            "{err}"
        );

        // "not a sa" as a little-endian length is enormous: refused, not
        // allocated for.
        let junk = dir.path().join("junk.safetensors");
        std::fs::write(&junk, b"not a safetensors file at all").unwrap();
        let err = safetensors_header(&junk).unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("not a safetensors file")),
            "{err}"
        );
    }
}
