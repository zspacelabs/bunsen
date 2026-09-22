//! # Reading a `transformers` Whisper checkpoint
//!
//! `transformers` exports a Whisper model as `model.safetensors`, or as
//! shards with an index, under its own parameter names:
//! `model.encoder.layers.N.self_attn.q_proj.weight` where `OpenAI`'s `.pt`
//! has `encoder.blocks.N.attn.query.weight`, `fc1` for `mlp.0`,
//! `embed_positions.weight` for `positional_embedding`, and so on.
//! [`SafetensorsWhisperScanner`] reads that layout: the geometry from the
//! files' headers alone, the weights through the generic
//! [`SafetensorsCheckpoint`] with the names mapped to bunsen's. The tied
//! output projection, when a repo carries it, is skipped: bunsen's decoder
//! projects through its token embedding, as upstream's does.

use burn::{
    config::Config,
    prelude::Backend,
};
use burn_store::PyTorchToBurnAdapter;

use crate::{
    burner::module::ModuleInit,
    data::pretrained::SafetensorsCheckpoint,
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

/// Reads a `transformers` Whisper checkpoint: `model.safetensors`, or its
/// shards.
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
    /// Scans a checkpoint for its config from the files' headers: the
    /// geometry from the tensors' shapes, the layer counts from their
    /// names. No tensor data is read.
    ///
    /// # Errors
    /// As [`SafetensorsCheckpoint::headers`]; [`BunsenError::Invalid`]
    /// naming a tensor the layout requires and the checkpoint lacks, or
    /// has at another rank.
    pub fn scan_cfg(
        &self,
        checkpoint: &SafetensorsCheckpoint,
    ) -> BunsenResult<WhisperApiConfig> {
        let header = checkpoint.headers()?;
        let name = || {
            checkpoint
                .shards
                .first()
                .map(|s| s.display().to_string())
                .unwrap_or_default()
        };
        let shape = |tensor: &str, rank: usize| -> BunsenResult<&[usize]> {
            let shape = header
                .get(tensor)
                .map(|e| e.shape.as_slice())
                .ok_or_else(|| {
                    BunsenError::Invalid(format!(
                        "{}: not a transformers Whisper checkpoint: no {tensor}",
                        name()
                    ))
                })?;
            if shape.len() == rank {
                Ok(shape)
            } else {
                Err(BunsenError::Invalid(format!(
                    "{}: {tensor} has shape {shape:?}, not rank {rank}",
                    name()
                )))
            }
        };

        let conv1 = shape("model.encoder.conv1.weight", 3)?;
        let (d_model, n_mels) = (conv1[0], conv1[1]);
        let vocab_size = shape("model.decoder.embed_tokens.weight", 2)?[0];
        let max_audio_ctx =
            shape("model.encoder.embed_positions.weight", 2)?[0] * AUDIO_ENCODER_STRIDE;
        let max_text_ctx = shape("model.decoder.embed_positions.weight", 2)?[0];
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

    /// Loads a checkpoint into a model at the precision the files store.
    ///
    /// # Errors
    /// As [`scan_cfg`](Self::scan_cfg) and
    /// [`SafetensorsCheckpoint::load_into`].
    pub fn load<B: Backend>(
        &self,
        checkpoint: &SafetensorsCheckpoint,
        device: &B::Device,
    ) -> BunsenResult<(Whisper<B>, WhisperApiConfig)> {
        let cfg = self.scan_cfg(checkpoint)?;
        let mut module: Whisper<B> = cfg.try_init(device)?;
        checkpoint.load_into(&mut module, |store| {
            let mut store = store.with_from_adapter(PyTorchToBurnAdapter);
            for (from, to) in HF_TO_BUNSEN {
                store = store.with_key_remapping(*from, *to);
            }
            store
        })?;
        Ok((module, cfg))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use burn::tensor::Tensor;
    use burn_store::{
        BurnToPyTorchAdapter,
        ModuleSnapshot,
        SafetensorsStore,
    };

    use super::*;
    use crate::{
        data::pretrained::safetensors_header,
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
    /// weight layout; `only` keeps the parameters matching a pattern, for
    /// a shard.
    fn write_hf(
        model: &Whisper<CpuBackend>,
        path: &Path,
        only: Option<&str>,
    ) {
        let mut store =
            SafetensorsStore::from_file(path.to_path_buf()).with_to_adapter(BurnToPyTorchAdapter);
        if let Some(pattern) = only {
            store = store.with_regex(pattern);
        }
        for (from, to) in BUNSEN_TO_HF {
            store = store.with_key_remapping(*from, *to);
        }
        model.save_into(&mut store).unwrap();
    }

    /// The parameters of two models that must agree, compared.
    fn assert_same_weights(
        a: &Whisper<CpuBackend>,
        b: &Whisper<CpuBackend>,
    ) {
        let same = |x: Tensor<CpuBackend, 2>, y: Tensor<CpuBackend, 2>| {
            assert_eq!(x.dims(), y.dims());
            x.into_data().assert_eq(&y.into_data(), false);
        };
        same(
            a.encoder.blocks[0].attn.query.weight.val(),
            b.encoder.blocks[0].attn.query.weight.val(),
        );
        same(
            a.decoder.blocks[2].cross_attn.output.weight.val(),
            b.decoder.blocks[2].cross_attn.output.weight.val(),
        );
        same(
            a.decoder.blocks[1].mlp.linear2.weight.val(),
            b.decoder.blocks[1].mlp.linear2.weight.val(),
        );
        same(
            a.encoder.positional_embedding.val(),
            b.encoder.positional_embedding.val(),
        );
        same(
            a.decoder.token_embedding.weight.val(),
            b.decoder.token_embedding.weight.val(),
        );
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
        write_hf(&model, &path, None);

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

        let checkpoint = SafetensorsCheckpoint::single(&path);
        let scanner = SafetensorsWhisperScanner::new().with_d_head(D_HEAD);
        let cfg = scanner.scan_cfg(&checkpoint).unwrap();
        assert_eq!(cfg.geometry(), toy().geometry());

        let (loaded, cfg) = scanner.load::<CpuBackend>(&checkpoint, &device).unwrap();
        assert_eq!(cfg.geometry(), toy().geometry());
        assert_eq!(loaded.n_mels(), 80);
        assert_eq!(loaded.vocab_size(), 64);
        assert_same_weights(&model, &loaded);
    }

    /// The same model in two shards, the encoder in one and the decoder
    /// in the other, with an index naming them: scanned and loaded whole;
    /// a shard held back is a load error naming what is missing.
    #[test]
    fn test_a_sharded_layout_round_trips() {
        let device = default_device();
        let dir = tempfile::tempdir().unwrap();
        let model: Whisper<CpuBackend> = toy().try_init(&device).unwrap();
        let s1 = dir.path().join("model-00001-of-00002.safetensors");
        let s2 = dir.path().join("model-00002-of-00002.safetensors");
        write_hf(&model, &s1, Some(r"^encoder\."));
        write_hf(&model, &s2, Some(r"^decoder\."));
        assert!(
            safetensors_header(&s1)
                .unwrap()
                .keys()
                .all(|k| k.starts_with("model.encoder.")),
            "the first shard is the encoder"
        );

        let checkpoint = SafetensorsCheckpoint {
            shards: vec![s1.clone(), s2.clone()],
        };
        let scanner = SafetensorsWhisperScanner::new().with_d_head(D_HEAD);
        let cfg = scanner.scan_cfg(&checkpoint).unwrap();
        assert_eq!(cfg.geometry(), toy().geometry());
        let (loaded, _) = scanner.load::<CpuBackend>(&checkpoint, &device).unwrap();
        assert_same_weights(&model, &loaded);

        let encoder_only = SafetensorsCheckpoint::single(&s1);
        let err = scanner.scan_cfg(&encoder_only).unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("no model.decoder.embed_tokens.weight")),
            "{err}"
        );
        let decoder_only = SafetensorsCheckpoint::single(&s2);
        assert!(scanner.scan_cfg(&decoder_only).is_err());
    }

    /// A safetensors file of something else is refused by name, from its
    /// header.
    #[test]
    fn test_a_file_of_another_shape_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("other.safetensors");
        let header = br#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 24]);
        std::fs::write(&path, bytes).unwrap();

        let err = SafetensorsWhisperScanner::new()
            .scan_cfg(&SafetensorsCheckpoint::single(&path))
            .unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("no model.encoder.conv1.weight")),
            "{err}"
        );
    }
}
