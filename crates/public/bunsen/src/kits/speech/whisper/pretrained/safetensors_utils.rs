//! # Reading a `transformers` Whisper checkpoint

use burn::{
    config::Config,
    tensor::Device,
};

use crate::{
    blocks::{
        conv::ConvSeq1d,
        transformers::{
            decoders::cross::CrossDecoderBlock,
            encoders::residual::ResidualEncoderAttentionBlock,
        },
    },
    burner::module::{
        ModuleInit,
        Rekey,
    },
    data::pretrained::{
        SafetensorsCheckpoint,
        apply_rekeyed_tensors,
    },
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

/// `transformers`' `WhisperForConditionalGeneration` state-dict names onto
/// a [`Whisper`]'s module paths: under `model.`, the layers by their blocks'
/// maps ([`ResidualEncoderAttentionBlock::whisper_hf_rekey`],
/// [`CrossDecoderBlock::whisper_hf_rekey`]), `layers.N` as `blocks.N`; the
/// two convolutions, `conv1` and `conv2`, the encoder head's blocks
/// ([`ConvSeq1d::one_based_rekey`]); the positional tables
/// without their `.weight`; `embed_tokens` the token embedding; the final
/// norms `ln_post` and `ln`; everything else as it is. Anything outside
/// `model.` (a tied `proj_out.weight`) is dropped.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`apply_rekeyed_tensors`] resolves to `gamma` / `beta`.
pub fn hf_rekey() -> Rekey {
    Rekey::drop_rest().mount(
        "model.",
        "",
        Rekey::keep_rest()
            .mount(
                "encoder.conv",
                "encoder.head.",
                ConvSeq1d::one_based_rekey(),
            )
            .exact(
                "encoder.embed_positions.weight",
                "encoder.positional_embedding",
            )
            .exact(
                "decoder.embed_positions.weight",
                "decoder.positional_embedding",
            )
            .rename("decoder.embed_tokens.", "decoder.token_embedding.")
            .mount_indexed(
                "encoder.layers.",
                "encoder.blocks.",
                ResidualEncoderAttentionBlock::whisper_hf_rekey(),
            )
            .mount_indexed(
                "decoder.layers.",
                "decoder.blocks.",
                CrossDecoderBlock::whisper_hf_rekey(),
            )
            .rename("encoder.layer_norm.", "encoder.ln_post.")
            .rename("decoder.layer_norm.", "decoder.ln."),
    )
}

/// Reads a `transformers` Whisper checkpoint: `model.safetensors`, or its
/// shards.
///
/// `transformers` exports a Whisper model as `model.safetensors`, or as
/// shards with an index, under its own parameter names:
/// `model.encoder.layers.N.self_attn.q_proj.weight` where `OpenAI`'s `.pt`
/// has `encoder.blocks.N.attn.query.weight`, `fc1` for `mlp.0`,
/// `embed_positions.weight` for `positional_embedding`, and so on. This
/// reads that layout: the geometry from the files' headers alone
/// ([`scan_cfg`](Self::scan_cfg)), the weights through the generic
/// [`SafetensorsCheckpoint`] with the names mapped to bunsen's
/// ([`hf_rekey`]). The tied output projection, when a repo carries it,
/// is left unused: bunsen's decoder projects through its token embedding,
/// as upstream's does.
///
/// [`WhisperReader`](super::WhisperReader) picks this for a checkpoint
/// whose `kind` is safetensors, which is what the `hf:` provider serves.
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
    /// As [`SafetensorsCheckpoint::headers`];
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// naming a tensor the layout requires and the checkpoint lacks, or has
    /// at another rank.
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
                    BunsenError::invalid_resource(format!(
                        "{}: not a transformers Whisper checkpoint: no {tensor}",
                        name()
                    ))
                })?;
            if shape.len() == rank {
                Ok(shape)
            } else {
                Err(BunsenError::invalid_resource(format!(
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
    /// As [`scan_cfg`](Self::scan_cfg),
    /// [`SafetensorsCheckpoint::tensors`] and [`apply_rekeyed_tensors`].
    pub fn load(
        &self,
        checkpoint: &SafetensorsCheckpoint,
        device: &Device,
    ) -> BunsenResult<(Whisper, WhisperApiConfig)> {
        let cfg = self.scan_cfg(checkpoint)?;
        let mut module: Whisper = cfg.try_init(device)?;
        apply_rekeyed_tensors(&mut module, checkpoint.tensors()?, &hf_rekey())?;
        Ok((module, cfg))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use burn::{
        store::{
            BurnToPyTorchAdapter,
            ModuleSnapshot,
            SafetensorsStore,
        },
        tensor::Tensor,
    };

    use super::*;
    use crate::{
        data::pretrained::safetensors_header,
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        kits::speech::whisper::WhisperMeta,
        support::testing::cpu_device,
    };

    /// The key remappings from bunsen's names to `transformers`': the inverse
    /// of [`hf_rekey`], for writing a file in that layout.
    const BUNSEN_TO_HF: &[(&str, &str)] = &[
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
        (r"\.attn_norm\.", ".self_attn_layer_norm."),
        (r"\.cross_attn_norm\.", ".encoder_attn_layer_norm."),
        (r"\.mlp_norm\.", ".final_layer_norm."),
        (r"\.mlp\.linear1\.", ".fc1."),
        (r"\.mlp\.linear2\.", ".fc2."),
        (r"^encoder\.ln_post\.", "encoder.layer_norm."),
        (r"^decoder\.ln\.", "decoder.layer_norm."),
        (r"^", "model."),
    ];

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
        model: &Whisper,
        path: &Path,
        only: Option<&str>,
    ) {
        let mut store = SafetensorsStore::from_file(path.to_path_buf())
            .with_to_adapter(BurnToPyTorchAdapter)
            .skip_enum_variants(true);
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
        a: &Whisper,
        b: &Whisper,
    ) {
        let same = |x: Tensor<2>, y: Tensor<2>| {
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

    /// `transformers`' names onto bunsen's, one of each kind; a tied
    /// output projection is dropped.
    #[test]
    fn test_hf_rekey() {
        let rekey = hf_rekey();
        for (hf, bunsen) in [
            (
                "model.encoder.conv1.weight",
                Some("encoder.head.blocks.0.conv.weight"),
            ),
            (
                "model.encoder.conv2.bias",
                Some("encoder.head.blocks.1.conv.bias"),
            ),
            (
                "model.encoder.embed_positions.weight",
                Some("encoder.positional_embedding"),
            ),
            (
                "model.decoder.embed_positions.weight",
                Some("decoder.positional_embedding"),
            ),
            (
                "model.decoder.embed_tokens.weight",
                Some("decoder.token_embedding.weight"),
            ),
            (
                "model.encoder.layers.3.self_attn.out_proj.weight",
                Some("encoder.blocks.3.attn.output.weight"),
            ),
            (
                "model.encoder.layers.3.self_attn_layer_norm.bias",
                Some("encoder.blocks.3.attn_norm.bias"),
            ),
            (
                "model.decoder.layers.1.self_attn.k_proj.weight",
                Some("decoder.blocks.1.attn.key.weight"),
            ),
            (
                "model.decoder.layers.1.encoder_attn.q_proj.bias",
                Some("decoder.blocks.1.cross_attn.query.bias"),
            ),
            (
                "model.decoder.layers.1.encoder_attn_layer_norm.weight",
                Some("decoder.blocks.1.cross_attn_norm.weight"),
            ),
            (
                "model.decoder.layers.1.final_layer_norm.weight",
                Some("decoder.blocks.1.mlp_norm.weight"),
            ),
            (
                "model.decoder.layers.1.fc1.weight",
                Some("decoder.blocks.1.mlp.linear1.weight"),
            ),
            (
                "model.encoder.layer_norm.weight",
                Some("encoder.ln_post.weight"),
            ),
            ("model.decoder.layer_norm.bias", Some("decoder.ln.bias")),
            ("proj_out.weight", None),
        ] {
            assert_eq!(rekey.apply(hf).as_deref(), bunsen, "{hf}");
        }
    }

    /// A toy model written in `transformers`' layout comes back with the
    /// same geometry and the same weights, through the names and the
    /// transposition both ways.
    #[test]
    fn test_a_transformers_layout_round_trips() {
        let device = cpu_device();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let model: Whisper = toy().try_init(&device).unwrap();
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

        let (loaded, cfg) = scanner.load(&checkpoint, &device).unwrap();
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
        let device = cpu_device();
        let dir = tempfile::tempdir().unwrap();
        let model: Whisper = toy().try_init(&device).unwrap();
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
        let (loaded, _) = scanner.load(&checkpoint, &device).unwrap();
        assert_same_weights(&model, &loaded);

        let encoder_only = SafetensorsCheckpoint::single(&s1);
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("no model.decoder.embed_tokens.weight")
            .assert_err(&scanner.scan_cfg(&encoder_only));
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

        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("no model.encoder.conv1.weight")
            .assert_err(
                &SafetensorsWhisperScanner::new().scan_cfg(&SafetensorsCheckpoint::single(&path)),
            );
    }
}
