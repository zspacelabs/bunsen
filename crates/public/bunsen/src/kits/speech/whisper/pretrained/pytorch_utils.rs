use std::path::{
    Path,
    PathBuf,
};

use burn::{
    config::Config,
    store::{
        ModuleSnapshot,
        ModuleStore,
        PytorchStore,
        PytorchStoreError,
        pytorch::PytorchError,
    },
    tensor::Device,
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenErrorKind,
        BunsenResult,
        LookupError,
        LookupProblem,
        io_error_kind,
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

/// Sorts a [`PytorchStoreError`] from reading the checkpoint at `path` by
/// what it means: a missing or forbidden file is a
/// [`Lookup`](BunsenErrorKind::Lookup), another failure to read it is by
/// its `io::ErrorKind`, and the rest is a file that is not the checkpoint
/// it should be, [`InvalidResource`](BunsenErrorKind::InvalidResource).
fn pytorch_store_error(path: &Path) -> impl FnOnce(PytorchStoreError) -> BunsenError + '_ {
    move |error| {
        let kind = match &error {
            PytorchStoreError::Io(io) | PytorchStoreError::Reader(PytorchError::Io(io)) => {
                io_error_kind(io)
            }
            _ => BunsenErrorKind::InvalidResource,
        };
        if kind == BunsenErrorKind::Lookup {
            let problem = match &error {
                PytorchStoreError::Io(io) | PytorchStoreError::Reader(PytorchError::Io(io))
                    if io.kind() == std::io::ErrorKind::PermissionDenied =>
                {
                    LookupProblem::Denied
                }
                _ => LookupProblem::Missing,
            };
            BunsenError::lookup(LookupError::path(path, problem).with_source(error))
        } else {
            BunsenError::from_cause(kind, error).context(path.display())
        }
    }
}

fn block_layers_from_keys<S: AsRef<str>>(
    kind: &str,
    keys: &[S],
) -> usize {
    keys.iter()
        .filter(|k| {
            let k = k.as_ref();
            k.starts_with(&format!("{kind}.blocks.")) && k.ends_with(".attn.key.weight")
        })
        .count()
}

/// Pytorch Whisper Model Scanner.
#[derive(Debug, Config)]
pub struct PytorchWhisperScanner {
    /// Top-level key in the model state dict.
    #[config(default_value = "Some(\"model_state_dict\".to_string())")]
    pub top_level_key: Option<String>,

    /// The audio front end to declare on the scanned config.
    ///
    /// A checkpoint does not record it; `OpenAI`'s were all trained with
    /// the default.
    #[config(default = "WhisperFrontEndConfig::new()")]
    pub front_end: WhisperFrontEndConfig,

    /// The token layout to declare on the scanned config: upstream's.
    #[config(default = "WhisperTokenLayoutConfig::new()")]
    pub token_layout: WhisperTokenLayoutConfig,

    /// Head Dimensionality.
    #[config(default = "WHISPER_DEFAULT_D_MODEL")]
    pub d_head: usize,
}

impl PytorchWhisperScanner {
    /// Scan a pytorch whisper checkpoint for configuration.
    ///
    /// Due to the reader, this will always set `n_heads` to 1.
    ///
    /// # Errors
    /// - [`Lookup`](BunsenErrorKind::Lookup) when `path` is missing or
    ///   forbidden;
    /// - [`InvalidResource`](BunsenErrorKind::InvalidResource), with the
    ///   [`PytorchStoreError`] as its cause, when the file is not a `PyTorch`
    ///   checkpoint; without one, naming a tensor the layout requires and the
    ///   checkpoint lacks, or has at another rank;
    /// - [`Sys`](BunsenErrorKind::Sys) when reading it fails otherwise.
    pub fn scan_cfg<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> BunsenResult<(PytorchStore, WhisperApiConfig)> {
        let path = path.as_ref();
        let path: PathBuf = path.to_path_buf();

        let store = PytorchStore::from_file(path.clone())
            .map_indices_contiguous(false)
            // PyTorch's 1-based `conv1`/`conv2` map to the 0-based
            // `ConvSeq1d` blocks of the encoder head.
            .with_key_remapping(r"encoder\.conv1\.", "encoder.head.blocks.0.conv.")
            .with_key_remapping(r"encoder\.conv2\.", "encoder.head.blocks.1.conv.")
            .with_key_remapping(r"\.attn\.out", ".attn.output")
            .with_key_remapping(r"\.cross_attn\.out", ".cross_attn.output")
            .with_key_remapping(r"\.mlp\.2", ".mlp.linear2")
            .with_key_remapping(r"\.mlp\.0", ".mlp.linear1");

        let store = match &self.top_level_key {
            Some(k) => store.with_top_level_key(k),
            None => store,
        };

        let mut store = store;

        let keys = store.keys().map_err(pytorch_store_error(&path))?;

        // The shape of a tensor the layout requires, at the rank it
        // requires: the same refusals as the safetensors scanner's.
        let mut shape = |tensor: &str, rank: usize| -> BunsenResult<Vec<usize>> {
            let shape = store
                .get_tensor(tensor)
                .map_err(pytorch_store_error(&path))?
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
        let max_audio_ctx = shape("encoder.positional_embedding", 2)?[0] * AUDIO_ENCODER_STRIDE;
        let max_text_ctx = shape("decoder.positional_embedding", 2)?[0];

        let encoder_layers = block_layers_from_keys("encoder", &keys);
        let decoder_layers = block_layers_from_keys("decoder", &keys);

        Ok((
            store,
            WhisperApiConfig::new(
                n_mels,
                vocab_size,
                d_model,
                max_audio_ctx,
                encoder_layers,
                max_text_ctx,
                decoder_layers,
            )
            .with_d_head(self.d_head)
            .with_front_end(self.front_end.clone())
            .with_token_layout(self.token_layout.clone()),
        ))
    }

    /// Loads a pytorch whisper model from a checkpoint.
    ///
    /// # Errors
    /// As [`scan_cfg`](Self::scan_cfg), and for loading the weights;
    /// as [`ModuleInit::try_init`] for the scanned config.
    pub fn load<P: AsRef<Path>>(
        &self,
        path: P,
        device: &Device,
    ) -> BunsenResult<(Whisper, WhisperApiConfig)> {
        let path = path.as_ref();
        let (mut store, cfg) = self.scan_cfg(path)?;

        let mut module: Whisper = cfg.try_init(device)?;

        module
            .load_from(&mut store)
            .map_err(pytorch_store_error(path))?;

        Ok((module, cfg))
    }
}
