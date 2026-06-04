use std::path::{
    Path,
    PathBuf,
};

use burn::{
    config::Config,
    prelude::Backend,
};
use burn_store::{
    ModuleSnapshot,
    ModuleStore,
    PytorchStore,
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::whisper::blocks::{
        WHISPER_DEFAULT_D_MODEL,
        Whisper,
        WhisperApiConfig,
    },
};

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

    /// Head Dimensionality.
    #[config(defaul_value = "WHISPER_DEFAULT_D_MODEL")]
    pub d_head: usize,
}

impl PytorchWhisperScanner {
    /// Scan a pytorch whisper checkpoint for configuration.
    ///
    /// Due to the reader, this will always set `n_heads` to 1.
    pub fn scan_cfg<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> BunsenResult<(PytorchStore, WhisperApiConfig)> {
        let path = path.as_ref();
        let path: PathBuf = path.to_path_buf();

        let store = PytorchStore::from_file(path.clone())
            .map_indices_contiguous(false)
            .with_key_remapping(r"\.attn\.out", ".attn.output")
            .with_key_remapping(r"\.cross_attn\.out", ".cross_attn.output")
            .with_key_remapping(r"\.mlp\.2", ".mlp.linear2")
            .with_key_remapping(r"\.mlp\.0", ".mlp.linear1");

        let store = match &self.top_level_key {
            Some(k) => store.with_top_level_key(k),
            None => store,
        };

        let mut store = store;

        let keys = store.keys().map_err(BunsenError::external)?;

        let [d_model, n_mels] = store
            .get_snapshot("encoder.conv1.weight")
            .map_err(BunsenError::external)?
            .unwrap()
            .shape
            .dims();

        let [vocab_size, _] = store
            .get_snapshot("decoder.token_embedding.weight")
            .map_err(BunsenError::external)?
            .unwrap()
            .shape
            .dims();

        let [k, _] = store
            .get_snapshot("encoder.positional_embedding")
            .map_err(BunsenError::external)?
            .unwrap()
            .shape
            .dims();
        let max_audio_ctx = k * 2;

        let [max_text_ctx, _] = store
            .get_snapshot("decoder.positional_embedding")
            .map_err(BunsenError::external)?
            .unwrap()
            .shape
            .dims();

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
            .with_d_head(self.d_head),
        ))
    }

    /// Load a pytorch whisper model from a checkpoint.
    pub fn load<B: Backend, P: AsRef<Path>>(
        &self,
        path: P,
        device: &B::Device,
    ) -> BunsenResult<(Whisper<B>, WhisperApiConfig)> {
        let (mut store, cfg) = self.scan_cfg(path)?;

        let mut module = cfg.try_init(device)?;
        module
            .load_from(&mut store)
            .map_err(BunsenError::external)?;

        Ok((module, cfg))
    }
}
