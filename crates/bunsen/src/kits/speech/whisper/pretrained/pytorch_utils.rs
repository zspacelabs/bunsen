use std::path::{
    Path,
    PathBuf,
};

use burn::config::Config;
use burn_store::{
    ModuleStore,
    PytorchStore,
};

use crate::kits::speech::whisper::blocks::WhisperApiConfig;

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

    /// Head Dimension.
    #[config(default = "64")]
    pub d_head: usize,
}

impl PytorchWhisperScanner {
    /// Scan a pytorch whisper checkpoint for configuration.
    ///
    /// Due to the reader, this will always set `n_heads` to 1.
    pub fn scan_cfg<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<WhisperApiConfig, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let path: PathBuf = path.to_path_buf();

        let store = PytorchStore::from_file(path.clone());
        let mut store = match &self.top_level_key {
            Some(k) => store.with_top_level_key(k),
            None => store,
        };
        let keys = store.keys()?;

        let [d_model, n_mels] = store
            .get_snapshot("encoder.conv1.weight")?
            .unwrap()
            .shape
            .dims();

        let [vocab_size, _] = store
            .get_snapshot("decoder.token_embedding.weight")?
            .unwrap()
            .shape
            .dims();

        let [k, _] = store
            .get_snapshot("encoder.positional_embedding")?
            .unwrap()
            .shape
            .dims();
        let max_audio_ctx = k * 2;

        let [max_text_ctx, _] = store
            .get_snapshot("decoder.positional_embedding")?
            .unwrap()
            .shape
            .dims();

        let encoder_layers = block_layers_from_keys("encoder", &keys);
        let decoder_layers = block_layers_from_keys("decoder", &keys);

        Ok(WhisperApiConfig::new(
            n_mels,
            vocab_size,
            d_model,
            max_audio_ctx,
            encoder_layers,
            max_text_ctx,
            decoder_layers,
        )
        .with_d_head(self.d_head))
    }
}
