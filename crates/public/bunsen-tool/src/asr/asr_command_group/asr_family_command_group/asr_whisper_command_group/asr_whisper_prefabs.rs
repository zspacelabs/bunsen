//! `asr family whisper prefabs`.

use bunsen::{
    data::pretrained::Prefabs,
    errors::BunsenResult,
    kits::speech::asr::whisper::WhisperApiConfig,
};
use bunsen_app::logging::{
    LevelFilter,
    LogArgs,
    LogPolicy,
};

/// Lists Whisper's prefabs: the geometries, without weights.
#[derive(clap::Args, Debug)]
pub struct AsrWhisperPrefabsCommand {
    #[clap(flatten)]
    pub logging: LogArgs,
}

impl AsrWhisperPrefabsCommand {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        println!(
            "{}: OpenAI Whisper geometries",
            WhisperApiConfig::PREFAB_TABLE
        );
        println!("  NAME             MELS  VOCAB D_MODEL HEADS  ENC  DEC AUDIO_CTX TEXT_CTX");
        for (name, prefab) in WhisperApiConfig::PREFABS {
            let g = prefab().geometry();
            println!(
                "  {:<16} {:>4} {:>6} {:>7} {:>5} {:>4} {:>4} {:>9} {:>8}",
                name,
                g.n_mels,
                g.vocab_size,
                g.d_model,
                g.n_heads(),
                g.n_encoder_layers,
                g.n_decoder_layers,
                g.max_audio_ctx,
                g.max_text_ctx,
            );
        }
        Ok(())
    }
}
