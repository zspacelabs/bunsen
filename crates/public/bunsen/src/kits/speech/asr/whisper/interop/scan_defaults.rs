//! # What a Whisper checkpoint does not record

use burn::config::Config;

use crate::kits::speech::{
    asr::whisper::blocks::{
        WHISPER_DEFAULT_D_HEAD,
        WhisperApiConfig,
        WhisperTokenLayoutConfig,
    },
    support::log_mel::LogMelFrontEndConfig,
};

/// What a scanned Whisper config declares that the checkpoint does not
/// record, in every ecosystem: the head size, the audio front end and the
/// token layout. The geometry comes from the tensors' shapes; these come from
/// here.
#[derive(Config, Debug, PartialEq)]
pub struct WhisperScanDefaults {
    /// The audio front end. A checkpoint does not record it; `OpenAI`'s were
    /// all trained with the default, and so is every conversion of them.
    #[config(default = "LogMelFrontEndConfig::new()")]
    pub front_end: LogMelFrontEndConfig,

    /// The token layout: upstream's.
    #[config(default = "WhisperTokenLayoutConfig::new()")]
    pub token_layout: WhisperTokenLayoutConfig,

    /// The head dimensionality: every `OpenAI` checkpoint's is 64, and a
    /// checkpoint does not record it.
    #[config(default = "WHISPER_DEFAULT_D_HEAD")]
    pub d_head: usize,
}

impl WhisperScanDefaults {
    /// `config` declaring these.
    pub fn declare(
        &self,
        config: WhisperApiConfig,
    ) -> WhisperApiConfig {
        config
            .with_d_head(self.d_head)
            .with_front_end(self.front_end.clone())
            .with_token_layout(self.token_layout.clone())
    }
}
