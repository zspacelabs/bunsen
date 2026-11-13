use std::sync::Arc;

use bunsen::{
    data::resources::ResourceCache,
    errors::BunsenResult,
    kits::speech::diarization::{
        SpeakerActivityModel,
        sortformer::{
            SortformerConstruct,
            pretrained::DEFAULT_SORTFORMER_MODEL,
        },
    },
};
use burn::tensor::Device;

use super::load_diarizer;
use crate::support::PresetArgs;

/// The diarizer: its model, and the presets that say how far behind the
/// audio it runs (`latency`: `--preset very-low`) and how its activity
/// becomes turns (`postprocess`: `--preset dihard3`); `diarize list` shows
/// each model's.
#[derive(clap::Args, Debug)]
pub struct AsrDiarizerArgs {
    /// The model, as a lookup expression: `diarize list` shows them.
    ///
    /// `nvidia/nemotron-3-diarization` when omitted.
    #[arg(long)]
    pub model: Option<String>,

    #[clap(flatten)]
    pub presets: PresetArgs,
}

impl AsrDiarizerArgs {
    /// Loads the model through `cache` under its presets, logging its info
    /// card.
    ///
    /// # Errors
    /// As [`load_diarizer`].
    pub fn load(
        &self,
        cache: &ResourceCache,
        device: &Device,
    ) -> BunsenResult<Arc<SortformerConstruct>> {
        let model = self
            .presets
            .apply(self.model.as_deref().unwrap_or(DEFAULT_SORTFORMER_MODEL))?;
        load_diarizer(&model, cache, device)
    }
}

/// A speech recognizer's diarizer: `--diarize` on every family's
/// `transcribe` and `live`.
#[derive(clap::Args, Debug)]
pub struct AsrSpeakerArgs {
    /// Label each word or segment with its speaker, through a diarizer.
    ///
    /// `--diarize` alone for `nvidia/nemotron-3-diarization`, `--diarize=EXPR`
    /// for another, or its presets
    /// (`--diarize='nvidia/nemotron-3-diarization[very-low]'`; `diarize list`
    /// shows them).
    #[arg(
        long,
        value_name = "EXPR",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = DEFAULT_SORTFORMER_MODEL
    )]
    pub diarize: Option<String>,
}

impl AsrSpeakerArgs {
    /// The diarizer `--diarize` names, loaded through `cache`; `None`
    /// without the flag. The recognizers see only the trait object.
    ///
    /// # Errors
    /// As [`load_diarizer`].
    pub fn diarizer(
        &self,
        cache: &ResourceCache,
        device: &Device,
    ) -> BunsenResult<Option<Arc<dyn SpeakerActivityModel>>> {
        let Some(model) = &self.diarize else {
            return Ok(None);
        };
        let bundle = load_diarizer(model, cache, device)?;
        Ok(Some(Arc::new(bundle.diarizer(bundle.streaming()?)?)))
    }
}

/// How turns are printed.
#[derive(clap::Args, Debug)]
#[group(multiple = false)]
pub struct AsrDiarizeOutputArgs {
    /// Print NIST RTTM: a `SPEAKER` line per turn, the file's stem as its
    /// recording id.
    #[arg(long)]
    pub rttm: bool,

    /// Print JSON: for `run`, an object per file; for `live`, one per turn.
    ///
    /// A file's object has its turns and each speaker's total.
    #[arg(long)]
    pub json: bool,
}
