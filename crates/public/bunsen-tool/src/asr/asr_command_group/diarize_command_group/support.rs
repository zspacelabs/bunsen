use std::sync::Arc;

use bunsen::{
    data::resources::ResourceCache,
    errors::BunsenResult,
    kits::speech::diarization::{
        SpeakerTurn,
        sortformer::{
            SortformerConstruct,
            pretrained::default_sortformer_factory,
        },
    },
};
use burn::tensor::Device;
use serde::Serialize;

use crate::asr::{
    load_logged,
    millis,
    speaker_name,
};

/// Loads diarizer `model` through `cache`, logging its info card.
///
/// # Errors
/// As the factory resolving the name, and the load.
pub fn load_diarizer(
    model: &str,
    cache: &ResourceCache,
    device: &Device,
) -> BunsenResult<Arc<SortformerConstruct>> {
    let bundle = load_logged(
        default_sortformer_factory()?.resolve(model, cache)?,
        cache,
        device,
    )?;
    log::info!("  {bundle}");
    Ok(bundle)
}

/// A turn, as `--json` prints it.
#[derive(Debug, Serialize)]
pub(super) struct JsonTurn {
    pub(super) speaker: String,
    pub(super) start: f64,
    pub(super) end: f64,
}

impl From<&SpeakerTurn> for JsonTurn {
    fn from(turn: &SpeakerTurn) -> Self {
        Self {
            speaker: speaker_name(turn.speaker),
            start: millis(turn.start),
            end: millis(turn.end),
        }
    }
}

/// A speaker's total speech, as `run --json` prints it.
#[derive(Debug, Serialize)]
pub(super) struct JsonSpeaker {
    pub(super) speaker: String,
    pub(super) seconds: f64,
    pub(super) turns: usize,
}

/// One file, as `run --json` prints it.
#[derive(Debug, Serialize)]
pub(super) struct JsonDiarization {
    pub(super) file: String,
    pub(super) turns: Vec<JsonTurn>,
    pub(super) speakers: Vec<JsonSpeaker>,
}

/// A turn as a text line: `{start} {end} speaker_{k}`, in seconds.
pub(super) fn text_line(turn: &SpeakerTurn) -> String {
    format!(
        "{:.3} {:.3} {}",
        turn.start,
        turn.end,
        speaker_name(turn.speaker)
    )
}

/// Each speaker's total speech and turn count, by speaker.
pub(super) fn speaker_totals(turns: &[SpeakerTurn]) -> Vec<JsonSpeaker> {
    let mut totals: std::collections::BTreeMap<usize, (f64, usize)> = Default::default();
    for turn in turns {
        let total = totals.entry(turn.speaker).or_default();
        total.0 += turn.duration();
        total.1 += 1;
    }
    totals
        .into_iter()
        .map(|(speaker, (seconds, turns))| JsonSpeaker {
            speaker: speaker_name(speaker),
            seconds: millis(seconds),
            turns,
        })
        .collect()
}
