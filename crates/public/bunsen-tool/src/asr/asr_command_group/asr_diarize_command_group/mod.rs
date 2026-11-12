//! `asr diarize`: the command group.

#[cfg(feature = "live")]
mod asr_diarize_live;
mod asr_diarize_run;

use std::sync::Arc;

#[cfg(feature = "live")]
pub use asr_diarize_live::*;
pub use asr_diarize_run::*;
use bunsen::{
    data::resources::PretrainedCache,
    errors::BunsenResult,
    kits::speech::diarization::{
        SpeakerActivityModel,
        SpeakerTurn,
        sortformer::{
            SortformerConstruct,
            pretrained::{
                DEFAULT_SORTFORMER_MODEL,
                default_sortformer_factory,
            },
        },
    },
};
use burn::tensor::Device;
use serde::Serialize;

use crate::{
    asr::{
        AsrKitFetchCommand,
        AsrKitListCommand,
        Kit,
        load_logged,
        millis,
        speaker_name,
    },
    support::PresetArgs,
};

/// The diarizer's commands.
#[derive(clap::Args, Debug)]
pub struct AsrDiarizeCommandGroup {
    #[command(subcommand)]
    action: AsrDiarizeSubCommand,
}

#[derive(clap::Subcommand, Debug)]
enum AsrDiarizeSubCommand {
    /// List the models `--model` can name, their licences and files.
    ///
    /// Each file is marked with whether it is cached.
    List(AsrKitListCommand),

    /// Bring every file of the named models into the weights cache.
    Fetch(AsrKitFetchCommand),

    /// Diarize audio files: each one's speaker turns.
    Run(AsrDiarizeRunCommand),

    /// Diarize the microphone: each speaker turn as it closes.
    #[cfg(feature = "live")]
    Live(AsrDiarizeLiveCommand),
}

impl AsrDiarizeCommandGroup {
    pub fn run(&self) -> BunsenResult<()> {
        match &self.action {
            AsrDiarizeSubCommand::List(list) => list.run(Kit::Diarizer),
            AsrDiarizeSubCommand::Fetch(fetch) => fetch.run(Kit::Diarizer),
            AsrDiarizeSubCommand::Run(run) => run.run(),
            #[cfg(feature = "live")]
            AsrDiarizeSubCommand::Live(live) => live.run(),
        }
    }
}

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
        cache: &PretrainedCache,
        device: &Device,
    ) -> BunsenResult<Arc<SortformerConstruct>> {
        let model = self
            .presets
            .apply(self.model.as_deref().unwrap_or(DEFAULT_SORTFORMER_MODEL))?;
        load_diarizer(&model, cache, device)
    }
}

/// Loads diarizer `model` through `cache`, logging its info card.
///
/// # Errors
/// As the factory resolving the name, and the load.
pub fn load_diarizer(
    model: &str,
    cache: &PretrainedCache,
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
        cache: &PretrainedCache,
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

/// A turn, as `--json` prints it.
#[derive(Debug, Serialize)]
struct JsonTurn {
    speaker: String,
    start: f64,
    end: f64,
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
struct JsonSpeaker {
    speaker: String,
    seconds: f64,
    turns: usize,
}

/// One file, as `run --json` prints it.
#[derive(Debug, Serialize)]
struct JsonDiarization {
    file: String,
    turns: Vec<JsonTurn>,
    speakers: Vec<JsonSpeaker>,
}

/// A turn as a text line: `{start} {end} speaker_{k}`, in seconds.
fn text_line(turn: &SpeakerTurn) -> String {
    format!(
        "{:.3} {:.3} {}",
        turn.start,
        turn.end,
        speaker_name(turn.speaker)
    )
}

/// Each speaker's total speech and turn count, by speaker.
fn speaker_totals(turns: &[SpeakerTurn]) -> Vec<JsonSpeaker> {
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use bunsen::kits::speech::diarization::sortformer::pretrained::default_sortformer_factory;

    use super::*;
    use crate::{
        asr::recording_id,
        support::with_presets,
    };

    /// Text lines and JSON name speakers as `NeMo` and the RTTM do, and
    /// round times to the millisecond.
    #[test]
    fn test_formats() {
        let turn = SpeakerTurn {
            speaker: 2,
            start: 1.23456,
            end: 4.5,
        };
        assert_eq!(text_line(&turn), "1.235 4.500 speaker_2");
        let json = serde_json::to_string(&JsonTurn::from(&turn)).unwrap();
        assert_eq!(json, r#"{"speaker":"speaker_2","start":1.235,"end":4.5}"#);
        assert_eq!(recording_id(Path::new("/a/b/meeting.wav")), "meeting");
    }

    /// Totals are by speaker, in index order.
    #[test]
    fn test_speaker_totals() {
        let turn = |speaker, start, end| SpeakerTurn {
            speaker,
            start,
            end,
        };
        let totals = speaker_totals(&[turn(1, 0.0, 1.5), turn(0, 1.0, 2.0), turn(1, 3.0, 4.0)]);
        let got: Vec<(&str, f64, usize)> = totals
            .iter()
            .map(|t| (t.speaker.as_str(), t.seconds, t.turns))
            .collect();
        assert_eq!(got, [("speaker_0", 1.0, 1), ("speaker_1", 2.5, 2)]);
    }

    /// A diarizer's latency and post-processing are every row's axes: a
    /// model's box names them, and they default to `low` and `none`.
    #[test]
    fn test_presets() {
        assert_eq!(
            with_presets("sortformer-v2[dihard3]", &["high"]).unwrap(),
            "sortformer-v2[dihard3,high]"
        );

        let rows = default_sortformer_factory().unwrap().list();
        assert!(!rows.is_empty());
        for (_, row) in rows {
            let axes: Vec<&str> = row
                .presets
                .axes
                .iter()
                .map(|a| a.axis.name.leaf())
                .collect();
            assert_eq!(axes, ["latency", "postprocess"], "{}", row.name);
            assert_eq!(row.presets.default_names(), ["low", "none"], "{}", row.name);
        }
    }
}
