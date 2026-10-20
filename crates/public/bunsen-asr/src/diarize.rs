//! `bunsen-asr diarize`: who spoke when, through Sortformer.
use std::{
    path::{
        Path,
        PathBuf,
    },
    sync::Arc,
    time::Instant,
};

use bunsen::{
    data::pretrained::PretrainedCache,
    errors::BunsenResult,
    kits::speech::{
        diarization::{
            RttmSegment,
            SpeakerTurn,
            TurnFilterConfig,
        },
        sortformer::{
            SortformerBundle,
            SortformerStreamingConfig,
            pretrained::{
                DEFAULT_SORTFORMER_MODEL,
                default_sortformer_factory,
            },
        },
    },
    support::audio::load_audio_mono_sr,
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};
use burn::tensor::Device;
use serde::Serialize;

#[cfg(feature = "live")]
use crate::capture::CaptureArgs;
use crate::{
    args::DeviceArgs,
    family::{
        Kit,
        load_logged,
    },
    fetch_cmd::FetchCmd,
    list_cmd::ListCmd,
    transcribe_cmd::millis,
};

/// The diarizer's commands.
#[derive(clap::Args, Debug)]
pub struct DiarizeCmd {
    #[command(subcommand)]
    action: DiarizeAction,
}

#[derive(clap::Subcommand, Debug)]
enum DiarizeAction {
    /// List the models `--model` can name, their licences and files, and
    /// whether each file is cached.
    List(ListCmd),

    /// Bring every file of the named models into the weights cache.
    Fetch(FetchCmd),

    /// Diarize audio files: each one's speaker turns.
    Run(RunCmd),

    /// Diarize the microphone: each speaker turn as it closes.
    #[cfg(feature = "live")]
    Live(LiveCmd),
}

impl DiarizeCmd {
    pub fn run(&self) -> BunsenResult<()> {
        match &self.action {
            DiarizeAction::List(list) => list.run(Kit::Diarizer),
            DiarizeAction::Fetch(fetch) => fetch.run(Kit::Diarizer),
            DiarizeAction::Run(run) => run.run(),
            #[cfg(feature = "live")]
            DiarizeAction::Live(live) => live.run(),
        }
    }
}

/// How far behind the audio the diarizer runs: the streaming presets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Latency {
    /// 1.04 s: chunks of 9 frames with 4 of look-ahead.
    #[default]
    Low,

    /// 0.64 s: chunks of 6 with 2.
    VeryLow,

    /// 0.32 s: chunks of 3 with 1.
    UltraLow,

    /// 30.4 s: chunks of 340 with 40; the most accurate, for files.
    Offline,
}

impl From<Latency> for SortformerStreamingConfig {
    fn from(latency: Latency) -> Self {
        match latency {
            Latency::Low => Self::nemotron_low_latency(),
            Latency::VeryLow => Self::nemotron_very_low_latency(),
            Latency::UltraLow => Self::nemotron_ultra_low_latency(),
            Latency::Offline => Self::nemotron_offline(),
        }
    }
}

/// How speaker activity becomes turns: `NeMo`'s post-processing presets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Postprocess {
    /// Thresholds of 0.5, no padding, no filters.
    #[default]
    None,

    /// `NeMo`'s parameters tuned on DIHARD III.
    Dihard3,

    /// `NeMo`'s parameters tuned on CALLHOME.
    Callhome,
}

impl From<Postprocess> for TurnFilterConfig {
    fn from(postprocess: Postprocess) -> Self {
        match postprocess {
            Postprocess::None => Self::none(),
            Postprocess::Dihard3 => Self::dihard3_v2(),
            Postprocess::Callhome => Self::callhome_v2(),
        }
    }
}

/// The diarizer, and how its activity becomes turns.
#[derive(clap::Args, Debug)]
pub struct DiarizerArgs {
    /// The model, by name: `diarize list` shows them;
    /// `nvidia/nemotron-3-diarization` when omitted.
    #[arg(long)]
    pub model: Option<String>,

    /// How far behind the audio the diarizer runs.
    #[arg(long, value_enum, default_value_t)]
    pub latency: Latency,

    /// How speaker activity becomes turns.
    #[arg(long, value_enum, default_value_t)]
    pub postprocess: Postprocess,
}

impl DiarizerArgs {
    /// Loads the model through `cache`, logging its info card.
    ///
    /// # Errors
    /// As the factory resolving the name, and the load.
    pub fn load(
        &self,
        cache: &PretrainedCache,
        device: &Device,
    ) -> BunsenResult<Arc<SortformerBundle>> {
        let model = self.model.as_deref().unwrap_or(DEFAULT_SORTFORMER_MODEL);
        let bundle = load_logged(
            default_sortformer_factory()?.resolve(model, cache)?,
            cache,
            device,
        )?;
        log::info!("  {bundle}");
        Ok(bundle)
    }
}

/// How turns are printed.
#[derive(clap::Args, Debug)]
#[group(multiple = false)]
pub struct OutputArgs {
    /// Print NIST RTTM: a `SPEAKER` line per turn, the file's stem as its
    /// recording id.
    #[arg(long)]
    pub rttm: bool,

    /// Print JSON: for `run`, an object per file with its turns and each
    /// speaker's total; for `live`, an object per turn.
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

/// A speaker's name, as `NeMo` and the RTTM output name it.
fn speaker_name(index: usize) -> String {
    format!("speaker_{index}")
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

/// The recording id `--rttm` gives `path`: its stem.
fn recording_id(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
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

/// Diarizes audio files: each one's speaker turns, as lines of
/// `{start} {end} speaker_{k}` (seconds), RTTM, or JSON.
///
/// Each file is decoded at the model's rate (16 kHz, mono) and streamed
/// through the diarizer under `--latency`; speakers are numbered in the
/// order they first speak. The model's info card, and after each file how
/// long it took, are logged at `INFO` (`-v`).
#[derive(clap::Args, Debug)]
pub struct RunCmd {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub diarizer: DiarizerArgs,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    #[clap(flatten)]
    pub device: DeviceArgs,

    #[clap(flatten)]
    pub output: OutputArgs,

    /// The audio files.
    #[arg(required = true)]
    pub files: Vec<PathBuf>,
}

impl RunCmd {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let device = self.device.device()?;
        let cache = self.cache.init()?;
        let bundle = self.diarizer.load(&cache, &device)?;
        let streaming: SortformerStreamingConfig = self.diarizer.latency.into();
        let filter: TurnFilterConfig = self.diarizer.postprocess.into();
        let rate = bundle.sample_rate();

        for path in &self.files {
            let samples = load_audio_mono_sr(path, rate)?;
            let t0 = Instant::now();
            let turns = bundle.diarize(&samples, streaming.clone(), &filter)?;
            let elapsed = t0.elapsed();

            if self.output.json {
                let json = JsonDiarization {
                    file: path.display().to_string(),
                    turns: turns.iter().map(JsonTurn::from).collect(),
                    speakers: speaker_totals(&turns),
                };
                println!(
                    "{}",
                    serde_json::to_string(&json).expect("a diarization serializes")
                );
            } else if self.output.rttm {
                let id = recording_id(path);
                for turn in &turns {
                    println!("{}", RttmSegment::from_turn(&id, turn));
                }
            } else {
                if self.files.len() > 1 {
                    println!("{}:", path.display());
                }
                for turn in &turns {
                    println!("{}", text_line(turn));
                }
            }

            let seconds = samples.len() as f64 / rate as f64;
            log::info!(
                "{}: {seconds:.1} s of audio in {elapsed:.2?} ({:.1}x realtime)",
                path.display(),
                seconds / elapsed.as_secs_f64()
            );
        }
        Ok(())
    }
}

/// Diarizes the microphone: each speaker turn printed as it closes, as a
/// text line, an RTTM line (recording id `live`), or a JSON object.
///
/// Times are seconds from the start of capture. The input device is opened
/// as `<family> live` opens it; Ctrl-C, or `--seconds`, ends the stream
/// cleanly, closing the turns still open.
#[cfg(feature = "live")]
#[derive(clap::Args, Debug)]
pub struct LiveCmd {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub diarizer: DiarizerArgs,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    #[clap(flatten)]
    pub device: DeviceArgs,

    #[clap(flatten)]
    pub output: OutputArgs,

    /// Milliseconds of audio per push.
    #[arg(long, default_value_t = 250)]
    pub chunk_ms: usize,

    #[clap(flatten)]
    pub capture: CaptureArgs,
}

#[cfg(feature = "live")]
impl LiveCmd {
    pub fn run(&self) -> BunsenResult<()> {
        use bunsen::kits::speech::diarization::{
            SpeakerActivityModel,
            SpeakerActivityStream,
        };

        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let Some(capture) = self.capture.open()? else {
            return Ok(());
        };
        let device = self.device.device()?;
        let cache = self.cache.init()?;
        let bundle = self.diarizer.load(&cache, &device)?;
        let diarizer = bundle.diarizer(self.diarizer.latency.into())?;
        let rate = bundle.sample_rate();
        let frame_seconds =
            bunsen::kits::speech::diarization::SpeakerActivityModel::frame_samples(&diarizer)
                as f64
                / rate as f64;
        let mut filter = TurnFilterConfig::from(self.diarizer.postprocess)
            .init(diarizer.max_speakers(), frame_seconds);
        let mut stream = diarizer.stream()?;
        let chunk = (rate * self.chunk_ms / 1000).max(1);

        let print = |turns: Vec<SpeakerTurn>| {
            for turn in &turns {
                if self.output.json {
                    println!(
                        "{}",
                        serde_json::to_string(&JsonTurn::from(turn)).expect("a turn serializes")
                    );
                } else if self.output.rttm {
                    println!("{}", RttmSegment::from_turn("live", turn));
                } else {
                    println!("{}", text_line(turn));
                }
            }
        };
        capture.run(rate, chunk, |samples, _time| {
            let activity = stream.push(samples)?;
            print(filter.push(&activity.probs));
            Ok(())
        })?;
        let activity = stream.finish()?;
        print(filter.push(&activity.probs));
        print(filter.finish());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Every preset maps to its config.
    #[test]
    fn test_presets() {
        let low: SortformerStreamingConfig = Latency::Low.into();
        assert_eq!(low.latency_frames(), 13);
        let offline: SortformerStreamingConfig = Latency::Offline.into();
        assert_eq!(offline.latency_frames(), 380);
        assert_eq!(
            TurnFilterConfig::from(Postprocess::None),
            TurnFilterConfig::none()
        );
        assert_eq!(
            TurnFilterConfig::from(Postprocess::Dihard3),
            TurnFilterConfig::dihard3_v2()
        );
    }
}
