//! `bunsen-asr <family> transcribe`.
use std::{
    path::PathBuf,
    time::{
        Duration,
        Instant,
    },
};

use bunsen::{
    data::pretrained::{
        PretrainedCache,
        PretrainedCacheOptions,
    },
    errors::{
        BunsenResult,
        ResultContext,
    },
    kits::speech::{
        asr_driver::TranscriptSegment,
        diarization::RttmSegment,
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
use serde::Serialize;

use crate::{
    args::DeviceArgs,
    diarize::{
        SpeakerArgs,
        recording_id,
        speaker_name,
    },
    driver::Driver,
    family::Family,
    parakeet::ParakeetFlags,
    qwen3_asr::Qwen3AsrFlags,
    stream::Preset,
    whisper,
};

/// A family's own transcription flags.
pub trait FamilyFlags: std::fmt::Debug {
    /// Parakeet's, when these are Parakeet's.
    fn parakeet(&self) -> Option<&ParakeetFlags> {
        None
    }

    /// Qwen3-ASR's, when these are Qwen3-ASR's.
    fn qwen3_asr(&self) -> Option<&Qwen3AsrFlags> {
        None
    }

    /// Whisper's, when these are Whisper's.
    fn whisper(&self) -> Option<&whisper::WhisperFlags> {
        None
    }

    /// The cache options these flags call for, over `options`.
    fn cache_options(
        &self,
        options: PretrainedCacheOptions,
    ) -> PretrainedCacheOptions {
        match self.whisper() {
            Some(whisper) => whisper.cache_options(options),
            None => options,
        }
    }
}

impl FamilyFlags for whisper::WhisperFlags {
    fn whisper(&self) -> Option<&whisper::WhisperFlags> {
        Some(self)
    }
}

impl FamilyFlags for ParakeetFlags {
    fn parakeet(&self) -> Option<&ParakeetFlags> {
        Some(self)
    }
}

impl FamilyFlags for Qwen3AsrFlags {
    fn qwen3_asr(&self) -> Option<&Qwen3AsrFlags> {
        Some(self)
    }
}

/// Transcribes audio files.
///
/// Each file is decoded at the model's rate (16 kHz, mono) and streamed
/// through the family's stream driver, about a second a push. `--model`
/// names a row of `list`, or `hf:org/repo` for a Hugging Face repo in
/// `transformers`' layout; the model's info card, and after each transcript
/// how long it took, are logged at `INFO` (`-v`).
#[derive(clap::Args, Debug)]
pub struct TranscribeCmd<F: clap::Args> {
    #[clap(flatten)]
    pub logging: LogArgs,

    /// The model, by name: `list` shows them; the family's default when
    /// omitted. Whisper also takes a checkpoint's path.
    #[arg(long)]
    pub model: Option<String>,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    #[clap(flatten)]
    pub device: DeviceArgs,

    /// When to decode.
    #[arg(long, value_enum, default_value_t = Preset::Offline)]
    pub preset: Preset,

    /// Milliseconds of audio per push: each file is fed to the driver as a
    /// live loop would feed it, in pieces of this.
    #[arg(long, default_value_t = 1000)]
    pub chunk_ms: usize,

    /// Print one JSON object per file: the text, the language when the model
    /// names one, each word with its start and end in seconds when the model
    /// times words, and each segment when it emits segments; under
    /// `--diarize`, each word's or segment's speaker, and the turns.
    #[arg(long)]
    pub json: bool,

    /// Print NIST RTTM: a `SPEAKER` line per turn, the file's stem as its
    /// recording id. Needs `--diarize`.
    #[arg(long, requires = "diarize", conflicts_with = "json")]
    pub rttm: bool,

    #[clap(flatten)]
    pub speakers: SpeakerArgs,

    #[clap(flatten)]
    pub flags: F,

    /// The audio files.
    #[arg(required = true)]
    pub files: Vec<PathBuf>,
}

/// A transcript, whichever family made it.
#[derive(Clone, Debug, Serialize)]
pub struct Transcript {
    /// The text.
    pub text: String,

    /// The language the model named, if it names one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Each word with its start and end in seconds, if the model times them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<Word>>,

    /// Each segment with its start and end in seconds, and its token ids,
    /// if the model emits segments.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<Segment>>,

    /// The speaker turns, under `--diarize`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turns: Option<Vec<Turn>>,
}

/// A timed word.
#[derive(Clone, Debug, Serialize)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
    /// The token ids the model decoded it as.
    pub tokens: Vec<i64>,
    /// Its speaker, under `--diarize`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

/// A timed segment.
#[derive(Clone, Debug, Serialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// The token ids the model decoded it as.
    pub tokens: Vec<i64>,
    /// Its speaker, under `--diarize`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

/// One speaker's turn: a run of consecutive words or segments attributed to
/// them.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Turn {
    /// The speaker, `speaker_{k}`; `null` where the diarizer heard nobody.
    pub speaker: Option<String>,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// `segment`'s speaker, as the output names it.
pub fn segment_speaker(segment: &TranscriptSegment) -> Option<String> {
    segment.speaker.map(|label| speaker_name(label.index))
}

/// The turns of `segments`: runs of one speaker, their texts joined; the
/// segments with no text are left out.
pub fn speaker_turns(segments: &[TranscriptSegment]) -> Vec<Turn> {
    let mut turns: Vec<Turn> = Vec::new();
    for segment in segments {
        let Some(text) = segment
            .text
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
        else {
            continue;
        };
        let speaker = segment_speaker(segment);
        match turns.last_mut() {
            Some(turn) if turn.speaker == speaker => {
                turn.end = millis(segment.end);
                turn.text.push(' ');
                turn.text.push_str(text);
            }
            _ => turns.push(Turn {
                speaker,
                start: millis(segment.start),
                end: millis(segment.end),
                text: text.to_string(),
            }),
        }
    }
    turns
}

/// A turn as a text line: `[speaker_k] text`, `[?]` for nobody.
pub fn turn_line(turn: &Turn) -> String {
    format!("[{}] {}", turn.speaker.as_deref().unwrap_or("?"), turn.text)
}

/// One file, as `--json` prints it.
#[derive(Serialize)]
struct JsonTranscript<'a> {
    file: String,
    #[serde(flatten)]
    transcript: &'a Transcript,
}

impl<F: clap::Args + FamilyFlags> TranscribeCmd<F> {
    pub fn run(
        &self,
        family: Family,
    ) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let device = self.device.device()?;
        // The options come from the command line.
        let cache =
            PretrainedCache::new(self.flags.cache_options(self.cache.options())).as_policy()?;
        let model = self.model.as_deref().unwrap_or(family.default_model());
        let loaded = family.load(model, &self.flags, &cache, &device)?;
        let speakers = self.speakers.diarizer(&cache, &device)?;

        let driver = Driver::build(loaded, &self.flags, self.preset, speakers, &cache, &device)?;
        let rate = driver.sample_rate();
        let chunk = driver.push_samples(self.chunk_ms);

        let (mut audio, mut busy) = (0.0, Duration::ZERO);
        for path in &self.files {
            let samples = load_audio_mono_sr(path, rate)?;
            let t0 = Instant::now();
            let transcript = driver.transcribe(&samples, chunk)?;
            let elapsed = t0.elapsed();

            if self.json {
                let json = JsonTranscript {
                    file: path.display().to_string(),
                    transcript: &transcript,
                };
                println!(
                    "{}",
                    serde_json::to_string(&json).expect("a transcript serializes")
                );
            } else if self.rttm {
                let id = recording_id(path);
                for turn in transcript.turns.iter().flatten() {
                    if let Some(speaker) = &turn.speaker {
                        let segment = RttmSegment {
                            file: id.clone(),
                            channel: 1,
                            start: turn.start,
                            duration: turn.end - turn.start,
                            speaker: speaker.clone(),
                        };
                        println!("{segment}");
                    }
                }
            } else if let Some(turns) = &transcript.turns {
                for turn in turns {
                    println!("{}", turn_line(turn));
                }
            } else {
                println!("{}", transcript.text);
            }

            let seconds = samples.len() as f64 / rate as f64;
            log::info!(
                "{}: {seconds:.1} s of audio in {elapsed:.2?} ({:.1}x realtime)",
                path.display(),
                seconds / elapsed.as_secs_f64()
            );
            audio += seconds;
            busy += elapsed;
        }
        if self.files.len() > 1 {
            log::info!(
                "{} files: {audio:.1} s of audio in {busy:.2?} ({:.1}x realtime)",
                self.files.len(),
                audio / busy.as_secs_f64()
            );
        }
        Ok(())
    }
}

/// Seconds, rounded to the millisecond.
pub fn millis(seconds: f64) -> f64 {
    (seconds * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use bunsen::kits::speech::asr_driver::SpeakerLabel;

    use super::*;

    fn segment(
        text: &str,
        start: f64,
        speaker: Option<usize>,
    ) -> TranscriptSegment {
        TranscriptSegment {
            start,
            end: start + 0.5,
            tokens: Vec::new(),
            text: Some(text.to_string()),
            speaker: speaker.map(|index| SpeakerLabel {
                index,
                confidence: 0.9,
            }),
        }
    }

    /// Consecutive segments of one speaker are one turn; empty ones are left
    /// out, and a change of speaker, nobody included, starts a turn.
    #[test]
    fn test_speaker_turns() {
        let turns = speaker_turns(&[
            segment(" hello", 0.0, Some(0)),
            segment("  ", 0.5, Some(1)),
            segment("there", 1.0, Some(0)),
            segment("hi", 2.0, Some(1)),
            segment("um", 3.0, None),
        ]);
        let lines: Vec<String> = turns.iter().map(turn_line).collect();
        assert_eq!(
            lines,
            ["[speaker_0] hello there", "[speaker_1] hi", "[?] um"]
        );
        assert_eq!((turns[0].start, turns[0].end), (0.0, 1.5));
    }
}
