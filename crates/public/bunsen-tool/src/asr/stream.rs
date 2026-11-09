//! Transcribing through a stream driver, whichever family's engine it runs.
use std::sync::Arc;

use bunsen::{
    data::pretrained::PretrainedCache,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::{
            asr::driver::{
                AsrEngine,
                AsrStreamContext,
                AsrStreamDriver,
                EmissionMode,
                EmissionPolicy,
                MeanActivity,
                StreamClock,
                TranscriptEvent,
                TranscriptSegment,
            },
            diarization::SpeakerActivityModel,
            vad::silero_vad::pretrained::default_silero_factory,
        },
        support::construct::LoadPolicy,
    },
};
use burn::tensor::Device;

use crate::asr::construct_config;

/// When the driver decodes.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Emission {
    /// Whole windows, one after another.
    Offline,

    /// Speech regions as well, found by the VAD (`--vad`; Silero's
    /// `silero/vad-16k`, through the same cache, by default): silence is
    /// never decoded.
    Conservative,

    /// Conservative, plus a draft of the speech so far every 600 ms of it:
    /// `live` shows the drafts as they come; `transcribe` prints only what
    /// is committed.
    Drafting,
}

impl From<Emission> for EmissionPolicy {
    fn from(emission: Emission) -> Self {
        match emission {
            Emission::Offline => EmissionMode::Offline,
            Emission::Conservative => EmissionMode::Conservative,
            Emission::Drafting => EmissionMode::Drafting,
        }
        .into()
    }
}

/// How a run streams: when it decodes, the VAD that finds the speech
/// regions, and how much audio each push carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Streaming {
    pub emission: Emission,

    /// The VAD slot's filler, a lookup expression; `None` under an emission
    /// that finds no speech regions ([`VadArgs::slot`]).
    pub vad: Option<String>,

    /// Milliseconds of audio per push.
    pub chunk_ms: usize,
}

/// The VAD slot: `--vad`, the model that finds the speech regions.
#[derive(clap::Args, Clone, Debug, Default)]
pub struct VadArgs {
    /// The VAD that finds the speech regions, as a lookup expression
    /// (`silero/vad-16k[silero]`); `asr presets list`'s when omitted. Only
    /// with an emission that finds them: an error with `--emission offline`.
    #[arg(long, value_name = "EXPR")]
    pub vad: Option<String>,
}

impl VadArgs {
    /// The VAD slot's filler under `emission`: `--vad`, else the table's;
    /// `None` under [`Offline`](Emission::Offline), which finds no speech
    /// regions.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) on `--vad` with
    /// [`Offline`](Emission::Offline): a slot flag for a slot the run does
    /// not use.
    pub fn slot(
        &self,
        emission: Emission,
    ) -> BunsenResult<Option<String>> {
        match (emission, &self.vad) {
            (Emission::Offline, Some(_)) => Err(BunsenError::policy(
                "`--vad` has no effect with `--emission offline`",
            )),
            (Emission::Offline, None) => Ok(None),
            (_, Some(vad)) => Ok(Some(vad.clone())),
            (_, None) => Ok(Some(construct_config().vad.clone())),
        }
    }
}

/// `driver`, with the VAD `vad` names attached; as it is without one.
pub fn with_vad<E: AsrEngine>(
    driver: AsrStreamDriver<E>,
    vad: Option<&str>,
    cache: &PretrainedCache,
    device: &Device,
) -> BunsenResult<AsrStreamDriver<E>> {
    let Some(vad) = vad else {
        return Ok(driver);
    };
    let vad = default_silero_factory()?
        .load(vad, cache, device, LoadPolicy::performance())?
        .handle;
    let rate = driver.sample_rate();
    driver.with_vad(
        Arc::new(vad.expect_branch(rate).clone()),
        vad.vad_filter().clone(),
    )
}

/// `driver`, with `speakers` attached when there is one: each segment is
/// attributed by its mean activity, `NeMo`'s word rule.
pub fn with_speakers<E: AsrEngine>(
    driver: AsrStreamDriver<E>,
    speakers: Option<Arc<dyn SpeakerActivityModel>>,
) -> BunsenResult<AsrStreamDriver<E>> {
    match speakers {
        Some(model) => driver.with_speakers(model, Arc::new(MeanActivity::default())),
        None => Ok(driver),
    }
}

/// Samples per push of `ms` milliseconds of audio, rounded up to a whole
/// number of the driver's grain (its grid and the VAD's chunk), so a push
/// leaves nothing staged behind it; at least one grain.
pub fn push_samples<E: AsrEngine>(
    driver: &AsrStreamDriver<E>,
    ms: usize,
) -> usize {
    let mut grain = driver.grid_samples();
    if let Some(vad) = driver.vad() {
        grain = lcm(grain, vad.chunk_size());
    }
    (ms * driver.sample_rate())
        .div_ceil(1000)
        .div_ceil(grain)
        .max(1)
        * grain
}

/// Opens a stream of `driver`: its clock at the model's rate from 0, its
/// windows packaged against the level the engine's reference uses
/// ([`level_policy`](AsrEngine::level_policy)).
///
/// # Errors
/// As [`AsrStreamDriver::new_context`].
pub fn open<E: AsrEngine>(driver: &AsrStreamDriver<E>) -> BunsenResult<AsrStreamContext<E>> {
    driver.new_context(
        StreamClock::uniform(driver.sample_rate()),
        driver.engine().level_policy(),
    )
}

/// Streams one recording through a new context of `driver` ([`open`]),
/// `chunk` samples a push.
///
/// # Returns
/// The committed segments, in order, and the finished context.
///
/// # Errors
/// As [`open`], and the context's writes and reads.
pub fn stream<E: AsrEngine>(
    driver: &AsrStreamDriver<E>,
    samples: &[f32],
    chunk: usize,
) -> BunsenResult<(Vec<TranscriptSegment>, AsrStreamContext<E>)> {
    let mut ctx = open(driver)?;

    let mut segments = Vec::new();
    let mut keep = |events: Vec<TranscriptEvent>| {
        segments.extend(events.into_iter().filter_map(|e| match e {
            TranscriptEvent::Committed(segment) => Some(segment),
            TranscriptEvent::Draft(_) => None,
        }))
    };
    for block in samples.chunks(chunk) {
        keep(ctx.write_read(block)?);
    }
    keep(ctx.end_read()?);
    Ok((segments, ctx))
}

/// The segments' texts, trimmed, the empty ones left out, joined with
/// spaces.
pub fn join_text(segments: &[TranscriptSegment]) -> String {
    segments
        .iter()
        .filter_map(|s| s.text.as_deref().map(str::trim))
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn gcd(
    a: usize,
    b: usize,
) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(
    a: usize,
    b: usize,
) -> usize {
    a / gcd(a, b) * b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcm() {
        assert_eq!(lcm(160, 320), 320);
        assert_eq!(lcm(320, 512), 2560);
    }

    #[test]
    fn test_join_text() {
        let segment = |text: Option<&str>| TranscriptSegment {
            start: 0.0,
            end: 0.0,
            tokens: Vec::new(),
            text: text.map(String::from),
            speaker: None,
        };
        let segments = [
            segment(Some(" the")),
            segment(None),
            segment(Some("  ")),
            segment(Some("moon. ")),
        ];
        assert_eq!(join_text(&segments), "the moon.");
    }
}
