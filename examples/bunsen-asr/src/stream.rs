//! Transcribing through a stream driver, whichever family's engine it runs.
use std::sync::Arc;

use bunsen::{
    data::pretrained::PretrainedCache,
    errors::BunsenResult,
    kits::speech::{
        asr_driver::{
            AsrEngine,
            AsrStreamContext,
            AsrStreamDriver,
            EmissionPolicy,
            LevelPolicy,
            PresetEmissionPolicy,
            StreamClock,
            TranscriptEvent,
            TranscriptSegment,
        },
        silero_vad::pretrained::default_silero_factory,
    },
};
use burn::tensor::Device;

/// When the driver decodes.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// Whole windows, one after another.
    Offline,

    /// Speech regions as well, found by Silero's VAD (`silero/vad-16k`,
    /// through the same cache): silence is never decoded.
    Conservative,
}

impl From<Preset> for EmissionPolicy {
    fn from(preset: Preset) -> Self {
        match preset {
            Preset::Offline => PresetEmissionPolicy::Offline,
            Preset::Conservative => PresetEmissionPolicy::Conservative,
        }
        .into()
    }
}

/// `driver`, with Silero's VAD attached when `preset` finds speech regions.
pub fn with_vad<E: AsrEngine>(
    driver: AsrStreamDriver<E>,
    preset: Preset,
    cache: &PretrainedCache,
    device: &Device,
) -> BunsenResult<AsrStreamDriver<E>> {
    if preset == Preset::Offline {
        return Ok(driver);
    }
    let vad = default_silero_factory()?
        .load("silero/vad-16k", cache, device)?
        .handle;
    let rate = driver.sample_rate();
    driver.with_vad(
        Arc::new(vad.expect_branch(rate).clone()),
        Default::default(),
    )
}

/// Streams one recording through a new context of `driver`.
///
/// # Returns
/// The committed segments, in order, and the finished context.
pub fn stream<E: AsrEngine>(
    driver: &AsrStreamDriver<E>,
    samples: &[f32],
    level: LevelPolicy,
) -> BunsenResult<(Vec<TranscriptSegment>, AsrStreamContext<E>)> {
    let mut ctx = driver.new_context(StreamClock::uniform(driver.sample_rate()), level)?;
    // About a second of audio a push, a whole number of the driver's grain:
    // its grid and the VAD's chunk, so a push leaves nothing staged behind it.
    let mut grain = driver.grid_samples();
    if let Some(vad) = driver.vad() {
        grain = lcm(grain, vad.chunk_size());
    }
    let chunk = driver.sample_rate().div_ceil(grain) * grain;

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
