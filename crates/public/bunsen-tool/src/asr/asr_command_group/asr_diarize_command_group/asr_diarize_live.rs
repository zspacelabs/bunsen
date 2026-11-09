//! `asr diarize live`.

use bunsen::{
    errors::BunsenResult,
    kits::speech::diarization::{
        RttmSegment,
        SpeakerActivityModel,
        SpeakerActivityStream,
        SpeakerTurn,
    },
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    device::BackendArgs,
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};

use super::{
    AsrDiarizeOutputArgs,
    AsrDiarizerArgs,
    JsonTurn,
    text_line,
};
use crate::asr::CaptureArgs;

/// Diarizes the microphone: each speaker turn printed as it closes, as a
/// text line, an RTTM line (recording id `live`), or a JSON object.
///
/// Times are seconds from the start of capture. The input device is opened
/// as `<family> live` opens it; Ctrl-C, or `--seconds`, ends the stream
/// cleanly, closing the turns still open.
#[derive(clap::Args, Debug)]
pub struct AsrDiarizeLiveCommand {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub diarizer: AsrDiarizerArgs,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    #[clap(flatten)]
    pub device: BackendArgs,

    #[clap(flatten)]
    pub output: AsrDiarizeOutputArgs,

    /// Milliseconds of audio per push.
    #[arg(long, default_value_t = 250)]
    pub chunk_ms: usize,

    #[clap(flatten)]
    pub capture: CaptureArgs,
}

impl AsrDiarizeLiveCommand {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let Some(capture) = self.capture.open()? else {
            return Ok(());
        };
        let device = self.device.open()?;
        let cache = self.cache.init()?;
        let bundle = self.diarizer.load(&cache, &device)?;
        let diarizer = bundle.diarizer(bundle.streaming()?)?;
        let rate = bundle.sample_rate();
        let frame_seconds = SpeakerActivityModel::frame_samples(&diarizer) as f64 / rate as f64;
        let mut filter = bundle.turns().init(diarizer.max_speakers(), frame_seconds);
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
