//! `bunsen-asr`: speech to text on bunsen, one subcommand per model family:
//! `bunsen-asr <parakeet|qwen3-asr|whisper> <list|transcribe> [FLAGS]`.

use bunsen::errors::BunsenResult;
use clap::Parser;

mod args;
mod family;
mod list_cmd;
mod parakeet;
mod qwen3_asr;
mod stream;
mod transcribe_cmd;
mod whisper;

use family::Family;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    family: FamilyCmd,
}

/// The model families, each with its own `list` and `transcribe`.
#[derive(clap::Subcommand, Debug)]
enum FamilyCmd {
    /// NVIDIA's Parakeet TDT: `FermionResearch`'s Phonon-2 by default.
    Parakeet(KitCmd<parakeet::ParakeetFlags>),

    /// Qwen's Qwen3-ASR: `FermionResearch`'s Phonon-1 by default.
    #[command(name = "qwen3-asr")]
    Qwen3Asr(KitCmd<qwen3_asr::Qwen3AsrFlags>),

    /// `OpenAI`'s Whisper: `openai/base` by default.
    Whisper(KitCmd<whisper::WhisperFlags>),
}

/// One family's commands; `F` is its own transcription flags.
#[derive(clap::Args, Debug)]
struct KitCmd<F: clap::Args> {
    #[command(subcommand)]
    action: Action<F>,
}

#[derive(clap::Subcommand, Debug)]
enum Action<F: clap::Args> {
    /// List the models `--model` can name, their licences and files, and
    /// whether each file is cached.
    List(list_cmd::ListCmd),

    /// Transcribe audio files.
    Transcribe(transcribe_cmd::TranscribeCmd<F>),
}

impl<F: clap::Args + transcribe_cmd::FamilyFlags> KitCmd<F> {
    fn run(
        &self,
        family: Family,
    ) -> BunsenResult<()> {
        match &self.action {
            Action::List(list) => list.run(family),
            Action::Transcribe(transcribe) => transcribe.run(family),
        }
    }
}

fn main() -> BunsenResult<()> {
    match &Args::parse().family {
        FamilyCmd::Parakeet(cmd) => cmd.run(Family::Parakeet),
        FamilyCmd::Qwen3Asr(cmd) => cmd.run(Family::Qwen3Asr),
        FamilyCmd::Whisper(cmd) => cmd.run(Family::Whisper),
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    /// The flags of every subcommand are well formed.
    #[test]
    fn test_cli_is_well_formed() {
        Args::command().debug_assert();
    }

    /// Each family parses its own `list` and `transcribe`.
    #[test]
    fn test_families_parse() {
        for family in ["parakeet", "qwen3-asr", "whisper"] {
            Args::try_parse_from(["bunsen-asr", family, "list"]).unwrap();
            Args::try_parse_from(["bunsen-asr", family, "transcribe", "a.wav"]).unwrap();
        }
        assert!(Args::try_parse_from(["bunsen-asr", "silero", "list"]).is_err());
        // Every family takes a preset.
        for family in ["parakeet", "qwen3-asr", "whisper"] {
            Args::try_parse_from([
                "bunsen-asr",
                family,
                "transcribe",
                "--preset",
                "conservative",
                "a.wav",
            ])
            .unwrap();
        }
        // Each family's flags are its own.
        Args::try_parse_from([
            "bunsen-asr",
            "parakeet",
            "transcribe",
            "--window-seconds",
            "20",
            "a.wav",
        ])
        .unwrap();
        Args::try_parse_from([
            "bunsen-asr",
            "whisper",
            "transcribe",
            "--language",
            "en",
            "a.wav",
        ])
        .unwrap();
        assert!(
            Args::try_parse_from([
                "bunsen-asr",
                "parakeet",
                "transcribe",
                "--batch",
                "2",
                "a.wav"
            ])
            .is_err()
        );
        assert!(
            Args::try_parse_from([
                "bunsen-asr",
                "parakeet",
                "transcribe",
                "--language",
                "en",
                "a.wav"
            ])
            .is_err()
        );
    }
}
