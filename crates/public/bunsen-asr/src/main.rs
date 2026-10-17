//! `bunsen-asr`: speech to text on bunsen, one subcommand per model family:
//! `bunsen-asr <parakeet|qwen3-asr|whisper> <list|fetch|transcribe|live>
//! [FLAGS]` (and `whisper inspect|prefabs`), and `bunsen-asr cache
//! <path|list|clear>` for the weights cache.

use bunsen::errors::BunsenResult;
use clap::Parser;

mod args;
mod driver;
mod family;
mod fetch_cmd;
mod list_cmd;
#[cfg(feature = "live")]
mod live_cmd;
mod parakeet;
mod qwen3_asr;
#[cfg(feature = "live")]
mod resample;
mod stream;
mod transcribe_cmd;
mod whisper;
mod whisper_models;

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
    Whisper(WhisperCmd),

    /// The weights cache every family loads through: where it is, what it
    /// holds (by kit: `parakeet`, `qwen3-asr`, `whisper`, `silero_vad`), and
    /// clearing it.
    Cache(bunsen_app::cache::CacheCommand),
}

/// One family's commands; `F` is its own transcription flags.
#[derive(clap::Args, Debug)]
struct KitCmd<F: clap::Args> {
    #[command(subcommand)]
    action: Action<F>,
}

/// Whisper's commands: every family's, and its own.
#[derive(clap::Args, Debug)]
struct WhisperCmd {
    #[command(subcommand)]
    action: WhisperAction,
}

#[derive(clap::Subcommand, Debug)]
enum WhisperAction {
    #[command(flatten)]
    Kit(Action<whisper::WhisperFlags>),

    /// Scan a checkpoint and check it against its prefab.
    Inspect(whisper_models::InspectCmd),

    /// List the prefabs: the geometries, without weights.
    Prefabs(whisper_models::PrefabsCmd),
}

impl WhisperCmd {
    fn run(&self) -> BunsenResult<()> {
        match &self.action {
            WhisperAction::Kit(action) => action.run(Family::Whisper),
            WhisperAction::Inspect(inspect) => inspect.run(),
            WhisperAction::Prefabs(prefabs) => prefabs.run(),
        }
    }
}

#[derive(clap::Subcommand, Debug)]
enum Action<F: clap::Args> {
    /// List the models `--model` can name, their licences and files, and
    /// whether each file is cached.
    List(list_cmd::ListCmd),

    /// Bring every file of the named models into the weights cache.
    Fetch(fetch_cmd::FetchCmd),

    /// Transcribe audio files.
    Transcribe(transcribe_cmd::TranscribeCmd<F>),

    /// Transcribe the microphone, as it speaks.
    #[cfg(feature = "live")]
    Live(live_cmd::LiveCmd<F>),
}

impl<F: clap::Args + transcribe_cmd::FamilyFlags> KitCmd<F> {
    fn run(
        &self,
        family: Family,
    ) -> BunsenResult<()> {
        self.action.run(family)
    }
}

impl<F: clap::Args + transcribe_cmd::FamilyFlags> Action<F> {
    fn run(
        &self,
        family: Family,
    ) -> BunsenResult<()> {
        match self {
            Action::List(list) => list.run(family),
            Action::Fetch(fetch) => fetch.run(family),
            Action::Transcribe(transcribe) => transcribe.run(family),
            #[cfg(feature = "live")]
            Action::Live(live) => live.run(family),
        }
    }
}

fn main() -> BunsenResult<()> {
    match &Args::parse().family {
        FamilyCmd::Parakeet(cmd) => cmd.run(Family::Parakeet),
        FamilyCmd::Qwen3Asr(cmd) => cmd.run(Family::Qwen3Asr),
        FamilyCmd::Whisper(cmd) => cmd.run(),
        FamilyCmd::Cache(cmd) => cmd.run(),
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
        // Whisper's own: its decode flags, a checkpoint path, inspect and
        // prefabs.
        Args::try_parse_from([
            "bunsen-asr",
            "whisper",
            "transcribe",
            "--timestamps",
            "--max-tokens",
            "100",
            "--prompt-carry",
            "false",
            "--fallback",
            "false",
            "--vocab",
            "v.tiktoken",
            "--upstream-cache-dir",
            "/tmp/w",
            "--model",
            "ckpt.pt",
            "a.wav",
        ])
        .unwrap();
        Args::try_parse_from(["bunsen-asr", "whisper", "inspect", "openai/tiny"]).unwrap();
        Args::try_parse_from(["bunsen-asr", "whisper", "prefabs"]).unwrap();
        assert!(Args::try_parse_from(["bunsen-asr", "parakeet", "inspect", "x"]).is_err());
        assert!(
            Args::try_parse_from([
                "bunsen-asr",
                "parakeet",
                "transcribe",
                "--timestamps",
                "a.wav"
            ])
            .is_err()
        );
        // Every family fetches, at least one name.
        for family in ["parakeet", "qwen3-asr", "whisper"] {
            Args::try_parse_from(["bunsen-asr", family, "fetch", "--verify", "a", "b"]).unwrap();
            assert!(Args::try_parse_from(["bunsen-asr", family, "fetch"]).is_err());
        }
        // The weights cache's commands.
        for action in [
            &["path"][..],
            &["list", "whisper"],
            &["clear", "qwen3-asr/fermion", "--yes"],
        ] {
            let mut argv = vec!["bunsen-asr", "cache", "--cache-dir", "/tmp/c"];
            argv.extend_from_slice(action);
            Args::try_parse_from(argv).unwrap();
        }
        // Every family listens live, under any preset.
        #[cfg(feature = "live")]
        for family in ["parakeet", "qwen3-asr", "whisper"] {
            Args::try_parse_from([
                "bunsen-asr",
                family,
                "live",
                "--preset",
                "responsive",
                "--chunk-ms",
                "100",
                "--seconds",
                "5",
            ])
            .unwrap();
        }
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
