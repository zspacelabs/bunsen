//! `bunsen-asr`: speech to text on bunsen. `bunsen-asr <transcribe|live>
//! [--profile NAME] [FLAGS]` runs a profile, a model chosen by what it is
//! for, and `bunsen-asr profiles <list|show|fetch>` shows them;
//! `bunsen-asr family <parakeet|qwen3-asr|voxtral|whisper>
//! <list|fetch|transcribe|live> [FLAGS]` (and `family whisper
//! inspect|prefabs`) runs a family's model by name; `bunsen-asr diarize
//! <list|fetch|run|live>` says who spoke when, and `bunsen-asr cache
//! <path|list|clear>` manages the weights cache.

use bunsen::errors::BunsenResult;
use clap::Parser;

#[cfg(feature = "live")]
mod capture;
mod diarize;
mod driver;
mod family;
mod fetch_cmd;
mod list_cmd;
#[cfg(feature = "live")]
mod live_cmd;
mod parakeet;
mod profile;
mod profile_cmd;
mod publish;
mod qwen3_asr;
#[cfg(feature = "live")]
mod resample;
mod stream;
mod transcribe_cmd;
mod voxtral;
mod whisper;
mod whisper_models;

use family::Family;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Transcribe audio files with a profile: a model chosen by what it is
    /// for (`en` by default).
    Transcribe(Box<profile_cmd::ProfileTranscribeCmd>),

    /// Transcribe the microphone, as it speaks, with a profile.
    #[cfg(feature = "live")]
    Live(Box<profile_cmd::ProfileLiveCmd>),

    /// The profiles: what each is for, and what it runs.
    Profiles(profile_cmd::ProfilesCmd),

    /// The model families, each with its own `list` and `transcribe`, and
    /// only the flags that apply to it.
    Family(FamilyCmd),

    /// NVIDIA's Sortformer, who spoke when: `nvidia/nemotron-3-diarization`
    /// by default.
    Diarize(diarize::DiarizeCmd),

    /// The weights cache every family loads through: where it is, what it
    /// holds (by kit: `parakeet`, `qwen3-asr`, `voxtral`, `whisper`,
    /// `sortformer`, `silero_vad`), and clearing it.
    Cache(bunsen_app::cache::CacheCommand),
}

/// The model families' commands.
#[derive(clap::Args, Debug)]
struct FamilyCmd {
    #[command(subcommand)]
    family: FamilySub,
}

/// The model families, each with its own `list` and `transcribe`.
#[derive(clap::Subcommand, Debug)]
enum FamilySub {
    /// NVIDIA's Parakeet TDT: `FermionResearch`'s Phonon-2 by default.
    Parakeet(KitCmd<parakeet::ParakeetFlags>),

    /// Qwen's Qwen3-ASR: `FermionResearch`'s Phonon-1 by default.
    #[command(name = "qwen3-asr")]
    Qwen3Asr(KitCmd<qwen3_asr::Qwen3AsrFlags>),

    /// Mistral's Voxtral Realtime: `mistralai/voxtral-mini-4b-realtime-2602`
    /// by default.
    Voxtral(KitCmd<voxtral::VoxtralFlags>),

    /// `OpenAI`'s Whisper: `openai/base` by default.
    Whisper(WhisperCmd),
}

impl FamilyCmd {
    fn run(&self) -> BunsenResult<()> {
        match &self.family {
            FamilySub::Parakeet(cmd) => cmd.run(Family::Parakeet),
            FamilySub::Qwen3Asr(cmd) => cmd.run(Family::Qwen3Asr),
            FamilySub::Voxtral(cmd) => cmd.run(Family::Voxtral),
            FamilySub::Whisper(cmd) => cmd.run(),
        }
    }
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
    Kit(Box<Action<whisper::WhisperFlags>>),

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
            Action::List(list) => list.run(family.into()),
            Action::Fetch(fetch) => fetch.run(family.into()),
            Action::Transcribe(transcribe) => transcribe.run(family),
            #[cfg(feature = "live")]
            Action::Live(live) => live.run(family),
        }
    }
}

fn main() -> BunsenResult<()> {
    bunsen_app::logging::add_target_prefix(env!("CARGO_CRATE_NAME"));
    match &Args::parse().command {
        Command::Transcribe(cmd) => cmd.run(),
        #[cfg(feature = "live")]
        Command::Live(cmd) => cmd.run(),
        Command::Profiles(cmd) => cmd.run(),
        Command::Family(cmd) => cmd.run(),
        Command::Diarize(cmd) => cmd.run(),
        Command::Cache(cmd) => cmd.run(),
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
        for family in ["parakeet", "qwen3-asr", "voxtral", "whisper"] {
            Args::try_parse_from(["bunsen-asr", "family", family, "list"]).unwrap();
            Args::try_parse_from(["bunsen-asr", "family", family, "transcribe", "a.wav"]).unwrap();
        }
        assert!(Args::try_parse_from(["bunsen-asr", "silero", "list"]).is_err());
        // The diarizer: its presets and its outputs, one at a time.
        Args::try_parse_from(["bunsen-asr", "diarize", "list"]).unwrap();
        Args::try_parse_from([
            "bunsen-asr",
            "diarize",
            "fetch",
            "nvidia/nemotron-3-diarization",
        ])
        .unwrap();
        Args::try_parse_from([
            "bunsen-asr",
            "diarize",
            "run",
            "--latency",
            "very-low",
            "--postprocess",
            "dihard3",
            "--rttm",
            "a.wav",
        ])
        .unwrap();
        assert!(
            Args::try_parse_from(["bunsen-asr", "diarize", "run", "--rttm", "--json", "a.wav"])
                .is_err()
        );
        assert!(Args::try_parse_from(["bunsen-asr", "diarize", "run"]).is_err());
        assert!(Args::try_parse_from(["bunsen-asr", "diarize", "transcribe", "a.wav"]).is_err());
        // Every family diarizes: `--diarize` alone is the default model, and
        // never takes the file that follows it.
        for family in ["parakeet", "qwen3-asr", "voxtral", "whisper"] {
            let Args { command: parsed } = Args::try_parse_from([
                "bunsen-asr",
                "family",
                family,
                "transcribe",
                "--diarize",
                "a.wav",
            ])
            .unwrap();
            let debug = format!("{parsed:?}");
            assert!(debug.contains("nvidia/nemotron-3-diarization"), "{debug}");
            assert!(debug.contains("\"a.wav\""), "{debug}");
            Args::try_parse_from([
                "bunsen-asr",
                "family",
                family,
                "transcribe",
                "--diarize=hf:org/repo",
                "--diarize-latency",
                "ultra-low",
                "--rttm",
                "a.wav",
            ])
            .unwrap();
            assert!(
                Args::try_parse_from([
                    "bunsen-asr",
                    "family",
                    family,
                    "transcribe",
                    "--rttm",
                    "a.wav"
                ])
                .is_err(),
                "{family}: --rttm without --diarize"
            );
        }
        // Whisper's own: its decode flags, a checkpoint path, inspect and
        // prefabs.
        Args::try_parse_from([
            "bunsen-asr",
            "family",
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
        Args::try_parse_from(["bunsen-asr", "family", "whisper", "inspect", "openai/tiny"])
            .unwrap();
        Args::try_parse_from(["bunsen-asr", "family", "whisper", "prefabs"]).unwrap();
        assert!(
            Args::try_parse_from(["bunsen-asr", "family", "parakeet", "inspect", "x"]).is_err()
        );
        assert!(
            Args::try_parse_from([
                "bunsen-asr",
                "family",
                "parakeet",
                "transcribe",
                "--timestamps",
                "a.wav"
            ])
            .is_err()
        );
        // Every family fetches, at least one name.
        for family in ["parakeet", "qwen3-asr", "voxtral", "whisper"] {
            Args::try_parse_from([
                "bunsen-asr",
                "family",
                family,
                "fetch",
                "--verify",
                "a",
                "b",
            ])
            .unwrap();
            assert!(Args::try_parse_from(["bunsen-asr", "family", family, "fetch"]).is_err());
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
        for family in ["parakeet", "qwen3-asr", "voxtral", "whisper"] {
            Args::try_parse_from([
                "bunsen-asr",
                "family",
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
        for family in ["parakeet", "qwen3-asr", "voxtral", "whisper"] {
            Args::try_parse_from([
                "bunsen-asr",
                "family",
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
            "family",
            "parakeet",
            "transcribe",
            "--window-seconds",
            "20",
            "a.wav",
        ])
        .unwrap();
        Args::try_parse_from([
            "bunsen-asr",
            "family",
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
                "family",
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
                "family",
                "parakeet",
                "transcribe",
                "--language",
                "en",
                "a.wav"
            ])
            .is_err()
        );
    }
    /// The profiles' commands parse; the families are under `family` alone.
    #[test]
    fn test_profiles_parse() {
        Args::try_parse_from(["bunsen-asr", "transcribe", "a.wav"]).unwrap();
        Args::try_parse_from([
            "bunsen-asr",
            "transcribe",
            "--profile",
            "multi",
            "--language",
            "de",
            "--opt",
            "whisper.beam-size=1",
            "--opt",
            "window-seconds=20",
            "--preset",
            "conservative",
            "--chunk-ms",
            "500",
            "--diarize",
            "--json",
            "a.wav",
            "b.wav",
        ])
        .unwrap();
        assert!(Args::try_parse_from(["bunsen-asr", "transcribe"]).is_err());
        assert!(
            Args::try_parse_from(["bunsen-asr", "transcribe", "--model", "x", "a.wav"]).is_err()
        );
        assert!(
            Args::try_parse_from(["bunsen-asr", "transcribe", "--opt", "klingon.x=1", "a.wav"])
                .is_err()
        );
        #[cfg(feature = "live")]
        Args::try_parse_from([
            "bunsen-asr",
            "live",
            "--profile",
            "translate-en",
            "--opt",
            "whisper.timestamps",
            "--seconds",
            "5",
        ])
        .unwrap();
        for argv in [
            &["profiles"][..],
            &["profiles", "list"],
            &["profiles", "show"],
            &["profiles", "show", "en", "multi"],
            &["profiles", "fetch", "--verify", "en"],
        ] {
            let mut full = vec!["bunsen-asr"];
            full.extend_from_slice(argv);
            Args::try_parse_from(full).unwrap();
        }
        assert!(Args::try_parse_from(["bunsen-asr", "profiles", "fetch"]).is_err());
        // The families moved under `family`.
        for family in ["parakeet", "qwen3-asr", "voxtral", "whisper"] {
            assert!(
                Args::try_parse_from(["bunsen-asr", family, "transcribe", "a.wav"]).is_err(),
                "{family}"
            );
        }
    }

    /// Each recipe's family command line, as `profiles show` prints it,
    /// parses as that command.
    #[test]
    fn test_profile_command_lines_parse() {
        for profile in profile::PROFILES {
            for mode in [profile::Mode::Offline, profile::Mode::Live] {
                #[cfg(not(feature = "live"))]
                if mode == profile::Mode::Live {
                    continue;
                }
                let resolved = profile.resolve(mode).unwrap();
                let mut line = resolved.command_line().unwrap();
                if mode == profile::Mode::Offline {
                    line.push("a.wav".into());
                }
                let Args { command } = Args::try_parse_from(&line)
                    .unwrap_or_else(|e| panic!("{}: {e}", profile::shell_line(&line)));
                assert!(matches!(command, Command::Family(_)), "{command:?}");
            }
        }
    }
}
