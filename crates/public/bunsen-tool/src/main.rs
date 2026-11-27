//! `bunsen-tool`: bunsen's models on the command line. `bunsen-tool asr`
//! is speech to text, by preset or by model family, and who spoke when;
//! `bunsen-tool tts` is text to speech with ZONOS3-Mini or Qwen3-TTS;
//! `bunsen-tool chat` chats with a Qwen3, Qwen3.5, Gemma or nanochat
//! checkpoint; `bunsen-tool decide` answers typed questions about a context
//! with a Laya decision model; `bunsen-tool text` normalizes text on its own;
//! `bunsen-tool cache <path|list|clear>` manages
//! the weights cache every command loads through; `bunsen-tool summary` lists
//! the builds it runs, in brief; `bunsen-tool catalog` imports
//! build catalogs, and dumps, browses, resolves and selects over them.

use std::process::ExitCode;

use bunsen::errors::BunsenResult;
use bunsen_app::{
    cache::CacheCommand,
    catalog::CatalogCommand,
};
#[cfg(feature = "speech")]
use bunsen_tool::{
    asr,
    tts,
};
use bunsen_tool::{
    chat::ChatCommandGroup,
    classify::ClassifyCommandGroup,
    decide::DecideCommandGroup,
    devices::DevicesCommand,
    embed::EmbedCommandGroup,
    summary::SummaryCommand,
    support::did_you_mean,
    text::TextCommandGroup,
};
use clap::{
    ArgMatches,
    CommandFactory,
    FromArgMatches,
    Parser,
};

/// bunsen's models on the command line.
///
/// `bunsen-tool asr` is speech to text, by preset or by model family, and who
/// spoke when; `bunsen-tool tts` is text to speech with ZONOS3-Mini or
/// Qwen3-TTS;
/// `bunsen-tool chat` chats with a Qwen3, Qwen3.5, Gemma or
/// nanochat checkpoint; `bunsen-tool decide` answers typed questions about a
/// context with a Laya decision model; `bunsen-tool classify` and
/// `bunsen-tool embed` classify images and compute their features with a timm
/// or torchvision checkpoint; `bunsen-tool text` normalizes text on
/// its own; `bunsen-tool cache` manages the weights
/// cache every command loads through; `bunsen-tool summary` lists the builds
/// it runs, in brief; `bunsen-tool devices` lists the backends
/// and their devices; `bunsen-tool catalog` browses, resolves and selects over
/// the model catalog.
#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
    #[clap(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Speech to text, by preset or by model family, and who spoke when.
    ///
    /// By preset (`asr transcribe`, `asr live`) or by model family (`asr
    /// family`), and who spoke when (`asr diarize`).
    #[cfg(feature = "speech")]
    Asr(asr::AsrCommandGroup),

    /// Text to speech with ZONOS3-Mini or Qwen3-TTS.
    ///
    /// A text spoken to a WAV file or a speaker (`tts run`), the stock voices
    /// (`tts voices`), and the checkpoints (`tts list`, `tts fetch`).
    #[cfg(feature = "speech")]
    Tts(tts::TtsCommandGroup),

    /// Chat with a Qwen3, Qwen3.5, Gemma or nanochat checkpoint.
    ///
    /// One prompt (`chat run`), a conversation (`chat repl`), and the
    /// checkpoints (`chat list`, `chat fetch`); `chat family` for one family's.
    Chat(Box<ChatCommandGroup>),

    /// Typed questions about a context, answered by a Laya decision model.
    ///
    /// Answered with calibrated probabilities: a request (YAML or JSON) in
    /// (`decide run`), one result per question out; and the checkpoints
    /// (`decide list`, `decide fetch`).
    Decide(DecideCommandGroup),

    /// Image classification with a timm or torchvision checkpoint.
    ///
    /// Image files to their top classes, each through the checkpoint's own
    /// eval transform (`classify run`); and the checkpoints (`classify list`,
    /// `classify fetch`).
    Classify(ClassifyCommandGroup),

    /// Image features with a timm or torchvision checkpoint.
    ///
    /// Image files to their pooled features, written as safetensors (`embed
    /// run`).
    Embed(EmbedCommandGroup),

    /// Text normalization on its own: what TTS speaks and WER compares.
    ///
    /// A text through normalizers in order, to stdout (`text normalize`), and
    /// the normalizers this build has (`text normalizers`).
    Text(TextCommandGroup),

    /// The weights cache every command loads through.
    ///
    /// Where it is, what it holds (each file under its resource's full name:
    /// `qwen/qwen3/qwen3-0.6b/checkpoint`, an `hf:` repo's
    /// `hf/<org>/<repo>/<slot>`), and clearing it by name (`qwen`,
    /// `hf/openai`).
    Cache(CacheCommand),

    /// The built-in model catalog: browse it, resolve builds, select them.
    ///
    /// bunsen's built-in catalog, with any `--catalog` extensions imported over
    /// it in order: the state as a directory view (`tree`), each catalog as
    /// written (`dump`), one entry (`show`), a resolved build (`resolve`), and
    /// selection pipelines, each stage explained (`select`).
    Catalog(CatalogCommand),

    /// The builds this binary runs, in brief: by command, contract and
    /// publisher.
    ///
    /// One section per command that runs builds (`asr`, `chat`, `decide`); in
    /// each, the builds by contract, then by publisher, each by its full name
    /// with its size, dtypes, preset axes (the default starred) and aliases,
    /// and its description wrapped below it.
    Summary(SummaryCommand),

    /// The backends this build can run on and their default devices, as YAML or
    /// JSON.
    ///
    /// Each backend's default device, and what the device does with each float
    /// dtype (store, compute, accelerate).
    Devices(DevicesCommand),
}

/// Runs the command; a failure is printed as its report, then, for a name
/// a table does not have, the names nearest it ([`did_you_mean`]).
fn main() -> ExitCode {
    // A load keeps the converted copy of the weights it read, and uses it
    // next time, unless `--converted-weights false`.
    let matches = bunsen_app::cache::converted_weights_by_default(Args::command()).get_matches();
    let args = match Args::from_arg_matches(&matches) {
        Ok(args) => args,
        Err(error) => error.exit(),
    };
    match run(&args, &matches) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error:?}");
            if let Some(hint) = did_you_mean::hint(&error) {
                eprintln!("{hint}");
            }
            ExitCode::FAILURE
        }
    }
}

/// The matches of the subcommand `matches` ends in: the leaf command's
/// own, whose flags it reads in command-line order.
fn leaf(matches: &ArgMatches) -> &ArgMatches {
    let mut at = matches;
    while let Some((_, sub)) = at.subcommand() {
        at = sub;
    }
    at
}

/// Runs the command `args`, parsed from `matches`: `catalog select`, `chat`
/// and `decide` read their selection flags' order from them.
fn run(
    args: &Args,
    matches: &ArgMatches,
) -> BunsenResult<()> {
    bunsen_app::logging::add_target_prefix(env!("CARGO_CRATE_NAME"));
    match &args.command {
        #[cfg(feature = "speech")]
        Command::Asr(cmd) => cmd.run(),
        #[cfg(feature = "speech")]
        Command::Tts(cmd) => cmd.run(leaf(matches)),
        Command::Chat(cmd) => cmd.run(leaf(matches)),
        Command::Decide(cmd) => cmd.run(leaf(matches)),
        Command::Classify(cmd) => cmd.run(),
        Command::Embed(cmd) => cmd.run(),
        Command::Text(cmd) => cmd.run(),
        Command::Cache(cmd) => cmd.run(),
        Command::Catalog(cmd) => cmd.run(
            matches
                .subcommand_matches("catalog")
                .expect("the catalog subcommand"),
            bunsen::data::catalog::builtin_stack()?,
        ),
        Command::Summary(cmd) => cmd.run(),
        Command::Devices(cmd) => cmd.run(),
    }
}

#[cfg(test)]
mod tests {
    use bunsen::support::testing::expect_variant;

    use super::*;

    /// The flags of every subcommand are well formed.
    #[test]
    fn test_cli_is_well_formed() {
        Args::command().debug_assert();
    }

    /// `decide`'s commands parse.
    #[test]
    fn test_decide_parses() {
        for argv in [
            &[
                "bunsen-tool",
                "decide",
                "run",
                "--model",
                "laya-multilingual",
                "--format",
                "json",
                "req.yml",
            ][..],
            &["bunsen-tool", "decide", "run"],
            &["bunsen-tool", "decide", "list", "--urls"],
            &["bunsen-tool", "decide", "fetch", "laya"],
        ] {
            Args::try_parse_from(argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
        }
    }

    /// `classify`'s and `embed`'s commands parse.
    #[test]
    fn test_classify_and_embed_parse() {
        for argv in [
            &[
                "bunsen-tool",
                "classify",
                "run",
                "--model",
                "torchvision/resnet50",
                "-k",
                "3",
                "--format",
                "json",
                "cat.jpg",
                "dog.png",
            ][..],
            &["bunsen-tool", "classify", "run", "cat.jpg"],
            &["bunsen-tool", "classify", "list", "--urls"],
            &["bunsen-tool", "classify", "fetch", "timm/resnet50.a1_in1k"],
            &[
                "bunsen-tool",
                "embed",
                "run",
                "-o",
                "f.safetensors",
                "cat.jpg",
            ],
        ] {
            Args::try_parse_from(argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
        }
        assert!(Args::try_parse_from(["bunsen-tool", "classify", "run"]).is_err());
    }

    /// `tts`'s commands parse.
    #[cfg(feature = "speech")]
    #[test]
    fn test_tts_parses() {
        for argv in [
            &[
                "bunsen-tool",
                "tts",
                "run",
                "--voice",
                "Cillian",
                "--seed",
                "7",
                "-o",
                "out.wav",
                "Hello",
                "there.",
            ][..],
            &["bunsen-tool", "tts", "run", "--out", "out.wav"],
            &["bunsen-tool", "tts", "list", "--urls"],
            &["bunsen-tool", "tts", "voices", "--yes"],
            &["bunsen-tool", "tts", "run", "--play", "hi"],
            &["bunsen-tool", "tts", "fetch", "zonos3-mini"],
        ] {
            Args::try_parse_from(argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
        }
    }

    /// The weights cache's commands parse.
    #[test]
    fn test_cache_parses() {
        for action in [
            &["path"][..],
            &["list", "whisper"],
            &["clear", "qwen3-asr/fermion", "--yes"],
        ] {
            let mut argv = vec!["bunsen-tool", "cache", "--cache-dir", "/tmp/c"];
            argv.extend_from_slice(action);
            Args::try_parse_from(argv).unwrap();
        }
    }

    /// Every command's and argument's short help is one summary of at most
    /// [`SUMMARY`] characters: clap's derive takes a doc comment's first
    /// paragraph, up to its first blank line, as the short help (`-h`, and
    /// the command and flag lists), and the whole comment as the long help
    /// (`--help`).
    #[test]
    fn test_help_summaries_are_short() {
        const SUMMARY: usize = 80;
        fn walk(
            command: &clap::Command,
            path: &str,
            long: &mut Vec<String>,
        ) {
            let path = format!("{path} {}", command.get_name()).trim().to_string();
            let mut check = |what: String, text: String| {
                if text.chars().count() > SUMMARY {
                    long.push(format!("{what} ({}): {text}", text.chars().count()));
                }
            };
            if let Some(about) = command.get_about() {
                check(path.clone(), about.to_string());
            }
            for arg in command.get_arguments() {
                if let Some(help) = arg.get_help() {
                    check(format!("{path} {}", arg.get_id()), help.to_string());
                }
            }
            for sub in command.get_subcommands() {
                walk(sub, &path, long);
            }
        }
        let mut long = Vec::new();
        walk(&Args::command(), "", &mut long);
        long.sort();
        long.dedup();
        assert!(
            long.is_empty(),
            "{} long summaries:\n{}",
            long.len(),
            long.join("\n")
        );
    }

    /// `catalog` runs over the built-in catalog, at its own level of the
    /// matches, so `select` sees its flags in command-line order.
    #[test]
    fn test_catalog_runs_through_its_matches() {
        let argv = [
            "bunsen-tool".to_string(),
            "catalog".into(),
            "select".into(),
            "qwen3".into(),
            "--off".into(),
            "thinking".into(),
            "--desc".into(),
            "parameters".into(),
            "--limit".into(),
            "1".into(),
        ];
        let matches = Args::command().try_get_matches_from(argv).unwrap();
        let args = Args::from_arg_matches(&matches).unwrap();
        expect_variant!(let Command::Catalog(cmd) = &args.command, "the catalog command");
        let mut out = String::new();
        cmd.run_into(
            matches.subcommand_matches("catalog").unwrap(),
            bunsen::data::catalog::builtin_stack().unwrap(),
            &mut out,
        )
        .unwrap();
        let results: Vec<&str> = out.lines().filter(|l| l.starts_with("- name:")).collect();
        assert_eq!(results, ["- name: qwen/qwen3/qwen3-8b"], "{out}");
    }
}
