use clap::Parser;

use super::*;
use crate::asr::{
    Mode,
    presets,
    shell_line,
};

/// `asr`'s commands alone, as `bunsen-tool asr` parses them.
#[derive(clap::Parser, Debug)]
struct Args {
    #[clap(subcommand)]
    command: AsrSubCommand,
}

/// Each family parses its own `list` and `transcribe`.
#[test]
fn test_families_parse() {
    for family in ["parakeet", "qwen3-asr", "voxtral", "whisper"] {
        Args::try_parse_from(["asr", "family", family, "list"]).unwrap();
        Args::try_parse_from(["asr", "family", family, "transcribe", "a.wav"]).unwrap();
    }
    assert!(Args::try_parse_from(["asr", "silero", "list"]).is_err());
    // The diarizer: its presets and its outputs, one at a time.
    Args::try_parse_from(["asr", "diarize", "list"]).unwrap();
    Args::try_parse_from([
        "asr",
        "diarize",
        "fetch",
        "nvidia/sortformer/nemotron-3-diarization",
    ])
    .unwrap();
    Args::try_parse_from([
        "asr",
        "diarize",
        "run",
        "--preset",
        "very-low,dihard3",
        "--rttm",
        "a.wav",
    ])
    .unwrap();
    // Its latency and post-processing are the model's presets, not flags.
    for flag in ["--latency", "--postprocess"] {
        assert!(
            Args::try_parse_from(["asr", "diarize", "run", flag, "low", "a.wav"]).is_err(),
            "{flag}"
        );
    }
    assert!(Args::try_parse_from(["asr", "diarize", "run", "--rttm", "--json", "a.wav"]).is_err());
    assert!(Args::try_parse_from(["asr", "diarize", "run"]).is_err());
    assert!(Args::try_parse_from(["asr", "diarize", "transcribe", "a.wav"]).is_err());
    // Every family diarizes: `--diarize` alone is the default model, and
    // never takes the file that follows it.
    for family in ["parakeet", "qwen3-asr", "voxtral", "whisper"] {
        let Args { command: parsed } =
            Args::try_parse_from(["asr", "family", family, "transcribe", "--diarize", "a.wav"])
                .unwrap();
        let debug = format!("{parsed:?}");
        assert!(
            debug.contains("nvidia/sortformer/nemotron-3-diarization"),
            "{debug}"
        );
        assert!(debug.contains("\"a.wav\""), "{debug}");
        Args::try_parse_from([
            "asr",
            "family",
            family,
            "transcribe",
            "--diarize=nvidia/sortformer/nemotron-3-diarization[very-low]",
            "--rttm",
            "a.wav",
        ])
        .unwrap();
        assert!(
            Args::try_parse_from(["asr", "family", family, "transcribe", "--rttm", "a.wav"])
                .is_err(),
            "{family}: --rttm without --diarize"
        );
        assert!(
            Args::try_parse_from([
                "asr",
                "family",
                family,
                "transcribe",
                "--diarize",
                "--diarize-latency",
                "low",
                "a.wav"
            ])
            .is_err(),
            "{family}: --diarize-latency"
        );
    }
    // Whisper's own: its paths, a checkpoint path, inspect and prefabs;
    // its policy is the model's (`--model 'openai/whisper/tiny{timestamps:
    // true}'`).
    Args::try_parse_from([
        "asr",
        "family",
        "whisper",
        "transcribe",
        "--vocab",
        "v.tiktoken",
        "--upstream-cache-dir",
        "/tmp/w",
        "--model",
        "ckpt.pt",
        "a.wav",
    ])
    .unwrap();
    Args::try_parse_from(["asr", "family", "whisper", "inspect", "openai/whisper/tiny"]).unwrap();
    Args::try_parse_from(["asr", "family", "whisper", "prefabs"]).unwrap();
    assert!(Args::try_parse_from(["asr", "family", "parakeet", "inspect", "x"]).is_err());
    // The policy flags are gone from every family.
    for (family, flag) in [
        ("whisper", "--language=de"),
        ("whisper", "--task=translate"),
        ("whisper", "--beam-size=1"),
        ("whisper", "--timestamps"),
        ("whisper", "--max-tokens=100"),
        ("whisper", "--prompt-carry=false"),
        ("whisper", "--fallback=false"),
        ("parakeet", "--window-seconds=20"),
        ("qwen3-asr", "--window-seconds=20"),
        ("voxtral", "--window-seconds=20"),
        ("voxtral", "--delay-ms=960"),
    ] {
        assert!(
            Args::try_parse_from(["asr", "family", family, "transcribe", flag, "a.wav"]).is_err(),
            "{family} {flag}"
        );
    }
    // Every family fetches, at least one name.
    for family in ["parakeet", "qwen3-asr", "voxtral", "whisper"] {
        Args::try_parse_from(["asr", "family", family, "fetch", "--verify", "a", "b"]).unwrap();
        assert!(Args::try_parse_from(["asr", "family", family, "fetch"]).is_err());
    }
    // Every family listens live, under any emission.
    #[cfg(feature = "live")]
    for family in ["parakeet", "qwen3-asr", "voxtral", "whisper"] {
        Args::try_parse_from([
            "asr",
            "family",
            family,
            "live",
            "--emission",
            "drafting",
            "--chunk-ms",
            "100",
            "--seconds",
            "5",
        ])
        .unwrap();
    }
    // Every family takes an emission.
    for family in ["parakeet", "qwen3-asr", "voxtral", "whisper"] {
        Args::try_parse_from([
            "asr",
            "family",
            family,
            "transcribe",
            "--emission",
            "conservative",
            "a.wav",
        ])
        .unwrap();
    }
    // Each family's flags are its own; its policy is its model's patch.
    Args::try_parse_from([
        "asr",
        "family",
        "parakeet",
        "transcribe",
        "--model",
        "fermion/phonon-2{window_seconds: 20}",
        "a.wav",
    ])
    .unwrap();
    assert!(
        Args::try_parse_from([
            "asr",
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
            "asr",
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
/// The presets' commands parse; the families are under `family` alone.
#[test]
fn test_profiles_parse() {
    Args::try_parse_from(["asr", "transcribe", "a.wav"]).unwrap();
    Args::try_parse_from([
        "asr",
        "transcribe",
        "--preset",
        "multi",
        "--language",
        "de",
        "--emission",
        "conservative",
        "--chunk-ms",
        "500",
        "--diarize",
        "--json",
        "a.wav",
        "b.wav",
    ])
    .unwrap();
    assert!(Args::try_parse_from(["asr", "transcribe"]).is_err());
    // The slots: the engine's model and the VAD.
    Args::try_parse_from([
        "asr",
        "transcribe",
        "--model",
        "fermion/phonon-1",
        "--vad",
        "silero/vad-16k[silero]",
        "a.wav",
    ])
    .unwrap();
    Args::try_parse_from([
        "asr",
        "family",
        "whisper",
        "transcribe",
        "--vad",
        "silero/vad-16k[silero]",
        "a.wav",
    ])
    .unwrap();
    // `--opt` is gone: a model's settings are its expression's.
    assert!(Args::try_parse_from(["asr", "transcribe", "--opt", "beam-size=1", "a.wav"]).is_err());
    #[cfg(feature = "live")]
    Args::try_parse_from([
        "asr",
        "live",
        "--preset",
        "translate-en",
        "--model",
        "openai/whisper/small[translate]{timestamps: true}",
        "--seconds",
        "5",
    ])
    .unwrap();
    for argv in [
        &["presets"][..],
        &["presets", "list"],
        &["presets", "show"],
        &["presets", "show", "en", "multi"],
        &["presets", "fetch", "--verify", "en"],
    ] {
        let mut full = vec!["asr"];
        full.extend_from_slice(argv);
        Args::try_parse_from(full).unwrap();
    }
    assert!(Args::try_parse_from(["asr", "presets", "fetch"]).is_err());
    // `presets` and `--profile` are gone, with no alias.
    assert!(Args::try_parse_from(["asr", "profiles"]).is_err());
    assert!(Args::try_parse_from(["asr", "transcribe", "--profile", "en", "a.wav"]).is_err());
    // The families moved under `family`.
    for family in ["parakeet", "qwen3-asr", "voxtral", "whisper"] {
        assert!(
            Args::try_parse_from(["asr", family, "transcribe", "a.wav"]).is_err(),
            "{family}"
        );
    }
}

/// Each recipe's family command line, as `presets show` prints it,
/// parses as that command.
#[test]
fn test_profile_command_lines_parse() {
    for preset in presets() {
        for mode in [Mode::Offline, Mode::Live] {
            #[cfg(not(feature = "live"))]
            if mode == Mode::Live {
                continue;
            }
            let resolved = preset.resolve(mode).unwrap();
            let mut line = resolved.command_line();
            if mode == Mode::Offline {
                line.push("a.wav".into());
            }
            // `bunsen-tool asr ...`: `asr`'s own parser takes `asr` as the
            // program name, so the line parses from its second word.
            assert_eq!(line[..2], ["bunsen-tool", "asr"]);
            let Args { command } = Args::try_parse_from(&line[1..])
                .unwrap_or_else(|e| panic!("{}: {e}", shell_line(&line)));
            assert!(matches!(&command, AsrSubCommand::Family(_)), "{command:?}");
        }
    }
}
