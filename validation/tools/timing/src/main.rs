//! `cargo timing`: the driver. `run` chooses the models as `cargo validate`
//! does and builds the runner for the backend; `compare` sets two runs side
//! by side. See the library's docs for what is timed.

use std::{
    path::PathBuf,
    process::{
        Command,
        ExitCode,
    },
    time::{
        SystemTime,
        UNIX_EPOCH,
    },
};

use clap::{
    Parser,
    Subcommand,
};
use validate::{
    profile,
    profile::Backend,
    schedule::{
        Dtype,
        ValidationSchedule,
    },
    validation_dir,
};

/// The validation crates whose models the runner can time: those whose kit
/// has a `TimedDecode`: every `CachedLm`, Whisper, Voxtral, ZONOS3's
/// language model and Qwen3-TTS.
const TIMED: &[&str] = &[
    "qwen3",
    "qwen3-5",
    "gemma",
    "nanochat",
    "whisper",
    "voxtral",
    "zonos3",
    "qwen3-tts",
];

/// Times bunsen's decodes on the validation catalog's checkpoints.
#[derive(Parser)]
#[command(name = "timing")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Times the models a profile chooses, on one backend, in each dtype.
    Run {
        /// A profile's name in `validation/profiles/`, or a path to one.
        #[arg(long, env = "BUNSEN_VALIDATE_PROFILE", default_value = "small-core")]
        profile: String,

        /// Only this crate (repeatable).
        #[arg(long = "crate")]
        crates: Vec<String>,

        /// Only this model, as `models.yml` names it (repeatable).
        #[arg(long = "model")]
        models: Vec<String>,

        /// The backend; else the profile's, else `wgpu`.
        #[arg(long, value_enum)]
        backend: Option<Backend>,

        /// Over the profile's: `f32`, `bf16` (on cuda alone).
        #[arg(long, value_enum, value_delimiter = ',')]
        dtypes: Option<Vec<Dtype>>,

        /// Build the runner with burn's fusion.
        #[arg(long)]
        fusion: bool,

        /// The JSON-lines file the records go to; else
        /// `validation/target/timing/{unix time}.jsonl`.
        #[arg(long)]
        out: Option<PathBuf>,

        /// The records' label; else the commit (`git describe --always
        /// --dirty`).
        #[arg(long)]
        label: Option<String>,

        /// Passed on to the runner: `--rows`, `--prefill`, `--steps`,
        /// `--prefill-reps`, `--buckets`, `--storage`.
        #[arg(last = true)]
        runner_args: Vec<String>,
    },

    /// Two runs' records side by side (`b / a`: above 1 is slower in `b`).
    Compare { a: PathBuf, b: PathBuf },
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("timing: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode, String> {
    match cli.command {
        Cmd::Compare { a, b } => {
            let (a, b) = (
                timing::compare::read_run(&a)?,
                timing::compare::read_run(&b)?,
            );
            timing::compare::print_compare(&a, &b);
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Run {
            profile: profile_name,
            crates,
            models,
            backend,
            dtypes,
            fusion,
            out,
            label,
            runner_args,
        } => {
            let validation = validation_dir();
            let schedule = ValidationSchedule::load(&validation.join("models.yml"))?;
            let profile = profile::load(&profile_name, &validation.join("profiles"))?;
            let backend = backend.or(profile.backend).unwrap_or(Backend::Wgpu);
            let dtypes = dtypes.unwrap_or_else(|| profile.dtypes());
            let out = out.unwrap_or_else(|| {
                let unix = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs());
                validation.join(format!("target/timing/{unix}.jsonl"))
            });
            let label = label.unwrap_or_else(|| describe(&validation));

            let chosen = profile::choose(&schedule, &profile, &crates, &dtypes);
            let mut code = ExitCode::SUCCESS;
            for dtype in dtypes {
                if dtype == Dtype::Bf16 && backend != Backend::Cuda {
                    eprintln!("timing: bf16 on cuda alone; skipped on {backend:?}");
                    continue;
                }
                let mut rows: Vec<String> = Vec::new();
                for (krate, picks) in &chosen {
                    if !TIMED.contains(&krate.as_str()) {
                        continue;
                    }
                    for (model, keep) in picks {
                        if *keep
                            && (models.is_empty() || models.contains(model))
                            && schedule.crates[krate].models[model].dtypes.contains(&dtype)
                        {
                            rows.push(format!("{krate}:{model}"));
                        }
                    }
                }
                let untimed: Vec<&String> = chosen
                    .iter()
                    .filter(|(krate, picks)| {
                        !TIMED.contains(&krate.as_str()) && picks.values().any(|k| *k)
                    })
                    .map(|(krate, _)| krate)
                    .collect();
                if !untimed.is_empty() {
                    eprintln!("timing: no timing adapter yet for {untimed:?}; skipped");
                }
                if rows.is_empty() {
                    eprintln!(
                        "timing: the profile chooses no timed model in {}",
                        dtype.name()
                    );
                    continue;
                }
                let mut features: Vec<&str> = backend.feature().into_iter().collect();
                if fusion {
                    features.push("fusion");
                }
                let mut command =
                    Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
                command.current_dir(&validation).args([
                    "run",
                    "-q",
                    "--profile",
                    "validate",
                    "-p",
                    "timing",
                    "--bin",
                    "timing-runner",
                ]);
                if !features.is_empty() {
                    command.args(["--features", &features.join(",")]);
                }
                command.arg("--");
                for row in &rows {
                    command.args(["--row", row]);
                }
                command
                    .args(["--dtype", dtype.name(), "--label", &label])
                    .arg("--out")
                    .arg(&out)
                    .args(&runner_args);
                eprintln!(
                    "timing: {} in {} -> {}",
                    rows.join(" "),
                    dtype.name(),
                    out.display()
                );
                let status = command
                    .status()
                    .map_err(|e| format!("run the timing runner: {e}"))?;
                if !status.success() {
                    code = ExitCode::FAILURE;
                }
            }
            Ok(code)
        }
    }
}

/// `git describe --always --dirty` of the checkout, or `unknown`.
fn describe(validation: &std::path::Path) -> String {
    Command::new("git")
        .current_dir(validation)
        .args(["describe", "--always", "--dirty"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
