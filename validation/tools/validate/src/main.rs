//! Runs the validation crates under a profile: the models it chooses, their
//! tests alone.
//!
//! ```sh
//! cd validation
//! cargo validate list  [--profile P] [--crate C]...
//! cargo validate run   [--profile P] [--crate C]... [--backend B] [--dtypes D,...] [--dry-run] [--nextest] [-- TEST_ARGS]
//! cargo validate check [--crate C]... [--reference]
//! cargo validate catalog (--build | --check | --update) [--kit K]... [--verify-remote]
//! ```
//!
//! `models.yml` is the validation schedule: every model a crate checks, with
//! its size, whether it is a representative (`core`), what it varies, and the
//! dtypes it is checked in. A profile (`profiles/<name>.yml`, or a path)
//! chooses models from it:
//!
//! ```yaml
//! extends: medium-core          # a profile, or a list of them, whose choice this adds to
//! include: [{size: medium, core: true}, {crate: qwen3, model: qwen3-1.7b}]
//! deny:    [{crate: voxtral}]   # removed from everything included, extended ones too
//! dtypes:  [f32, bf16]          # `bf16` checks run on cuda alone; default [f32]
//! checks:  [driver]             # `driver`, `model`, or both (the default)
//! backend: cuda                 # the default for --backend
//! ```
//!
//! A selector matches a model when every key it has does: `crate`, `model`,
//! `size` (one or a list) and `core`. The profile is `--profile`, else
//! `BUNSEN_VALIDATE_PROFILE`, else `small-core`.
//!
//! A *driver* check is a test with a `driver` segment in its path (a stream
//! driver held to the same model's one-shot decode); every other test of a
//! model is a *model* check. `checks: [driver]` runs the chosen models'
//! driver checks alone, and builds no ONNX reference, which no driver check
//! needs.
//!
//! `run` builds each crate with a chosen model (`cargo test --no-run`,
//! under the `validate` cargo profile: release's optimization, with debug
//! assertions) with the crate's `features`, its `reference` feature when a
//! chosen model's model checks need it, and the backend's; lists each test
//! target's tests; and runs the chosen ones, one at a time, through
//! `cargo test -- --exact`. A test of a model has the model's id in its path
//! (`qwen3-0.6b` is the segment `qwen3_0_6b`); the run drops the tests of
//! every model not chosen and, without `bf16`, the `cuda_bf16` targets.
//! Tests of no model (the fixture checks) run with their crate. `--nextest`
//! runs the same choice through `cargo nextest` instead, under its
//! `validate` profile (`.config/nextest.toml`), a process per test.
//!
//! The checkpoints are fetched by the tests into the weights cache
//! (`BUNSEN_CACHE_DIR`, or the user's cache directory);
//! `BUNSEN_VALIDATE_OFFLINE=1` forbids it.
//!
//! `check` builds and lists the crates' tests (without their ONNX references
//! unless `--reference`) and checks the validation schedule against them: every
//! model has a test.
//!
//! `catalog` builds each kit's pretrained catalog (`validation/catalogs/
//! {kit}.yml`, the hand-written source) into its `catalog.json`. A file on
//! the Hub takes its digest and size from the repo's listing at the pinned
//! revision, or, a small one, from its bytes; a checkpoint takes the dtype
//! its `config.json` names. A digest the source gives is authoritative: it
//! must agree with the derived one, and a file off the Hub must have one.
//! `--build` builds into `target/catalog/` (once per source and tool
//! version), `--check` fails if the checked-in catalog differs, `--update`
//! overwrites it. Only `--verify-remote` downloads the weights, to hash them
//! against their pins.
//!
//! A build holds every row's tags to bunsen's tag vocabulary ([`tags`]).

use std::{
    path::PathBuf,
    process::{
        Command,
        ExitCode,
    },
};

use clap::{
    Parser,
    Subcommand,
};
use indexmap::IndexMap;

use crate::{
    profile::{
        Backend,
        Check,
        Profile,
    },
    run::{
        Choice,
        DRIVER,
    },
    schedule::{
        Dtype,
        ValidationSchedule,
        segment,
    },
};

mod catalog;
mod profile;
mod run;
mod schedule;
pub mod tags;

/// The validation workspace: `validation/`.
pub fn validation_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs the validation crates under a profile: the models it chooses, their
/// tests alone.
#[derive(Parser)]
#[command(name = "validate")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// What a profile chooses, crate by crate, and what it fetches.
    List {
        #[command(flatten)]
        select: Select,
    },

    /// Builds the chosen models' crates and runs their tests.
    Run {
        #[command(flatten)]
        select: Select,

        /// The backend; else the profile's, else `wgpu`.
        #[arg(long, value_enum)]
        backend: Option<Backend>,

        /// Over the profile's: `f32`, `bf16` (on cuda alone).
        #[arg(long, value_enum, value_delimiter = ',')]
        dtypes: Option<Vec<Dtype>>,

        /// Build and list, print what would run, and run nothing.
        #[arg(long)]
        dry_run: bool,

        /// Run through `cargo nextest`, a process per test.
        #[arg(long)]
        nextest: bool,

        /// Passed on to the test binaries (to nextest, with `--nextest`).
        #[arg(last = true)]
        test_args: Vec<String>,
    },

    /// Checks the validation schedule against the crates' tests: every model
    /// has one.
    Check {
        /// Only this crate (repeatable).
        #[arg(long = "crate")]
        crates: Vec<String>,

        /// Build the ONNX references too (Voxtral's export is 19.4 GB).
        #[arg(long)]
        reference: bool,

        /// Build with `cuda` too, which the bf16 test binaries need: without
        /// it, a model checked in bf16 alone is reported, not failed.
        #[arg(long)]
        cuda: bool,
    },

    /// Builds each kit's pretrained catalog from its source in
    /// `validation/catalogs/`.
    #[command(group(clap::ArgGroup::new("mode").required(true).args(["build", "check", "update"])))]
    Catalog {
        /// Build into `target/catalog/`, unless this source's build is there.
        #[arg(long)]
        build: bool,

        /// Build, then fail if a checked-in catalog differs.
        #[arg(long)]
        check: bool,

        /// Build, then overwrite the checked-in catalogs.
        #[arg(long)]
        update: bool,

        /// Only this kit's source (repeatable).
        #[arg(long)]
        kit: Vec<String>,

        /// Download every file and check it hashes to its pin.
        #[arg(long)]
        verify_remote: bool,
    },
}

#[derive(clap::Args)]
struct Select {
    /// A profile's name in `validation/profiles/`, or a path to one.
    #[arg(long, env = "BUNSEN_VALIDATE_PROFILE", default_value = "small-core")]
    profile: String,

    /// Only this crate (repeatable).
    #[arg(long = "crate")]
    crates: Vec<String>,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("validate: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode, String> {
    let validation = validation_dir();
    let schedule = ValidationSchedule::load(&validation.join("models.yml"))?;
    match cli.command {
        Cmd::List { select } => {
            let profile = profile::load(&select.profile, &validation.join("profiles"))?;
            check_crates(&schedule, &select.crates)?;
            list(&schedule, &profile, &select);
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Run {
            select,
            backend,
            dtypes,
            dry_run,
            nextest,
            test_args,
        } => {
            let profile = profile::load(&select.profile, &validation.join("profiles"))?;
            check_crates(&schedule, &select.crates)?;
            let backend = backend.or(profile.backend).unwrap_or(Backend::Wgpu);
            let mut dtypes = dtypes.unwrap_or_else(|| profile.dtypes());
            if dtypes.contains(&Dtype::Bf16) && backend != Backend::Cuda {
                eprintln!("bf16 checks run on cuda alone; dropping them on {backend:?}");
                dtypes.retain(|d| *d != Dtype::Bf16);
            }
            let plan = Plan::new(&schedule, &profile, &select.crates, backend, &dtypes);
            if plan.crates.is_empty() {
                return Err("the profile chooses no model".into());
            }
            if nextest {
                plan.nextest(&validation, dry_run, &test_args)
            } else {
                plan.cargo_test(&validation, dry_run, &test_args)
            }
        }
        Cmd::Check {
            crates,
            reference,
            cuda,
        } => {
            check_crates(&schedule, &crates)?;
            check(&schedule, &validation, &crates, reference, cuda)
        }
        Cmd::Catalog {
            build,
            check,
            update: _,
            kit,
            verify_remote,
        } => {
            let mode = if build {
                catalog::Mode::Build
            } else if check {
                catalog::Mode::Check
            } else {
                catalog::Mode::Update
            };
            let remote = catalog::Http::new();
            let mut current = true;
            for path in catalog::sources(&validation, &kit)? {
                current &= catalog::run(&remote, &validation, &path, mode, verify_remote)?;
            }
            Ok(if current {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
    }
}

/// Refuses a `--crate` the validation schedule does not have.
fn check_crates(
    schedule: &ValidationSchedule,
    crates: &[String],
) -> Result<(), String> {
    match crates.iter().find(|c| !schedule.crates.contains_key(*c)) {
        Some(c) => Err(format!(
            "no crate {c:?} in models.yml: one of {:?}",
            schedule.crates.keys().collect::<Vec<_>>()
        )),
        None => Ok(()),
    }
}

fn check_names(checks: &[Check]) -> Vec<&'static str> {
    let mut names: Vec<&str> = checks
        .iter()
        .map(|c| match c {
            Check::Driver => "driver",
            Check::Model => "model",
        })
        .collect();
    names.sort();
    names.dedup();
    names
}

fn list(
    schedule: &ValidationSchedule,
    profile: &Profile,
    select: &Select,
) {
    let chosen = profile::choose(schedule, profile, &select.crates, &profile.dtypes());
    println!(
        "profile {}; dtypes {:?}; checks {:?}",
        select.profile,
        profile
            .dtypes()
            .iter()
            .map(|d| d.name())
            .collect::<Vec<_>>(),
        check_names(&profile.checks())
    );
    let mut total = 0.0;
    for (krate, models) in &chosen {
        println!("\n{krate}");
        for (model, keep) in models {
            let m = &schedule.crates[krate].models[model];
            if *keep {
                total += m.gb;
            }
            println!(
                "  {} {model:42} {:6} {} {:9} {:5.1} GB  {}",
                if *keep { "+" } else { "-" },
                m.size.name(),
                if m.core { "core" } else { "    " },
                m.dtypes
                    .iter()
                    .map(|d| d.name())
                    .collect::<Vec<_>>()
                    .join(","),
                m.gb,
                m.varies
            );
        }
    }
    println!("\nchosen: {total:.1} GB on a cold cache");
}

/// One crate of a run.
struct PlanCrate {
    package: String,
    features: Vec<String>,
    models: IndexMap<String, bool>,
}

/// A run: the crates with a chosen model, each with its features.
struct Plan {
    crates: Vec<PlanCrate>,
    checks: Vec<Check>,
    bf16: bool,
}

impl Plan {
    fn new(
        schedule: &ValidationSchedule,
        profile: &Profile,
        crates: &[String],
        backend: Backend,
        dtypes: &[Dtype],
    ) -> Plan {
        let checks = profile.checks();
        let crates = profile::choose(schedule, profile, crates, dtypes)
            .into_iter()
            // A crate with no chosen model is left out, its fixture checks too
            // (`cargo make check-validation` runs those).
            .filter(|(_, models)| models.values().any(|keep| *keep))
            .map(|(krate, models)| {
                let spec = &schedule.crates[&krate];
                let mut features = spec.features.clone();
                // Only a model check needs a reference.
                let needs_reference = checks.contains(&Check::Model)
                    && models
                        .iter()
                        .any(|(m, keep)| *keep && spec.models[m].reference);
                if let (true, Some(reference)) = (needs_reference, &spec.reference) {
                    features.push(reference.clone());
                }
                // `flex` is the CPU, which every build has: no backend feature.
                features.extend(backend.feature().map(str::to_string));
                PlanCrate {
                    package: spec.package.clone(),
                    features,
                    models,
                }
            })
            .collect();
        Plan {
            crates,
            checks,
            bf16: dtypes.contains(&Dtype::Bf16),
        }
    }

    fn cargo_test(
        &self,
        validation: &std::path::Path,
        dry_run: bool,
        test_args: &[String],
    ) -> Result<ExitCode, String> {
        let mut failed = Vec::new();
        let mut ran = 0;
        for c in &self.crates {
            let choice = Choice {
                models: &c.models,
                checks: &self.checks,
                bf16: self.bf16,
            };
            for target in run::build_and_list(validation, &c.package, &c.features)? {
                let tests: Vec<&str> = target
                    .tests
                    .iter()
                    .filter(|t| choice.keeps(&target.name, t))
                    .map(String::as_str)
                    .collect();
                if tests.is_empty() {
                    continue;
                }
                ran += tests.len();
                if dry_run {
                    println!("{} {}:", c.package, target.select.join(" "));
                    for t in &tests {
                        println!("  {t}");
                    }
                    continue;
                }
                if !run::run_tests(
                    validation,
                    &c.package,
                    &c.features,
                    &target,
                    &tests,
                    test_args,
                )? {
                    failed.push(format!("{} {}", c.package, target.select.join(" ")));
                }
            }
        }
        if dry_run {
            println!("\n{ran} tests chosen");
            return Ok(ExitCode::SUCCESS);
        }
        eprintln!("\n{ran} tests run; {} targets failed", failed.len());
        for f in &failed {
            eprintln!("  FAILED: {f}");
        }
        Ok(if failed.is_empty() {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        })
    }

    /// The same choice as a nextest filterset.
    fn nextest_filter(&self) -> String {
        let segments = |models: Vec<&String>| {
            let names: Vec<String> = models.iter().map(|m| segment(m)).collect();
            format!("(^|::)({})(::|$)", names.join("|"))
        };
        let driver = format!("(^|::){DRIVER}(::|$)");
        self.crates
            .iter()
            .map(|c| {
                let mut expr = format!("package({})", c.package);
                let dropped: Vec<&String> = c
                    .models
                    .iter()
                    .filter(|(_, k)| !**k)
                    .map(|(m, _)| m)
                    .collect();
                if !dropped.is_empty() {
                    expr += &format!(" - test(/{}/)", segments(dropped));
                }
                if !self.checks.contains(&Check::Model) {
                    expr += &format!(
                        " - (test(/{}/) - test(/{driver}/))",
                        segments(c.models.keys().collect())
                    );
                }
                if !self.checks.contains(&Check::Driver) {
                    expr += &format!(" - test(/{driver}/)");
                }
                if !self.bf16 {
                    expr += " - binary(cuda_bf16)";
                }
                format!("({expr})")
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    fn nextest(
        &self,
        validation: &std::path::Path,
        dry_run: bool,
        test_args: &[String],
    ) -> Result<ExitCode, String> {
        let features: Vec<String> = self
            .crates
            .iter()
            .flat_map(|c| c.features.iter().map(move |f| format!("{}/{f}", c.package)))
            .collect();
        let mut command = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
        command.current_dir(validation).args([
            "nextest",
            "run",
            "--cargo-profile",
            run::PROFILE,
            "--profile",
            "validate",
            "--no-tests",
            "warn",
        ]);
        for c in &self.crates {
            command.args(["-p", &c.package]);
        }
        command
            .args(["--features", &features.join(",")])
            .args(["-E", &self.nextest_filter()])
            .args(test_args);
        eprintln!("+ {}", run::show(&command));
        if dry_run {
            return Ok(ExitCode::SUCCESS);
        }
        let status = command
            .status()
            .map_err(|e| format!("cargo nextest: {e}"))?;
        Ok(if status.success() {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        })
    }
}

/// Every scheduled model has a test; prints each crate's tests of no model.
fn check(
    schedule: &ValidationSchedule,
    validation: &std::path::Path,
    crates: &[String],
    reference: bool,
    cuda: bool,
) -> Result<ExitCode, String> {
    let mut failed = false;
    for (krate, spec) in &schedule.crates {
        if !crates.is_empty() && !crates.contains(krate) {
            continue;
        }
        let mut features = spec.features.clone();
        // A reference costs its export: only on asking.
        if let (true, Some(r)) = (reference, &spec.reference) {
            features.push(r.clone());
        }
        if cuda {
            features.extend(Backend::Cuda.feature().map(str::to_string));
        }
        let tests: Vec<String> = run::build_and_list(validation, &spec.package, &features)?
            .into_iter()
            .flat_map(|t| t.tests)
            .collect();
        let segments: Vec<String> = spec.models.keys().map(|m| segment(m)).collect();
        for ((model, m), seg) in spec.models.iter().zip(&segments) {
            if tests.iter().any(|t| t.split("::").any(|s| s == seg)) {
                continue;
            }
            if !cuda && !m.dtypes.contains(&Dtype::F32) {
                println!("{krate}: {model} is checked in bf16 alone, on cuda: not built (--cuda)");
                continue;
            }
            println!("{krate}: {model} has no test (no `{seg}` in a test path)");
            failed = true;
        }
        let free: Vec<&String> = tests
            .iter()
            .filter(|t| !t.split("::").any(|s| segments.iter().any(|seg| seg == s)))
            .collect();
        println!(
            "{krate}: {} tests; of no model, always run: {free:?}",
            tests.len()
        );
    }
    Ok(if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ci`'s plan: the four crates, no reference, no backend feature, and a
    /// filterset that keeps driver checks alone.
    #[test]
    fn test_ci_plan() {
        let validation = validation_dir();
        let schedule = ValidationSchedule::load(&validation.join("models.yml")).unwrap();
        let profile = profile::load("ci", &validation.join("profiles")).unwrap();
        let plan = Plan::new(&schedule, &profile, &[], Backend::Flex, &[Dtype::F32]);
        let packages: Vec<&str> = plan.crates.iter().map(|c| c.package.as_str()).collect();
        assert_eq!(
            packages,
            [
                "silero-model-validation",
                "whisper-model-validation",
                "parakeet-model-validation",
                "qwen3-asr-model-validation"
            ]
        );
        for c in &plan.crates {
            assert_eq!(c.features, ["checkpoints", "gpu-tests"], "{}", c.package);
        }
        let filter = plan.nextest_filter();
        assert!(filter.contains("package(whisper-model-validation) - test(/(^|::)(base)(::|$)/)"));
        assert!(filter.contains("- binary(cuda_bf16)"));
    }
}
