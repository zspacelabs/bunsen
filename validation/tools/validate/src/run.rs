//! Builds the crates' tests, lists them, and runs the chosen ones.

use std::{
    path::{
        Path,
        PathBuf,
    },
    process::{
        Command,
        Stdio,
    },
};

use indexmap::IndexMap;

use crate::{
    catalog::segment,
    profile::Check,
};

/// One test target of a crate, built: its tests, by name.
#[derive(Debug)]
pub struct Target {
    /// The target's name (`cuda_bf16`, or the crate's for its lib).
    pub name: String,

    /// How `cargo test` names it: `--lib`, or `--test {name}`.
    pub select: Vec<String>,

    /// Its tests.
    pub tests: Vec<String>,
}

/// The cargo profile the crates are built with (`validation/Cargo.toml`):
/// release's optimization, with debug assertions.
pub const PROFILE: &str = "validate";

/// `cargo test --profile validate -p {package} --features {features}`, the
/// command every build and run of a crate shares, in the validation workspace.
pub fn cargo_test(
    validation: &Path,
    package: &str,
    features: &[String],
) -> Command {
    let mut command = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    command
        .current_dir(validation)
        .args(["test", "--profile", PROFILE, "-p", package]);
    if !features.is_empty() {
        command.args(["--features", &features.join(",")]);
    }
    command
}

/// Builds `package`'s test targets with `features` and lists each one's
/// tests.
pub fn build_and_list(
    validation: &Path,
    package: &str,
    features: &[String],
) -> Result<Vec<Target>, String> {
    let mut command = cargo_test(validation, package, features);
    command
        .args(["--no-run", "--message-format", "json-render-diagnostics"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    eprintln!("+ {}", show(&command));
    let out = command
        .output()
        .map_err(|e| format!("cargo test --no-run: {e}"))?;
    if !out.status.success() {
        return Err(format!("{package}: the build failed"));
    }
    let mut targets = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Ok(message) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if message["reason"] != "compiler-artifact" || message["profile"]["test"] != true {
            continue;
        }
        let Some(executable) = message["executable"].as_str() else {
            continue;
        };
        let name = message["target"]["name"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let kinds = &message["target"]["kind"];
        let select = if kinds
            .as_array()
            .is_some_and(|k| k.iter().any(|k| k == "lib"))
        {
            vec!["--lib".to_string()]
        } else if kinds
            .as_array()
            .is_some_and(|k| k.iter().any(|k| k == "test"))
        {
            vec!["--test".to_string(), name.clone()]
        } else {
            continue;
        };
        let tests = list_tests(Path::new(executable))?;
        targets.push(Target {
            name,
            select,
            tests,
        });
    }
    Ok(targets)
}

/// The tests a test binary has: `--list --format terse`.
fn list_tests(executable: &Path) -> Result<Vec<String>, String> {
    let out = Command::new(executable)
        .args(["--list", "--format", "terse"])
        .output()
        .map_err(|e| format!("{}: {e}", executable.display()))?;
    if !out.status.success() {
        return Err(format!("{}: --list failed", executable.display()));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| line.strip_suffix(": test"))
        .map(str::to_string)
        .collect())
}

/// What a run keeps of a crate's tests.
pub struct Choice<'a> {
    /// The crate's models: each chosen or not.
    pub models: &'a IndexMap<String, bool>,

    /// The kinds of check it runs.
    pub checks: &'a [Check],

    /// Whether it runs the `cuda_bf16` targets.
    pub bf16: bool,
}

/// The segment that makes a test a driver check.
pub const DRIVER: &str = "driver";

impl Choice<'_> {
    /// Whether the run keeps `test`, of target `target`: not a test of a model
    /// it does not choose, of a kind of check it does not run, or a bf16 check
    /// without bf16. A test of no model (a fixture check) is kept unless it is
    /// a driver check and drivers are not run.
    pub fn keeps(
        &self,
        target: &str,
        test: &str,
    ) -> bool {
        let segments: Vec<&str> = test.split("::").collect();
        let has = |model: &str| segments.contains(&segment(model).as_str());
        let of_model = self.models.keys().any(|m| has(m));
        let dropped = self.models.iter().any(|(m, keep)| !keep && has(m));
        let driver = segments.contains(&DRIVER);
        !dropped
            && (self.checks.contains(&Check::Model) || !of_model || driver)
            && (self.checks.contains(&Check::Driver) || !driver)
            && (self.bf16 || target != "cuda_bf16")
    }
}

/// Runs `tests` of `target`, one at a time, through `cargo test`.
pub fn run_tests(
    validation: &Path,
    package: &str,
    features: &[String],
    target: &Target,
    tests: &[&str],
    extra: &[String],
) -> Result<bool, String> {
    let mut command = cargo_test(validation, package, features);
    command
        .args(&target.select)
        .args(["--", "--exact", "--test-threads", "1"])
        .args(extra)
        .args(tests);
    eprintln!("+ {}", show(&command));
    let status = command.status().map_err(|e| format!("cargo test: {e}"))?;
    Ok(status.success())
}

/// A command, as a shell would read it.
pub fn show(command: &Command) -> String {
    let dir = command
        .get_current_dir()
        .map(PathBuf::from)
        .map(|d| format!("(cd {} && ", d.display()))
        .unwrap_or_default();
    let words: Vec<String> = std::iter::once(command.get_program())
        .chain(command.get_args())
        .map(|w| quote(&w.to_string_lossy()))
        .collect();
    let close = if dir.is_empty() { "" } else { ")" };
    format!("{dir}{}{close}", words.join(" "))
}

fn quote(word: &str) -> String {
    let plain = word
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "-_./,:=+@".contains(c));
    if plain && !word.is_empty() {
        word.to_string()
    } else {
        format!("'{}'", word.replace('\'', "'\\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models() -> IndexMap<String, bool> {
        [("qwen3-0.6b", true), ("qwen3-4b", false)]
            .into_iter()
            .map(|(m, k)| (m.to_string(), k))
            .collect()
    }

    /// Each rule of `keeps`, one test name at a time.
    #[test]
    fn test_keeps() {
        let models = models();
        let both = Choice {
            models: &models,
            checks: &[Check::Model, Check::Driver],
            bf16: false,
        };
        assert!(both.keeps("lib", "greedy::qwen3_0_6b"));
        assert!(!both.keeps("lib", "greedy::qwen3_4b"));
        assert!(!both.keeps("lib", "cross::qwen3_4b::tracks_the_reference"));
        assert!(both.keeps("lib", "tests::test_fixture_integrity"));
        assert!(both.keeps("lib", "speech::driver::qwen3_0_6b"));
        assert!(!both.keeps("cuda_bf16", "bf16::qwen3_0_6b"));
        // A segment is whole: `qwen3_0_6b_x` is not `qwen3_0_6b`'s.
        assert!(both.keeps("lib", "greedy::qwen3_4b_x"));

        let drivers = Choice {
            checks: &[Check::Driver],
            ..both
        };
        assert!(!drivers.keeps("lib", "greedy::qwen3_0_6b"));
        assert!(drivers.keeps("lib", "speech::driver::qwen3_0_6b"));
        assert!(!drivers.keeps("lib", "speech::driver::qwen3_4b"));
        assert!(drivers.keeps("lib", "tests::test_fixture_integrity"));

        let model_checks = Choice {
            checks: &[Check::Model],
            bf16: true,
            ..drivers
        };
        assert!(model_checks.keeps("lib", "greedy::qwen3_0_6b"));
        assert!(!model_checks.keeps("lib", "speech::driver::qwen3_0_6b"));
        assert!(model_checks.keeps("cuda_bf16", "bf16::qwen3_0_6b"));
    }

    #[test]
    fn test_show() {
        let mut command = Command::new("cargo");
        command
            .current_dir("/v")
            .args(["test", "--", "a::b", "it's"]);
        assert_eq!(show(&command), "(cd /v && cargo test -- a::b 'it'\\''s')");
    }
}
