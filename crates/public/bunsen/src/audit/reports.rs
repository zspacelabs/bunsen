//! Where test reports and recorded baselines live: `{root}/{backend}/{name}`.
//!
//! The library never picks the root: each call site passes a
//! [`ReportsOptions`] naming it. A repository that wants one standard location
//! writes a constructor for its own default options:
//!
//! ```
//! # #[cfg(feature = "audit")] {
//! use bunsen::{
//!     audit::reports::{
//!         BaselineMode,
//!         ReportsOptions,
//!         cargo_target_dir,
//!     },
//!     errors::BunsenResult,
//! };
//!
//! /// This repository's reports: `{target}/my-reports`, mode from the env.
//! fn my_reports() -> BunsenResult<ReportsOptions> {
//!     let root = cargo_target_dir()
//!         .unwrap_or_else(std::env::temp_dir)
//!         .join("my-reports");
//!     Ok(ReportsOptions::new(root)
//!         .with_mode(BaselineMode::from_env("MY_REPORTS_MODE")?))
//! }
//! # }
//! ```
//!
//! Reports are not meant to be tracked in git.

use std::{
    env,
    fs,
    path::{
        Component,
        Path,
        PathBuf,
    },
};

use burn::prelude::Backend;

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    ParseError,
    ResultContext,
    sys_at,
};

/// The cargo target directory: `$CARGO_TARGET_DIR` if set, else the one
/// holding the running binary.
///
/// For building a repository's default [`ReportsOptions`] root.
pub fn cargo_target_dir() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("CARGO_TARGET_DIR") {
        return Some(PathBuf::from(dir));
    }
    env::current_exe().ok().and_then(|exe| target_dir_of(&exe))
}

/// The cargo target directory of a binary at `target/{profile}/deps/{bin}` or
/// `target/{profile}/{bin}`.
fn target_dir_of(exe: &Path) -> Option<PathBuf> {
    let mut dir = exe.parent()?;
    if dir.file_name()? == "deps" {
        dir = dir.parent()?;
    }
    Some(dir.parent()?.to_path_buf())
}

/// A path-safe label for backend `B`: its name, with each run of characters
/// other than `[A-Za-z0-9.-]` replaced by one `_`, and no leading or trailing
/// `_`; `cubecl<wgpu<spirv>>` gives `cubecl_wgpu_spirv`.
pub fn backend_label<B: Backend>(device: &B::Device) -> String {
    path_label(&B::name(device))
}

fn path_label(name: &str) -> String {
    name.split(|c: char| !(c.is_ascii_alphanumeric() || ".-".contains(c)))
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

/// Where reports and baselines are stored, and how stored baselines are
/// treated.
///
/// The call site always names the root: [`audit_baseline`] and
/// [`SeriesReport::write_report`] take these options, and place each file at
/// [`report_path`](Self::report_path), `{root}/{backend}/{name}`.
/// [`mode`](Self::mode) is the [`BaselineMode`] that `audit_baseline` applies.
/// See the module docs for a repository-default constructor.
///
/// [`audit_baseline`]: crate::audit::audit_baseline
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportsOptions {
    root: PathBuf,
    mode: BaselineMode,
}

impl ReportsOptions {
    /// Reports under `root`, in [`BaselineMode::Auto`].
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            mode: BaselineMode::default(),
        }
    }

    /// These options with baseline mode `mode`.
    pub fn with_mode(
        self,
        mode: BaselineMode,
    ) -> Self {
        Self { mode, ..self }
    }

    /// The reports root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The baseline mode.
    pub fn mode(&self) -> BaselineMode {
        self.mode
    }

    /// The path of report `name` for `backend`: `{root}/{backend}/{name}`.
    ///
    /// `name` may contain `/` to group reports; it must be relative and must
    /// not contain `..`.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if `name` escapes
    /// the backend directory.
    pub fn report_path(
        &self,
        backend: &str,
        name: &str,
    ) -> BunsenResult<PathBuf> {
        let rel = Path::new(name);
        if name.is_empty() || !rel.components().all(|c| matches!(c, Component::Normal(_))) {
            return Err(BunsenError::illegal(format!(
                "report name {name:?} must be a relative path without `..`"
            )));
        }
        Ok(self.root.join(backend).join(rel))
    }
}

/// How [`audit_baseline`] treats a stored baseline.
///
/// Carried by [`ReportsOptions::mode`], or passed directly to
/// [`audit_baseline_at`]; the [`BaselineOutcome`] says which happened. Read
/// from an environment variable with [`from_env`](Self::from_env).
///
/// [`audit_baseline`]: crate::audit::audit_baseline
/// [`audit_baseline_at`]: crate::audit::audit_baseline_at
/// [`BaselineOutcome`]: crate::audit::BaselineOutcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BaselineMode {
    /// Record when no baseline exists; verify when one does.
    #[default]
    Auto,

    /// Always record, replacing any stored baseline.
    Record,

    /// Always verify; a missing baseline is an error.
    Verify,
}

impl BaselineMode {
    /// Reads the mode from environment variable `var` (`auto`, `record`,
    /// `verify`); unset is `Auto`.
    ///
    /// # Errors
    /// As [`parse`](Self::parse) on any other value, with a frame naming
    /// `var`.
    pub fn from_env(var: &str) -> BunsenResult<Self> {
        match env::var(var) {
            Err(_) => Ok(Self::Auto),
            Ok(value) => {
                Self::parse(&value).with_context(|| format!("reading environment variable {var}"))
            }
        }
    }

    /// Parses `auto`, `record`, or `verify` (case-insensitive).
    ///
    /// # Errors
    /// [`Policy`](crate::errors::BunsenErrorKind::Policy), with a
    /// [`ParseError`] cause, on any other value.
    pub fn parse(value: &str) -> BunsenResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "record" => Ok(Self::Record),
            "verify" => Ok(Self::Verify),
            _ => Err(BunsenError::from_cause(
                BunsenErrorKind::Policy,
                ParseError::new("baseline mode")
                    .input(value)
                    .because("expected auto, record, or verify"),
            )),
        }
    }
}

/// Named `f64` columns over a step index, written as CSV (`step,<names>`).
///
/// For logs a reviewer reads or plots: energy drift, reversal error, fit
/// inputs. Values are written in Rust's shortest round-trip form. This is not
/// an audit stream, and nothing verifies it. Write it with
/// [`write`](Self::write), or under a [`ReportsOptions`] root with
/// [`write_report`](Self::write_report), beside the backend's baselines.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SeriesReport {
    columns: Vec<(String, Vec<f64>)>,
}

impl SeriesReport {
    /// An empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// This report with column `name`.
    ///
    /// # Panics
    /// If `values` has a different length from the columns already added,
    /// or `name` is empty or contains `,`.
    #[track_caller]
    pub fn column(
        mut self,
        name: &str,
        values: Vec<f64>,
    ) -> Self {
        assert!(
            !name.is_empty() && !name.contains(','),
            "column name {name:?} must be non-empty, without ','"
        );
        if let Some((first, v)) = self.columns.first() {
            assert_eq!(
                values.len(),
                v.len(),
                "column {name:?} has {} values; {first:?} has {}",
                values.len(),
                v.len()
            );
        }
        self.columns.push((name.to_string(), values));
        self
    }

    /// The column `name`, if present.
    pub fn get(
        &self,
        name: &str,
    ) -> Option<&[f64]> {
        self.columns
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }

    /// The report as CSV text.
    pub fn to_csv(&self) -> String {
        let mut out = String::from("step");
        for (name, _) in &self.columns {
            out.push(',');
            out.push_str(name);
        }
        out.push('\n');
        let rows = self.columns.first().map_or(0, |(_, v)| v.len());
        for i in 0..rows {
            out.push_str(&i.to_string());
            for (_, v) in &self.columns {
                out.push(',');
                out.push_str(&v[i].to_string());
            }
            out.push('\n');
        }
        out
    }

    /// Parses CSV text written by [`to_csv`](Self::to_csv).
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
    /// with a [`ParseError`] cause, on a malformed header, row, or value.
    pub fn from_csv(text: &str) -> BunsenResult<Self> {
        let invalid = |e: ParseError| BunsenError::from_cause(BunsenErrorKind::InvalidResource, e);
        let mut lines = text.lines();
        let header = lines
            .next()
            .ok_or_else(|| invalid(ParseError::new("series CSV").because("empty text")))?;
        let mut names = header.split(',');
        if names.next() != Some("step") {
            return Err(invalid(
                ParseError::new("series CSV header")
                    .input(header)
                    .because("must start with \"step\""),
            ));
        }
        let mut columns: Vec<(String, Vec<f64>)> =
            names.map(|n| (n.to_string(), Vec::new())).collect();
        for (row, line) in lines.enumerate() {
            let cells: Vec<&str> = line.split(',').collect();
            if cells.len() != columns.len() + 1 || cells[0] != row.to_string() {
                return Err(invalid(
                    ParseError::new("series CSV row")
                        .at(format!("row {row}"))
                        .input(line)
                        .because(format!(
                            "expected {} cells, starting with step {row}",
                            columns.len() + 1
                        )),
                ));
            }
            for ((_, col), cell) in columns.iter_mut().zip(&cells[1..]) {
                col.push(cell.parse().map_err(|e| {
                    invalid(
                        ParseError::new("series CSV value")
                            .at(format!("row {row}"))
                            .input(cell)
                            .with_source(e),
                    )
                })?);
            }
        }
        Ok(Self { columns })
    }

    /// Writes the report to `path`, creating parent directories.
    ///
    /// # Errors
    /// An I/O failure, sorted by [`sys_at`]: usually
    /// [`Sys`](crate::errors::BunsenErrorKind::Sys), or
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a forbidden path.
    pub fn write(
        &self,
        path: &Path,
    ) -> BunsenResult<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(sys_at("create directory", parent))?;
        }
        fs::write(path, self.to_csv()).map_err(sys_at("write", path))
    }

    /// Writes the report for backend `B` as `{root}/{backend}/{name}.csv`.
    ///
    /// # Returns
    /// The path written.
    ///
    /// # Errors
    /// As [`ReportsOptions::report_path`] and [`write`](Self::write).
    pub fn write_report<B: Backend>(
        &self,
        options: &ReportsOptions,
        device: &B::Device,
        name: &str,
    ) -> BunsenResult<PathBuf> {
        let path = options.report_path(&backend_label::<B>(device), &format!("{name}.csv"))?;
        self.write(&path)?;
        Ok(path)
    }

    /// Reads a report written by [`write`](Self::write).
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) if `path` does not
    /// exist or cannot be read; another I/O failure, sorted by [`sys_at`]; as
    /// [`from_csv`](Self::from_csv), with a frame naming `path`.
    pub fn read(path: &Path) -> BunsenResult<Self> {
        let text = fs::read_to_string(path).map_err(sys_at("read", path))?;
        Self::from_csv(&text).with_context(|| format!("reading {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::testing::{
            ErrorMatcher,
            predicate,
        },
        support::testing::{
            CpuBackend,
            backend_device,
        },
    };

    #[test]
    fn test_target_dir_of() {
        let exe = Path::new("/w/target/debug/deps/foo-1234");
        assert_eq!(target_dir_of(exe), Some(PathBuf::from("/w/target")));
        let exe = Path::new("/w/target/release/foo");
        assert_eq!(target_dir_of(exe), Some(PathBuf::from("/w/target")));
    }

    #[test]
    fn test_report_path_rejects_escapes() {
        let options = ReportsOptions::new("r");
        assert_eq!(
            options.report_path("flex", "a/b.cbor").unwrap(),
            Path::new("r").join("flex").join("a/b.cbor")
        );
        for name in ["", "../x", "/abs"] {
            ErrorMatcher::kind(BunsenErrorKind::Illegal)
                .message_contains("must be a relative path without `..`")
                .assert_err(&options.report_path("flex", name));
        }
    }

    #[test]
    fn test_backend_label() {
        assert_eq!(path_label("cubecl<wgpu<spirv>>"), "cubecl_wgpu_spirv");
        assert_eq!(path_label("fusion<cubecl<cuda>>"), "fusion_cubecl_cuda");
        assert_eq!(path_label("ndarray"), "ndarray");

        let label = backend_label::<CpuBackend>(&backend_device::<CpuBackend>());
        assert!(!label.is_empty());
        assert!(
            label
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c)),
            "{label}"
        );
    }

    #[test]
    fn test_baseline_mode_parse() {
        assert_eq!(BaselineMode::parse("").unwrap(), BaselineMode::Auto);
        assert_eq!(BaselineMode::parse("Record").unwrap(), BaselineMode::Record);
        assert_eq!(BaselineMode::parse("verify").unwrap(), BaselineMode::Verify);
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_eq("cannot parse baseline mode \"replay\": expected auto, record, or verify")
            .has_cause::<ParseError>()
            .assert_err(&BaselineMode::parse("replay"));
    }

    #[test]
    fn test_reports_options() {
        let options = ReportsOptions::new("r");
        assert_eq!(options.root(), Path::new("r"));
        assert_eq!(options.mode(), BaselineMode::Auto);

        let options = options.with_mode(BaselineMode::Verify);
        assert_eq!(options.root(), Path::new("r"));
        assert_eq!(options.mode(), BaselineMode::Verify);
    }

    #[test]
    fn test_series_report_round_trip() {
        let report = SeriesReport::new()
            .column("energy", vec![1.0, 1.0 + 1e-15, 0.1 + 0.2])
            .column("error", vec![0.0, -2.5e-300, f64::MAX]);
        let csv = report.to_csv();
        assert!(csv.starts_with("step,energy,error\n0,1,0\n"), "{csv}");
        assert_eq!(SeriesReport::from_csv(&csv).unwrap(), report);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("series").join("report.csv");
        report.write(&path).unwrap();
        assert_eq!(SeriesReport::read(&path).unwrap(), report);

        let options = ReportsOptions::new(dir.path());
        let device = backend_device::<CpuBackend>();
        let path = report
            .write_report::<CpuBackend>(&options, &device, "g/s")
            .unwrap();
        assert_eq!(
            path,
            options
                .report_path(&backend_label::<CpuBackend>(&device), "g/s.csv")
                .unwrap()
        );
        assert_eq!(SeriesReport::read(&path).unwrap(), report);
        assert_eq!(report.get("error").unwrap()[1], -2.5e-300);

        let bad_csv = |at: Option<&str>| {
            let m = ErrorMatcher::kind(BunsenErrorKind::InvalidResource);
            let at = at.map(str::to_string);
            m.cause(predicate(
                "a series CSV parse error",
                move |c: &ParseError| c.what.starts_with("series CSV") && c.at == at,
            ))
        };
        bad_csv(None).assert_err(&SeriesReport::from_csv(""));
        bad_csv(None).assert_err(&SeriesReport::from_csv("x,y\n"));
        bad_csv(Some("row 0")).assert_err(&SeriesReport::from_csv("step,y\n0,abc\n"));
        bad_csv(Some("row 0")).assert_err(&SeriesReport::from_csv("step,y\n1,2\n"));

        fs::write(&path, "step,y\n0,abc\n").unwrap();
        bad_csv(Some("row 0"))
            .frame_contains("reading ")
            .assert_err(&SeriesReport::read(&path));
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .assert_err(&SeriesReport::read(&dir.path().join("absent.csv")));
    }

    #[test]
    #[should_panic = "column \"b\" has 1 values; \"a\" has 2"]
    fn test_series_report_rejects_ragged_columns() {
        let _ = SeriesReport::new()
            .column("a", vec![1.0, 2.0])
            .column("b", vec![1.0]);
    }
}
