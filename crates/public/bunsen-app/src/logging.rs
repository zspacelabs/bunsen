use std::sync::{
    LazyLock,
    Mutex,
    MutexGuard,
    PoisonError,
};

use bunsen::errors::{
    BunsenError,
    BunsenResult,
};
use env_logger::{
    Builder,
    DEFAULT_FILTER_ENV,
    DEFAULT_WRITE_STYLE_ENV,
    Logger,
    TimestampPrecision,
};
pub use log::LevelFilter;
use log::{
    Log,
    Metadata,
};

/// The log target prefix for bunsen's crates.
///
/// Targets match by string prefix, so this covers `bunsen`, `bunsen_app`,
/// `bunsen_tool` and every other `bunsen*` crate.
pub const BUNSEN_TARGET: &str = "bunsen";

/// The process-wide target prefixes logged at a [`LogPolicy`]'s `ours` level.
static TARGET_PREFIXES: LazyLock<Mutex<Vec<String>>> = LazyLock::new(|| {
    Mutex::new(vec![
        BUNSEN_TARGET.to_string(),
        env!("CARGO_CRATE_NAME").to_string(),
    ])
});

fn target_prefixes_lock() -> MutexGuard<'static, Vec<String>> {
    TARGET_PREFIXES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Adds `prefix` to the target prefixes logged at a [`LogPolicy`]'s `ours`
/// level; adding a prefix twice keeps one copy.
///
/// An app adds its own crate from `main()`, before any [`LogArgs::init`]:
///
/// ```rust
/// bunsen_app::logging::add_target_prefix(env!("CARGO_CRATE_NAME"));
/// ```
pub fn add_target_prefix(prefix: impl Into<String>) {
    let prefix = prefix.into();
    let mut prefixes = target_prefixes_lock();
    if !prefixes.contains(&prefix) {
        prefixes.push(prefix);
    }
}

/// The target prefixes logged at a [`LogPolicy`]'s `ours` level:
/// [`BUNSEN_TARGET`], `bunsen_app`, then each [`add_target_prefix`] in order.
pub fn target_prefixes() -> Vec<String> {
    target_prefixes_lock().clone()
}

/// A command's default log levels: one for the [`target_prefixes`], one for
/// everything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogPolicy {
    /// The level for the [`target_prefixes`].
    pub ours: LevelFilter,

    /// The level for every other target.
    pub rest: LevelFilter,
}

impl Default for LogPolicy {
    /// `Warn` for every target.
    fn default() -> Self {
        Self::new(LevelFilter::Warn, LevelFilter::Warn)
    }
}

impl LogPolicy {
    /// A policy logging the [`target_prefixes`] at `ours` and everything else
    /// at `rest`.
    pub fn new(
        ours: LevelFilter,
        rest: LevelFilter,
    ) -> Self {
        Self { ours, rest }
    }
}

/// Logging setup arg group.
///
/// The filter comes from `RUST_LOG` when it is set, and from the command's
/// [`LogPolicy`] otherwise. `-v` raises the levels of the
/// [`target_prefixes`] one step per `v` above what that filter gives them,
/// and never lowers them; a more specific `RUST_LOG` directive, such as
/// `bunsen_tool=error`, keeps its own level. `-q` turns logging off
/// regardless.
///
/// # Example
///
/// ```rust,ignore
/// pub struct Args {
///    # ...
///
///    #[clap(flatten)]
///    pub logging: LogArgs,
/// }
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///    bunsen_app::logging::add_target_prefix(env!("CARGO_CRATE_NAME"));
///    let args = Args::parse();
///    args.logging.init(LogPolicy::new(LevelFilter::Info, LevelFilter::Warn))?;
///
///    ...
/// }
/// ```
#[derive(clap::Args, Debug)]
pub struct LogArgs {
    /// Silence log messages.
    #[clap(short, long)]
    pub quiet: bool,

    /// Raise the bunsen and app log levels (-v, -vv, -vvv).
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Enable timestamped logging.
    #[clap(short, long)]
    pub ts: bool,
}

impl LogArgs {
    /// Initialize logging.
    ///
    /// # Args
    ///
    /// * `policy` - the default levels, used when `RUST_LOG` is not set.
    ///
    /// # Errors
    /// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal), with the
    /// `log::SetLoggerError` as the cause, if a logger is already installed.
    pub fn init(
        &self,
        policy: LogPolicy,
    ) -> BunsenResult<()> {
        let rust_log = std::env::var(DEFAULT_FILTER_ENV).ok();
        let mut builder = self.builder(policy, &target_prefixes(), rust_log.as_deref());
        if let Ok(style) = std::env::var(DEFAULT_WRITE_STYLE_ENV) {
            builder.parse_write_style(&style);
        }
        builder
            .try_init()
            .map_err(|e| BunsenError::illegal("a logger is already installed").with_cause(e))
    }

    /// The logger builder for `policy` over `prefixes`, with `rust_log` as
    /// the `RUST_LOG` value.
    fn builder(
        &self,
        policy: LogPolicy,
        prefixes: &[String],
        rust_log: Option<&str>,
    ) -> Builder {
        let mut builder = self.filters(policy, prefixes, rust_log);
        if !self.quiet && self.verbose > 0 {
            let base = self.filters(policy, prefixes, rust_log).build();
            for prefix in prefixes {
                builder.filter_module(prefix, raise(level_for(&base, prefix), self.verbose));
            }
        }
        builder.format_timestamp(self.ts.then_some(TimestampPrecision::Seconds));
        builder
    }

    /// The filters before `-v`: off under `-q`, else `rust_log`, else
    /// `policy` over `prefixes`.
    fn filters(
        &self,
        policy: LogPolicy,
        prefixes: &[String],
        rust_log: Option<&str>,
    ) -> Builder {
        let mut builder = Builder::new();
        if self.quiet {
            builder.filter_level(LevelFilter::Off);
        } else if let Some(spec) = rust_log {
            builder.parse_filters(spec);
        } else {
            builder.filter_level(policy.rest);
            for prefix in prefixes {
                builder.filter_module(prefix, policy.ours);
            }
        }
        builder
    }
}

/// The most verbose level `logger` enables for `target`.
fn level_for(
    logger: &Logger,
    target: &str,
) -> LevelFilter {
    LevelFilter::iter()
        .take_while(|filter| {
            filter.to_level().is_none_or(|level| {
                logger.enabled(&Metadata::builder().target(target).level(level).build())
            })
        })
        .last()
        .unwrap_or(LevelFilter::Off)
}

/// `level` raised `steps` levels, capped at `Trace`.
fn raise(
    level: LevelFilter,
    steps: u8,
) -> LevelFilter {
    LevelFilter::iter()
        .nth(level as usize + steps as usize)
        .unwrap_or(LevelFilter::Trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(verbose: u8) -> LogArgs {
        LogArgs {
            quiet: false,
            verbose,
            ts: false,
        }
    }

    /// [`BUNSEN_TARGET`] and `conway`.
    fn prefixes() -> Vec<String> {
        vec![BUNSEN_TARGET.to_string(), "conway".to_string()]
    }

    fn level(
        args: &LogArgs,
        policy: LogPolicy,
        rust_log: Option<&str>,
        target: &str,
    ) -> LevelFilter {
        level_for(&args.builder(policy, &prefixes(), rust_log).build(), target)
    }

    #[test]
    fn test_target_prefixes_start_with_bunsen_and_bunsen_app() {
        assert_eq!(target_prefixes()[..2], ["bunsen", "bunsen_app"]);
    }

    #[test]
    fn test_add_target_prefix_keeps_one_copy() {
        add_target_prefix("logging_test_app");
        add_target_prefix("logging_test_app");

        let prefixes = target_prefixes();
        assert_eq!(
            prefixes.iter().filter(|p| *p == "logging_test_app").count(),
            1
        );
    }

    #[test]
    fn test_policy_splits_ours_from_rest() {
        let policy = LogPolicy::new(LevelFilter::Info, LevelFilter::Warn);
        let args = args(0);

        assert_eq!(level(&args, policy, None, "bunsen"), LevelFilter::Info);
        assert_eq!(
            level(&args, policy, None, "bunsen_tool::live_cmd"),
            LevelFilter::Info
        );
        assert_eq!(
            level(&args, policy, None, "conway::commands"),
            LevelFilter::Info
        );
        assert_eq!(
            level(&args, policy, None, "wgpu_core::device"),
            LevelFilter::Warn
        );
    }

    #[test]
    fn test_verbose_raises_only_ours() {
        let policy = LogPolicy::new(LevelFilter::Info, LevelFilter::Warn);

        assert_eq!(
            level(&args(1), policy, None, "bunsen_app"),
            LevelFilter::Debug
        );
        assert_eq!(level(&args(1), policy, None, "conway"), LevelFilter::Debug);
        assert_eq!(
            level(&args(1), policy, None, "wgpu_core"),
            LevelFilter::Warn
        );
        assert_eq!(
            level(&args(9), policy, None, "bunsen_app"),
            LevelFilter::Trace
        );
    }

    #[test]
    fn test_verbose_raises_from_off() {
        let policy = LogPolicy::new(LevelFilter::Off, LevelFilter::Off);

        assert_eq!(level(&args(0), policy, None, "bunsen"), LevelFilter::Off);
        assert_eq!(level(&args(1), policy, None, "bunsen"), LevelFilter::Error);
        assert_eq!(level(&args(1), policy, None, "burn"), LevelFilter::Off);
    }

    #[test]
    fn test_rust_log_replaces_policy() {
        let policy = LogPolicy::new(LevelFilter::Info, LevelFilter::Warn);
        let rust_log = Some("error,wgpu_core=debug");

        assert_eq!(
            level(&args(0), policy, rust_log, "bunsen"),
            LevelFilter::Error
        );
        assert_eq!(
            level(&args(0), policy, rust_log, "wgpu_core"),
            LevelFilter::Debug
        );
        assert_eq!(
            level(&args(0), policy, rust_log, "burn"),
            LevelFilter::Error
        );
    }

    #[test]
    fn test_verbose_raises_rust_log_for_ours() {
        let policy = LogPolicy::new(LevelFilter::Info, LevelFilter::Warn);
        let rust_log = Some("warn,conway=info");

        assert_eq!(
            level(&args(1), policy, rust_log, "bunsen_app"),
            LevelFilter::Info
        );
        assert_eq!(
            level(&args(1), policy, rust_log, "conway"),
            LevelFilter::Debug
        );
        assert_eq!(level(&args(1), policy, rust_log, "burn"), LevelFilter::Warn);
    }

    #[test]
    fn test_verbose_never_lowers() {
        let policy = LogPolicy::default();
        let rust_log = Some("warn,bunsen=trace");

        assert_eq!(
            level(&args(1), policy, rust_log, "bunsen"),
            LevelFilter::Trace
        );
    }

    #[test]
    fn test_specific_rust_log_directive_keeps_its_level() {
        let policy = LogPolicy::default();
        let rust_log = Some("info,bunsen_tool=error");

        assert_eq!(
            level(&args(1), policy, rust_log, "bunsen_app"),
            LevelFilter::Debug
        );
        assert_eq!(
            level(&args(1), policy, rust_log, "bunsen_tool"),
            LevelFilter::Error
        );
    }

    #[test]
    fn test_quiet_turns_everything_off() {
        let quiet = LogArgs {
            quiet: true,
            verbose: 3,
            ts: false,
        };
        let policy = LogPolicy::new(LevelFilter::Info, LevelFilter::Warn);

        assert_eq!(level(&quiet, policy, None, "bunsen"), LevelFilter::Off);
        assert_eq!(
            level(&quiet, policy, Some("trace"), "bunsen"),
            LevelFilter::Off
        );
        assert_eq!(level(&quiet, policy, None, "burn"), LevelFilter::Off);
    }

    #[test]
    fn test_flags_parse() {
        #[derive(clap::Parser)]
        struct Cli {
            #[clap(flatten)]
            logging: LogArgs,
        }

        let cli = <Cli as clap::Parser>::parse_from(["app", "-vv", "--ts"]);
        assert_eq!(cli.logging.verbose, 2);
        assert!(cli.logging.ts);
        assert!(!cli.logging.quiet);
    }
}
