use bunsen::errors::{
    BunsenError,
    BunsenResult,
};
pub use stderrlog::LogLevelNum;
use stderrlog::Timestamp;

/// Logging setup arg group.
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
///    let args = Args::parse();
///    args.logging.init(None)?;
///
///    ...
/// }
/// ```
#[derive(clap::Args, Debug)]
pub struct LogArgs {
    /// Silence log messages.
    #[clap(short, long)]
    pub quiet: bool,

    /// Turn debugging information on (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, default_value = None)]
    verbose: Option<u8>,

    /// Enable timestamped logging.
    #[clap(short, long)]
    pub ts: bool,
}

fn log_level_num(level: LogLevelNum) -> Option<usize> {
    match level {
        LogLevelNum::Off => None,
        LogLevelNum::Error => Some(0),
        LogLevelNum::Warn => Some(1),
        LogLevelNum::Info => Some(2),
        LogLevelNum::Debug => Some(3),
        LogLevelNum::Trace => Some(4),
    }
}

impl LogArgs {
    /// Initialize logging.
    ///
    /// # Args
    ///
    /// * `default` - Default log level; if None, defaults to Warn.
    pub fn init(
        &self,
        default: impl Into<Option<LogLevelNum>>,
    ) -> BunsenResult<()> {
        let log_level = log_level_num(default.into().unwrap_or(LogLevelNum::Warn));

        let log_level = if let Some(verbose) = self.verbose {
            let verbose = verbose as usize;
            Some(
                log_level
                    .map(|level| level + verbose)
                    .unwrap_or_else(|| verbose),
            )
        } else {
            log_level
        };

        if let Some(log_level) = log_level {
            stderrlog::new()
                .quiet(self.quiet)
                .verbosity(log_level)
                .timestamp(if self.ts {
                    Timestamp::Second
                } else {
                    Timestamp::Off
                })
                .init()
                .map_err(BunsenError::external)?;
        }

        Ok(())
    }
}
