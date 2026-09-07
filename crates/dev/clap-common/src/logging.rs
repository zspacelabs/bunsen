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

impl LogArgs {
    pub fn init(
        &self,
        default: impl Into<Option<LogLevelNum>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let log_level = if let Some(verbose) = self.verbose
            && verbose > 0
        {
            LogLevelNum::from(verbose as usize)
        } else {
            default.into().unwrap_or(LogLevelNum::Info)
        };

        stderrlog::new()
            .quiet(self.quiet)
            .verbosity(log_level)
            .timestamp(if self.ts {
                Timestamp::Second
            } else {
                Timestamp::Off
            })
            .init()?;

        Ok(())
    }
}
