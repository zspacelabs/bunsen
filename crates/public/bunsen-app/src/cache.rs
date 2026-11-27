//! The weights cache: the flags that open it, and the commands that show,
//! list and clear it.
//!
//! A binary flattens [`WeightsCacheArgs`] into every command that loads
//! weights (one that keeps converted weights by default passes its command
//! through [`converted_weights_by_default`]), and mounts [`CacheCommand`] as
//! a `cache` subcommand:
//! `cache path`, `cache list [NAME]`,
//! `cache clear [NAME] [--partial] [--age AGE] [--yes]`, `NAME` selecting the
//! files cached under a name (`qwen`, `openai/whisper/base`, `hf/openai`), and
//! `cache gc [--max-age AGE] [--max-cache SIZE] [--keep-shadowed]
//! [--keep-stale-converted] [--yes]`, collecting what no load needs
//! ([`ResourceCache::gc_plan`]).
use std::{
    io::Write,
    path::PathBuf,
    time::{
        Duration,
        SystemTime,
    },
};

use bunsen::{
    data::{
        cache::BunsenDiskCacheOptions,
        resources::{
            CacheGcOptions,
            CacheSelector,
            CachedFile,
            DEFAULT_GC_MAX_AGE,
            GcPlan,
            ResourceCache,
            ResourceCacheOptions,
        },
    },
    errors::{
        BunsenResult,
        ResultContext,
        sys_at,
    },
};

use crate::logging::{
    LevelFilter,
    LogArgs,
    LogPolicy,
};

/// Where fetched weights live, and whether fetching is allowed.
#[derive(clap::Args, Clone, Debug, Default)]
pub struct WeightsCacheArgs {
    /// Directory for fetched weights.
    ///
    /// When omitted: `$BUNSEN_CACHE_DIR`, then the platform's cache
    /// directory.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,

    /// Never reach the network: a model not already local is an error.
    #[arg(long)]
    pub offline: bool,

    /// Keep each model's weights converted, and load them from that copy.
    ///
    /// `keep`: the first load writes the copy; later loads read it. A copy
    /// is the weights again on disk, listed under the build's `converted`
    /// name (`cache list`); with it, a load skips reading and converting the
    /// checkpoint, and a build whose checkpoint was cleared loads without
    /// fetching it. `drop-downloads`: as `keep`, and a load that downloaded
    /// the weights it converts removes them once the copy is saved. `off`:
    /// no copies. The language models keep copies.
    #[arg(long, value_enum, default_value_t = ConvertedWeightsArg::Off)]
    pub converted_weights: ConvertedWeightsArg,
}

/// `--converted-weights`: what loads do with converted copies of the weights
/// ([`ConvertedWeights`](bunsen::data::resources::ConvertedWeights)).
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConvertedWeightsArg {
    /// No copies.
    #[default]
    #[value(alias = "false")]
    Off,
    /// Keep a copy, and load from it.
    #[value(alias = "true")]
    Keep,
    /// Keep a copy, and remove the downloads it was made from.
    DropDownloads,
}

impl From<ConvertedWeightsArg> for bunsen::data::resources::ConvertedWeights {
    fn from(arg: ConvertedWeightsArg) -> Self {
        match arg {
            ConvertedWeightsArg::Off => Self::Off,
            ConvertedWeightsArg::Keep => Self::Keep,
            ConvertedWeightsArg::DropDownloads => Self::KeepAndDropDownloads,
        }
    }
}

impl WeightsCacheArgs {
    /// The cache options these flags give.
    pub fn options(&self) -> ResourceCacheOptions {
        ResourceCacheOptions::default()
            .with_disk(BunsenDiskCacheOptions::default().with_cache_dir(self.cache_dir.clone()))
            .with_offline(self.offline)
            .with_converted_weights(self.converted_weights.into())
    }

    /// Opens the cache.
    ///
    /// # Errors
    /// As [`ResourceCache::new`], as policy: the options come from the
    /// command line.
    pub fn init(&self) -> BunsenResult<ResourceCache> {
        ResourceCache::new(self.options()).as_policy()
    }
}

/// `command`, and every subcommand under it, with `--converted-weights`
/// ([`WeightsCacheArgs::converted_weights`]) defaulting to `keep`: for a
/// binary that keeps converted weights unless told not to.
pub fn converted_weights_by_default(command: clap::Command) -> clap::Command {
    let command = if command
        .get_arguments()
        .any(|arg| arg.get_id() == "converted_weights")
    {
        command.mut_arg("converted_weights", |arg| arg.default_value("keep"))
    } else {
        command
    };
    let names: Vec<String> = command
        .get_subcommands()
        .map(|sub| sub.get_name().to_string())
        .collect();
    names.into_iter().fold(command, |command, name| {
        command.mut_subcommand(name, converted_weights_by_default)
    })
}

/// The weights cache: where it is, what it holds, and clearing it.
#[derive(clap::Args, Debug)]
pub struct CacheCommand {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    #[command(subcommand)]
    pub action: CacheAction,
}

/// What [`CacheCommand`] does.
#[derive(clap::Subcommand, Clone, Debug)]
pub enum CacheAction {
    /// Print the cache directory.
    Path,

    /// List the cached weights, file by file, then the total.
    ///
    /// Each file's name (its resource's full name: an `hf:` repo's under
    /// `hf/<org>/<repo>/<slot>`, a path's `given`), size, when it was fetched
    /// and last loaded (UTC), and its pin. Listing loads nothing, so it changes
    /// no file's last use.
    List {
        /// The files cached under this name; every file when omitted.
        ///
        /// A catalog name (`qwen`, `openai/whisper/base`), or `hf`
        /// (`hf/openai`) or `given` for the files no catalog names.
        selector: Option<CacheSelector>,
    },

    /// Remove cached weights; without `--yes`, only list what would go.
    Clear {
        /// The files cached under this name; every file when omitted.
        ///
        /// A catalog name (`qwen`, `openai/whisper/base`), or `hf`
        /// (`hf/openai`) or `given` for the files no catalog names.
        selector: Option<CacheSelector>,

        /// Only unfinished downloads.
        #[arg(long)]
        partial: bool,

        /// Only files not loaded for at least this long (`90m`, `12h`, `30d`,
        /// `2w`).
        ///
        /// Units: `s`, `m`, `h`, `d`, `w`. A file whose last use the file
        /// system does not record is kept.
        #[arg(long, value_parser = parse_age)]
        age: Option<Duration>,

        /// Remove them.
        #[arg(long)]
        yes: bool,
    },

    /// Remove what no load needs; without `--yes`, only list what would go.
    ///
    /// Each file the first reason that holds: unfinished downloads, always;
    /// converted copies no load of this bunsen reads (another version's, or
    /// from before copies had manifests); files not loaded within
    /// `--max-age`; a checkpoint every converted copy standing for it is
    /// newer than (it was not read since); then, with `--max-cache`, the
    /// least recently used until the rest fits. A copy and its manifest go
    /// together. Only the cache's own files: never a local directory's.
    Gc {
        /// Remove files not loaded for this long (`7d`), or `none`.
        ///
        /// A count and a unit (`90m`, `12h`, `7d`, `2w`); `none` keeps files
        /// whatever their age. Units: `s`, `m`, `h`, `d`, `w`. A file whose
        /// last use the file system does not record is kept.
        #[arg(long, value_parser = parse_max_age, default_value = "7d")]
        max_age: MaxAge,

        /// Then remove the least recently used until the cache fits this size.
        ///
        /// A count and a decimal unit: `500M`, `50G`, `1.5T`.
        #[arg(long, value_parser = parse_size)]
        max_cache: Option<u64>,

        /// Keep checkpoints that converted copies stand for.
        #[arg(long)]
        keep_shadowed: bool,

        /// Keep converted copies no load of this bunsen reads.
        #[arg(long)]
        keep_stale_converted: bool,

        /// Remove them.
        #[arg(long)]
        yes: bool,
    },
}

/// `cache gc --max-age`: a maximum age, or none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaxAge(pub Option<Duration>);

impl Default for MaxAge {
    fn default() -> Self {
        Self(Some(DEFAULT_GC_MAX_AGE))
    }
}

impl CacheCommand {
    /// Runs the command, writing to stdout.
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let cache = self.cache.init()?;
        self.action.run(&cache, &mut std::io::stdout().lock())
    }
}

impl CacheAction {
    /// Runs the action on `cache`, writing to `out`.
    ///
    /// # Errors
    /// As [`ResourceCache::cached_files`] and
    /// [`ResourceCache::remove_cached`]; a
    /// [`Sys`](bunsen::errors::BunsenErrorKind::Sys) error if `out` cannot be
    /// written.
    pub fn run(
        &self,
        cache: &ResourceCache,
        out: &mut impl Write,
    ) -> BunsenResult<()> {
        let w = sys_at("write to", "stdout");
        match self {
            CacheAction::Path => writeln!(out, "{}", cache.cache_dir().display()).map_err(w),
            CacheAction::List { selector } => {
                let selector = selector.clone().unwrap_or_default();
                let files = cache.cached_files(&selector)?;
                write_table(&files, out).map_err(sys_at("write to", "stdout"))?;
                writeln!(
                    out,
                    "{} in {}",
                    summary(&files),
                    cache.resources_dir().display()
                )
                .map_err(w)
            }
            CacheAction::Clear {
                selector,
                partial,
                age,
                yes,
            } => {
                let selector = selector.clone().unwrap_or_default();
                let now = SystemTime::now();
                let old_enough = |f: &CachedFile| match age {
                    None => true,
                    Some(age) => f
                        .last_used
                        .and_then(|used| now.duration_since(used).ok())
                        .is_some_and(|idle| idle >= *age),
                };
                let files: Vec<CachedFile> = cache
                    .cached_files(&selector)?
                    .into_iter()
                    .filter(|f| (!partial || f.partial) && old_enough(f))
                    .collect();
                if files.is_empty() {
                    return writeln!(out, "nothing cached for {selector}").map_err(w);
                }
                write_table(&files, out).map_err(sys_at("write to", "stdout"))?;
                if !yes {
                    return writeln!(
                        out,
                        "{} would be removed; pass --yes to remove them",
                        summary(&files)
                    )
                    .map_err(w);
                }
                let freed = cache.remove_cached(&files)?;
                writeln!(
                    out,
                    "removed {} file{}, {}",
                    files.len(),
                    if files.len() == 1 { "" } else { "s" },
                    human_bytes(freed)
                )
                .map_err(w)
            }
            CacheAction::Gc {
                max_age,
                max_cache,
                keep_shadowed,
                keep_stale_converted,
                yes,
            } => {
                let options = CacheGcOptions::default()
                    .with_max_age(max_age.0)
                    .with_max_cache(*max_cache)
                    .with_shadowed(!keep_shadowed)
                    .with_stale_converted(!keep_stale_converted);
                let plan = cache.gc_plan(&options, SystemTime::now())?;
                let kept = summary(&plan.keep);
                if plan.remove.is_empty() {
                    return writeln!(out, "nothing to collect; keeping {kept}").map_err(w);
                }
                write_gc_table(&plan, out).map_err(sys_at("write to", "stdout"))?;
                let files = plan.files();
                if !yes {
                    return writeln!(
                        out,
                        "{} would be removed, keeping {kept}; pass --yes to remove them",
                        summary(&files)
                    )
                    .map_err(w);
                }
                let freed = cache.remove_cached(&files)?;
                writeln!(
                    out,
                    "removed {} file{}, {}; kept {kept}",
                    files.len(),
                    if files.len() == 1 { "" } else { "s" },
                    human_bytes(freed)
                )
                .map_err(w)
            }
        }
    }
}

/// `--max-age`: an age as [`parse_age`] reads it, or `none`.
fn parse_max_age(text: &str) -> Result<MaxAge, String> {
    if text == "none" {
        return Ok(MaxAge(None));
    }
    parse_age(text).map(|age| MaxAge(Some(age)))
}

/// `--max-cache`: a size, a count (decimals allowed) and an optional decimal
/// unit, `B`, `K`, `M`, `G` or `T` (`kB`, `MB`, `GB`, `TB` too), any case.
fn parse_size(text: &str) -> Result<u64, String> {
    let bad = || format!("{text:?}: a size, as 500M, 50G or 1.5T");
    let split = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(text.len());
    let (count, unit) = text.split_at(split);
    let count: f64 = count.parse().map_err(|_| bad())?;
    let scale: u64 = match unit.to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" => 1_000,
        "m" | "mb" => 1_000_000,
        "g" | "gb" => 1_000_000_000,
        "t" | "tb" => 1_000_000_000_000,
        _ => return Err(bad()),
    };
    Ok((count * scale as f64) as u64)
}

/// `n files, size`.
fn summary(files: &[CachedFile]) -> String {
    let bytes = files.iter().map(|f| f.bytes).sum();
    format!(
        "{} file{}, {}",
        files.len(),
        if files.len() == 1 { "" } else { "s" },
        human_bytes(bytes)
    )
}

/// `--age`: a count and a unit, `s`, `m`, `h`, `d` or `w`.
fn parse_age(text: &str) -> Result<Duration, String> {
    let split = text
        .find(|c: char| !c.is_ascii_digit())
        .ok_or_else(|| format!("{text:?}: a count and a unit (s, m, h, d, w), as 30d"))?;
    let (count, unit) = text.split_at(split);
    let count: u64 = count
        .parse()
        .map_err(|_| format!("{text:?}: a count and a unit (s, m, h, d, w), as 30d"))?;
    let seconds = match unit {
        "s" => 1,
        "m" => 60,
        "h" => 3_600,
        "d" => 86_400,
        "w" => 604_800,
        _ => return Err(format!("{text:?}: the unit is one of s, m, h, d, w")),
    };
    Ok(Duration::from_secs(count * seconds))
}

/// A time as `cache list` shows it: `2026-10-31 16:40`, UTC; `-` when the
/// file system does not record it.
fn utc_minute(time: Option<SystemTime>) -> String {
    let format = time::macros::format_description!("[year]-[month]-[day] [hour]:[minute]");
    time.and_then(|t| time::OffsetDateTime::from(t).format(&format).ok())
        .unwrap_or_else(|| "-".to_string())
}

/// One row per file: name, file (marked when partial), size, when it was
/// fetched and last loaded, and the pin's first 12 characters.
fn write_table(
    files: &[CachedFile],
    out: &mut impl Write,
) -> std::io::Result<()> {
    let rows: Vec<(&CachedFile, String)> = files.iter().map(|f| (f, String::new())).collect();
    write_rows(&rows, None, out)
}

/// [`write_table`] for a gc plan's removals, each row led by its reason.
fn write_gc_table(
    plan: &GcPlan,
    out: &mut impl Write,
) -> std::io::Result<()> {
    let rows: Vec<(&CachedFile, String)> = plan
        .remove
        .iter()
        .map(|c| (&c.file, c.reason.to_string()))
        .collect();
    write_rows(&rows, Some("REASON"), out)
}

/// [`write_table`]'s rows, led by a column headed `lead` when given.
fn write_rows(
    rows: &[(&CachedFile, String)],
    lead: Option<&str>,
    out: &mut impl Write,
) -> std::io::Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let r = lead.map_or(0, |head| {
        rows.iter()
            .fold(head.len(), |r, (_, reason)| r.max(reason.len()))
    });
    let led = |text: &str| match lead {
        Some(_) => format!("{text:r$}  "),
        None => String::new(),
    };
    let files: Vec<&CachedFile> = rows.iter().map(|(f, _)| *f).collect();
    if files.is_empty() {
        return Ok(());
    }
    let name = |f: &CachedFile| {
        if f.partial {
            format!("{} (partial)", f.file)
        } else {
            f.file.clone()
        }
    };
    let widths = files.iter().fold([9, 4], |[n, f], file| {
        [n.max(file.name.len()), f.max(name(file).len())]
    });
    let [n, f] = widths;
    writeln!(
        out,
        "{}{:n$}  {:f$}  {:>9}  {:16}  {:16}  PIN",
        led(lead.unwrap_or("")),
        "NAME",
        "FILE",
        "SIZE",
        "FETCHED (UTC)",
        "LAST USED (UTC)"
    )?;
    for (file, reason) in rows {
        let pin: String = file.pin.chars().take(12).collect();
        writeln!(
            out,
            "{}{:n$}  {:f$}  {:>9}  {:16}  {:16}  {pin}",
            led(reason),
            file.name,
            name(file),
            human_bytes(file.bytes),
            utc_minute(file.created),
            utc_minute(file.last_used),
        )?;
    }
    Ok(())
}

/// A size in decimal units: `512 B`, `72.1 kB`, `151.1 MB`, `6.17 GB`.
pub fn human_bytes(bytes: u64) -> String {
    let b = bytes as f64;
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", b / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", b / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.1} kB", b / 1e3)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use clap::Parser;

    use super::*;

    #[derive(Parser, Debug)]
    struct Cli {
        #[command(subcommand)]
        cmd: Cmd,
    }

    #[derive(clap::Subcommand, Debug)]
    enum Cmd {
        Cache(CacheCommand),
    }

    fn cache_in(dir: &std::path::Path) -> ResourceCache {
        WeightsCacheArgs {
            cache_dir: Some(dir.to_path_buf()),
            offline: true,
            converted_weights: ConvertedWeightsArg::Off,
        }
        .init()
        .unwrap()
    }

    #[derive(Parser, Debug)]
    struct Loader {
        #[command(subcommand)]
        cmd: LoaderCmd,
    }

    #[derive(clap::Subcommand, Debug)]
    enum LoaderCmd {
        Run {
            #[command(flatten)]
            cache: WeightsCacheArgs,
        },
    }

    /// `--converted-weights` is off unless a binary turns it on by default,
    /// which reaches every subcommand and still yields to the flag.
    #[test]
    fn test_converted_weights_by_default() {
        use clap::{
            CommandFactory,
            FromArgMatches,
        };
        let parse = |command: clap::Command, args: &[&str]| {
            let matches = command.try_get_matches_from(args).unwrap();
            let LoaderCmd::Run { cache } = Loader::from_arg_matches(&matches).unwrap().cmd;
            cache.options().converted_weights
        };
        use bunsen::data::resources::ConvertedWeights;
        assert_eq!(
            parse(Loader::command(), &["t", "run"]),
            ConvertedWeights::Off
        );
        assert_eq!(
            parse(
                converted_weights_by_default(Loader::command()),
                &["t", "run"]
            ),
            ConvertedWeights::Keep
        );
        for (flag, want) in [
            ("off", ConvertedWeights::Off),
            ("false", ConvertedWeights::Off),
            ("true", ConvertedWeights::Keep),
            ("drop-downloads", ConvertedWeights::KeepAndDropDownloads),
        ] {
            assert_eq!(
                parse(
                    converted_weights_by_default(Loader::command()),
                    &["t", "run", "--converted-weights", flag]
                ),
                want,
                "{flag}"
            );
        }
    }

    fn put(
        cache: &ResourceCache,
        rel: &str,
        bytes: usize,
    ) {
        let path = cache.resources_dir().join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, vec![0u8; bytes]).unwrap();
    }

    fn run(
        action: &CacheAction,
        cache: &ResourceCache,
    ) -> String {
        let mut out = Vec::new();
        action.run(cache, &mut out).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn test_human_bytes() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(72_100), "72.1 kB");
        assert_eq!(human_bytes(151_100_000), "151.1 MB");
        assert_eq!(human_bytes(6_170_000_000), "6.17 GB");
    }

    #[test]
    fn test_flags_parse() {
        let cli = Cli::try_parse_from([
            "x",
            "cache",
            "--cache-dir",
            "/tmp/c",
            "clear",
            "openai/whisper",
            "--partial",
        ])
        .unwrap();
        let Cmd::Cache(cmd) = cli.cmd;
        assert_eq!(cmd.cache.cache_dir, Some(PathBuf::from("/tmp/c")));
        let CacheAction::Clear {
            selector,
            partial,
            yes,
            ..
        } = cmd.action
        else {
            panic!("clear");
        };
        assert_eq!(selector, Some("openai/whisper".parse().unwrap()));
        assert!(partial && !yes);
        assert!(Cli::try_parse_from(["x", "cache", "list", "a//b"]).is_err());
    }

    #[test]
    fn test_parse_age() {
        assert_eq!(parse_age("90m"), Ok(Duration::from_secs(5_400)));
        assert_eq!(parse_age("12h"), Ok(Duration::from_secs(43_200)));
        assert_eq!(parse_age("30d"), Ok(Duration::from_secs(2_592_000)));
        assert_eq!(parse_age("2w"), Ok(Duration::from_secs(1_209_600)));
        assert_eq!(parse_age("45s"), Ok(Duration::from_secs(45)));
        for bad in ["", "30", "d", "30y", "-1d", "1.5d"] {
            assert!(parse_age(bad).is_err(), "{bad:?}");
        }
        let cli = Cli::try_parse_from(["x", "cache", "clear", "--age", "30d"]).unwrap();
        let Cmd::Cache(cmd) = cli.cmd;
        assert!(matches!(
            cmd.action,
            CacheAction::Clear { age: Some(age), .. } if age == Duration::from_secs(2_592_000)
        ));
    }

    #[test]
    fn test_parse_gc_flags() {
        assert_eq!(parse_max_age("none"), Ok(MaxAge(None)));
        assert_eq!(
            parse_max_age("2d"),
            Ok(MaxAge(Some(Duration::from_secs(172_800))))
        );
        assert!(parse_max_age("never").is_err());
        assert_eq!(parse_size("500M"), Ok(500_000_000));
        assert_eq!(parse_size("50gb"), Ok(50_000_000_000));
        assert_eq!(parse_size("1.5T"), Ok(1_500_000_000_000));
        assert_eq!(parse_size("1024"), Ok(1_024));
        for bad in ["", "G", "5Q", "1..5G", "-5G"] {
            assert!(parse_size(bad).is_err(), "{bad:?}");
        }
        let cli = Cli::try_parse_from(["x", "cache", "gc"]).unwrap();
        let Cmd::Cache(cmd) = cli.cmd;
        assert!(matches!(
            cmd.action,
            CacheAction::Gc {
                max_age,
                max_cache: None,
                keep_shadowed: false,
                keep_stale_converted: false,
                yes: false,
            } if max_age == MaxAge::default()
        ));
    }

    /// `gc` lists what it would collect, each file's reason first, and with
    /// `--yes` removes it, keeping the rest.
    #[test]
    fn test_gc() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        let (a, b) = ("a".repeat(64), "b".repeat(64));
        put(&cache, &format!("m/checkpoint/{a}/w.bin.partial"), 500);
        put(&cache, &format!("m/checkpoint/{b}/old.bin"), 1_000);
        put(&cache, &format!("m/config/{b}/config.json"), 10);
        let old = cache
            .resources_dir()
            .join(format!("m/checkpoint/{b}/old.bin"));
        fs::File::options()
            .write(true)
            .open(&old)
            .unwrap()
            .set_modified(SystemTime::now() - Duration::from_secs(30 * 86_400))
            .unwrap();

        let gc = |yes| CacheAction::Gc {
            max_age: MaxAge::default(),
            max_cache: None,
            keep_shadowed: false,
            keep_stale_converted: false,
            yes,
        };
        let shown = run(&gc(false), &cache);
        let lines: Vec<&str> = shown.lines().collect();
        assert!(lines[0].starts_with("REASON   NAME"), "{shown}");
        assert!(lines[1].starts_with("partial  m/checkpoint"), "{shown}");
        assert!(lines[2].starts_with("aged     m/checkpoint"), "{shown}");
        assert!(
            lines[3].starts_with("2 files, 1.5 kB would be removed, keeping 1 file, 10 B"),
            "{shown}"
        );
        assert_eq!(
            cache.cached_files(&CacheSelector::default()).unwrap().len(),
            3
        );
        assert!(run(&gc(true), &cache).contains("removed 2 files, 1.5 kB; kept 1 file, 10 B"));
        assert!(run(&gc(true), &cache).contains("nothing to collect; keeping 1 file, 10 B"));
    }

    /// `list` shows when each file was fetched and last loaded, in UTC;
    /// `clear --age` takes only the files not loaded for that long.
    #[test]
    fn test_dates_and_clear_by_age() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        let (a, b) = ("a".repeat(64), "b".repeat(64));
        put(
            &cache,
            &format!("openai/whisper/old/checkpoint/{a}/old.pt"),
            1_000,
        );
        put(
            &cache,
            &format!("openai/whisper/new/checkpoint/{b}/new.pt"),
            2_000,
        );
        let old = cache
            .resources_dir()
            .join(format!("openai/whisper/old/checkpoint/{a}/old.pt"));
        let then = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000);
        fs::File::options()
            .write(true)
            .open(&old)
            .unwrap()
            .set_modified(then)
            .unwrap();

        let list = run(&CacheAction::List { selector: None }, &cache);
        assert!(
            list.lines().next().unwrap().contains("LAST USED (UTC)"),
            "{list}"
        );
        let row = list.lines().find(|l| l.contains("old.pt")).unwrap();
        assert!(row.contains("2001-09-09 01:46"), "{row}");
        assert_eq!(utc_minute(None), "-");

        let by_age = CacheAction::Clear {
            selector: None,
            partial: false,
            age: Some(Duration::from_secs(30 * 86_400)),
            yes: true,
        };
        assert!(run(&by_age, &cache).contains("removed 1 file, 1.0 kB"));
        let left = cache.cached_files(&CacheSelector::default()).unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].file, "new.pt");
    }

    /// `path` prints the directory; `list` a row per file and the total;
    /// `clear` only lists without `--yes`, then removes what it selects.
    #[test]
    fn test_path_list_and_clear() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        let tiny = format!("65147644a518{}", "d".repeat(52));
        let (a, b) = ("a".repeat(64), "b".repeat(64));
        put(
            &cache,
            &format!("openai/whisper/tiny/checkpoint/{tiny}/tiny.pt"),
            2_000,
        );
        put(
            &cache,
            &format!("openai/whisper/base/checkpoint/{a}/base.pt.partial"),
            500,
        );
        put(
            &cache,
            &format!("nvidia/parakeet-tdt-0.6b-v3/checkpoint/{b}/model.safetensors"),
            3_000_000,
        );

        assert_eq!(
            run(&CacheAction::Path, &cache).trim(),
            dir.path().display().to_string()
        );

        let list = run(&CacheAction::List { selector: None }, &cache);
        let lines: Vec<&str> = list.lines().collect();
        assert_eq!(lines.len(), 5, "{list}");
        assert!(lines[0].starts_with("NAME "));
        assert!(lines[1].starts_with("nvidia/parakeet") && lines[1].contains("3.0 MB"));
        assert!(lines[2].contains("base.pt (partial)"));
        assert!(lines[3].ends_with("65147644a518"));
        assert!(lines[4].starts_with("3 files, 3.0 MB"), "{}", lines[4]);

        let dry = CacheAction::Clear {
            selector: Some("openai/whisper".parse().unwrap()),
            partial: true,
            age: None,
            yes: false,
        };
        let shown = run(&dry, &cache);
        assert!(shown.contains("1 file, 500 B would be removed"), "{shown}");
        assert_eq!(
            cache.cached_files(&CacheSelector::default()).unwrap().len(),
            3
        );

        let partial = CacheAction::Clear {
            selector: Some("openai/whisper".parse().unwrap()),
            partial: true,
            age: None,
            yes: true,
        };
        assert!(run(&partial, &cache).contains("removed 1 file, 500 B"));
        let all = CacheAction::Clear {
            selector: None,
            partial: false,
            age: None,
            yes: true,
        };
        assert!(run(&all, &cache).contains("removed 2 files, 3.0 MB"));
        assert!(run(&all, &cache).contains("nothing cached for every resource"));
    }
}
