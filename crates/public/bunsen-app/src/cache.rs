//! The weights cache: the flags that open it, and the commands that show,
//! list and clear it.
//!
//! A binary flattens [`WeightsCacheArgs`] into every command that loads
//! weights, and mounts [`CacheCommand`] as a `cache` subcommand:
//! `cache path`, `cache list [KIT[/NAMESPACE]]`, and
//! `cache clear [KIT[/NAMESPACE]] [--partial] [--yes]`.
use std::{
    io::Write,
    path::PathBuf,
};

use bunsen::{
    data::{
        cache::BunsenDiskCacheOptions,
        pretrained::{
            CacheSelector,
            CachedFile,
            PretrainedCache,
            PretrainedCacheOptions,
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
    /// Directory for fetched weights; `$BUNSEN_CACHE_DIR`, then the
    /// platform's cache directory, when omitted.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,

    /// Never reach the network: a model not already local is an error.
    #[arg(long)]
    pub offline: bool,
}

impl WeightsCacheArgs {
    /// The cache options these flags give.
    pub fn options(&self) -> PretrainedCacheOptions {
        PretrainedCacheOptions::default()
            .with_disk(BunsenDiskCacheOptions::default().with_cache_dir(self.cache_dir.clone()))
            .with_offline(self.offline)
    }

    /// Opens the cache.
    ///
    /// # Errors
    /// As [`PretrainedCache::new`], as policy: the options come from the
    /// command line.
    pub fn init(&self) -> BunsenResult<PretrainedCache> {
        PretrainedCache::new(self.options()).as_policy()
    }
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

    /// List the cached weights: each file's kit, namespace, size and pin,
    /// then the total.
    List {
        /// `kit` or `kit/namespace`; every file when omitted.
        selector: Option<CacheSelector>,
    },

    /// Remove cached weights; without `--yes`, only list what would go.
    Clear {
        /// `kit` or `kit/namespace`; every file when omitted.
        selector: Option<CacheSelector>,

        /// Only unfinished downloads.
        #[arg(long)]
        partial: bool,

        /// Remove them.
        #[arg(long)]
        yes: bool,
    },
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
    /// As [`PretrainedCache::cached_files`] and
    /// [`PretrainedCache::remove_cached`]; a
    /// [`Sys`](bunsen::errors::BunsenErrorKind::Sys) error if `out` cannot be
    /// written.
    pub fn run(
        &self,
        cache: &PretrainedCache,
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
                    cache.pretrained_dir().display()
                )
                .map_err(w)
            }
            CacheAction::Clear {
                selector,
                partial,
                yes,
            } => {
                let selector = selector.clone().unwrap_or_default();
                let files: Vec<CachedFile> = cache
                    .cached_files(&selector)?
                    .into_iter()
                    .filter(|f| !partial || f.partial)
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
        }
    }
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

/// One row per file: kit, namespace, file (marked when partial), size, and
/// the pin's first 12 characters.
fn write_table(
    files: &[CachedFile],
    out: &mut impl Write,
) -> std::io::Result<()> {
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
    let widths = files.iter().fold([3, 9, 4], |[k, n, f], file| {
        [
            k.max(file.kit.len()),
            n.max(file.namespace.len()),
            f.max(name(file).len()),
        ]
    });
    let [k, n, f] = widths;
    writeln!(
        out,
        "{:k$}  {:n$}  {:f$}  {:>9}  PIN",
        "KIT", "NAMESPACE", "FILE", "SIZE"
    )?;
    for file in files {
        let pin: String = file.pin.chars().take(12).collect();
        writeln!(
            out,
            "{:k$}  {:n$}  {:f$}  {:>9}  {pin}",
            file.kit,
            file.namespace,
            name(file),
            human_bytes(file.bytes),
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

    fn cache_in(dir: &std::path::Path) -> PretrainedCache {
        WeightsCacheArgs {
            cache_dir: Some(dir.to_path_buf()),
            offline: true,
        }
        .init()
        .unwrap()
    }

    fn put(
        cache: &PretrainedCache,
        rel: &str,
        bytes: usize,
    ) {
        let path = cache.pretrained_dir().join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, vec![0u8; bytes]).unwrap();
    }

    fn run(
        action: &CacheAction,
        cache: &PretrainedCache,
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
            "whisper/openai",
            "--partial",
        ])
        .unwrap();
        let Cmd::Cache(cmd) = cli.cmd;
        assert_eq!(cmd.cache.cache_dir, Some(PathBuf::from("/tmp/c")));
        let CacheAction::Clear {
            selector,
            partial,
            yes,
        } = cmd.action
        else {
            panic!("clear");
        };
        assert_eq!(selector, Some("whisper/openai".parse().unwrap()));
        assert!(partial && !yes);
        assert!(Cli::try_parse_from(["x", "cache", "list", "a/b/c"]).is_err());
    }

    /// `path` prints the directory; `list` a row per file and the total;
    /// `clear` only lists without `--yes`, then removes what it selects.
    #[test]
    fn test_path_list_and_clear() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        put(&cache, "whisper/openai/65147644a518d12f/tiny.pt", 2_000);
        put(&cache, "whisper/openai/aaaa/base.pt.partial", 500);
        put(&cache, "parakeet/nvidia/bbbb/model.safetensors", 3_000_000);

        assert_eq!(
            run(&CacheAction::Path, &cache).trim(),
            dir.path().display().to_string()
        );

        let list = run(&CacheAction::List { selector: None }, &cache);
        let lines: Vec<&str> = list.lines().collect();
        assert_eq!(lines.len(), 5, "{list}");
        assert!(lines[0].starts_with("KIT"));
        assert!(lines[1].starts_with("parakeet") && lines[1].contains("3.0 MB"));
        assert!(lines[3].contains("base.pt (partial)"));
        assert!(lines[2].ends_with("65147644a518"));
        assert!(lines[4].starts_with("3 files, 3.0 MB"), "{}", lines[4]);

        let dry = CacheAction::Clear {
            selector: Some("whisper".parse().unwrap()),
            partial: true,
            yes: false,
        };
        let shown = run(&dry, &cache);
        assert!(shown.contains("1 file, 500 B would be removed"), "{shown}");
        assert_eq!(
            cache.cached_files(&CacheSelector::default()).unwrap().len(),
            3
        );

        let partial = CacheAction::Clear {
            selector: Some("whisper".parse().unwrap()),
            partial: true,
            yes: true,
        };
        assert!(run(&partial, &cache).contains("removed 1 file, 500 B"));
        let all = CacheAction::Clear {
            selector: None,
            partial: false,
            yes: true,
        };
        assert!(run(&all, &cache).contains("removed 2 files, 3.0 MB"));
        assert!(run(&all, &cache).contains("nothing cached for every kit"));
    }
}
