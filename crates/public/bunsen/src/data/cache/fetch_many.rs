//! # Parallel fetch
//!
//! Many files at once, under a policy: how many transfers run together, how
//! many times a file is retried, and whether one failure stops the rest. The
//! result is a [`FetchReport`] with one [`FetchOutcome`] per job, in job
//! order; the batch itself never fails. [`FetchReport::paths`] is the
//! all-or-nothing view a caller takes when every file has to be there, and
//! the report's `Display` is the one-line summary a CLI prints.
//!
//! A stop cannot cut a transfer in flight: the jobs already running finish,
//! nothing new starts, and the rest are [`FetchOutcome::Skipped`]. Files that
//! landed stay on disk under either policy.

use std::{
    fmt,
    path::{
        Path,
        PathBuf,
    },
    sync::{
        Mutex,
        PoisonError,
        atomic::{
            AtomicBool,
            AtomicUsize,
            Ordering,
        },
    },
    thread,
};

use super::BunsenDiskCache;
use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// One file to bring in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchJob {
    /// Where to fetch it from: mirrors, tried in order.
    pub urls: Vec<String>,

    /// Where it lands.
    pub dest: PathBuf,

    /// The lowercase hex SHA-256 it must have, when pinned.
    pub sha256: Option<String>,
}

impl FetchJob {
    /// A job for `dest` from `urls`, unpinned.
    pub fn new<U, S, P>(
        urls: U,
        dest: P,
    ) -> Self
    where
        U: IntoIterator<Item = S>,
        S: Into<String>,
        P: Into<PathBuf>,
    {
        Self {
            urls: urls.into_iter().map(Into::into).collect(),
            dest: dest.into(),
            sha256: None,
        }
    }

    /// Pins the job to a digest.
    pub fn with_sha256(
        mut self,
        sha256: impl Into<String>,
    ) -> Self {
        self.sha256 = Some(sha256.into());
        self
    }
}

/// What one failure does to the jobs not yet started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OnFailure {
    /// Start nothing new; the rest of the batch is skipped.
    #[default]
    Stop,

    /// Run every job; the report says what fell over.
    Continue,
}

/// How a batch runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchPolicy {
    /// Transfers in flight at once. `0` runs one.
    pub parallel: usize,

    /// Extra attempts per job after its first, each over every mirror again.
    pub retries: u32,

    /// What one failure does to the jobs not yet started.
    pub on_failure: OnFailure,
}

impl Default for FetchPolicy {
    /// Eight at once, one attempt, stop on the first failure.
    fn default() -> Self {
        Self {
            parallel: 8,
            retries: 0,
            on_failure: OnFailure::Stop,
        }
    }
}

impl FetchPolicy {
    /// Sets the transfers in flight at once.
    pub fn with_parallel(
        mut self,
        parallel: usize,
    ) -> Self {
        self.parallel = parallel;
        self
    }

    /// Sets the extra attempts per job.
    pub fn with_retries(
        mut self,
        retries: u32,
    ) -> Self {
        self.retries = retries;
        self
    }

    /// Sets what one failure does to the rest.
    pub fn with_on_failure(
        mut self,
        on_failure: OnFailure,
    ) -> Self {
        self.on_failure = on_failure;
        self
    }
}

/// How one job ended.
#[derive(Debug)]
pub enum FetchOutcome {
    /// The file was already there; nothing was fetched.
    Cached(PathBuf),

    /// The file was fetched and checked.
    Fetched(PathBuf),

    /// Every attempt failed; `error` is the last one.
    Failed {
        /// Where the file was going.
        dest: PathBuf,
        /// The last attempt's error.
        error: BunsenError,
        /// Attempts made, the first included.
        attempts: u32,
    },

    /// Never started: an earlier failure stopped the batch.
    Skipped {
        /// Where the file was going.
        dest: PathBuf,
    },
}

impl FetchOutcome {
    /// Where the file was going, whatever happened.
    pub fn dest(&self) -> &Path {
        match self {
            Self::Cached(p) | Self::Fetched(p) => p,
            Self::Failed { dest, .. } | Self::Skipped { dest } => dest,
        }
    }

    /// The file's path, when it is there.
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::Cached(p) | Self::Fetched(p) => Some(p),
            Self::Failed { .. } | Self::Skipped { .. } => None,
        }
    }

    /// `true` when the file is there.
    pub fn is_ok(&self) -> bool {
        self.path().is_some()
    }
}

/// A batch, job by job, in job order.
#[derive(Debug, Default)]
pub struct FetchReport {
    /// One outcome per job, in the order the jobs were given.
    pub outcomes: Vec<FetchOutcome>,
}

impl FetchReport {
    /// Jobs that fetched a file.
    pub fn fetched(&self) -> usize {
        self.count(|o| matches!(o, FetchOutcome::Fetched(_)))
    }

    /// Jobs whose file was already there.
    pub fn cached(&self) -> usize {
        self.count(|o| matches!(o, FetchOutcome::Cached(_)))
    }

    /// Jobs whose every attempt failed.
    pub fn failed(&self) -> usize {
        self.count(|o| matches!(o, FetchOutcome::Failed { .. }))
    }

    /// Jobs that never started.
    pub fn skipped(&self) -> usize {
        self.count(|o| matches!(o, FetchOutcome::Skipped { .. }))
    }

    /// `true` when every file is there.
    pub fn is_complete(&self) -> bool {
        self.outcomes.iter().all(FetchOutcome::is_ok)
    }

    /// The failed jobs, by index.
    pub fn failures(&self) -> impl Iterator<Item = (usize, &FetchOutcome)> {
        self.outcomes
            .iter()
            .enumerate()
            .filter(|(_, o)| matches!(o, FetchOutcome::Failed { .. }))
    }

    /// Every file's path, in job order, when every file is there.
    ///
    /// # Errors
    /// [`BunsenError::External`] naming each job that failed or was skipped.
    pub fn paths(&self) -> BunsenResult<Vec<PathBuf>> {
        if self.is_complete() {
            return Ok(self
                .outcomes
                .iter()
                .filter_map(|o| o.path().map(Path::to_path_buf))
                .collect());
        }
        let missing: Vec<String> = self
            .outcomes
            .iter()
            .filter_map(|o| match o {
                FetchOutcome::Failed { dest, error, .. } => {
                    Some(format!("{}: {error}", file_name(dest)))
                }
                FetchOutcome::Skipped { dest } => Some(format!("{}: skipped", file_name(dest))),
                _ => None,
            })
            .collect();
        Err(BunsenError::External(format!(
            "{} of {} fetches did not land: {}",
            missing.len(),
            self.outcomes.len(),
            missing.join("; ")
        )))
    }

    fn count(
        &self,
        pred: impl Fn(&FetchOutcome) -> bool,
    ) -> usize {
        self.outcomes.iter().filter(|o| pred(o)).count()
    }
}

impl fmt::Display for FetchReport {
    /// `12 fetched, 3 cached, 1 failed, 0 skipped (shard_00007.parquet: …)`.
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            f,
            "{} fetched, {} cached, {} failed, {} skipped",
            self.fetched(),
            self.cached(),
            self.failed(),
            self.skipped()
        )?;
        let failures: Vec<String> = self
            .failures()
            .filter_map(|(_, o)| match o {
                FetchOutcome::Failed { dest, error, .. } => {
                    Some(format!("{}: {error}", file_name(dest)))
                }
                _ => None,
            })
            .collect();
        if !failures.is_empty() {
            write!(f, " ({})", failures.join("; "))?;
        }
        Ok(())
    }
}

/// The file name of a path, for messages.
fn file_name(path: &Path) -> String {
    path.file_name().unwrap_or_default().display().to_string()
}

impl BunsenDiskCache {
    /// Brings every job's file in, `policy.parallel` at a time.
    ///
    /// A job whose file is already there is [`FetchOutcome::Cached`]; the
    /// rest go through [`fetch_from_urls`](Self::fetch_from_urls), up to
    /// `policy.retries` more times each on failure. Under
    /// [`OnFailure::Stop`], the first failure keeps any job not yet started
    /// from starting. Every transfer reports to the observer stack.
    pub fn fetch_many(
        &self,
        jobs: &[FetchJob],
        policy: &FetchPolicy,
    ) -> FetchReport {
        let slots: Vec<Mutex<Option<FetchOutcome>>> =
            jobs.iter().map(|_| Mutex::new(None)).collect();
        let next = AtomicUsize::new(0);
        let stop = AtomicBool::new(false);
        let workers = policy.parallel.clamp(1, jobs.len().max(1));

        thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::SeqCst);
                        let Some(job) = jobs.get(i) else {
                            break;
                        };
                        let outcome = if stop.load(Ordering::SeqCst) {
                            FetchOutcome::Skipped {
                                dest: job.dest.clone(),
                            }
                        } else {
                            self.fetch_job(job, policy.retries)
                        };
                        if policy.on_failure == OnFailure::Stop
                            && matches!(outcome, FetchOutcome::Failed { .. })
                        {
                            stop.store(true, Ordering::SeqCst);
                        }
                        *slots[i].lock().unwrap_or_else(PoisonError::into_inner) = Some(outcome);
                    }
                });
            }
        });

        FetchReport {
            outcomes: slots
                .into_iter()
                .map(|slot| {
                    slot.into_inner()
                        .unwrap_or_else(PoisonError::into_inner)
                        .expect("every job is visited once")
                })
                .collect(),
        }
    }

    /// One job: cached, or fetched with retries, or failed.
    fn fetch_job(
        &self,
        job: &FetchJob,
        retries: u32,
    ) -> FetchOutcome {
        if job.dest.is_file() {
            return FetchOutcome::Cached(job.dest.clone());
        }
        let urls: Vec<&str> = job.urls.iter().map(String::as_str).collect();
        let mut attempts = 0;
        let mut last = None;
        while attempts <= retries {
            attempts += 1;
            match self.fetch_from_urls(&urls, &job.dest, job.sha256.as_deref()) {
                Ok(()) => return FetchOutcome::Fetched(job.dest.clone()),
                Err(e) => last = Some(e),
            }
        }
        FetchOutcome::Failed {
            dest: job.dest.clone(),
            error: last.expect("at least one attempt was made"),
            attempts,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::Path,
    };

    use super::*;
    use crate::{
        data::cache::{
            BunsenDiskCacheOptions,
            transfer::testing::{
                ABC_SHA256,
                refused_url,
                serve_flaky,
                serve_n,
                serve_once,
            },
        },
        errors::BunsenError,
    };

    fn cache_in(dir: &Path) -> BunsenDiskCache {
        BunsenDiskCache::new(
            BunsenDiskCacheOptions::default()
                .with_cache_dir(Some(dir.join("cache")))
                .without_transfer_observers(),
        )
        .unwrap()
    }

    #[test]
    fn test_policy_and_job_builders() {
        let policy = FetchPolicy::default();
        assert_eq!(policy.parallel, 8);
        assert_eq!(policy.retries, 0);
        assert_eq!(policy.on_failure, OnFailure::Stop);

        let policy = policy
            .with_parallel(2)
            .with_retries(3)
            .with_on_failure(OnFailure::Continue);
        assert_eq!(policy.parallel, 2);
        assert_eq!(policy.retries, 3);
        assert_eq!(policy.on_failure, OnFailure::Continue);

        let job = FetchJob::new(["a", "b"], "/d/f.bin").with_sha256(ABC_SHA256);
        assert_eq!(job.urls, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(job.dest, Path::new("/d/f.bin"));
        assert_eq!(job.sha256.as_deref(), Some(ABC_SHA256));
    }

    /// Under `Continue` every job runs: one was cached, one lands off its
    /// second mirror, one has no live mirror; the report says so, `paths()`
    /// names the one that did not land, and the summary reads right.
    #[test]
    fn test_continue_reports_every_job() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        let root = dir.path().join("shards");

        let cached = root.join("shard_0.bin");
        fs::create_dir_all(&root).unwrap();
        fs::write(&cached, b"abc").unwrap();
        let live = serve_once("shard_1.bin", b"abc");
        let jobs = vec![
            FetchJob::new(["http://127.0.0.1:1/never"], &cached),
            FetchJob::new([refused_url("shard_1.bin"), live], root.join("shard_1.bin"))
                .with_sha256(ABC_SHA256),
            FetchJob::new([refused_url("shard_2.bin")], root.join("shard_2.bin")),
        ];
        let policy = FetchPolicy::default()
            .with_parallel(2)
            .with_on_failure(OnFailure::Continue);

        let report = cache.fetch_many(&jobs, &policy);

        assert!(matches!(report.outcomes[0], FetchOutcome::Cached(_)));
        assert!(matches!(report.outcomes[1], FetchOutcome::Fetched(_)));
        assert!(matches!(
            report.outcomes[2],
            FetchOutcome::Failed { attempts: 1, .. }
        ));
        assert_eq!(
            (
                report.fetched(),
                report.cached(),
                report.failed(),
                report.skipped()
            ),
            (1, 1, 1, 0)
        );
        assert!(!report.is_complete());
        assert_eq!(
            report.failures().map(|(i, _)| i).collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(fs::read(root.join("shard_1.bin")).unwrap(), b"abc");

        match report.paths() {
            Err(BunsenError::External(message)) => {
                assert!(
                    message.starts_with("1 of 3 fetches did not land: shard_2.bin: "),
                    "{message}"
                );
            }
            other => panic!("expected External, got {other:?}"),
        }
        let summary = report.to_string();
        assert!(
            summary.starts_with("1 fetched, 1 cached, 1 failed, 0 skipped (shard_2.bin: "),
            "{summary}"
        );
    }

    /// Under `Stop` the first failure keeps the rest from starting, and
    /// `paths()` names the skipped job too.
    #[test]
    fn test_stop_skips_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        let live = serve_once("b.bin", b"abc");
        let jobs = vec![
            FetchJob::new([refused_url("a.bin")], dir.path().join("a.bin")),
            FetchJob::new([live], dir.path().join("b.bin")),
        ];
        let policy = FetchPolicy::default().with_parallel(1);

        let report = cache.fetch_many(&jobs, &policy);

        assert!(matches!(report.outcomes[0], FetchOutcome::Failed { .. }));
        assert!(matches!(report.outcomes[1], FetchOutcome::Skipped { .. }));
        assert_eq!(report.skipped(), 1);
        assert!(!dir.path().join("b.bin").exists());
        match report.paths() {
            Err(BunsenError::External(message)) => {
                assert!(message.contains("b.bin: skipped"), "{message}");
            }
            other => panic!("expected External, got {other:?}"),
        }
        assert_eq!(
            report.to_string().split(" (").next().unwrap(),
            "0 fetched, 0 cached, 1 failed, 1 skipped"
        );
    }

    /// A mirror that drops the first connection comes good on the retry; the
    /// attempt count says how many it took, and without a retry it fails.
    #[test]
    fn test_retries_come_good() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());

        let flaky = serve_flaky("r.bin", b"abc", 1);
        let jobs = vec![FetchJob::new([flaky], dir.path().join("r.bin"))];
        let report = cache.fetch_many(&jobs, &FetchPolicy::default().with_retries(1));
        assert!(
            matches!(report.outcomes[0], FetchOutcome::Fetched(_)),
            "{report}"
        );
        assert_eq!(fs::read(dir.path().join("r.bin")).unwrap(), b"abc");

        let flaky = serve_flaky("n.bin", b"abc", 1);
        let jobs = vec![FetchJob::new([flaky], dir.path().join("n.bin"))];
        let report = cache.fetch_many(&jobs, &FetchPolicy::default());
        assert!(
            matches!(report.outcomes[0], FetchOutcome::Failed { attempts: 1, .. }),
            "{report}"
        );
    }

    /// Several jobs off one server, more jobs than workers; every path comes
    /// back in job order.
    #[test]
    fn test_paths_come_back_in_job_order() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        let url = serve_n("shard.bin", b"abc", 3);
        let dests: Vec<_> = (0..3)
            .map(|i| dir.path().join(format!("s{i}.bin")))
            .collect();
        let jobs: Vec<_> = dests
            .iter()
            .map(|d| FetchJob::new([url.clone()], d).with_sha256(ABC_SHA256))
            .collect();

        let report = cache.fetch_many(&jobs, &FetchPolicy::default().with_parallel(2));

        assert!(report.is_complete(), "{report}");
        assert_eq!(report.paths().unwrap(), dests);
        assert_eq!(
            report.to_string(),
            "3 fetched, 0 cached, 0 failed, 0 skipped"
        );
        assert!(cache.fetch_many(&[], &FetchPolicy::default()).is_complete());
    }
}
