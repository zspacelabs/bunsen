//! # Shard sets on disk
//!
//! A [`ShardSetDescriptor`] bound to a directory, with the disk cache to
//! bring shards in.

use std::{
    fs,
    path::{
        Path,
        PathBuf,
    },
};

use super::{
    ShardId,
    ShardSetDescriptor,
};
use crate::{
    data::cache::{
        BunsenDiskCache,
        FetchJob,
        FetchPolicy,
        FetchReport,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// The directory under the data dir that holds shard sets, one per name.
pub const SHARDS_DIR: &str = "shards";

/// A shard set bound to a directory.
///
/// Two ways to bind: [`in_cache`](Self::in_cache) puts the set under the
/// cache's data directory, at `<data_dir>/shards/<name>/`;
/// [`at_dir`](Self::at_dir) uses a directory the caller names, where the
/// shard files sit directly. The second is what a `--dataset-dir` flag maps
/// to, and it keeps a tree downloaded before this type existed valid.
pub struct ShardSet<'c> {
    cache: &'c BunsenDiskCache,
    desc: ShardSetDescriptor,
    root: PathBuf,
}

impl<'c> ShardSet<'c> {
    /// Binds `desc` under the cache's data directory.
    pub fn in_cache(
        cache: &'c BunsenDiskCache,
        desc: ShardSetDescriptor,
    ) -> Self {
        let root = cache.data_path(&[SHARDS_DIR], &desc.name);
        Self { cache, desc, root }
    }

    /// Binds `desc` to `dir`, where its shard files sit directly.
    pub fn at_dir(
        cache: &'c BunsenDiskCache,
        desc: ShardSetDescriptor,
        dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            cache,
            desc,
            root: dir.into(),
        }
    }

    /// The set.
    pub fn descriptor(&self) -> &ShardSetDescriptor {
        &self.desc
    }

    /// The directory the shard files sit in.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where shard `id` sits, present or not.
    pub fn path(
        &self,
        id: ShardId,
    ) -> PathBuf {
        self.root.join(self.desc.file_name(id))
    }

    /// `true` when shard `id` is on disk.
    pub fn is_cached(
        &self,
        id: ShardId,
    ) -> bool {
        self.path(id).is_file()
    }

    /// The ids of the shards on disk, in order. A root that does not exist
    /// yet holds none.
    ///
    /// # Errors
    /// [`BunsenError::External`] if the root cannot be read.
    pub fn cached_ids(&self) -> BunsenResult<Vec<ShardId>> {
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(BunsenError::external(e)),
        };
        let mut ids = Vec::new();
        for entry in entries {
            let entry = entry.map_err(BunsenError::external)?;
            if !entry.file_type().map_err(BunsenError::external)?.is_file() {
                continue;
            }
            let name = entry.file_name();
            if let Some(id) = name.to_str().and_then(|n| self.desc.parse_file_name(n)) {
                ids.push(id);
            }
        }
        ids.sort_unstable();
        Ok(ids)
    }

    /// Shard `id`'s path: on disk already, or fetched when `download` is on.
    ///
    /// # Errors
    /// [`BunsenError::InvalidArgument`] for an id outside the set;
    /// [`BunsenError::ResourceNotFound`] if it is not on disk and `download`
    /// is off; otherwise as [`fetch`](Self::fetch).
    pub fn locate(
        &self,
        id: ShardId,
        download: bool,
    ) -> BunsenResult<PathBuf> {
        self.check(id)?;
        let path = self.path(id);
        if path.is_file() {
            return Ok(path);
        }
        if !download {
            return Err(BunsenError::ResourceNotFound(format!(
                "{}: shard {id} is not at {}",
                self.desc.name,
                path.display()
            )));
        }
        self.fetch(id)
    }

    /// Brings shard `id` in if it is not on disk, and returns its path.
    ///
    /// Mirrors are tried in order, the shard is checked against its digest
    /// when the set is pinned, and the transfer reports to the cache's
    /// observers.
    ///
    /// # Errors
    /// [`BunsenError::InvalidArgument`] for an id outside the set; otherwise
    /// as [`BunsenDiskCache::fetch_from_urls`].
    pub fn fetch(
        &self,
        id: ShardId,
    ) -> BunsenResult<PathBuf> {
        self.check(id)?;
        let path = self.path(id);
        if path.is_file() {
            return Ok(path);
        }
        let urls = self.desc.urls(id);
        let urls: Vec<&str> = urls.iter().map(String::as_str).collect();
        self.cache
            .fetch_from_urls(&urls, &path, self.desc.digest(id))?;
        Ok(path)
    }

    /// The fetch jobs for `ids`, in the order given.
    ///
    /// # Errors
    /// [`BunsenError::InvalidArgument`] for an id outside the set.
    pub fn jobs(
        &self,
        ids: &[ShardId],
    ) -> BunsenResult<Vec<FetchJob>> {
        ids.iter()
            .map(|&id| {
                self.check(id)?;
                let mut job = FetchJob::new(self.desc.urls(id), self.path(id));
                job.sha256 = self.desc.digest(id).map(str::to_string);
                Ok(job)
            })
            .collect()
    }

    /// Brings shards `ids` in under `policy`, and reports on each. Shards
    /// already on disk are reported as cached.
    ///
    /// # Errors
    /// [`BunsenError::InvalidArgument`] for an id outside the set. A shard
    /// that fails to land is in the report, not an error here.
    pub fn fetch_many(
        &self,
        ids: &[ShardId],
        policy: &FetchPolicy,
    ) -> BunsenResult<FetchReport> {
        let jobs = self.jobs(ids)?;
        Ok(self.cache.fetch_many(&jobs, policy))
    }

    fn check(
        &self,
        id: ShardId,
    ) -> BunsenResult<()> {
        if self.desc.contains(id) {
            Ok(())
        } else {
            Err(BunsenError::InvalidArgument {
                msg: format!(
                    "{}: shard {id} is out of range; the set has {} shards",
                    self.desc.name, self.desc.count
                ),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{
        cache::{
            BunsenDiskCacheOptions,
            FetchOutcome,
            OnFailure,
            testing::{
                ABC_SHA256,
                refused_url,
                serve_n,
            },
        },
        shards::{
            ShardDigests,
            StaticShardSetDescriptor,
        },
    };

    fn cache_in(dir: &Path) -> BunsenDiskCache {
        BunsenDiskCache::new(
            BunsenDiskCacheOptions::default()
                .with_cache_dir(Some(dir.join("cache")))
                .with_data_dir(Some(dir.join("data")))
                .without_transfer_observers(),
        )
        .unwrap()
    }

    /// A set whose three shards all resolve to the same `serve_n` body.
    fn served(
        base_urls: &'static [&'static str],
        count: usize,
    ) -> ShardSetDescriptor {
        StaticShardSetDescriptor {
            name: "served",
            description: "served off a loopback port",
            license: None,
            origin: None,
            base_urls,
            template: "shard_{index}.bin",
            index_width: 2,
            count,
            format: "bin",
            digests: ShardDigests::Unpinned,
        }
        .to_descriptor()
    }

    #[test]
    fn test_roots_and_paths() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        let desc = served(&["https://a.example"], 3);

        let set = ShardSet::in_cache(&cache, desc.clone());
        assert_eq!(
            set.root(),
            dir.path().join("data").join("shards").join("served")
        );
        assert_eq!(set.path(ShardId(2)), set.root().join("shard_02.bin"));
        assert_eq!(set.descriptor(), &desc);
        assert_eq!(set.cached_ids().unwrap(), vec![], "no root yet");

        let set = ShardSet::at_dir(&cache, desc, dir.path().join("elsewhere"));
        assert_eq!(set.root(), dir.path().join("elsewhere"));
        assert!(!set.is_cached(ShardId(0)));
        assert!(matches!(
            set.locate(ShardId(0), false),
            Err(BunsenError::ResourceNotFound(_))
        ));
        assert!(matches!(
            set.locate(ShardId(3), true),
            Err(BunsenError::InvalidArgument { .. })
        ));
        assert!(matches!(
            set.fetch_many(&[ShardId(3)], &FetchPolicy::default()),
            Err(BunsenError::InvalidArgument { .. })
        ));
    }

    /// `cached_ids` reads what is on disk through the template, ignoring
    /// partials, other files and directories.
    #[test]
    fn test_cached_ids_reads_the_root() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        let set = ShardSet::at_dir(&cache, served(&["https://a.example"], 10), dir.path());

        for name in [
            "shard_07.bin",
            "shard_02.bin",
            "shard_02.bin.partial",
            "notes.txt",
        ] {
            fs::write(dir.path().join(name), b"x").unwrap();
        }
        fs::create_dir(dir.path().join("shard_05.bin")).unwrap();

        assert_eq!(set.cached_ids().unwrap(), vec![ShardId(2), ShardId(7)]);
        assert!(set.is_cached(ShardId(7)));
        assert!(!set.is_cached(ShardId(5)), "a directory is not a shard");
        assert_eq!(
            set.locate(ShardId(7), false).unwrap(),
            dir.path().join("shard_07.bin")
        );
    }

    /// `fetch` brings one shard in off the first live mirror; `fetch_many`
    /// brings the rest in under the policy and reports the one on disk as
    /// cached.
    #[test]
    fn test_fetch_and_fetch_many() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        let dead = refused_url("");
        let live = serve_n("", b"abc", 3);
        let bases: &'static [&'static str] = Box::leak(
            vec![
                Box::leak(dead.into_boxed_str()) as &str,
                Box::leak(live.into_boxed_str()) as &str,
            ]
            .into_boxed_slice(),
        );
        let set = ShardSet::in_cache(&cache, served(bases, 3));

        let path = set.fetch(ShardId(1)).unwrap();
        assert_eq!(path, set.path(ShardId(1)));
        assert_eq!(fs::read(&path).unwrap(), b"abc");
        assert_eq!(set.locate(ShardId(1), false).unwrap(), path);

        let ids = set
            .descriptor()
            .select(&[burn::tensor::Slice::from(..)])
            .unwrap();
        let jobs = set.jobs(&ids).unwrap();
        assert_eq!(jobs.len(), 3);
        assert_eq!(jobs[0].urls.len(), 2);
        assert_eq!(jobs[0].sha256, None);

        let policy = FetchPolicy::default()
            .with_parallel(2)
            .with_on_failure(OnFailure::Continue);
        let report = set.fetch_many(&ids, &policy).unwrap();
        assert!(
            matches!(report.outcomes[1], FetchOutcome::Cached(_)),
            "{report}"
        );
        assert!(report.is_complete(), "{report}");
        assert_eq!(
            report.paths().unwrap(),
            ids.iter().map(|&id| set.path(id)).collect::<Vec<_>>()
        );
        assert_eq!(set.cached_ids().unwrap(), ids);
        let _ = ABC_SHA256;
    }
}
