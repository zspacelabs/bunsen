//! Collecting a resource cache's garbage: what `cache gc` removes, ranked.
use alloc::{
    collections::BTreeMap,
    fmt,
};
use core::time::Duration;
use std::{
    path::Path,
    time::SystemTime,
};

use crate::{
    data::resources::{
        CONVERTED_DIR,
        CacheSelector,
        CachedFile,
        ConvertedManifest,
        ResourceCache,
    },
    errors::BunsenResult,
};

/// [`CacheGcOptions::max_age`]'s default: a week.
pub const DEFAULT_GC_MAX_AGE: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// What [`ResourceCache::gc_plan`] collects, besides unfinished downloads,
/// which it always does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheGcOptions {
    /// Collect files not loaded for this long (by their last use, the
    /// modification time a load sets); [`DEFAULT_GC_MAX_AGE`] by default,
    /// none when `None`. A file whose last use the file system does not
    /// record is kept.
    pub max_age: Option<Duration>,

    /// Then, while the cache holds more than this many bytes, collect what
    /// is left, least recently used first; no bound when `None` (the
    /// default).
    pub max_cache: Option<u64>,

    /// Collect a cached resource a converted copy shadows: one that every
    /// current copy's manifest listing it ([`ConvertedManifest::shadows`])
    /// is newer than, so no load has read it since those copies were made or
    /// last used. On by default.
    pub shadowed: bool,

    /// Collect converted copies no load of this bunsen reads: those whose
    /// manifest is not current ([`ConvertedManifest::is_current`]), or that
    /// have none (written before manifests); and a manifest whose copy is
    /// gone. On by default.
    pub stale_converted: bool,
}

impl Default for CacheGcOptions {
    fn default() -> Self {
        Self {
            max_age: Some(DEFAULT_GC_MAX_AGE),
            max_cache: None,
            shadowed: true,
            stale_converted: true,
        }
    }
}

impl CacheGcOptions {
    /// Sets [`max_age`](Self::max_age).
    pub fn with_max_age(
        mut self,
        max_age: Option<Duration>,
    ) -> Self {
        self.max_age = max_age;
        self
    }

    /// Sets [`max_cache`](Self::max_cache).
    pub fn with_max_cache(
        mut self,
        max_cache: Option<u64>,
    ) -> Self {
        self.max_cache = max_cache;
        self
    }

    /// Sets [`shadowed`](Self::shadowed).
    pub fn with_shadowed(
        mut self,
        shadowed: bool,
    ) -> Self {
        self.shadowed = shadowed;
        self
    }

    /// Sets [`stale_converted`](Self::stale_converted).
    pub fn with_stale_converted(
        mut self,
        stale_converted: bool,
    ) -> Self {
        self.stale_converted = stale_converted;
        self
    }
}

/// Why [`ResourceCache::gc_plan`] collects a file, in rank order: a file is
/// given the first that holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GcReason {
    /// An unfinished download, or an unfinished converted copy.
    Partial,

    /// A converted copy, or its manifest, that no load of this bunsen reads
    /// ([`CacheGcOptions::stale_converted`]).
    StaleConverted,

    /// Not loaded within [`CacheGcOptions::max_age`].
    Aged,

    /// Shadowed by current converted copies ([`CacheGcOptions::shadowed`]).
    Shadowed,

    /// Least recently used while the cache is over
    /// [`CacheGcOptions::max_cache`].
    OverMaxCache,
}

impl fmt::Display for GcReason {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(match self {
            Self::Partial => "partial",
            Self::StaleConverted => "stale converted",
            Self::Aged => "aged",
            Self::Shadowed => "shadowed",
            Self::OverMaxCache => "over max cache",
        })
    }
}

/// A file [`ResourceCache::gc_plan`] collects, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GcCandidate {
    /// The file.
    pub file: CachedFile,

    /// Why it goes.
    pub reason: GcReason,
}

/// What [`ResourceCache::gc_plan`] would remove, and what it keeps.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GcPlan {
    /// The files to remove, by reason, then path.
    pub remove: Vec<GcCandidate>,

    /// The files kept, in path order.
    pub keep: Vec<CachedFile>,
}

impl GcPlan {
    /// The files to remove, for
    /// [`ResourceCache::remove_cached`].
    pub fn files(&self) -> Vec<CachedFile> {
        self.remove.iter().map(|c| c.file.clone()).collect()
    }
}

/// Files that go together: a converted copy and its manifest, or one file.
struct Unit {
    files: Vec<CachedFile>,
    /// The copy's last use for a copy, else the file's.
    last_used: Option<SystemTime>,
    kind: UnitKind,
}

enum UnitKind {
    Partial,
    Resource,
    Converted {
        /// The copy's manifest, when the copy is here and current.
        current: Option<ConvertedManifest>,
        /// When the manifest was last touched (with the copy).
        manifest_used: Option<SystemTime>,
    },
}

impl Unit {
    fn bytes(&self) -> u64 {
        self.files.iter().map(|f| f.bytes).sum()
    }
}

/// Whether `file` is a converted copy or its manifest:
/// `<build>/converted/<key>/<kit>.bpk[.json]`.
fn is_converted(file: &CachedFile) -> bool {
    file.name
        .rsplit_once('/')
        .is_some_and(|(_, last)| last == CONVERTED_DIR)
        && (file.file.ends_with(".bpk") || file.file.ends_with(".bpk.json"))
}

impl ResourceCache {
    /// What `cache gc` removes from the whole cache under `options`, judged
    /// at `now`, in rank order ([`GcReason`]):
    ///
    /// 1. unfinished downloads and copies, always;
    /// 2. stale converted copies and stray manifests;
    /// 3. files not loaded within the maximum age;
    /// 4. resources current converted copies shadow;
    /// 5. then, least recently used first, what keeps the cache over its
    ///    maximum size.
    ///
    /// A converted copy and its manifest go together, by the copy's last use.
    /// Only the cache's own files are judged: a local directory's or a given
    /// path's never are. Nothing is removed: pass the plan's
    /// [`files`](GcPlan::files) to [`remove_cached`](Self::remove_cached).
    ///
    /// # Errors
    /// As [`cached_files`](Self::cached_files).
    pub fn gc_plan(
        &self,
        options: &CacheGcOptions,
        now: SystemTime,
    ) -> BunsenResult<GcPlan> {
        let mut units = units(self.cached_files(&CacheSelector::default())?);
        let mut reasons: Vec<Option<GcReason>> = vec![None; units.len()];

        let aged = |used: Option<SystemTime>| {
            options.max_age.is_some_and(|age| {
                used.and_then(|u| now.duration_since(u).ok())
                    .is_some_and(|idle| idle >= age)
            })
        };
        for (unit, reason) in units.iter().zip(reasons.iter_mut()) {
            *reason = match &unit.kind {
                UnitKind::Partial => Some(GcReason::Partial),
                UnitKind::Converted { current: None, .. } if options.stale_converted => {
                    Some(GcReason::StaleConverted)
                }
                _ if aged(unit.last_used) => Some(GcReason::Aged),
                _ => None,
            };
        }

        if options.shadowed {
            // Each shadowed resource's file, and when each current copy
            // listing it was last used, from the copies that stay.
            let mut listed: BTreeMap<(&str, &str, &str), Vec<Option<SystemTime>>> = BTreeMap::new();
            for (unit, reason) in units.iter().zip(&reasons) {
                if let (
                    None,
                    UnitKind::Converted {
                        current: Some(manifest),
                        manifest_used,
                    },
                ) = (reason, &unit.kind)
                {
                    for s in &manifest.shadows {
                        listed
                            .entry((&s.name, &s.pin, &s.file))
                            .or_default()
                            .push(*manifest_used);
                    }
                }
            }
            let shadowed: Vec<bool> = units
                .iter()
                .zip(&reasons)
                .map(|(unit, reason)| {
                    let (None, UnitKind::Resource, [file]) =
                        (reason, &unit.kind, unit.files.as_slice())
                    else {
                        return false;
                    };
                    let Some(used) = file.last_used else {
                        return false;
                    };
                    listed
                        .get(&(file.name.as_str(), file.pin.as_str(), file.file.as_str()))
                        .is_some_and(|times| times.iter().all(|t| t.is_some_and(|t| t > used)))
                })
                .collect();
            for (reason, shadowed) in reasons.iter_mut().zip(shadowed) {
                if shadowed {
                    *reason = Some(GcReason::Shadowed);
                }
            }
        }

        if let Some(max) = options.max_cache {
            let mut kept: Vec<usize> = (0..units.len()).filter(|&i| reasons[i].is_none()).collect();
            let mut total: u64 = kept.iter().map(|&i| units[i].bytes()).sum();
            // Least recently used first; an unrecorded last use, last.
            kept.sort_by_key(|&i| (units[i].last_used.is_none(), units[i].last_used));
            for i in kept {
                if total <= max {
                    break;
                }
                total -= units[i].bytes();
                reasons[i] = Some(GcReason::OverMaxCache);
            }
        }

        let mut plan = GcPlan::default();
        for (unit, reason) in units.drain(..).zip(reasons) {
            match reason {
                Some(reason) => plan.remove.extend(
                    unit.files
                        .into_iter()
                        .map(|file| GcCandidate { file, reason }),
                ),
                None => plan.keep.extend(unit.files),
            }
        }
        plan.remove
            .sort_by(|a, b| (a.reason, &a.file.path).cmp(&(b.reason, &b.file.path)));
        plan.keep.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(plan)
    }
}

/// `files` grouped into what goes together.
fn units(files: Vec<CachedFile>) -> Vec<Unit> {
    let mut units = Vec::new();
    // A converted copy and its manifest, by the copy's path.
    let mut copies: BTreeMap<std::path::PathBuf, Vec<CachedFile>> = BTreeMap::new();
    for file in files {
        if file.partial {
            units.push(Unit {
                last_used: file.last_used,
                files: vec![file],
                kind: UnitKind::Partial,
            });
        } else if is_converted(&file) {
            let copy = match file.file.strip_suffix(".json") {
                Some(_) => file.path.with_extension(""),
                None => file.path.clone(),
            };
            copies.entry(copy).or_default().push(file);
        } else {
            units.push(Unit {
                last_used: file.last_used,
                files: vec![file],
                kind: UnitKind::Resource,
            });
        }
    }
    for (copy, files) in copies {
        let bpk = files.iter().find(|f| f.path == copy);
        let manifest_file = files.iter().find(|f| f.path != copy);
        let current = bpk
            .and(manifest_file)
            .and_then(|_| ConvertedManifest::read(Path::new(&copy)).ok().flatten())
            .filter(ConvertedManifest::is_current);
        let last_used = bpk.or(manifest_file).and_then(|f| f.last_used);
        let manifest_used = manifest_file.and_then(|f| f.last_used);
        units.push(Unit {
            files,
            last_used,
            kind: UnitKind::Converted {
                current,
                manifest_used,
            },
        });
    }
    units
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::data::{
        cache::BunsenDiskCacheOptions,
        resources::{
            CONVERTED_WEIGHTS_REVISION,
            ResourceCacheOptions,
            ShadowedResource,
            converted_manifest_path,
        },
    };

    const DAY: Duration = Duration::from_secs(24 * 60 * 60);

    fn cache_in(dir: &Path) -> ResourceCache {
        ResourceCache::new(
            ResourceCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap()
    }

    /// Writes `bytes` bytes at `rel` under the resources directory, last
    /// used `days` days before `now`.
    fn put(
        cache: &ResourceCache,
        rel: &str,
        bytes: usize,
        now: SystemTime,
        days: u32,
    ) -> std::path::PathBuf {
        let path = cache.resources_dir().join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, vec![0u8; bytes]).unwrap();
        touch(&path, now - DAY * days);
        path
    }

    fn touch(
        path: &Path,
        when: SystemTime,
    ) {
        fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(when)
            .unwrap();
    }

    /// A converted copy of `shadows` at `rel`, with a manifest written by
    /// `bunsen`, both last used `days` days before `now`.
    fn put_copy(
        cache: &ResourceCache,
        rel: &str,
        bunsen: &str,
        shadows: &[(&str, &str, &str)],
        now: SystemTime,
        days: u32,
    ) {
        let copy = put(cache, rel, 100, now, days);
        ConvertedManifest {
            revision: CONVERTED_WEIGHTS_REVISION,
            bunsen: bunsen.to_string(),
            kit: "toy".to_string(),
            key: "k".to_string(),
            facets: "policy x".to_string(),
            shadows: shadows
                .iter()
                .map(|(name, pin, file)| ShadowedResource {
                    name: name.to_string(),
                    pin: pin.to_string(),
                    file: file.to_string(),
                })
                .collect(),
        }
        .write(&copy)
        .unwrap();
        touch(&converted_manifest_path(&copy), now - DAY * days);
    }

    fn plan(
        cache: &ResourceCache,
        options: &CacheGcOptions,
        now: SystemTime,
    ) -> Vec<(String, GcReason)> {
        let root = cache.resources_dir();
        cache
            .gc_plan(options, now)
            .unwrap()
            .remove
            .into_iter()
            .map(|c| {
                (
                    c.file
                        .path
                        .strip_prefix(&root)
                        .unwrap()
                        .display()
                        .to_string(),
                    c.reason,
                )
            })
            .collect()
    }

    /// Each rule on its own cache: partials always go, then stale copies, aged
    /// files, shadowed resources, and the least recently used over the size
    /// bound; each optional rule can be turned off.
    #[test]
    fn test_gc_plan() {
        use GcReason::*;
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        let now = SystemTime::now();
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        let c = "c".repeat(64);
        let k = "e".repeat(64);
        let old = "f".repeat(64);
        let here = env!("CARGO_PKG_VERSION");

        // The checkpoint was read 2 days ago; its current copy was used 1 day
        // ago. The config was read since: nothing shadows it anyway.
        put(&cache, &format!("m/checkpoint/{a}/w.bin"), 1000, now, 2);
        put(&cache, &format!("m/config/{b}/config.json"), 10, now, 0);
        put_copy(
            &cache,
            &format!("m/converted/{k}/toy.bpk"),
            here,
            &[("m/checkpoint", &a, "w.bin")],
            now,
            1,
        );
        // Another version's copy, and a copy from before manifests.
        put_copy(
            &cache,
            &format!("m/converted/{old}/toy.bpk"),
            "0.0.1",
            &[],
            now,
            0,
        );
        put(&cache, &format!("n/converted/{old}/toy.bpk"), 100, now, 0);
        // A download in flight, and a file not loaded in 9 days.
        put(
            &cache,
            &format!("n/checkpoint/{c}/w.bin.partial"),
            5,
            now,
            0,
        );
        put(&cache, &format!("o/checkpoint/{c}/w.bin"), 50, now, 9);

        let stale_m = (format!("m/converted/{old}/toy.bpk"), StaleConverted);
        let stale_m_json = (format!("m/converted/{old}/toy.bpk.json"), StaleConverted);
        let stale_n = (format!("n/converted/{old}/toy.bpk"), StaleConverted);
        let partial = (format!("n/checkpoint/{c}/w.bin.partial"), Partial);
        let aged = (format!("o/checkpoint/{c}/w.bin"), Aged);
        let shadowed = (format!("m/checkpoint/{a}/w.bin"), Shadowed);
        assert_eq!(
            plan(&cache, &CacheGcOptions::default(), now),
            [
                partial.clone(),
                stale_m.clone(),
                stale_m_json.clone(),
                stale_n.clone(),
                aged.clone(),
                shadowed.clone(),
            ]
        );

        assert_eq!(
            plan(
                &cache,
                &CacheGcOptions::default()
                    .with_max_age(None)
                    .with_shadowed(false)
                    .with_stale_converted(false),
                now
            ),
            [partial.clone()],
            "partials always go"
        );

        // A load of the checkpoint since the copy was used keeps it.
        touch(&cache.resources_dir().join(&shadowed.0), now);
        assert_eq!(
            plan(&cache, &CacheGcOptions::default(), now),
            [
                partial.clone(),
                stale_m.clone(),
                stale_m_json.clone(),
                stale_n.clone(),
                aged.clone(),
            ]
        );
    }

    /// Over the size bound, the least recently used goes first, a copy with
    /// its manifest, until the rest fits.
    #[test]
    fn test_gc_max_cache() {
        use GcReason::*;
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        let now = SystemTime::now();
        let a = "a".repeat(64);
        let k = "e".repeat(64);
        put(&cache, &format!("x/checkpoint/{a}/w.bin"), 100, now, 4);
        put_copy(
            &cache,
            &format!("y/converted/{k}/toy.bpk"),
            env!("CARGO_PKG_VERSION"),
            &[],
            now,
            3,
        );
        put(&cache, &format!("z/checkpoint/{a}/w.bin"), 100, now, 2);
        put(&cache, &format!("w/checkpoint/{a}/w.bin"), 100, now, 1);
        let options = CacheGcOptions::default()
            .with_max_age(None)
            .with_max_cache(Some(250));
        assert_eq!(
            plan(&cache, &options, now),
            [
                (format!("x/checkpoint/{a}/w.bin"), OverMaxCache),
                (format!("y/converted/{k}/toy.bpk"), OverMaxCache),
                (format!("y/converted/{k}/toy.bpk.json"), OverMaxCache),
            ]
        );
        let kept: u64 = cache
            .gc_plan(&options, now)
            .unwrap()
            .keep
            .iter()
            .map(|f| f.bytes)
            .sum();
        assert_eq!(kept, 200);
    }

    /// A removed plan leaves the cache what it kept.
    #[test]
    fn test_gc_removes_the_plan() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        let now = SystemTime::now();
        let a = "a".repeat(64);
        put(&cache, &format!("m/checkpoint/{a}/w.bin"), 10, now, 30);
        put(&cache, &format!("m/config/{a}/config.json"), 10, now, 0);
        let plan = cache.gc_plan(&CacheGcOptions::default(), now).unwrap();
        assert_eq!(cache.remove_cached(&plan.files()).unwrap(), 10);
        assert_eq!(
            cache.cached_files(&CacheSelector::default()).unwrap(),
            plan.keep
        );
    }
}
