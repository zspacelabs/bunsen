//! `--cpus`: the CPUs the process's threads run on.
//!
//! A decode on an accelerator keeps two host threads busy, the caller's and the
//! runtime's device thread, handing each kernel launch from one to the other.
//! On a CPU of several L3 caches (AMD's CCXs, a multi-socket host), the
//! scheduler may place them under different ones, and every hand-off crosses
//! the interconnect: a Qwen3-0.6B decode step on CUDA (a Threadripper 3990X)
//! takes ~28 ms unpinned and ~16 ms with both threads under one L3.
//!
//! [`Placement::Auto`], the default, pins the process to the CPUs that share
//! an L3 with the one it starts on, when its device is an accelerator. On a CPU
//! with a single L3, that is every CPU the process may use, and nothing
//! changes. The CPU backend is never pinned: it computes on every core.
//!
//! Pinning is Linux's (`sched_setaffinity`, the topology from
//! `/sys/devices/system/cpu`). Elsewhere, `auto` and `all` leave the threads
//! where they are, and a list is [`Unsupported`](bunsen::errors::BunsenErrorKind::Unsupported).
use std::{
    fmt,
    str::FromStr,
};

use bunsen::errors::{
    BunsenError,
    BunsenResult,
};

/// A set of CPUs, as Linux lists them: `0-3,64-67`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuList {
    cpus: Vec<usize>,
}

impl CpuList {
    /// The CPUs, ascending and distinct.
    pub fn new(cpus: impl IntoIterator<Item = usize>) -> Self {
        let mut cpus: Vec<usize> = cpus.into_iter().collect();
        cpus.sort_unstable();
        cpus.dedup();
        Self { cpus }
    }

    /// The CPUs, ascending.
    pub fn cpus(&self) -> &[usize] {
        &self.cpus
    }

    /// Whether it lists no CPU.
    pub fn is_empty(&self) -> bool {
        self.cpus.is_empty()
    }

    /// The CPUs in both.
    pub fn intersection(
        &self,
        other: &CpuList,
    ) -> CpuList {
        CpuList::new(self.cpus.iter().copied().filter(|c| other.cpus.contains(c)))
    }
}

impl FromStr for CpuList {
    type Err = String;

    /// Comma-separated CPUs and inclusive ranges: `0-3,64-67`, `5`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut cpus = Vec::new();
        for part in s.trim().split(',') {
            let cpu = |t: &str| {
                t.trim()
                    .parse::<usize>()
                    .map_err(|_| format!("`{part}` is not a CPU or a range of them"))
            };
            match part.split_once('-') {
                Some((lo, hi)) => {
                    let (lo, hi) = (cpu(lo)?, cpu(hi)?);
                    if lo > hi {
                        return Err(format!("`{part}` is an empty range"));
                    }
                    cpus.extend(lo..=hi);
                }
                None => cpus.push(cpu(part)?),
            }
        }
        Ok(CpuList::new(cpus))
    }
}

impl fmt::Display for CpuList {
    /// As Linux lists them: runs as ranges, `0-3,64-67`.
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let mut runs = Vec::new();
        let mut cpus = self.cpus.iter().copied().peekable();
        while let Some(lo) = cpus.next() {
            let mut hi = lo;
            while cpus.peek() == Some(&(hi + 1)) {
                hi = cpus.next().unwrap_or(hi);
            }
            runs.push(if lo == hi {
                lo.to_string()
            } else {
                format!("{lo}-{hi}")
            });
        }
        f.write_str(&runs.join(","))
    }
}

/// Where the process's threads run: `--cpus auto|all|<list>`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Placement {
    /// For an accelerator, the CPUs sharing an L3 with the one the process
    /// starts on; the threads stay where they are for the CPU backend.
    #[default]
    Auto,

    /// Wherever the scheduler puts them: the threads are not moved.
    All,

    /// These CPUs (Linux's list form, `0-3,64-67`).
    Cpus(CpuList),
}

impl FromStr for Placement {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "auto" => Ok(Placement::Auto),
            "all" => Ok(Placement::All),
            list => list.parse().map(Placement::Cpus),
        }
    }
}

impl fmt::Display for Placement {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Placement::Auto => f.write_str("auto"),
            Placement::All => f.write_str("all"),
            Placement::Cpus(list) => list.fmt(f),
        }
    }
}

impl Placement {
    /// Pins every thread of the process as the placement says, for a device
    /// that is an accelerator or not; the CPUs pinned to, when it pins.
    ///
    /// Threads started later inherit their starter's CPUs, so a device's
    /// runtime thread, started by its first work, runs where the process was
    /// pinned.
    ///
    /// `auto` is best effort: a host whose topology it cannot read stays as
    /// it is.
    ///
    /// # Errors
    /// For a list: [`Unsupported`](bunsen::errors::BunsenErrorKind::Unsupported) off Linux;
    /// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) when it names no CPU the process
    /// may use; [`Sys`](bunsen::errors::BunsenErrorKind::Sys) when the host refuses.
    pub fn apply(
        &self,
        accelerated: bool,
    ) -> BunsenResult<Option<CpuList>> {
        match self {
            Placement::All => Ok(None),
            Placement::Auto if !accelerated => Ok(None),
            Placement::Auto => Ok(auto()),
            Placement::Cpus(list) => pin_list(list).map(Some),
        }
    }
}

#[cfg(target_os = "linux")]
fn auto() -> Option<CpuList> {
    let allowed = linux::allowed().ok()?;
    let shared = linux::l3_of(linux::current_cpu()?)?.intersection(&allowed);
    if shared.is_empty() || shared == allowed {
        return None;
    }
    match linux::pin(&shared) {
        Ok(()) => {
            log::info!("pinned to CPUs {shared}: one L3 cache (`--cpus all` to not pin)");
            Some(shared)
        }
        Err(e) => {
            log::debug!("not pinned to CPUs {shared}: {e}");
            None
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn auto() -> Option<CpuList> {
    None
}

#[cfg(target_os = "linux")]
fn pin_list(list: &CpuList) -> BunsenResult<CpuList> {
    let allowed = linux::allowed().map_err(bunsen::errors::sys_op("read the CPUs allowed"))?;
    let cpus = list.intersection(&allowed);
    if cpus.is_empty() {
        return Err(BunsenError::illegal(format!(
            "--cpus {list}: none of them is one this process may use ({allowed})"
        )));
    }
    linux::pin(&cpus).map_err(bunsen::errors::sys_op("pin the process's threads"))?;
    log::info!("pinned to CPUs {cpus}");
    Ok(cpus)
}

#[cfg(not(target_os = "linux"))]
fn pin_list(list: &CpuList) -> BunsenResult<CpuList> {
    Err(BunsenError::unsupported(format!(
        "--cpus {list}: pinning is Linux's"
    )))
}

#[cfg(target_os = "linux")]
mod linux {
    use std::{
        fs,
        io,
        mem,
    };

    use super::CpuList;

    /// The CPUs the calling thread may run on.
    pub fn allowed() -> io::Result<CpuList> {
        // SAFETY: a zeroed `cpu_set_t` is an empty set, and the kernel writes
        // at most `size_of::<cpu_set_t>()` bytes into it.
        let set = unsafe {
            let mut set: libc::cpu_set_t = mem::zeroed();
            if libc::sched_getaffinity(0, mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
                return Err(io::Error::last_os_error());
            }
            set
        };
        let size = 8 * mem::size_of::<libc::cpu_set_t>();
        // SAFETY: every index is below the set's size.
        Ok(CpuList::new(
            (0..size).filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &set) }),
        ))
    }

    /// The CPU the calling thread is on.
    pub fn current_cpu() -> Option<usize> {
        // SAFETY: no arguments; -1 on failure.
        usize::try_from(unsafe { libc::sched_getcpu() }).ok()
    }

    /// The CPUs sharing `cpu`'s L3 cache, from sysfs.
    pub fn l3_of(cpu: usize) -> Option<CpuList> {
        let caches = fs::read_dir(format!("/sys/devices/system/cpu/cpu{cpu}/cache")).ok()?;
        for cache in caches.flatten() {
            let path = cache.path();
            // Not every entry is a cache (`uevent`).
            let Ok(level) = fs::read_to_string(path.join("level")) else {
                continue;
            };
            if level.trim() == "3" {
                return fs::read_to_string(path.join("shared_cpu_list"))
                    .ok()?
                    .parse()
                    .ok();
            }
        }
        None
    }

    /// Pins every thread of the process to `cpus`.
    pub fn pin(cpus: &CpuList) -> io::Result<()> {
        // SAFETY: a zeroed `cpu_set_t` is an empty set; only CPUs below its
        // size are added.
        let set = unsafe {
            let mut set: libc::cpu_set_t = mem::zeroed();
            for &cpu in cpus.cpus() {
                if cpu < 8 * mem::size_of::<libc::cpu_set_t>() {
                    libc::CPU_SET(cpu, &mut set);
                }
            }
            set
        };
        for task in fs::read_dir("/proc/self/task")?.flatten() {
            let Some(tid) = task.file_name().to_str().and_then(|t| t.parse().ok()) else {
                continue;
            };
            // SAFETY: the set is initialized and its size is given. A thread
            // that ended since the listing reports ESRCH, which is skipped.
            if unsafe { libc::sched_setaffinity(tid, mem::size_of::<libc::cpu_set_t>(), &set) } != 0
            {
                let e = io::Error::last_os_error();
                if e.raw_os_error() != Some(libc::ESRCH) {
                    return Err(e);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_list_round_trip() {
        let list: CpuList = "64-67,0-3,5,3".parse().unwrap();
        assert_eq!(list.cpus(), [0, 1, 2, 3, 5, 64, 65, 66, 67]);
        assert_eq!(list.to_string(), "0-3,5,64-67");
        assert!("3-1".parse::<CpuList>().is_err());
        assert!("x".parse::<CpuList>().is_err());
        assert_eq!(
            CpuList::new([0, 1, 2]).intersection(&CpuList::new([1, 2, 9])),
            CpuList::new([1, 2])
        );
    }

    #[test]
    fn test_placement_parse() {
        assert_eq!("auto".parse::<Placement>(), Ok(Placement::Auto));
        assert_eq!("all".parse::<Placement>(), Ok(Placement::All));
        assert_eq!(
            "0-1".parse::<Placement>(),
            Ok(Placement::Cpus(CpuList::new([0, 1])))
        );
        assert_eq!(Placement::default().to_string(), "auto");
    }

    /// The CPU backend is never pinned, and `all` never moves a thread.
    #[test]
    fn test_unpinned() {
        assert_eq!(Placement::Auto.apply(false).unwrap(), None);
        assert_eq!(Placement::All.apply(true).unwrap(), None);
    }

    /// The topology this host reports holds the CPU asked about, when the
    /// host reports one (a container may hide sysfs).
    #[cfg(target_os = "linux")]
    #[test]
    fn test_l3_holds_its_cpu() {
        let cpu = linux::current_cpu().expect("the current CPU");
        let has_l3 = std::fs::read_dir(format!("/sys/devices/system/cpu/cpu{cpu}/cache"))
            .into_iter()
            .flatten()
            .flatten()
            .any(|e| {
                std::fs::read_to_string(e.path().join("level")).is_ok_and(|l| l.trim() == "3")
            });
        match linux::l3_of(cpu) {
            Some(l3) => assert!(l3.cpus().contains(&cpu)),
            None => assert!(!has_l3, "an L3 the lookup missed"),
        }
    }
}
