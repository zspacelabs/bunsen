//! [`TimingRecord`]: one model's timing in one dtype, as a run writes it.

use std::time::Duration;

use cubecl_common::{
    benchmark::{
        BenchmarkComputations,
        BenchmarkDurations,
    },
    profile::TimingMethod,
};
use serde::{
    Deserialize,
    Serialize,
};

/// One model's timing in one dtype: what it ran on, and each phase.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimingRecord {
    /// The validation crate's name in `models.yml` (`qwen3`).
    pub krate: String,

    /// The model's name in `models.yml` (`qwen3-0.6b`).
    pub model: String,

    /// The catalog build it loaded (`qwen/qwen3-0.6b`).
    pub build: String,

    /// `f32` or `bf16`.
    pub dtype: String,

    /// The backend stack the device dispatches to
    /// (`dispatch<fusion<cubecl<cuda>>>`): which runtime, and whether fusion
    /// was on.
    pub backend: String,

    /// The KV cache storage asked for, and what it resolved to on the device.
    pub storage: String,

    /// What the run was labeled (a commit, a note).
    pub label: String,

    /// Rows decoded at once.
    pub rows: usize,

    /// Positions each row's prompt prefills.
    pub prefill: usize,

    /// Steps each row decodes after the prompt.
    pub steps: usize,

    /// What a step makes (`token`).
    pub unit: String,

    /// The load, fetching included when the checkpoint was not cached, in
    /// seconds.
    pub load_s: f64,

    /// The first prefill, on cold kernels.
    pub prefill_cold: Window,

    /// The prefills after it, each into a fresh state.
    pub prefill_warm: Phase,

    /// The first step after the cold prefill.
    pub step_cold: Window,

    /// Every step after it.
    pub step_warm: Phase,

    /// The warm steps in contiguous groups by cache position: how a step's
    /// cost grows with what the cache holds.
    pub step_buckets: Vec<Bucket>,

    /// Every warm step's wall time, in ms, in order: step `i` ran with
    /// `prefill + 1 + i` positions cached (where an outlier falls). Empty
    /// in a record written before runs kept it.
    #[serde(default)]
    pub step_wall_ms: Vec<f64>,

    /// The device memory pool at its largest across the phases, if the
    /// backend reports one.
    pub memory: Option<Memory>,
}

impl TimingRecord {
    /// Prefill throughput, rows times positions per warm prefill (median),
    /// per second of wall time.
    pub fn prefill_per_s(&self) -> f64 {
        per_s(self.rows * self.prefill, self.prefill_warm.wall.median_ms)
    }

    /// Decode throughput, rows per warm step (median), per second of wall
    /// time.
    pub fn decode_per_s(&self) -> f64 {
        per_s(self.rows, self.step_warm.wall.median_ms)
    }

    /// The key two runs' records are matched by: the model, the dtype, the
    /// shape of the work.
    pub fn key(&self) -> (String, String, String, usize, usize, usize) {
        (
            self.krate.clone(),
            self.model.clone(),
            self.dtype.clone(),
            self.rows,
            self.prefill,
            self.steps,
        )
    }
}

fn per_s(
    units: usize,
    ms: f64,
) -> f64 {
    if ms > 0.0 {
        units as f64 * 1000.0 / ms
    } else {
        f64::NAN
    }
}

/// One timed window: wall time, and device time where the device has a
/// clock (else the same wall time, between syncs).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Window {
    /// Host wall time, from an idle device to the phase's read, in ms.
    pub wall_ms: f64,

    /// Device time, in ms, as burn's profiling window reports it; `None`
    /// when the window carried no measurement. Not pure kernel time: on the
    /// backends measured so far (wgpu, CUDA) it reads above the wall time.
    pub device_ms: Option<f64>,
}

/// Many windows of one phase, summarized.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Phase {
    /// The windows.
    pub count: usize,

    /// Their wall times.
    pub wall: Stats,

    /// Their device times, where every window carried one.
    pub device: Option<Stats>,
}

impl Phase {
    /// `windows` summarized.
    pub fn of(windows: &[Window]) -> Phase {
        let wall = Stats::of(windows.iter().map(|w| w.wall_ms));
        let device = windows
            .iter()
            .map(|w| w.device_ms)
            .collect::<Option<Vec<f64>>>()
            .map(Stats::of);
        Phase {
            count: windows.len(),
            wall,
            device,
        }
    }
}

/// cubecl's statistics of a set of durations, in ms.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Stats {
    pub mean_ms: f64,
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    /// The standard deviation, from cubecl's variance (coarse below ~30 µs).
    pub std_ms: f64,
}

impl Stats {
    /// [`BenchmarkComputations`] of `ms`, in ms; all zero for none.
    pub fn of(ms: impl IntoIterator<Item = f64>) -> Stats {
        let durations: Vec<Duration> = ms
            .into_iter()
            .map(|ms| Duration::from_secs_f64(ms.max(0.0) / 1000.0))
            .collect();
        if durations.is_empty() {
            return Stats {
                mean_ms: 0.0,
                median_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                std_ms: 0.0,
            };
        }
        let c = BenchmarkComputations::new(&BenchmarkDurations::from_durations(
            TimingMethod::System,
            durations,
        ));
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        Stats {
            mean_ms: ms(c.mean),
            median_ms: ms(c.median),
            min_ms: ms(c.min),
            max_ms: ms(c.max),
            // cubecl keeps the variance as a `Duration` of that many seconds
            // squared (rounded to the nanosecond: deviations under ~30 µs
            // read as none).
            std_ms: c.variance.as_secs_f64().sqrt() * 1000.0,
        }
    }
}

/// The warm steps whose cache held positions `from..to` when they ran.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bucket {
    pub from: usize,
    pub to: usize,
    pub wall: Stats,
}

/// The device memory pool, in bytes.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Memory {
    pub bytes_in_use: u64,
    pub bytes_reserved: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// cubecl's statistics, in ms.
    #[test]
    fn test_stats() {
        let s = Stats::of([1.0, 2.0, 3.0, 10.0]);
        assert_eq!((s.min_ms, s.max_ms), (1.0, 10.0));
        assert!((s.mean_ms - 4.0).abs() < 1e-6);
        assert!(s.median_ms >= 2.0 && s.median_ms <= 3.0);
        assert!(s.std_ms > 0.0);
        assert_eq!(Stats::of([]).median_ms, 0.0);
    }

    /// A phase's device statistics need every window's device time.
    #[test]
    fn test_phase() {
        let w = |device_ms| Window {
            wall_ms: 2.0,
            device_ms,
        };
        let full = Phase::of(&[w(Some(1.0)), w(Some(1.0))]);
        assert_eq!(full.count, 2);
        assert!(full.device.is_some());
        assert!(Phase::of(&[w(Some(1.0)), w(None)]).device.is_none());
    }
}
