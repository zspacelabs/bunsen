//! Two runs' records side by side: `cargo timing compare`.

use std::path::Path;

use indexmap::IndexMap;

use crate::TimingRecord;

/// The records of a run's JSON-lines file, in order.
pub fn read_run(path: &Path) -> Result<Vec<TimingRecord>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(line).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))
        })
        .collect()
}

/// One phase of a model in both runs: `b / a`, so above 1 is slower in `b`.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub phase: &'static str,
    pub a_ms: f64,
    pub b_ms: f64,
}

impl Row {
    /// `b / a`: above 1, `b` is slower.
    pub fn ratio(&self) -> f64 {
        self.b_ms / self.a_ms
    }
}

/// The phases two records of one model share, by wall time (medians for the
/// warm phases).
pub fn rows(
    a: &TimingRecord,
    b: &TimingRecord,
) -> Vec<Row> {
    let row = |phase, a_ms, b_ms| Row { phase, a_ms, b_ms };
    vec![
        row("load s", a.load_s, b.load_s),
        row(
            "prefill cold",
            a.prefill_cold.wall_ms,
            b.prefill_cold.wall_ms,
        ),
        row(
            "prefill warm",
            a.prefill_warm.wall.median_ms,
            b.prefill_warm.wall.median_ms,
        ),
        row("step cold", a.step_cold.wall_ms, b.step_cold.wall_ms),
        row(
            "step warm",
            a.step_warm.wall.median_ms,
            b.step_warm.wall.median_ms,
        ),
    ]
}

/// Prints `a` and `b` side by side: each model both runs timed (matched by
/// model, dtype and the shape of the work), phase by phase, and the models
/// only one of them has.
pub fn print_compare(
    a: &[TimingRecord],
    b: &[TimingRecord],
) {
    let index = |run: &[TimingRecord]| -> IndexMap<_, TimingRecord> {
        run.iter().map(|r| (r.key(), r.clone())).collect()
    };
    let (a, b) = (index(a), index(b));
    for (key, ra) in &a {
        let Some(rb) = b.get(key) else {
            println!("{} {}: only in a", ra.build, ra.dtype);
            continue;
        };
        println!(
            "{} {} rows={} prefill={} steps={}",
            ra.build, ra.dtype, ra.rows, ra.prefill, ra.steps
        );
        println!("  a: [{}] {} {}", ra.backend, ra.storage, ra.label);
        println!("  b: [{}] {} {}", rb.backend, rb.storage, rb.label);
        println!(
            "  {:<14} {:>10} {:>10} {:>8}",
            "phase", "a ms", "b ms", "b / a"
        );
        for r in rows(ra, rb) {
            println!(
                "  {:<14} {:>10.2} {:>10.2} {:>8.3}",
                r.phase,
                r.a_ms,
                r.b_ms,
                r.ratio()
            );
        }
        println!(
            "  {:<14} {:>10.1} {:>10.1} {:>8.3}",
            format!("{}/s", ra.unit),
            ra.decode_per_s(),
            rb.decode_per_s(),
            rb.decode_per_s() / ra.decode_per_s()
        );
    }
    for (key, rb) in &b {
        if !a.contains_key(key) {
            println!("{} {}: only in b", rb.build, rb.dtype);
        }
    }
}
