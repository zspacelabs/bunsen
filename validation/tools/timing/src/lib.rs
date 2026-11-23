//! # `cargo timing`: bunsen's decodes, timed
//!
//! Times each architecture's decode on the validation catalog's checkpoints,
//! phase by phase, through its
//! [`TimedDecode`](bunsen::kits::support::timing::TimedDecode): a cold
//! prefill, warm prefills, a cold first step, then warm steps by cache
//! position. Each phase is a [`Device::profile`](burn::tensor::Device::profile)
//! window, closed on a read of the phase's last token (as a decode reads its
//! token), and timed twice: on the host's wall clock (what a caller waits)
//! and on the device's clock (what the device works); the gap between them is
//! the host's share. Statistics are cubecl's
//! ([`BenchmarkComputations`](cubecl_common::benchmark::BenchmarkComputations)).
//!
//! - `cargo timing run`: the models a profile chooses (`validation/models.yml`
//!   and `validation/profiles/`, as `cargo validate` chooses), built for one
//!   backend (`--backend`, `--fusion`) and timed in each dtype; one
//!   [`TimingRecord`] per model and dtype, as a JSON line.
//! - `cargo timing compare a.jsonl b.jsonl`: two runs side by side, model by
//!   model and phase by phase, whatever they differ in (commits, backends,
//!   dtypes, fusion, storage).
//!
//! Timing is on demand, never a check: it reports. Run it under the device
//! lock, one heavy run at a time.

pub mod compare;
pub mod record;

pub use record::*;
