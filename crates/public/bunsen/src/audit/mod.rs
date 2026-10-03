//! # Audit Probes
//!
//! Record what a computation produced at named checkpoints, then check that
//! another run produces the same thing: on another backend, after a refactor,
//! or against a baseline stored on disk.
//!
//! A unit test asserts what a value *should* be. An audit asserts that a value
//! *has not changed*: you do not need to know the right answer, only that some
//! run you trust produced it. This makes audits a good fit for numerical
//! code whose exact outputs are hard to derive by hand, and for proving that
//! two backends agree.
//!
//! # Concepts
//!
//! * **Checkpoint**: a call such as [`AuditProbe::assert_eq_as`] in the code
//!   under test. It names a value (`"x"`, `"logits"`) and says how to compare
//!   it: exactly, or within a [`TolerancePolicy`] at a given float type.
//! * **Event**: what a checkpoint emits: an [`AuditProbeEventHeader`] (label,
//!   source location, timestamp), [`AuditProbeEventParams`] (the comparison
//!   kind, and its tolerance), and a data map of [`TensorData`]. Events are
//!   passed to handlers as a borrowed [`AuditProbeEventStub`], and stored as an
//!   owned [`AuditProbeEvent`].
//! * **Probe**: an [`AuditProbe`] fans each event out to a list of handlers.
//!   The code under test takes a `&mut AuditProbe` and does not know what the
//!   handlers do.
//! * **Handler**: an [`AuditProbeEventHandler`]. The two that matter are a
//!   *recorder*, which appends each event to a stream, and a *verifier*, which
//!   compares each event to the next one in an expected stream.
//! * **Stream**: the events of one run, in order. A stream can live in memory
//!   ([`AuditStreamRecorder`] / [`AuditStreamVerifier`]) or on disk as a CBOR
//!   [`AuditStreamFile`].
//! * **Body**: an [`AuditBody`] is the code under test, generic over the
//!   backend, so a harness can run it once to record and once to verify.
//! * **Baseline**: a stream stored on disk, under a location the call site
//!   chooses with [`ReportsOptions`]. [`audit_baseline`] records it on the
//!   first run and verifies against it after that.
//!
//! The flow is always the same: run once with a recorder, then run again with
//! a verifier fed from what was recorded.
//!
//! ```text
//!   run A ── checkpoints ──▶ AuditProbe ──▶ recorder ──▶ stream ──┐
//!                                                                 │ (memory or .cbor)
//!   run B ── checkpoints ──▶ AuditProbe ──▶ verifier ◀────────────┘
//!                                              │
//!                                              └──▶ Err on the first mismatch
//! ```
//!
//! # Quick Start
//!
//! This module is built with the `audit` feature, which turns on `testing`. A
//! crate that audits in its tests enables it on its dev-dependency:
//!
//! ```toml
//! [dev-dependencies]
//! bunsen = { version = "*", features = ["audit"] }
//! ```
//!
//! ## Write a body
//!
//! A closure cannot be generic over the backend, so a body is a type with a
//! generic `run`. Put checkpoints wherever an intermediate value is worth
//! pinning down. Inputs must be the same on every backend; use
//! [`seeded_tensor`](crate::support::testing::seeded_tensor) rather than
//! `Tensor::random`.
//!
//! ```
//! # #[cfg(feature = "audit")] {
//! use bunsen::{
//!     audit::{
//!         AuditBody,
//!         AuditProbe,
//!     },
//!     burner::descriptors::TolerancePolicy,
//!     errors::BunsenResult,
//!     support::testing::seeded_tensor,
//! };
//! use burn::{
//!     prelude::Backend,
//!     tensor::Distribution,
//! };
//!
//! struct Softmax;
//!
//! impl AuditBody for Softmax {
//!     fn run<B: Backend>(
//!         &self,
//!         probe: &mut AuditProbe<'_>,
//!         device: &B::Device,
//!     ) -> BunsenResult<()> {
//!         let x =
//!             seeded_tensor::<B, 2>(7, [4, 8], Distribution::Default, device);
//!         // Uploaded host values: bit-identical everywhere.
//!         probe.assert_eq_as::<f32>("x", &x)?;
//!
//!         // Computed values: close, not identical, across backends.
//!         let y = burn::tensor::activation::softmax(x, 1);
//!         probe.assert_approx_eq_as::<f32>(
//!             "softmax(x)",
//!             &y,
//!             TolerancePolicy::Balanced,
//!         )?;
//!         Ok(())
//!     }
//! }
//! # }
//! ```
//!
//! ## Compare two backends
//!
//! [`audit_across`] runs the body on the reference backend `R`, recording,
//! then on the target backend `T`, verifying. Nothing touches the disk.
//!
//! [`PerformanceBackend`](crate::support::testing::PerformanceBackend) is the
//! CPU unless a backend feature is on, so the example below compares the CPU
//! with itself in a bare `cargo test`. Run it with a backend feature (e.g.
//! `--features wgpu`) for the comparison to mean anything; see
//! [Test backends](crate::support::testing#test-backends).
//!
//! ```
//! # #[cfg(feature = "audit")] {
//! # use bunsen::{audit::{AuditBody, AuditProbe}, errors::BunsenResult};
//! # use burn::prelude::Backend;
//! # struct Softmax;
//! # impl AuditBody for Softmax {
//! #     fn run<B: Backend>(&self, _: &mut AuditProbe<'_>, _: &B::Device) -> BunsenResult<()> {
//! #         Ok(())
//! #     }
//! # }
//! use bunsen::{
//!     audit::audit_across,
//!     support::testing::{
//!         CpuBackend,
//!         PerformanceBackend,
//!     },
//! };
//!
//! audit_across::<CpuBackend, PerformanceBackend>(&Softmax).unwrap();
//! # }
//! ```
//!
//! ## Pin a baseline
//!
//! [`audit_baseline`] compares a run with a stream stored on disk. The call
//! site says where the reports live; see [Configuration](#configuration).
//!
//! ```
//! # #[cfg(feature = "audit")] {
//! # use bunsen::{audit::{AuditBody, AuditProbe}, errors::BunsenResult};
//! # use burn::prelude::Backend;
//! # struct Softmax;
//! # impl AuditBody for Softmax {
//! #     fn run<B: Backend>(&self, _: &mut AuditProbe<'_>, _: &B::Device) -> BunsenResult<()> {
//! #         Ok(())
//! #     }
//! # }
//! use bunsen::{
//!     audit::{
//!         BaselineOutcome,
//!         audit_baseline,
//!         reports::ReportsOptions,
//!     },
//!     support::testing::CpuBackend,
//! };
//!
//! let dir = tempfile::tempdir().unwrap();
//! let options = ReportsOptions::new(dir.path());
//!
//! // No baseline yet: this run is recorded.
//! let first = audit_baseline::<CpuBackend>(&options, "softmax", &Softmax).unwrap();
//! assert!(matches!(first, BaselineOutcome::Recorded(_)));
//!
//! // From now on, runs are verified against it.
//! let second = audit_baseline::<CpuBackend>(&options, "softmax", &Softmax).unwrap();
//! assert!(matches!(second, BaselineOutcome::Verified(_)));
//! # }
//! ```
//!
//! ## Drive a probe by hand
//!
//! The harness functions are thin. When a test needs something else, such as
//! two different code paths on the same backend, or extra handlers, build the
//! probes yourself:
//!
//! ```
//! # #[cfg(feature = "audit")] {
//! use bunsen::{
//!     audit::{
//!         AuditProbe,
//!         AuditStreamRecorder,
//!     },
//!     support::testing::{
//!         CpuBackend,
//!         backend_device,
//!     },
//! };
//! use burn::prelude::Tensor;
//!
//! let device = backend_device::<CpuBackend>();
//! let x: Tensor<CpuBackend, 1> = Tensor::arange(0..8, &device).float();
//!
//! let mut recorder = AuditStreamRecorder::default();
//! {
//!     let mut probe = AuditProbe::new(vec![&mut recorder]);
//!     probe.assert_eq_as::<f32>("2x", &(x.clone() * 2.0)).unwrap();
//! }
//!
//! let mut verifier = recorder.into_verifier();
//! {
//!     let mut probe = AuditProbe::new(vec![&mut verifier]);
//!     probe
//!         .assert_eq_as::<f32>("2x", &(x.clone() + x.clone()))
//!         .unwrap();
//! }
//! verifier.finish().unwrap();
//! # }
//! ```
//!
//! # Configuration
//!
//! ## Where baselines live
//!
//! The library never picks a reports location. Every call that touches the
//! disk takes it from the call site: [`audit_baseline`] and
//! [`SeriesReport::write_report`](reports::SeriesReport::write_report)
//! take a [`ReportsOptions`], and [`audit_baseline_at`] takes a path.
//!
//! A baseline named `name` for backend `B` is stored at
//!
//! ```text
//! {root}/{backend_label}/{name}.cbor
//! ```
//!
//! where [`backend_label`](reports::backend_label) is
//! `B::name(device)` made path-safe (`cubecl<wgpu<spirv>>` becomes
//! `cubecl_wgpu_spirv`). Baselines are per backend: a CPU baseline never
//! verifies a GPU run. To compare backends with each other, use
//! [`audit_across`]. `name` may contain `/` to group baselines, but must be
//! relative and must not contain `..`.
//!
//! ## A repository default
//!
//! A repository that wants one standard location writes a constructor for its
//! default options, and its tests call that:
//!
//! ```
//! # #[cfg(feature = "audit")] {
//! use bunsen::{
//!     audit::reports::{
//!         BaselineMode,
//!         ReportsOptions,
//!         cargo_target_dir,
//!     },
//!     errors::BunsenResult,
//! };
//!
//! /// This repository's reports: `{target}/my-reports`; mode from the env.
//! pub fn my_reports() -> BunsenResult<ReportsOptions> {
//!     let root = cargo_target_dir()
//!         .unwrap_or_else(std::env::temp_dir)
//!         .join("my-reports");
//!     Ok(ReportsOptions::new(root)
//!         .with_mode(BaselineMode::from_env("MY_REPORTS_MODE")?))
//! }
//! # }
//! ```
//!
//! [`cargo_target_dir`](reports::cargo_target_dir) is
//! `$CARGO_TARGET_DIR`, else the target directory holding the running test
//! binary, so reports land beside the build and stay out of git. To check
//! baselines *in* instead, point the root at a directory in the source tree,
//! e.g. under `env!("CARGO_MANIFEST_DIR")`.
//!
//! ## Record or verify
//!
//! [`BaselineMode`] says what [`audit_baseline`] does with a stored baseline:
//!
//! | Mode     | Baseline absent | Baseline present        |
//! |----------|-----------------|-------------------------|
//! | `Auto`   | record          | verify                  |
//! | `Record` | record          | record (replace it)     |
//! | `Verify` | error           | verify                  |
//!
//! `Auto` is the default. Use `Record` to accept an intended change, and
//! `Verify` in CI, so that a missing baseline fails rather than silently
//! becoming the new truth. [`BaselineMode::from_env`] reads the mode from an
//! environment variable that the repository names.
//!
//! ## Choosing a checkpoint
//!
//! | Method                                | Compares   | At                    |
//! |---------------------------------------|------------|-----------------------|
//! | [`assert_eq_as::<E>`]                 | exactly    | element type `E`      |
//! | [`assert_eq_cast`]                    | exactly    | a runtime [`DType`]   |
//! | [`assert_approx_eq_as::<F>`]          | tolerance  | float type `F`        |
//! | [`assert_approx_eq_cast`]             | tolerance  | a runtime [`FloatDType`] |
//!
//! Each takes a label and either a `&Tensor` or a `&TensorData`; the data is
//! converted to the comparison type before it is stored. Use exact checks for
//! values that are uploaded, copied or rearranged, and approximate checks for
//! values that are computed. Across backends, float arithmetic is rarely
//! bit-identical.
//!
//! The comparison type is part of the event: comparing at `f16` deliberately
//! drops precision, and a checkpoint that changes its type no longer matches
//! its recorded event. `BF16` is compared via `f16`.
//!
//! [`assert_eq_as::<E>`]: AuditProbe::assert_eq_as
//! [`assert_eq_cast`]: AuditProbe::assert_eq_cast
//! [`assert_approx_eq_as::<F>`]: AuditProbe::assert_approx_eq_as
//! [`assert_approx_eq_cast`]: AuditProbe::assert_approx_eq_cast
//!
//! # Details
//!
//! ## How events match
//!
//! A verifier compares an incoming event with the next expected one, in this
//! order ([`try_match_events`]):
//!
//! 1. **Params** must be equal: the same comparison kind and, for approximate
//!    checks, the same [`ToleranceDesc`], which holds both the float type and
//!    the [`TolerancePolicy`]. Policy equality is syntactic: `Balanced` does
//!    not match a `RelAbs` with the same numbers. A recorded tolerance cannot
//!    be loosened silently at verify time.
//! 2. **Shapes** must be equal: the same data keys, holding the same number of
//!    tensors of the same shapes.
//! 3. **Data** must match: exactly, with equal dtypes, or within the recorded
//!    tolerance at the recorded float type.
//!
//! The label, source location and timestamp are **not** compared. Matching
//! goes by position in the stream, so the *n*-th checkpoint of run B is
//! checked against the *n*-th of run A, whatever it is called. The header
//! appears in the error message, so a mismatch names the label and the
//! `file:line` of the checkpoint that failed. Keep the checkpoint sequence
//! deterministic: no checkpoints inside loops whose trip count depends on the
//! backend.
//!
//! ## Errors, not panics
//!
//! Checkpoints return [`BunsenResult`], and the first handler that fails stops
//! the event; the error carries a [`BunsenError::AssertionError`]. Use `?` to
//! stop the body at the first divergence, which is usually the interesting
//! one: later checkpoints only repeat it.
//!
//! [`AuditStreamVerifier`] reports both ways a stream can be the wrong length:
//! an event past the end is an error in `on_event`, and
//! [`finish`](AuditStreamVerifier::finish) reports expected events that
//! never arrived. Call `finish` when the run ends; [`audit_across`] and
//! [`audit_baseline`] do. The older in-memory pair,
//! [`AuditProbeVecRecorder`] / [`AuditProbeVecVerifier`], panics on an event
//! past the end and has no `finish`, so it cannot detect a short run. Prefer
//! the stream handlers.
//!
//! ## Several handlers
//!
//! [`AuditProbe::new`] takes any number of handlers, and calls each in order
//! for every event. A probe can record and verify at once, for example to
//! verify against a baseline while recording a fresh stream for inspection.
//! [`AuditProbe`] is itself a handler, so probes nest.
//!
//! ## Stream files
//!
//! [`save_audit_stream`] writes an [`AuditStreamFile`] as CBOR: a format tag
//! ([`AUDIT_STREAM_FORMAT`]), a version ([`AUDIT_STREAM_VERSION`]), and the
//! events as [`AuditEventRecord`]s, the serializable mirror of
//! [`AuditProbeEvent`]. [`load_audit_stream`] rejects a file with another tag
//! or version. A missing file is [`BunsenError::ResourceNotFound`]. Each data
//! map is written in key order, but timestamps are stored too, so two
//! recordings of the same run are equal as streams and not byte for byte.
//!
//! ## Custom handlers and event kinds
//!
//! A handler is any `Debug` type implementing
//! [`AuditProbeEventHandler::on_event`]; it sees each event as a borrowed
//! [`AuditProbeEventStub`] and calls
//! [`to_event`](AuditProbeEventView::to_event) to keep it. To read an
//! event's data map, use [`unpack_audit_probe_event_data!`], which unpacks one
//! or more events against a single exact pattern of keys and arities.
//!
//! ## Features
//!
//! The whole module is behind the `audit` feature. It turns on `testing`,
//! which provides the test backends and
//! [`seeded_tensor`](crate::support::testing::seeded_tensor) (and brings in
//! `rand`), and it brings in `ciborium` for the stream files.
//!
//! [`TensorData`]: burn::prelude::TensorData
//! [`DType`]: burn::tensor::DType
//! [`FloatDType`]: burn::tensor::FloatDType
//! [`TolerancePolicy`]: crate::burner::descriptors::TolerancePolicy
//! [`ToleranceDesc`]: crate::burner::descriptors::ToleranceDesc
//! [`BunsenResult`]: crate::errors::BunsenResult
//! [`BunsenError::AssertionError`]: crate::errors::BunsenError::AssertionError
//! [`BunsenError::ResourceNotFound`]: crate::errors::BunsenError::ResourceNotFound
//! [`ReportsOptions`]: reports::ReportsOptions
//! [`BaselineMode`]: reports::BaselineMode
//! [`BaselineMode::from_env`]: reports::BaselineMode::from_env

mod audit_body;
mod audit_event_record;
mod audit_probe;
mod audit_probe_event;
mod audit_stream;
mod event_data_unpack;
pub mod reports;
mod vec_handlers;

pub use audit_body::*;
pub use audit_event_record::*;
pub use audit_probe::*;
pub use audit_probe_event::*;
pub use audit_stream::*;
pub use event_data_unpack::*;
pub use vec_handlers::*;
