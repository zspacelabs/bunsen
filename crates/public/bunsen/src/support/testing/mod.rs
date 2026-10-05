//! # Test support
//!
//! What bunsen's own tests are written with, published for the tests of code
//! built on bunsen. This module exists only with the `testing` feature (and
//! in bunsen's own unit tests). The feature also turns on `flex`, the CPU
//! backend behind [`CpuBackend`].
//!
//! # Test backends
//!
//! A test names its backend through one of two aliases, not a concrete
//! backend:
//!
//! - [`PerformanceBackend`] is for anything that does tensor math. It names the
//!   best accelerator that this build of bunsen enables, by feature: `cuda`,
//!   then `metal`, then `vulkan`, then `wgpu`. With none of them it is
//!   [`CpuBackend`]. One suite then runs on whichever backend a developer
//!   builds, with no per-backend copies of the tests.
//! - [`CpuBackend`] is `Flex`, the CPU backend, always present under `testing`.
//!   It is for trivial plumbing tests: setup and teardown, config round trips,
//!   shape bookkeeping. It is not a numerical reference, and a test does not
//!   move to it to get trustworthy numbers.
//!
//! **The CPU fallback is silent.** Without a backend feature on bunsen,
//! `PerformanceBackend` *is* the CPU. A bare `cargo test` builds and passes
//! every test written against it without touching a GPU, so a regression that
//! only an accelerator shows passes too. Run tensor tests with a backend
//! feature: `cargo test -p bunsen --features wgpu`. A dependent forwards it
//! as `bunsen/wgpu`; enabling `burn/wgpu` alone leaves the alias on the CPU.
//!
//! **Tolerances, not bit-exactness.** A test written against
//! `PerformanceBackend` runs on whatever backend the developer has, and
//! backends do not agree bit for bit: kernels reduce in different orders, and
//! the CUDA backend compiles its kernels with fast math (`cubecl`'s CUDA
//! runtime turns `fast_math` on). Compare computed floats within a tolerance,
//! with [`assert_tensors_close`] or [`assert_tensor_close_to_vec`]. Keep exact
//! comparisons for values that are only uploaded, copied or rearranged.
//!
//! To check that two backends agree with each other, or that a result has not
//! drifted from a recorded run, use `bunsen::audit` (the `audit` feature).
//!
//! # Devices
//!
//! [`default_device`] and [`backend_device`] return one shared device per
//! device type, built on first use and cached for the life of the process.
//! Every test in a binary therefore uses the same device, and on an
//! accelerator the same client and memory pool. [`set_default_device`] pins
//! another device; [`reset_default_device`] drops the cached one.
//!
//! # Seeded inputs
//!
//! [`seeded_tensor`] draws its values on the host from a seeded RNG, then
//! uploads them, so a seed gives the same values on every backend.
//! `Tensor::random` does not: it uses the backend's own RNG. Use
//! `seeded_tensor` wherever two runs must see the same input: comparisons
//! across backends, stored baselines and golden values.
//!
//! # Device memory
//!
//! A `cargo test` binary runs all its tests in one process, so they share one
//! accelerator memory pool, and `cubecl`'s pools grow without giving pages
//! back on their own. A suite whose tests each load a model can fill the card,
//! and the out-of-memory error then lands in some later test. Bind a
//! [`DeviceMemoryGuard`] at the top of any test that loads a model, and mark
//! the test `#[serial]`. [`release_cached_device_memory`] is the call the
//! guard makes on drop, and its docs explain the page sizing.
//!
//! # Assertions
//!
//! - [`assert_tensors_close`]: two tensors of one shape, within a burn
//!   [`Tolerance`](burn::tensor::Tolerance).
//! - [`assert_tensor_close_to_vec`]: a tensor against a row-major host buffer
//!   of `f64`, within a `Tolerance`.
//! - [`assert_close_to_vec`]: two host slices, within an absolute tolerance.
//! - [`param_load_mapping`]: runs a parameter's load-path mapping, so a test
//!   can check which mapping a module attached.
//!
//! # Speech
//!
//! [`asr`] scores speech recognition: [`asr::text_error_rate`] is the word
//! error rate between two transcripts, after a light normalization.
//!
//! # Example
//!
//! A test of real tensor math, as it would sit in a `#[test] #[serial] fn`:
//!
//! ```
//! use bunsen::support::testing::{
//!     DeviceMemoryGuard,
//!     PerformanceBackend,
//!     assert_tensors_close,
//!     backend_device,
//!     seeded_tensor,
//! };
//! use burn::tensor::{
//!     Distribution,
//!     Tolerance,
//! };
//!
//! type B = PerformanceBackend;
//! let device = backend_device::<B>();
//! let _memory = DeviceMemoryGuard::<B>::new(&device);
//!
//! let x = seeded_tensor::<B, 2>(7, [4, 8], Distribution::Default, &device);
//! let y = (x.clone() * 3.0) / 3.0;
//! assert_tensors_close(&y, &x, Tolerance::default());
//! ```
pub mod asr;

mod common_assertions;
mod common_devices;
mod default_device;
mod device_memory;
mod seeded;

pub use common_assertions::*;
pub use common_devices::*;
pub use default_device::*;
pub use device_memory::*;
pub use seeded::*;
