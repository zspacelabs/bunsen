use burn::{
    prelude::Backend,
    tensor::backend::DeviceOps,
};

/// The CPU backend, `Flex`, for trivial plumbing tests.
///
/// Use it where a test checks mechanics rather than numbers: setup and
/// teardown, config round trips, shape bookkeeping. It is always present
/// under the `testing` feature, and it has no accelerator memory pool to
/// share, so such a test has no device memory to manage.
///
/// It is **not** the numerically trustworthy choice. A pass on the CPU says
/// nothing about the kernels an accelerator runs, and a test does not move
/// here to get better numbers. A test that does real tensor math uses
/// [`PerformanceBackend`], and compares within a tolerance. See
/// [Test backends](crate::support::testing#test-backends).
pub type CpuBackend = ::burn::backend::Flex;

/// Defines [`PerformanceBackend`] as `$backend`, so its docs are written once
/// for every arm of the `cfg_select!` below. `$selected` says which arm this
/// build took; it follows a heading, since rustfmt drops a blank doc line
/// that comes right before an attribute.
macro_rules! performance_backend {
    ($backend:ty, $selected:literal) => {
        /// The burn backend for tests that do tensor math: the best
        /// accelerator that this build of bunsen enables.
        ///
        /// # Selection
        ///
        /// The first backend feature enabled **on bunsen**, in this order,
        /// picks it:
        ///
        /// 1. `cuda`: `burn::backend::Cuda`;
        /// 2. `metal`: `burn::backend::Metal`;
        /// 3. `vulkan`: `burn::backend::Vulkan`;
        /// 4. `wgpu`: `burn::backend::Wgpu`;
        /// 5. none of them: [`CpuBackend`], which is `Flex`.
        ///
        /// Only `bunsen/<backend>` moves it. A dependent that enables
        /// `burn/wgpu` but not `bunsen/wgpu` still gets the CPU here.
        ///
        /// # This build
        #[doc = $selected]
        ///
        /// # The CPU fallback is silent
        ///
        /// Without a backend feature, this *is* [`CpuBackend`]. A bare
        /// `cargo test` builds and passes every test written against it,
        /// having run them all on the CPU, so a regression that only an
        /// accelerator shows passes too. Run tensor tests with a backend
        /// feature, e.g. `cargo test -p bunsen --features wgpu`.
        ///
        /// # Compare within a tolerance
        ///
        /// The same test runs on whichever backend a developer builds, and
        /// backends do not agree bit for bit. The CUDA backend, for one,
        /// compiles its kernels with fast math (`cubecl`'s CUDA runtime turns
        /// `fast_math` on). Assert computed floats within a tolerance, with
        /// [`assert_tensors_close`](crate::support::testing::assert_tensors_close)
        /// or
        /// [`assert_tensor_close_to_vec`](crate::support::testing::assert_tensor_close_to_vec),
        /// not with exact equality.
        ///
        /// See [Test backends](crate::support::testing#test-backends).
        pub type PerformanceBackend = $backend;
    };
}

cfg_select! {
    feature = "cuda" => {
        performance_backend!(::burn::backend::Cuda, "`Cuda`, from the `cuda` feature.");
    }
    feature = "metal" => {
        performance_backend!(::burn::backend::Metal, "`Metal`, from the `metal` feature.");
    }
    feature = "vulkan" => {
        performance_backend!(
            ::burn::backend::Vulkan,
            "`Vulkan`, from the `vulkan` feature."
        );
    }
    feature = "wgpu" => {
        performance_backend!(::burn::backend::Wgpu, "`Wgpu`, from the `wgpu` feature.");
    }
    _ => {
        performance_backend!(CpuBackend, "[`CpuBackend`]: no backend feature is on.");
    }
}
