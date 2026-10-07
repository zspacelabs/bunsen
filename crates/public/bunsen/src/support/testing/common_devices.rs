use burn::tensor::Device;

/// The CPU device, `Flex`, for trivial plumbing tests.
///
/// Use it where a test checks mechanics rather than numbers: setup and
/// teardown, config round trips, shape bookkeeping. It is always present
/// under the `testing` feature, and it has no accelerator memory pool to
/// share, so such a test has no device memory to manage.
///
/// It is **not** the numerically trustworthy choice. A pass on the CPU says
/// nothing about the kernels an accelerator runs, and a test does not move
/// here to get better numbers. A test that does real tensor math uses
/// [`performance_device`], and compares within a tolerance. See
/// [Test devices](crate::support::testing#test-devices).
pub fn cpu_device() -> Device {
    Device::flex()
}

/// The device for tests that do tensor math: the best accelerator that this
/// build of bunsen enables.
///
/// # Selection
///
/// 1. With `BURN_DEVICE` set in the environment, burn's own choice,
///    `Device::default()`, which reads it.
/// 2. Otherwise the first backend feature enabled **on bunsen**, in this order:
///    1. `cuda`: `Device::cuda(0)`;
///    2. `metal`: `Device::metal(DeviceKind::DefaultDevice)`;
///    3. `vulkan`: `Device::vulkan(DeviceKind::DefaultDevice)`;
///    4. `wgpu`: `Device::wgpu(DeviceKind::DefaultDevice)`;
///    5. none of them: [`cpu_device`], which is `Flex`.
///
/// Only `bunsen/<backend>` moves the choice. A dependent that enables
/// `burn/wgpu` but not `bunsen/wgpu` still gets the CPU here.
///
/// # The CPU fallback is silent
///
/// Without a backend feature, this *is* [`cpu_device`]. A bare `cargo test`
/// builds and passes every test written against it, having run them all on
/// the CPU, so a regression that only an accelerator shows passes too. Run
/// tensor tests with a backend feature, e.g.
/// `cargo test -p bunsen --features wgpu`.
///
/// # Compare within a tolerance
///
/// Backends do not agree bit for bit, so a test compares computed floats
/// within a tolerance, not with exact equality. The rule and the assertions
/// for it are under [Test devices](crate::support::testing#test-devices).
pub fn performance_device() -> Device {
    if std::env::var_os("BURN_DEVICE").is_some() {
        return Device::default();
    }
    feature_performance_device()
}

/// The [`performance_device`] that bunsen's backend features select.
fn feature_performance_device() -> Device {
    cfg_select! {
        feature = "cuda" => Device::cuda(0),
        feature = "metal" => Device::metal(burn::tensor::DeviceKind::DefaultDevice),
        feature = "vulkan" => Device::vulkan(burn::tensor::DeviceKind::DefaultDevice),
        feature = "wgpu" => Device::wgpu(burn::tensor::DeviceKind::DefaultDevice),
        _ => cpu_device(),
    }
}
