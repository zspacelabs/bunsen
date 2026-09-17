use burn::prelude::Backend;

/// Releases `device`'s cached memory pages.
///
/// A `cargo test` binary runs every test in **one process**, so every test
/// shares one accelerator client — and with it one memory pool. `cubecl`'s
/// pools hand out slices of large pages and, by design, never return a page on
/// their own: the pool keeps the high-water mark so the next allocation of that
/// size is free. Within a single model that is exactly right. Across a suite of
/// independent tests, each loading its own model, it means the pool only ever
/// grows — until the card is full and some later, innocent test fails its next
/// allocation.
///
/// How fast it grows depends on the page sizes, and those are **not** the same
/// on every backend. `cubecl`'s wgpu runtime sizes its pages from
/// `max_storage_buffer_binding_size`, but its Vulkan/SPIR-V path sizes them
/// from a quarter of the device heap instead — so on a 24 GiB card the largest
/// pool allocates in 6 GiB pages rather than 2 GiB ones. A suite whose peak
/// working set is a couple of GiB can still exhaust the card, and it shows up
/// as `wgpu error: Out of Memory` from `Device::create_buffer` in whichever
/// test was running when the card filled — not in whichever test did the
/// allocating.
///
/// Only pages that are **entirely free** are returned, so this reclaims a
/// previous test's pool rather than the caller's own live tensors. It is a
/// no-op on backends without a pooled allocator, [`CpuBackend`] among them.
///
/// Prefer [`DeviceMemoryGuard`] at a test's top over calling this at its
/// bottom: a trailing call is skipped by the panic it would be most useful
/// after.
///
/// [`CpuBackend`]: crate::support::testing::CpuBackend
pub fn release_cached_device_memory<B: Backend>(device: &B::Device) {
    B::memory_cleanup(device);
}

/// Releases a device's cached memory pages when it drops.
///
/// Bind it at the top of a test that loads a model, so the pool that test grows
/// is handed back to the tests running after it:
///
/// ```
/// use bunsen::support::testing::{
///     CpuBackend,
///     DeviceMemoryGuard,
///     default_device,
/// };
///
/// type B = CpuBackend;
/// let device = default_device();
/// let _memory = DeviceMemoryGuard::<B>::new(&device);
///
/// // ... load a model, run it, assert on it ...
/// ```
///
/// Two binding rules, both of them about *when* the guard drops:
///
/// * Bind it to a named `_`-prefixed local, never to bare `_`. `let _ = ..`
///   drops the guard immediately, releasing the pool before the test has
///   allocated anything.
/// * Bind it **before** the tensors it is meant to reclaim, and in a `let` of
///   its own. Locals drop in reverse declaration order, so a guard declared
///   first drops last — after the models and tensors, when their pages are
///   finally free. A guard declared after them, or destructured out of the same
///   tuple as them, drops first and reclaims nothing.
///
/// [`release_cached_device_memory`] says why a shared pool needs this at all,
/// and what "release" does and does not reclaim.
///
/// Pair it with `#[serial]`. The guard bounds what a suite accumulates *over*
/// time; it does nothing about what it holds *at once*, and tests running
/// concurrently each hold their own working set. A dozen of those is enough to
/// exhaust a 24 GiB card even though no single one comes close.
pub struct DeviceMemoryGuard<B: Backend> {
    /// The device whose pool is released on drop.
    device: B::Device,
}

impl<B: Backend> DeviceMemoryGuard<B> {
    /// Guards `device`'s memory pool, releasing it when the guard drops.
    pub fn new(device: &B::Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl<B: Backend> Drop for DeviceMemoryGuard<B> {
    fn drop(&mut self) {
        release_cached_device_memory::<B>(&self.device);
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::{
        Tensor,
        TensorData,
    };

    use super::*;
    use crate::support::testing::{
        CpuBackend,
        default_device,
    };

    /// The release path runs, and leaves live tensors alone.
    ///
    /// `CpuBackend` has no pooled allocator, so this asserts the contract
    /// rather than any reclamation: a guard around live work is harmless.
    #[test]
    fn test_guard_leaves_live_tensors_intact() {
        let device = default_device();
        let tensor = Tensor::<CpuBackend, 1>::from_data(TensorData::from([1.0, 2.0]), &device);

        {
            let _memory = DeviceMemoryGuard::<CpuBackend>::new(&device);
            release_cached_device_memory::<CpuBackend>(&device);
        }

        assert_eq!(tensor.into_data().to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    }
}
